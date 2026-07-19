"""
Fine-tune Resemble Enhance Denoiser (~5 epochs) from pretrained enhancer_stage2 weights
on VoiceBank-DEMAND, then evaluate SI-SDR / STOI / L1.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download
from pystoi import stoi
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from resemble_enhance.denoiser.denoiser import Denoiser
from resemble_enhance.denoiser.hparams import HParams

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "resemble-enhance" / "data" / "voicebank"
RUN = ROOT / "resemble-enhance" / "runs" / "denoiser_voicebank"
PRETRAINED = ROOT / "resemble-enhance" / "pretrained" / "enhancer_stage2"


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SDR. est/ref: (T,) or (B,T)."""
    if est.dim() == 1:
        est, ref = est.unsqueeze(0), ref.unsqueeze(0)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    s_ref = (ref * ref).sum(dim=-1, keepdim=True)
    proj = dot * ref / (s_ref + eps)
    noise = est - proj
    return 10 * torch.log10((proj * proj).sum(dim=-1) / ((noise * noise).sum(dim=-1) + eps))


def load_wav(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav).float()
    if sr != target_sr:
        t = AF.resample(t.unsqueeze(0), sr, target_sr).squeeze(0)
    peak = t.abs().max().clamp_min(1e-7)
    return t / peak


class PairDataset(Dataset):
    def __init__(self, split: str, hp: HParams, training: bool, max_items: int | None = None):
        self.fg_dir = DATA / "fg" / split
        self.noisy_dir = DATA / "noisy" / split
        self.ids = sorted(p.stem for p in self.fg_dir.glob("*.wav"))
        if max_items is not None:
            self.ids = self.ids[:max_items]
        self.hp = hp
        self.training = training
        self.seg = int(hp.training_seconds * hp.wav_rate)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        uid = self.ids[idx]
        clean = load_wav(self.fg_dir / f"{uid}.wav", self.hp.wav_rate)
        noisy = load_wav(self.noisy_dir / f"{uid}.wav", self.hp.wav_rate)
        n = min(clean.numel(), noisy.numel())
        clean, noisy = clean[:n], noisy[:n]
        if self.training:
            if n > self.seg:
                start = random.randint(0, n - self.seg)
                clean = clean[start : start + self.seg]
                noisy = noisy[start : start + self.seg]
            elif n < self.seg:
                pad = self.seg - n
                clean = F.pad(clean, (0, pad))
                noisy = F.pad(noisy, (0, pad))
        return {"clean": clean, "noisy": noisy, "id": uid}


def collate(batch: list[dict]) -> dict:
    # train: fixed length; eval: pad to max
    cleans = [b["clean"] for b in batch]
    noisys = [b["noisy"] for b in batch]
    ids = [b["id"] for b in batch]
    if all(c.numel() == cleans[0].numel() for c in cleans):
        return {
            "clean": torch.stack(cleans),
            "noisy": torch.stack(noisys),
            "id": ids,
        }
    T = max(c.numel() for c in cleans)
    clean = torch.stack([F.pad(c, (0, T - c.numel())) for c in cleans])
    noisy = torch.stack([F.pad(c, (0, T - c.numel())) for c in noisys])
    return {"clean": clean, "noisy": noisy, "id": ids}


def download_and_extract_pretrained(device: str) -> Denoiser:
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    ckpt = hf_hub_download(
        "ResembleAI/resemble-enhance",
        "enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt",
        local_dir=str(PRETRAINED.parent),
    )
    # hub may nest under local_dir/ResembleAI... or enhancer_stage2
    ckpt_path = Path(ckpt)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module = state["module"] if isinstance(state, dict) and "module" in state else state
    denoiser_sd = {
        k[len("denoiser.") :]: v for k, v in module.items() if k.startswith("denoiser.")
    }
    if not denoiser_sd:
        raise RuntimeError("No denoiser.* keys in enhancer_stage2 checkpoint")

    hp = HParams(
        fg_dir=DATA / "fg" / "train",
        bg_dir=DATA / "bg" / "train",
        rir_dir=DATA / "rir",
        batch_size_per_gpu=4,
        training_seconds=2.0,
        nj=0,
        max_steps=10_000,
        warmup_steps=100,
        min_lr=1e-5,
        max_lr=5e-5,
    )
    model = Denoiser(hp)
    missing, unexpected = model.load_state_dict(denoiser_sd, strict=False)
    print(f"Loaded pretrained denoiser | missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device), hp


@torch.no_grad()
def evaluate(model: Denoiser, loader: DataLoader, device: str, max_items: int | None = None) -> dict:
    model.eval()
    l1s, sdrs, stois = [], [], []
    n_done = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        pred = model(noisy)
        # align length
        T = min(pred.shape[-1], clean.shape[-1])
        pred, clean, noisy = pred[..., :T], clean[..., :T], noisy[..., :T]
        l1s.append(F.l1_loss(pred, clean).item())
        sdrs.extend(si_sdr(pred, clean).detach().cpu().tolist())

        # STOI @ 16k
        for i in range(pred.shape[0]):
            p = AF.resample(pred[i].cpu(), model.hp.wav_rate, 16000).numpy()
            c = AF.resample(clean[i].cpu(), model.hp.wav_rate, 16000).numpy()
            m = min(len(p), len(c))
            if m < 16000:  # stoi needs ~1s
                continue
            stois.append(float(stoi(c[:m], p[:m], 16000, extended=False)))
        n_done += pred.shape[0]
        if max_items is not None and n_done >= max_items:
            break

    # baseline noisy SI-SDR for reference — recompute quickly on last batch only omitted
    return {
        "l1": float(np.mean(l1s)) if l1s else float("nan"),
        "si_sdr": float(np.mean(sdrs)) if sdrs else float("nan"),
        "stoi": float(np.mean(stois)) if stois else float("nan"),
        "n": n_done,
    }


@torch.no_grad()
def baseline_noisy_metrics(loader: DataLoader, wav_rate: int, max_items: int = 100) -> dict:
    sdrs, stois = [], []
    n_done = 0
    for batch in loader:
        clean, noisy = batch["clean"], batch["noisy"]
        T = min(clean.shape[-1], noisy.shape[-1])
        clean, noisy = clean[..., :T], noisy[..., :T]
        sdrs.extend(si_sdr(noisy, clean).tolist())
        for i in range(clean.shape[0]):
            c = AF.resample(clean[i], wav_rate, 16000).numpy()
            n = AF.resample(noisy[i], wav_rate, 16000).numpy()
            m = min(len(c), len(n))
            if m >= 16000:
                stois.append(float(stoi(c[:m], n[:m], 16000, extended=False)))
        n_done += clean.shape[0]
        if n_done >= max_items:
            break
    return {
        "si_sdr": float(np.mean(sdrs)),
        "stoi": float(np.mean(stois)) if stois else float("nan"),
        "n": n_done,
    }


def train(
    epochs: int = 5,
    device: str = "cuda",
    max_train: int | None = 2000,
    max_eval: int = 100,
) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    model, hp = download_and_extract_pretrained(device)

    train_ds = PairDataset("train", hp, training=True, max_items=max_train)
    test_ds = PairDataset("test", hp, training=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=hp.batch_size_per_gpu,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate,
    )
    # eval uses shorter crops via training=False full files — subsample for speed
    eval_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    opt = torch.optim.Adam(model.parameters(), lr=hp.max_lr)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * epochs
    print(f"train_files={len(train_ds)} test_files={len(test_ds)}")
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} batch={hp.batch_size_per_gpu}")

    print(f"\n=== Baseline (noisy vs clean) on test[0:{max_eval}] ===")
    base = baseline_noisy_metrics(eval_dl, hp.wav_rate, max_items=max_eval)
    print(json.dumps(base, indent=2))

    print("\n=== Pretrained eval (before finetune) ===")
    pre = evaluate(model, eval_dl, device, max_items=max_eval)
    print(json.dumps(pre, indent=2))

    history = {"baseline_noisy": base, "pretrained": pre, "epochs": []}
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy, clean)
            loss = model.losses["l1"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
            opt.step()
            global_step += 1
            running.append(loss.item())
            if global_step % 20 == 0:
                pbar.set_postfix(loss=f"{np.mean(running[-20:]):.4f}")

        # epoch eval
        metrics = evaluate(model, eval_dl, device, max_items=max_eval)
        metrics["train_l1"] = float(np.mean(running))
        metrics["epoch"] = epoch
        history["epochs"].append(metrics)
        print(f"\n[epoch {epoch}] {json.dumps(metrics)}")

        ckpt_dir = RUN / "ds" / "G" / "default"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "module": model.state_dict(),
            "epoch": epoch,
            "hp": {
                "wav_rate": hp.wav_rate,
                "batch_size_per_gpu": hp.batch_size_per_gpu,
                "training_seconds": hp.training_seconds,
            },
        }
        torch.save(ckpt, ckpt_dir / "mp_rank_00_model_states.pt")
        (RUN / "ds" / "G" / "latest").write_text("default")
        from omegaconf import OmegaConf

        OmegaConf.save(
            {
                "wav_rate": hp.wav_rate,
                "n_fft": hp.n_fft,
                "win_size": hp.win_size,
                "hop_size": hp.hop_size,
                "num_mels": hp.num_mels,
                "batch_size_per_gpu": hp.batch_size_per_gpu,
                "training_seconds": hp.training_seconds,
                "fg_dir": str(hp.fg_dir),
                "bg_dir": str(hp.bg_dir),
                "rir_dir": str(hp.rir_dir),
            },
            str(RUN / "hparams.yaml"),
        )

    print("\n=== Final eval on full test set ===")
    final = evaluate(model, eval_dl, device, max_items=None)
    history["final"] = final
    print(json.dumps(final, indent=2))

    out = RUN / "metrics.json"
    out.write_text(json.dumps(history, indent=2))
    print(f"\nSaved metrics -> {out}")
    print(f"Checkpoint -> {RUN / 'ds' / 'G' / 'default' / 'mp_rank_00_model_states.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train", type=int, default=2000, help="Cap train samples (None=all)")
    parser.add_argument("--max-eval", type=int, default=100, help="Cap per-epoch eval samples")
    args = parser.parse_args()
    assert torch.cuda.is_available() or args.device == "cpu"
    train(
        epochs=args.epochs,
        device=args.device,
        max_train=args.max_train,
        max_eval=args.max_eval,
    )


if __name__ == "__main__":
    main()
