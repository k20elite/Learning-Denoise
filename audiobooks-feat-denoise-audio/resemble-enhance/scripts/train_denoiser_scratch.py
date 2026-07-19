"""
Denoiser Warmup from scratch (Resemble Enhance recipe) on VoiceBank-DEMAND.

Mirrors `python -m resemble_enhance.denoiser.train` but without DeepSpeed/WavAugment
(unsupported on this Windows setup): random-init Denoiser + mix_fg_bg + L1 loss.

Usage:
  python scripts/train_denoiser_scratch.py --epochs 5 --max-train 2000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from omegaconf import OmegaConf
from pystoi import stoi
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from resemble_enhance.data.utils import mix_fg_bg
from resemble_enhance.denoiser.denoiser import Denoiser
from resemble_enhance.denoiser.hparams import HParams

ROOT = Path(__file__).resolve().parents[2]
VB = ROOT / "resemble-enhance" / "data" / "voicebank"
RUN = ROOT / "resemble-enhance" / "runs" / "denoiser_scratch"


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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


class MixDataset(Dataset):
    """Official-style: fg=clean, bg=noise residual; mix on the fly."""

    def __init__(self, split: str, hp: HParams, training: bool, max_items: int | None = None):
        self.fg_dir = VB / "fg" / split
        self.bg_dir = VB / "bg" / split
        self.noisy_dir = VB / "noisy" / split  # for paired eval only
        self.ids = sorted(p.stem for p in self.fg_dir.glob("*.wav"))
        if max_items is not None:
            self.ids = self.ids[:max_items]
        self.bg_ids = sorted(p.stem for p in self.bg_dir.glob("*.wav"))
        self.hp = hp
        self.training = training
        self.seg = int(hp.training_seconds * hp.wav_rate)

    def __len__(self) -> int:
        return len(self.ids)

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.numel()
        if n > self.seg:
            start = random.randint(0, n - self.seg) if self.training else 0
            return wav[start : start + self.seg]
        if n < self.seg:
            return F.pad(wav, (0, self.seg - n))
        return wav

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        uid = self.ids[idx]
        fg = load_wav(self.fg_dir / f"{uid}.wav", self.hp.wav_rate)
        if self.training:
            bg_uid = random.choice(self.bg_ids)
            bg = load_wav(self.bg_dir / f"{bg_uid}.wav", self.hp.wav_rate)
            fg = self._crop_or_pad(fg)
            bg = self._crop_or_pad(bg)
            # placeholder; real mix happens in train loop with random alpha
            noisy = fg
        else:
            bg = load_wav(self.bg_dir / f"{self.bg_ids[idx % len(self.bg_ids)]}.wav", self.hp.wav_rate)
            noisy = load_wav(self.noisy_dir / f"{uid}.wav", self.hp.wav_rate)
            n = min(fg.numel(), noisy.numel())
            fg, noisy, bg = fg[:n], noisy[:n], bg[:n]
        return {"fg": fg, "bg": bg, "noisy": noisy, "id": uid}


def collate(batch: list[dict]) -> dict:
    fgs = [b["fg"] for b in batch]
    bgs = [b["bg"] for b in batch]
    noisys = [b["noisy"] for b in batch]
    ids = [b["id"] for b in batch]
    if all(x.numel() == fgs[0].numel() for x in fgs):
        return {"fg": torch.stack(fgs), "bg": torch.stack(bgs), "noisy": torch.stack(noisys), "id": ids}
    T = max(x.numel() for x in fgs)
    return {
        "fg": torch.stack([F.pad(x, (0, T - x.numel())) for x in fgs]),
        "bg": torch.stack([F.pad(x, (0, T - x.numel())) for x in bgs]),
        "noisy": torch.stack([F.pad(x, (0, T - x.numel())) for x in noisys]),
        "id": ids,
    }


@torch.no_grad()
def evaluate(model: Denoiser, loader: DataLoader, device: str, max_items: int | None) -> dict:
    model.eval()
    l1s, sdrs, stois = [], [], []
    n_done = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        noisy = batch["noisy"].to(device)
        clean = batch["fg"].to(device)
        pred = model(noisy)
        T = min(pred.shape[-1], clean.shape[-1])
        pred, clean = pred[..., :T], clean[..., :T]
        l1s.append(F.l1_loss(pred, clean).item())
        sdrs.extend(si_sdr(pred, clean).cpu().tolist())
        for i in range(pred.shape[0]):
            p = AF.resample(pred[i].cpu(), model.hp.wav_rate, 16000).numpy()
            c = AF.resample(clean[i].cpu(), model.hp.wav_rate, 16000).numpy()
            m = min(len(p), len(c))
            if m >= 16000:
                stois.append(float(stoi(c[:m], p[:m], 16000, extended=False)))
        n_done += pred.shape[0]
        if max_items is not None and n_done >= max_items:
            break
    return {
        "l1": float(np.mean(l1s)) if l1s else float("nan"),
        "si_sdr": float(np.mean(sdrs)) if sdrs else float("nan"),
        "stoi": float(np.mean(stois)) if stois else float("nan"),
        "n": n_done,
    }


@torch.no_grad()
def baseline_noisy(loader: DataLoader, wav_rate: int, max_items: int) -> dict:
    sdrs, stois = [], []
    n_done = 0
    for batch in loader:
        clean, noisy = batch["fg"], batch["noisy"]
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
    return {"si_sdr": float(np.mean(sdrs)), "stoi": float(np.mean(stois)) if stois else float("nan"), "n": n_done}


def plot_charts(history: dict, step_losses: list[dict], out_dir: Path) -> None:
    epochs = [e["epoch"] for e in history["epochs"]]
    train_l1 = [e["train_l1"] for e in history["epochs"]]
    val_l1 = [e["l1"] for e in history["epochs"]]
    si_sdr_e = [e["si_sdr"] for e in history["epochs"]]
    stoi_e = [e["stoi"] for e in history["epochs"]]
    base = history["baseline_noisy"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    ax = axes[0]
    ax.plot(epochs, train_l1, "o-", lw=2, color="#2563eb", label="train L1")
    ax.plot(epochs, val_l1, "s-", lw=2, color="#dc2626", label="val L1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("Loss (from-scratch warmup)")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(epochs, si_sdr_e, "o-", lw=2, color="#059669", label="val SI-SDR")
    ax.axhline(base["si_sdr"], color="#6b7280", ls=":", label=f"noisy ({base['si_sdr']:.1f} dB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("Quality (SI-SDR) — no clf accuracy")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "charts_loss_sisdr.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=140)
    ax2.plot(epochs, stoi_e, "o-", lw=2, color="#7c3aed", label="val STOI")
    ax2.axhline(base["stoi"], color="#6b7280", ls=":", label=f"noisy ({base['stoi']:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("STOI")
    ax2.set_title("STOI (intelligibility)")
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    p2 = out_dir / "charts_stoi.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)

    # step-level loss curve
    if step_losses:
        fig3, ax3 = plt.subplots(figsize=(10, 3.5), dpi=140)
        steps = [s["step"] for s in step_losses]
        losses = [s["loss"] for s in step_losses]
        ax3.plot(steps, losses, color="#2563eb", alpha=0.35, lw=0.8, label="step L1")
        # moving average
        w = max(10, len(losses) // 50)
        if len(losses) >= w:
            kernel = np.ones(w) / w
            smooth = np.convolve(losses, kernel, mode="valid")
            ax3.plot(steps[w - 1 :], smooth, color="#1d4ed8", lw=2, label=f"MA({w})")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("L1 Loss")
        ax3.set_title("Train loss (per step)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        p3 = out_dir / "charts_step_loss.png"
        fig3.savefig(p3, bbox_inches="tight")
        plt.close(fig3)
        print(f"saved {p3}")

    print(f"saved {p1}")
    print(f"saved {p2}")


def train(epochs: int, device: str, max_train: int | None, max_eval: int) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "ds" / "G" / "default").mkdir(parents=True, exist_ok=True)

    hp = HParams(
        fg_dir=VB / "fg" / "train",
        bg_dir=VB / "bg" / "train",
        rir_dir=VB / "rir",
        batch_size_per_gpu=4,
        training_seconds=2.0,
        nj=0,
        max_steps=50_000,
        warmup_steps=200,
        min_lr=1e-5,
        max_lr=1e-4,
        mix_alpha_range=(0.2, 0.8),
    )
    # FROM SCRATCH — random weights
    model = Denoiser(hp).to(device)
    print(f"Initialized Denoiser from scratch on {device}")

    train_ds = MixDataset("train", hp, training=True, max_items=max_train)
    # eval uses real VoiceBank noisy|clean pairs
    eval_ds = MixDataset("test", hp, training=False, max_items=None)
    train_dl = DataLoader(
        train_ds, batch_size=hp.batch_size_per_gpu, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate
    )
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    opt = torch.optim.Adam(model.parameters(), lr=hp.max_lr)
    print(f"train_files={len(train_ds)} test_files={len(eval_ds)} steps/epoch={len(train_dl)}")

    print(f"\n=== Baseline noisy (test[:{max_eval}]) ===")
    base = baseline_noisy(eval_dl, hp.wav_rate, max_eval)
    print(json.dumps(base, indent=2))

    print("\n=== Scratch init eval (before train) ===")
    init_m = evaluate(model, eval_dl, device, max_eval)
    print(json.dumps(init_m, indent=2))

    history = {"mode": "from_scratch_warmup", "baseline_noisy": base, "init": init_m, "epochs": []}
    step_losses: list[dict] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            fg = batch["fg"].to(device)
            bg = batch["bg"].to(device)
            alpha = random.uniform(*hp.mix_alpha_range)
            mx = mix_fg_bg(fg, bg, alpha=alpha)
            pred = model(mx, fg)
            loss = model.losses["l1"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
            opt.step()
            global_step += 1
            lv = loss.item()
            running.append(lv)
            step_losses.append({"step": global_step, "epoch": epoch, "loss": lv})
            if global_step % 20 == 0:
                pbar.set_postfix(loss=f"{np.mean(running[-20:]):.4f}")

        metrics = evaluate(model, eval_dl, device, max_eval)
        metrics["train_l1"] = float(np.mean(running))
        metrics["epoch"] = epoch
        history["epochs"].append(metrics)
        print(f"\n[epoch {epoch}] {json.dumps(metrics)}")

        torch.save({"module": model.state_dict(), "epoch": epoch}, RUN / "ds" / "G" / "default" / "mp_rank_00_model_states.pt")
        (RUN / "ds" / "G" / "latest").write_text("default")
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
                "mix_alpha_range": list(hp.mix_alpha_range),
            },
            str(RUN / "hparams.yaml"),
        )

    print("\n=== Final eval full test ===")
    final = evaluate(model, eval_dl, device, None)
    history["final"] = final
    print(json.dumps(final, indent=2))

    (RUN / "metrics.json").write_text(json.dumps(history, indent=2))
    (RUN / "step_losses.json").write_text(json.dumps(step_losses))
    plot_charts(history, step_losses, RUN)
    print(f"\nSaved -> {RUN}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-eval", type=int, default=100)
    args = p.parse_args()
    train(args.epochs, args.device, args.max_train, args.max_eval)


if __name__ == "__main__":
    main()
