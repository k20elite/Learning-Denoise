"""Export VoiceBank-DEMAND-16k to fg/ (clean) + bg/ (noise residual) + pairs/ (noisy|clean)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_from_disk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
HF_DIR = ROOT / "data" / "datasets" / "VoiceBank-DEMAND-16k"
OUT = ROOT / "resemble-enhance" / "data" / "voicebank"


def _read_audio(item: dict) -> tuple[np.ndarray, int]:
    if item.get("bytes"):
        wav, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32")
    else:
        wav, sr = sf.read(item["path"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def export_split(split: str, max_items: int | None = None) -> None:
    ds = load_from_disk(str(HF_DIR))[split]
    ds = ds.cast_column("clean", Audio(decode=False))
    ds = ds.cast_column("noisy", Audio(decode=False))
    n = len(ds) if max_items is None else min(max_items, len(ds))

    fg = OUT / "fg" / split
    bg = OUT / "bg" / split
    noisy_dir = OUT / "noisy" / split
    for d in (fg, bg, noisy_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n), desc=f"export {split}"):
        row = ds[i]
        uid = row["id"]
        clean, sr_c = _read_audio(row["clean"])
        noisy, sr_n = _read_audio(row["noisy"])
        assert sr_c == sr_n == 16000, (sr_c, sr_n)
        m = min(len(clean), len(noisy))
        clean, noisy = clean[:m], noisy[:m]
        residual = noisy - clean
        # avoid near-silent residuals
        if float(np.abs(residual).max()) < 1e-4:
            residual = np.random.randn(m).astype(np.float32) * 1e-3

        sf.write(fg / f"{uid}.wav", clean, sr_c)
        sf.write(bg / f"{uid}.wav", residual, sr_c)
        sf.write(noisy_dir / f"{uid}.wav", noisy, sr_c)


def main() -> None:
    export_split("train")
    export_split("test")
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
