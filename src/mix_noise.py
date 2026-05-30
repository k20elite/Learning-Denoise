from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def _match_length(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(noise) < length:
        repeats = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, repeats)
    start = rng.randint(0, max(0, len(noise) - length))
    return noise[start : start + length]


def mix_speech_noise(
    speech_path: Path,
    noise_path: Path,
    snr_db: float,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or random.Random()

    speech, _ = librosa.load(speech_path, sr=TARGET_SR, mono=True)
    noise, _ = librosa.load(noise_path, sr=TARGET_SR, mono=True)
    noise = _match_length(noise, len(speech), rng)

    speech = speech.astype(np.float64)
    noise = noise.astype(np.float64)

    speech_peak = np.max(np.abs(speech)) + 1e-8
    speech = speech / speech_peak

    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2) + 1e-8
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)

    mixed = speech + scaled_noise
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8)

    return speech, mixed, scaled_noise


def generate_noisy_samples(
    speech_dir: Path,
    noise_dir: Path,
    output_dir: Path,
    num_samples: int = 10,
    snr_range: tuple[float, float] = (0.0, 10.0),
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    speech_files = sorted(speech_dir.glob("*.mp3"))
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not speech_files:
        raise FileNotFoundError(f"No mp3 files found in {speech_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No wav files found in {noise_dir}")

    chosen_speech = rng.sample(speech_files, k=min(num_samples, len(speech_files)))

    clean_dir = output_dir / "clean"
    noisy_dir = output_dir / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for idx, speech_path in enumerate(chosen_speech, start=1):
        noise_path = rng.choice(noise_files)
        snr_db = rng.uniform(*snr_range)

        clean, noisy, scaled_noise = mix_speech_noise(speech_path, noise_path, snr_db, rng)

        sample_id = f"sample_{idx:02d}"
        clean_out = clean_dir / f"{sample_id}.wav"
        noisy_out = noisy_dir / f"{sample_id}.wav"
        noise_ref_dir = output_dir / "noise_ref"
        noise_ref_dir.mkdir(parents=True, exist_ok=True)
        noise_ref_out = noise_ref_dir / f"{sample_id}.wav"

        sf.write(clean_out, clean, TARGET_SR)
        sf.write(noisy_out, noisy, TARGET_SR)
        sf.write(noise_ref_out, scaled_noise, TARGET_SR)

        manifest.append(
            {
                "sample_id": sample_id,
                "speech_file": speech_path.name,
                "noise_file": noise_path.name,
                "snr_db": round(snr_db, 2),
                "clean_path": str(clean_out),
                "noisy_path": str(noisy_out),
                "noise_ref_path": str(noise_ref_out),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
