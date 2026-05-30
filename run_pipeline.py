from __future__ import annotations

import json
from pathlib import Path

from src.dsp_denoise import denoise_file
from src.evaluate import evaluate_samples
from src.mix_noise import generate_noisy_samples

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main(num_samples: int = 10) -> None:
    speech_dir = DATA_DIR / "mp3"
    noise_dir = DATA_DIR / "esc50_noise"

    print(f"[1/3] Creating {num_samples} noisy speech samples...")
    manifest = generate_noisy_samples(
        speech_dir=speech_dir,
        noise_dir=noise_dir,
        output_dir=OUTPUT_DIR,
        num_samples=num_samples,
        snr_range=(0.0, 10.0),
        seed=42,
    )
    print(f"      Saved to {OUTPUT_DIR / 'clean'} and {OUTPUT_DIR / 'noisy'}")

    denoised_dir = OUTPUT_DIR / "denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)

    print("[2/3] Running DSP denoise...")
    for item in manifest:
        sample_id = item["sample_id"]
        denoise_file(
            Path(item["noisy_path"]),
            denoised_dir / f"{sample_id}.wav",
            noise_ref_path=Path(item["noise_ref_path"]),
        )
        print(f"      Denoised {sample_id}")

    print("[3/3] Evaluating and plotting...")
    df = evaluate_samples(
        manifest=manifest,
        denoised_dir=denoised_dir,
        evaluation_dir=OUTPUT_DIR / "evaluation",
    )

    print("\n=== Summary ===")
    summary_path = OUTPUT_DIR / "evaluation" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\nMetrics table:\n{df.to_string(index=False)}")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(num_samples=10)
