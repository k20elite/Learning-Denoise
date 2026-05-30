from __future__ import annotations

import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_SR = 16_000


def _snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    noise = clean - estimate
    signal_power = np.mean(clean**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def _rmse(clean: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(clean), len(estimate))
    return float(np.sqrt(np.mean((clean[:length] - estimate[:length]) ** 2)))


def _correlation(clean: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(clean), len(estimate))
    clean = clean[:length]
    estimate = estimate[:length]
    if np.std(clean) < 1e-8 or np.std(estimate) < 1e-8:
        return 0.0
    return float(np.corrcoef(clean, estimate)[0, 1])


def compute_metrics(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    length = min(len(clean), len(estimate))
    clean = clean[:length]
    estimate = estimate[:length]
    return {
        "snr_db": round(_snr_db(clean, estimate), 2),
        "rmse": round(_rmse(clean, estimate), 6),
        "correlation": round(_correlation(clean, estimate), 4),
    }


def _plot_waveforms(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    sr: int,
    sample_id: str,
    output_path: Path,
) -> None:
    length = min(len(clean), len(noisy), len(denoised))
    time = np.arange(length) / sr

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = [
        ("Clean (gốc)", clean[:length]),
        ("Noisy (có nhiễu)", noisy[:length]),
        ("Denoised (sau xử lý)", denoised[:length]),
    ]
    for ax, (title, data) in zip(axes, labels):
        ax.plot(time, data, linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Waveform comparison - {sample_id}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_spectrograms(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    sr: int,
    sample_id: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    signals = [clean, noisy, denoised]
    titles = ["Clean", "Noisy", "Denoised"]

    for ax, data, title in zip(axes, signals, titles):
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        img = librosa.display.specshow(
            spec,
            sr=sr,
            hop_length=128,
            x_axis="time",
            y_axis="hz",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.suptitle(f"Spectrogram comparison - {sample_id}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_metrics_summary(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("snr_noisy_db", "snr_denoised_db", "SNR (dB)"),
        ("rmse_noisy", "rmse_denoised", "RMSE"),
        ("corr_noisy", "corr_denoised", "Correlation"),
    ]

    x = np.arange(len(df))
    width = 0.35

    for ax, (before_col, after_col, title) in zip(axes, metrics):
        ax.bar(x - width / 2, df[before_col], width, label="Noisy vs Clean")
        ax.bar(x + width / 2, df[after_col], width, label="Denoised vs Clean")
        ax.set_xticks(x)
        ax.set_xticklabels(df["sample_id"], rotation=45)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Evaluation metrics across samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_snr_improvement(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if val > 0 else "#e74c3c" for val in df["snr_improvement_db"]]
    ax.bar(df["sample_id"], df["snr_improvement_db"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("SNR improvement (dB)")
    ax.set_title("SNR improvement: Denoised vs Noisy (relative to Clean)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_samples(
    manifest: list[dict],
    denoised_dir: Path,
    evaluation_dir: Path,
) -> pd.DataFrame:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = evaluation_dir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for item in manifest:
        sample_id = item["sample_id"]
        clean, _ = librosa.load(item["clean_path"], sr=TARGET_SR, mono=True)
        noisy, _ = librosa.load(item["noisy_path"], sr=TARGET_SR, mono=True)
        denoised_path = denoised_dir / f"{sample_id}.wav"
        denoised, _ = librosa.load(denoised_path, sr=TARGET_SR, mono=True)

        noisy_metrics = compute_metrics(clean, noisy)
        denoised_metrics = compute_metrics(clean, denoised)

        rows.append(
            {
                "sample_id": sample_id,
                "speech_file": item["speech_file"],
                "noise_file": item["noise_file"],
                "target_snr_db": item["snr_db"],
                "snr_noisy_db": noisy_metrics["snr_db"],
                "snr_denoised_db": denoised_metrics["snr_db"],
                "snr_improvement_db": round(
                    denoised_metrics["snr_db"] - noisy_metrics["snr_db"], 2
                ),
                "rmse_noisy": noisy_metrics["rmse"],
                "rmse_denoised": denoised_metrics["rmse"],
                "corr_noisy": noisy_metrics["correlation"],
                "corr_denoised": denoised_metrics["correlation"],
            }
        )

        _plot_waveforms(
            clean,
            noisy,
            denoised,
            TARGET_SR,
            sample_id,
            per_sample_dir / f"{sample_id}_waveform.png",
        )
        _plot_spectrograms(
            clean,
            noisy,
            denoised,
            TARGET_SR,
            sample_id,
            per_sample_dir / f"{sample_id}_spectrogram.png",
        )

    df = pd.DataFrame(rows)
    df.to_csv(evaluation_dir / "metrics.csv", index=False)

    summary = {
        "num_samples": int(len(df)),
        "avg_snr_noisy_db": float(round(df["snr_noisy_db"].mean(), 2)),
        "avg_snr_denoised_db": float(round(df["snr_denoised_db"].mean(), 2)),
        "avg_snr_improvement_db": float(round(df["snr_improvement_db"].mean(), 2)),
        "avg_rmse_noisy": float(round(df["rmse_noisy"].mean(), 6)),
        "avg_rmse_denoised": float(round(df["rmse_denoised"].mean(), 6)),
        "avg_corr_noisy": float(round(df["corr_noisy"].mean(), 4)),
        "avg_corr_denoised": float(round(df["corr_denoised"].mean(), 4)),
    }
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_metrics_summary(df, evaluation_dir / "metrics_comparison.png")
    _plot_snr_improvement(df, evaluation_dir / "snr_improvement.png")

    return df
