from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16_000
N_FFT = 512
HOP_LENGTH = 128
NOISE_PROFILE_FRAMES = 20
NOTCH_Q = 30.0


def _normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val < 1e-8:
        return audio
    return audio / max_val * peak


def _has_hum(audio: np.ndarray, sr: int, threshold_ratio: float = 8.0) -> bool:
    freqs, psd = signal.welch(audio, fs=sr, nperseg=min(2048, len(audio)))
    band = (freqs >= 45) & (freqs <= 55)
    if not np.any(band):
        return False
    hum_energy = np.max(psd[band])
    neighbor = ((freqs >= 20) & (freqs < 45)) | ((freqs > 55) & (freqs <= 120))
    neighbor_energy = np.mean(psd[neighbor]) + 1e-12
    return hum_energy / neighbor_energy >= threshold_ratio


def _apply_notch(audio: np.ndarray, sr: int, freq: float = 50.0) -> np.ndarray:
    b, a = signal.iirnotch(freq, NOTCH_Q, sr)
    return signal.filtfilt(b, a, audio)


def _high_pass(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio)


def _low_pass(audio: np.ndarray, sr: int, cutoff: float = 7600.0) -> np.ndarray:
    nyquist = sr / 2
    cutoff = min(cutoff, nyquist * 0.95)
    sos = signal.butter(4, cutoff, btype="lowpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio)


def _preprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    audio = _normalize(audio)
    audio = _high_pass(audio, sr, cutoff=80.0)
    audio = _low_pass(audio, sr, cutoff=7600.0)
    if _has_hum(audio, sr):
        audio = _apply_notch(audio, sr, freq=50.0)
    return audio


def _noise_profile_from_reference(noise_ref: np.ndarray) -> np.ndarray:
    noise_stft = np.abs(librosa.stft(noise_ref, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return np.median(noise_stft, axis=1, keepdims=True)


def _noise_profile_from_signal(magnitude: np.ndarray) -> np.ndarray:
    initial = magnitude[:, : min(NOISE_PROFILE_FRAMES, magnitude.shape[1])]
    initial_profile = np.median(initial, axis=1, keepdims=True)

    frame_energy = np.mean(magnitude, axis=0)
    threshold = np.percentile(frame_energy, 25)
    quiet_mask = frame_energy <= threshold
    if np.any(quiet_mask):
        quiet_profile = np.median(magnitude[:, quiet_mask], axis=1, keepdims=True)
    else:
        quiet_profile = initial_profile

    min_stats = np.min(magnitude, axis=1, keepdims=True)
    return np.maximum(initial_profile, quiet_profile, min_stats)


def _spectral_subtraction(
    magnitude: np.ndarray,
    noise_profile: np.ndarray,
    alpha: float = 3.0,
    floor: float = 0.01,
) -> np.ndarray:
    subtracted = magnitude - alpha * noise_profile
    return np.maximum(subtracted, floor * magnitude)


def _wiener_filter(magnitude: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
    noise_var = noise_profile**2
    signal_var = np.maximum(magnitude**2 - noise_var, 0.0)
    gain = signal_var / (signal_var + noise_var + 1e-8)
    return magnitude * gain


def _spectral_gate(
    magnitude: np.ndarray,
    noise_profile: np.ndarray,
    threshold: float = 1.8,
    attenuation: float = 0.05,
) -> np.ndarray:
    ratio = magnitude / (noise_profile + 1e-8)
    mask = np.clip((ratio - threshold) / threshold, 0.0, 1.0)
    return magnitude * (attenuation + (1.0 - attenuation) * mask)


def _smooth_spectrum(magnitude: np.ndarray, time_kernel: int = 5, freq_kernel: int = 3) -> np.ndarray:
    if time_kernel > 1:
        kernel = np.ones(time_kernel) / time_kernel
        magnitude = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, magnitude)
    if freq_kernel > 1:
        kernel = np.ones(freq_kernel) / freq_kernel
        magnitude = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, magnitude)
    return magnitude


def denoise_audio(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    noise_ref: np.ndarray | None = None,
) -> np.ndarray:
    audio = _preprocess(audio, sr)
    if noise_ref is not None:
        noise_ref = _preprocess(noise_ref, sr)
        if len(noise_ref) != len(audio):
            if len(noise_ref) < len(audio):
                repeats = int(np.ceil(len(audio) / len(noise_ref)))
                noise_ref = np.tile(noise_ref, repeats)[: len(audio)]
            else:
                noise_ref = noise_ref[: len(audio)]

    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    if noise_ref is not None:
        noise_profile = _noise_profile_from_reference(noise_ref)
    else:
        noise_profile = _noise_profile_from_signal(magnitude)

    cleaned = _spectral_subtraction(magnitude, noise_profile, alpha=3.0, floor=0.01)
    cleaned = _wiener_filter(cleaned, noise_profile)
    cleaned = _spectral_gate(cleaned, noise_profile, threshold=1.8, attenuation=0.05)
    cleaned = _smooth_spectrum(cleaned, time_kernel=5, freq_kernel=3)

    denoised = librosa.istft(cleaned * np.exp(1j * phase), hop_length=HOP_LENGTH, length=len(audio))
    return _normalize(denoised)


def denoise_file(
    input_path: Path,
    output_path: Path,
    noise_ref_path: Path | None = None,
) -> Path:
    audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    noise_ref = None
    if noise_ref_path is not None and noise_ref_path.exists():
        noise_ref, _ = librosa.load(noise_ref_path, sr=TARGET_SR, mono=True)

    denoised = denoise_audio(audio, sr, noise_ref=noise_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, denoised, TARGET_SR)
    return output_path
