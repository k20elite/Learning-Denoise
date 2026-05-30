# Báo cáo: Pipeline Denoise DSP cho Tiếng nói Tiếng Việt

**Dự án:** SLP301 — Vietnamese AI Audiobook Platform  
**Giai đoạn:** Tiền xử lý audio (denoise) trước STT/TTS  
**Phiên bản pipeline:** 1.0

---

## 1. Mục tiêu

Xây dựng pipeline xử lý tín hiệu số (DSP) để:

1. Tạo mẫu tiếng người có nhiễu (speech + noise) từ dữ liệu thực tế.
2. Khử nhiễu, giữ lại giọng nói người.
3. Đánh giá chất lượng so với tín hiệu sạch (ground truth) và trực quan hóa kết quả.

Pipeline phục vụ bước **denoising** trong proposal (`Data/proposal.pdf`), trước khi đưa audio vào Speech-to-Text và Text-to-Speech.

---

## 2. Cấu trúc project

```
SLP301/Project/
├── Data/
│   ├── mp3/              # Tiếng người (FPT Open Speech Dataset)
│   ├── esc50_noise/      # Nhiễu môi trường (ESC-50)
│   └── proposal.pdf
├── src/
│   ├── mix_noise.py      # Trộn speech + noise
│   ├── dsp_denoise.py    # Khử nhiễu DSP
│   └── evaluate.py       # Đánh giá & biểu đồ
├── docs/
│   └── BAO_CAO.md        # Tài liệu này
├── output/               # Kết quả chạy pipeline
├── run_pipeline.py       # Script chạy toàn bộ
└── requirements.txt
```

---

## 3. Luồng xử lý tổng thể

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data/mp3       │     │ Data/esc50_noise│     │  Tham số SNR    │
│  (tiếng người)  │     │  (nhiễu)        │     │  seed, số mẫu   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  mix_noise.py          │
                    │  → clean, noisy,       │
                    │    noise_ref           │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  dsp_denoise.py        │
                    │  (noisy + noise_ref)   │
                    │  → denoised            │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  evaluate.py           │
                    │  clean vs noisy vs       │
                    │  denoised → metrics, PNG │
                    └────────────────────────┘
```

**Lệnh chạy:**

```powershell
cd d:\SLP301\Project
pip install -r requirements.txt
python run_pipeline.py
```

---

## 4. Dữ liệu đầu vào

| Nguồn | Đường dẫn | Định dạng | Mô tả |
|--------|-----------|-----------|--------|
| FPT Open Speech | `Data/mp3/` | `.mp3` | Tiếng người tiếng Việt (~60k file) |
| ESC-50 Noise | `Data/esc50_noise/` | `.wav` | Nhiễu môi trường (~2000 file) |

Tất cả audio được chuẩn hóa về **mono, 16 kHz** trước khi xử lý.

---

## 5. Giai đoạn 1 — Trộn nhiễu (`mix_noise.py`)

### 5.1. Mục đích

Tạo tín hiệu `noisy = speech + scaled_noise` với SNR kiểm soát được, đồng thời lưu `clean` và `noise_ref` để đánh giá và hỗ trợ khử nhiễu.

### 5.2. Tham số

| Tham số | Giá trị mặc định | Vị trí | Ý nghĩa |
|---------|------------------|--------|---------|
| `TARGET_SR` | `16000` | `mix_noise.py` | Tần số lấy mẫu (Hz) |
| `num_samples` | `10` | `run_pipeline.py` | Số mẫu tạo mỗi lần chạy |
| `snr_range` | `(0.0, 10.0)` dB | `run_pipeline.py` | Khoảng SNR ngẫu nhiên khi trộn |
| `seed` | `42` | `run_pipeline.py` | Hạt giống RNG (tái lập kết quả) |

### 5.3. Công thức trộn nhiễu

1. Chuẩn hóa biên độ speech: `speech = speech / max(|speech|)`
2. Công suất: `P_s = mean(speech²)`, `P_n = mean(noise²)`
3. Công suất nhiễu mục tiêu: `P_n_target = P_s / 10^(SNR_dB/10)`
4. Scale nhiễu: `scaled_noise = noise × √(P_n_target / P_n)`
5. Hỗn hợp: `mixed = speech + scaled_noise`, chuẩn hóa peak

### 5.4. Xử lý độ dài nhiễu

- Nếu `len(noise) < len(speech)`: lặp nối tiếp (`np.tile`) rồi cắt.
- Nếu dài hơn: cắt ngẫu nhiên một đoạn bằng độ dài speech.

### 5.5. Output

| Thư mục / file | Nội dung |
|----------------|----------|
| `output/clean/` | Tiếng người sạch (đã chuẩn hóa) |
| `output/noisy/` | Tiếng người + nhiễu |
| `output/noise_ref/` | Nhiễu đã scale (dùng cho noise profile) |
| `output/manifest.json` | Metadata từng mẫu |

**Cấu trúc `manifest.json` (mỗi phần tử):**

```json
{
  "sample_id": "sample_01",
  "speech_file": "FPTOpenSpeechData_....mp3",
  "noise_file": "4-175845-A-43.wav",
  "snr_db": 7.41,
  "clean_path": "...",
  "noisy_path": "...",
  "noise_ref_path": "..."
}
```

---

## 6. Giai đoạn 2 — Khử nhiễu DSP (`dsp_denoise.py`)

### 6.1. Sơ đồ 12 bước

| Bước | Thao tác | Hàm / tham số |
|------|----------|----------------|
| 1 | Load audio | `librosa.load(..., sr=16000, mono=True)` |
| 2 | Mono + 16 kHz | `TARGET_SR = 16000` |
| 3 | Normalize | `peak = 0.95` |
| 4 | High-pass 80 Hz | Butterworth bậc 4, `cutoff=80` |
| 5 | Low-pass 7600 Hz | Butterworth bậc 4, `cutoff=7600` |
| 6 | Notch 50 Hz (nếu có ù) | `freq=50`, `Q=30`, điều kiện `_has_hum` |
| 7 | STFT | `n_fft=512`, `hop_length=128` |
| 8 | Noise profile | Từ `noise_ref` hoặc ước lượng từ tín hiệu |
| 9 | Spectral subtraction | `alpha=3.0`, `floor=0.01` |
| 10 | Wiener filter + Spectral gate + Smooth | Xem bảng dưới |
| 11 | Inverse STFT | `librosa.istft`, giữ phase gốc |
| 12 | Save audio | `soundfile`, WAV 16 kHz |

### 6.2. Bảng tham số DSP

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `TARGET_SR` | 16000 | Tần số lấy mẫu |
| `N_FFT` | 512 | Cửa sổ FFT |
| `HOP_LENGTH` | 128 | Bước nhảy STFT (~8 ms @ 16 kHz) |
| `NOISE_PROFILE_FRAMES` | 20 | Số frame đầu dùng ước lượng nhiễu (fallback) |
| `NOTCH_Q` | 30.0 | Độ nhọn bộ lọc notch 50 Hz |
| High-pass cutoff | 80 Hz | Loại DC, rumble |
| Low-pass cutoff | 7600 Hz | Giới hạn dải thoại |
| Notch frequency | 50 Hz | Khử tiếng ù lưới (VN) |
| Hum detect `threshold_ratio` | 8.0 | Tỷ lệ năng lượng 50 Hz / vùng lân cận |
| Normalize `peak` | 0.95 | Tránh clipping |
| Spectral sub `alpha` | 3.0 | Hệ số trừ nhiễu (càng lớn càng mạnh) |
| Spectral sub `floor` | 0.01 | Sàn âm phổ (tránh artifact âm) |
| Gate `threshold` | 1.8 | Ngưỡng \|X\|/noise_profile |
| Gate `attenuation` | 0.05 | Mức giảm bin bị nhiễu chiếm ưu thế |
| Smooth `time_kernel` | 5 | Làm mịn theo thời gian |
| Smooth `freq_kernel` | 3 | Làm mịn theo tần số |

### 6.3. Noise profile

**Ưu tiên — từ `noise_ref` (khi có trong manifest):**

- STFT của file `output/noise_ref/sample_XX.wav`
- Profile = `median(|STFT_noise|, axis=time)` → vector theo tần số

**Fallback — ước lượng từ tín hiệu noisy:**

1. Median 20 frame STFT đầu.
2. Median các frame “yên lặng” (năng lượng ≤ percentile 25).
3. Minimum statistics theo trục thời gian.
4. Lấy `max` của ba ước lượng trên.

### 6.4. Spectral subtraction

```
|Y(f,t)| = max(|X(f,t)| - α · N(f), floor · |X(f,t)|)
```

- `X`: phổ noisy, `N`: noise profile, `α = 3.0`, `floor = 0.01`.

### 6.5. Wiener filter

```
σ²_noise = N²
σ²_signal = max(|X|² - σ²_noise, 0)
G = σ²_signal / (σ²_signal + σ²_noise)
|Y| = |X| · G
```

### 6.6. Spectral gate

```
ratio = |X| / (N + ε)
mask = clip((ratio - threshold) / threshold, 0, 1)
|Y| = |X| · (attenuation + (1 - attenuation) · mask)
```

- Bin có `ratio` thấp → giảm mạnh (gần `attenuation = 0.05`).
- Bin có `ratio` cao → giữ giọng nói.

### 6.7. Phát hiện tiếng ù 50 Hz (`_has_hum`)

- Welch PSD, so sánh năng lượng dải 45–55 Hz với vùng 20–45 Hz và 55–120 Hz.
- Nếu tỷ lệ ≥ `8.0` → áp dụng notch 50 Hz.

### 6.8. Output denoise

| Đường dẫn | Mô tả |
|-----------|--------|
| `output/denoised/sample_XX.wav` | Audio sau khử nhiễu |

---

## 7. Giai đoạn 3 — Đánh giá (`evaluate.py`)

### 7.1. Metric (so với `clean`)

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **SNR** | `10·log10(P_signal / P_error)` với `error = clean - estimate` | Càng cao càng gần bản gốc |
| **RMSE** | `√(mean((clean - estimate)²))` | Sai số biên độ, càng thấp càng tốt |
| **Correlation** | Pearson giữa `clean` và `estimate` | Độ tương quan [−1, 1], càng gần 1 càng tốt |

Với mỗi mẫu tính:

- `noisy` vs `clean`
- `denoised` vs `clean`
- `snr_improvement_db = snr_denoised - snr_noisy`

### 7.2. Tham số trực quan hóa

| Tham số | Giá trị |
|---------|---------|
| Waveform figure size | 12 × 8 inch |
| Spectrogram figure size | 14 × 4 inch |
| Summary figure size | 14 × 4 inch |
| DPI | 150 |
| STFT hop (plot) | 128 |
| Colormap spectrogram | `magma` |

### 7.3. Output đánh giá

| File | Mô tả |
|------|--------|
| `output/evaluation/metrics.csv` | Bảng metric từng mẫu |
| `output/evaluation/summary.json` | Trung bình toàn bộ mẫu |
| `output/evaluation/metrics_comparison.png` | SNR / RMSE / Correlation |
| `output/evaluation/snr_improvement.png` | Cải thiện SNR từng mẫu |
| `output/evaluation/per_sample/*_waveform.png` | Dạng sóng 3 đường |
| `output/evaluation/per_sample/*_spectrogram.png` | Spectrogram 3 cột |

---

## 8. Tham số đường chạy (`run_pipeline.py`)

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `PROJECT_ROOT` | Thư mục chứa `run_pipeline.py` | Tự động |
| `DATA_DIR` | `PROJECT_ROOT/Data` | |
| `OUTPUT_DIR` | `PROJECT_ROOT/output` | Ghi đè khi chạy lại |
| `speech_dir` | `Data/mp3` | |
| `noise_dir` | `Data/esc50_noise` | |
| `num_samples` | `10` | Đổi trong `main(num_samples=...)` |
| `snr_range` | `(0.0, 10.0)` | dB |
| `seed` | `42` | |

---

## 9. Kết quả thực nghiệm (10 mẫu, seed=42)

| Metric | Noisy vs Clean | Denoised vs Clean | Thay đổi |
|--------|----------------|-------------------|----------|
| SNR trung bình | 6.10 dB | 7.83 dB | **+1.73 dB** |
| RMSE trung bình | 0.0740 | 0.0608 | giảm |
| Correlation trung bình | 0.8685 | 0.9034 | tăng |

**Mẫu tốt nhất:** `sample_01` (SNR 11.25 dB, corr 0.97), `sample_02` (SNR 10.29 dB, corr 0.95).

**Hạn chế:** Một số loại nhiễu ESC-50 chồng phổ với giọng nói khiến DSP khó tách hoàn toàn (`sample_09` cải thiện ít).

---

## 10. Phụ thuộc Python

| Package | Phiên bản tối thiểu |
|---------|---------------------|
| librosa | 0.10.0 |
| scipy | 1.11.0 |
| soundfile | 0.12.0 |
| matplotlib | 3.8.0 |
| numpy | 1.24.0 |
| pandas | 2.0.0 |

---

## 11. Mở rộng đề xuất

| Hướng | Mô tả |
|-------|--------|
| Tăng `num_samples` | Đổi `main(num_samples=100)` trong `run_pipeline.py` |
| SNR khác | Sửa `snr_range` trong `generate_noisy_samples` |
| Denoise không cần `noise_ref` | Bỏ `noise_ref_path` → dùng `_noise_profile_from_signal` |
| Khử nhiễu AI | DeepFilterNet / RNNoise cho chất lượng cao hơn DSP thuần |

---

## 12. Tài liệu liên quan

- `Data/proposal.pdf` — Đề cương Vietnamese AI Audiobook Platform
- FPT Open Speech: https://data.mendeley.com/datasets/k9sxg2twv4/4
- ESC-50: Environmental Sound Classification dataset
