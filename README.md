# AudioCleaner

CLI denoise audio bằng spectral gating ([noisereduce](https://github.com/timsainb/noisereduce)). Thuần Python, không cần Docker, Rust hay GPU.

## Yêu cầu trên máy

| Thành phần | Bắt buộc | Ghi chú |
|------------|----------|---------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| pip | Có sẵn với Python | |
| ffmpeg | Khuyến nghị | Cần để đọc MP3/M4A. WAV/FLAC/OGG thường không cần |

**Cài ffmpeg (Windows):**

```powershell
winget install Gyan.FFmpeg
```

Sau khi cài, mở terminal mới để PATH có hiệu lực.

## Cài đặt

```powershell
git clone <repo-url>
cd AudioCleaner
.\setup.ps1
```

`setup.ps1` tạo `venv/` và cài dependencies từ `requirements.txt`.

**Cài tay (Linux/macOS):**

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Sử dụng

```powershell
.\venv\Scripts\python.exe denoise.py input.mp3
```

Output mặc định: `input_cleaned.wav` (cùng thư mục với file gốc).

**Chỉ định output:**

```powershell
.\venv\Scripts\python.exe denoise.py input.mp3 -o output.wav
```

**Tùy chọn:**

| Flag | Mô tả |
|------|--------|
| `--stationary` | Noise profile cố định (mặc định: non-stationary, phù hợp speech) |
| `--strength 0.8` | Độ mạnh denoise, 0–1 (mặc định: 1.0) |

**Ví dụ:**

```powershell
.\venv\Scripts\python.exe denoise.py "C:\Users\me\Desktop\recording.mp3"
.\venv\Scripts\python.exe denoise.py voice.wav -o voice_clean.wav --strength 0.85
```

## Định dạng hỗ trợ

Đọc: WAV, FLAC, OGG, MP3, M4A (MP3/M4A cần ffmpeg).  
Ghi: WAV.

## Cấu trúc project

```
AudioCleaner/
  denoise.py        # CLI chính
  requirements.txt  # Dependencies Python
  setup.ps1         # Setup nhanh trên Windows
```

## Ghi chú

- Thuật toán spectral gating, **không phải** model AI pretrained.
- Chất lượng tốt cho noise đơn giản / speech; noise phức tạp có thể còn artifact.
- `venv/` không commit — đã có trong `.gitignore`.
