# Audiobooks / Resemble Enhance Denoise

Training Denoiser (Resemble Enhance) on VoiceBank-DEMAND-16k.

## Runs

| Run | Path | Notes |
|-----|------|-------|
| Finetune pretrained | `resemble-enhance/runs/denoiser_voicebank` | extract `denoiser.*` from enhancer_stage2 |
| From-scratch warmup | `resemble-enhance/runs/denoiser_scratch` | random init + mix_fg_bg |

## Scripts (env: conda `cuda`)

```powershell
cd resemble-enhance
$env:PYTHONPATH = (Get-Location).Path
python scripts/prepare_voicebank.py
python scripts/finetune_denoiser.py --epochs 5 --max-train 2000
python scripts/train_denoiser_scratch.py --epochs 5 --max-train 2000
```

Charts + `metrics.json` are under each run folder.

Official `python -m resemble_enhance.denoiser.train` needs DeepSpeed/WavAugment (not used on Windows); scripts above mirror the recipe in pure PyTorch.
