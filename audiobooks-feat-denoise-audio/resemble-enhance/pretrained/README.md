# Pretrained weights

Download Resemble Enhance enhancer_stage2 (contains embedded denoiser):

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('ResembleAI/resemble-enhance','enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt', local_dir='.')"
```

Or run `scripts/finetune_denoiser.py` which downloads automatically.
