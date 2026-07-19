# Datasets

`VoiceBank-DEMAND-16k` is not stored in git (~2.3GB).

```python
from datasets import load_dataset
ds = load_dataset('JacobLinCool/VoiceBank-DEMAND-16k')
ds.save_to_disk('VoiceBank-DEMAND-16k')
```
