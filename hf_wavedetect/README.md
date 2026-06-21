# WaveDetect Inference

Usage:

```python
from wavedetect_hf import WaveDetectPredictor

predictor = WaveDetectPredictor(model_dir="hf_wavedetect")

result = predictor.predict("This is a sample text to be detected.")
print(result)
```

Run the example directly:

```bash
python hf_wavedetect/wavedetect_hf.py
```