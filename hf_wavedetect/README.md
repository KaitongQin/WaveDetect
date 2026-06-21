# WaveDetect Inference

目录结构：

```text
hf_wavedetect/
  wavedetect_hf.py
  stage2_model.bin
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
```

把训练好的模型放进来：

```bash
cp checkpoints/all-120901-weighted_loss_dynamic/stage2_model.bin hf_wavedetect/stage2_model.bin
```

再把训练时使用的 Qwen tokenizer/config 文件复制到同一个目录。

调用：

```python
from wavedetect_hf import WaveDetectPredictor

predictor = WaveDetectPredictor(model_dir="hf_wavedetect")
result = predictor.predict("需要检测的文本")
print(result)
```

直接运行示例：

```bash
python hf_wavedetect/wavedetect_hf.py
```
