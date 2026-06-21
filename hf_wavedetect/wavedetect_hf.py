import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    ResNetConfig,
    ResNetModel,
)


class WaveDetectConfig(PretrainedConfig):
    model_type = "wavedetect"

    def __init__(self, base_model_path, num_scales=16, kernel_size=128, n_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.n_labels = n_labels


def build_resnet18_config():
    return ResNetConfig(
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        layer_type="basic",
        hidden_act="relu",
        downsample_in_first_stage=False,
    )


class WaveletTransform(nn.Module):
    def __init__(self, num_scales=16, kernel_size=128):
        super().__init__()
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.sigma = 1.0
        self.freq0 = 5.0
        self.register_buffer("time", torch.linspace(-2, 2, steps=kernel_size))
        self.register_buffer("scales", torch.linspace(1.0, num_scales, num_scales))

    def morlet(self, scale):
        t = self.time / scale
        wave = torch.exp(-0.5 * (t / self.sigma) ** 2) * torch.cos(2 * math.pi * self.freq0 * t)
        wave = wave / (wave.abs().sum() + 1e-6)
        return wave.view(1, 1, -1)

    def forward(self, token_probs, attention_mask):
        batch_size, seq_len = token_probs.shape
        spectra = []

        for i in range(batch_size):
            valid_len = attention_mask[i].sum().long().item()
            x = token_probs[i, :valid_len].view(1, 1, -1)
            rows = []
            for scale in self.scales:
                kernel = self.morlet(scale.to(x.device)).to(x.dtype)
                rows.append(torch.nn.functional.conv1d(x, kernel, padding=self.kernel_size // 2))
            spectrum = torch.stack(rows, dim=2)
            if valid_len < seq_len:
                spectrum = torch.nn.functional.pad(spectrum, (0, seq_len - valid_len))
            spectra.append(spectrum)

        return torch.cat(spectra, dim=0).abs()


class WaveDetect(PreTrainedModel):
    config_class = WaveDetectConfig

    def __init__(self, config):
        super().__init__(config)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_config = AutoConfig.from_pretrained(config.base_model_path, trust_remote_code=True)

        self.qwen_model = AutoModelForCausalLM.from_config(base_config).to(dtype)
        self.wavelet = WaveletTransform(config.num_scales, config.kernel_size)
        self.cnn = ResNetModel(build_resnet18_config()).to(dtype)
        self.classifier = nn.Linear(512, config.n_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        labels = input_ids[:, 1:]
        mask = attention_mask[:, :-1]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_probs = torch.exp(log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)) * mask

        spectrum = self.wavelet(token_probs, mask).repeat(1, 3, 1, 1)
        cnn_out = self.cnn(spectrum, return_dict=True)
        pooled = cnn_out.last_hidden_state.mean(dim=[2, 3]).to(self.classifier.weight.dtype)
        return self.classifier(pooled)


def clean_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state_dict.items()}


class WaveDetectPredictor:
    def __init__(self, model_dir=None, device=None, max_length=1024):
        self.model_dir = Path(model_dir or Path(__file__).resolve().parent)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = WaveDetectConfig(base_model_path=str(self.model_dir))
        self.model = WaveDetect(config).to(self.device)
        state_dict = torch.load(self.model_dir / "wavedetect_all.bin", map_location="cpu")
        self.model.load_state_dict(clean_state_dict(state_dict), strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        logits = self.model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0]
        ai_score = float(probs[1].detach().cpu())

        return {
            "label": int(ai_score >= 0.5),
            "label_name": "ai_generated" if ai_score >= 0.5 else "human",
            "ai_score": ai_score,
        }


def predict(text, model_dir=None):
    return WaveDetectPredictor(model_dir=model_dir).predict(text)


def main():
    predictor = WaveDetectPredictor()
    text = "今天下午我去楼下买了一杯咖啡，然后把昨天没写完的报告补完了。"
    print(predictor.predict(text))


if __name__ == "__main__":
    main()
