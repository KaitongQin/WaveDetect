import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    ResNetModel,
)
from transformers.utils import ModelOutput


# ========== 1. Config ==========
class WaveDetectConfig(PretrainedConfig):
    model_type = "wavedetect"

    def __init__(
        self,
        base_model_name="qwen2",
        base_model_config=None,
        cnn_config=None,
        n_labels=2,
        wavelet="morl",
        num_scales=16,
        kernel_size=128,
        contrastive_weight=0.1,
        temperature=0.1,
        device='cpu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.base_model_config = base_model_config
        self.cnn_config = cnn_config
        self.n_labels = n_labels
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.device = device


@dataclass
class WaveDetectOutput(ModelOutput):
    loss: torch.FloatTensor = None
    ce_loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    cnn_logits: torch.FloatTensor = None
    cnn_embed: torch.FloatTensor = None
    base_model_logits: torch.FloatTensor = None
    token_probs: torch.FloatTensor = None
    wavelet_spectrum: torch.FloatTensor = None


# ========== 2. CWT ==========
class WaveletTransform(nn.Module):
    """
    Differentiable continuous wavelet transform (Morlet basis) with padding,
    cropping, and reconstruction.
    Inputs:
        token_probs (B, T)
        attention_mask (B, T) — optional but recommended
    Output:
        wavelet_spectrum (B, 1, num_scales, T)
    """
    def __init__(self, wavelet="morl", num_scales=32, kernel_size=128):
        super().__init__()
        assert wavelet == "morl", "Only Morlet wavelet is supported at this time"
        self.num_scales = num_scales
        self.kernel_size = kernel_size

        # Time axis in [-2, 2]
        self.register_buffer("time", torch.linspace(-2, 2, steps=kernel_size))
        self.sigma = 1.0
        self.freq0 = 5.0
        self.register_buffer("scales", torch.linspace(1.0, num_scales, num_scales))

    def morlet_wavelet(self, scale):
        t = self.time / scale
        wave = torch.exp(-0.5 * (t / self.sigma) ** 2) * torch.cos(2 * math.pi * self.freq0 * t)
        wave = wave / (wave.abs().sum() + 1e-6)
        return wave.view(1, 1, -1)

    def forward(self, token_probs: torch.FloatTensor, attention_mask: torch.FloatTensor = None):
        """
        token_probs: (B, T)
        attention_mask: (B, T)
        return: wavelet_spectrum (B, 1, num_scales, T)
        """
        B, T = token_probs.shape
        device = token_probs.device

        if attention_mask is not None:
            # Compute the valid sequence length for each sample
            valid_lengths = attention_mask.sum(dim=1).long()
        else:
            valid_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        # ========== Wavelet transform ==========
        responses = []
        for i in range(B):
            valid_len = valid_lengths[i].item()
            x_valid = token_probs[i, :valid_len].unsqueeze(0).unsqueeze(0)  # (1, 1, valid_len)
            res_list = []
            for s in self.scales:
                kernel = self.morlet_wavelet(s.to(device)).to(x_valid.dtype)
                pad = self.kernel_size // 2
                conv_real = torch.nn.functional.conv1d(x_valid, kernel, padding=pad)
                res_list.append(conv_real)
            # Stack responses into shape (1, 1, num_scales, valid_len)
            res = torch.stack(res_list, dim=2)
            
            # Restore the original length T by padding zeros on the right
            if valid_len < T:
                pad_size = T - valid_len
                res = torch.nn.functional.pad(res, (0, pad_size))
            responses.append(res)

        responses = torch.cat(responses, dim=0)  # (B, 1, num_scales, T)
        energy = responses.abs()

        return energy


# ========== 3. Loss ==========
def weighted_loss(logits, labels, config):
    """
    Dynamic inverse frequency weighting:
    classes with fewer samples in the batch receive larger weights.
    """

    ce_per_sample = nn.functional.cross_entropy(
        logits, labels, reduction="none"
    )

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    pos_count = pos_mask.sum().item()
    neg_count = neg_mask.sum().item()
    total = pos_count + neg_count

    if pos_count > 0:
        pos_w = total / (2.0 * pos_count)
    else:
        pos_w = 0.0

    if neg_count > 0:
        neg_w = total / (2.0 * neg_count)
    else:
        neg_w = 0.0

    pos_loss = ce_per_sample[pos_mask].mean() if pos_count > 0 else 0.0
    neg_loss = ce_per_sample[neg_mask].mean() if neg_count > 0 else 0.0

    loss = pos_w * pos_loss + neg_w * neg_loss

    return loss


# ========== 4. Main model ==========
class WaveDetect(PreTrainedModel):
    config_class = WaveDetectConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Base LM
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            dtype=torch.bfloat16
        )

        # Wavelet Transform
        self.wavelet = WaveletTransform(
            wavelet=config.wavelet,
            num_scales=config.num_scales,
            kernel_size=config.kernel_size,
        )

        # CNN Backbone
        self.cnn = ResNetModel(config.cnn_config)
        self.cnn = self.cnn.to(torch.bfloat16)

        # Classifier
        self.classifier = nn.Linear(config.cnn_config.hidden_sizes[-1], config.n_labels)

        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def compute_token_probs(self, logits, input_ids, attention_mask=None):
        """
        logits: (B, T, V)
        input_ids: (B, T)
        attention_mask: (B, T)
        return: token_probs (B, T)
        """
        labels = input_ids[:, 1:]
        logits = logits[:, :-1]
        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        token_probs = torch.exp(token_log_probs)  # (B, T)

        # Mask out padding positions
        if attention_mask is not None:
            token_probs = token_probs * attention_mask[:, :-1]

        return token_probs

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        """
        Forward:
            Qwen2 → logits → token_probs → WaveletTransform
            → CNN → embedding → classifier → loss
        """
        outputs = self.qwen_model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits_sequence = outputs.logits  # (B, T, V)
        # 1. Token probabilities
        token_probs = self.compute_token_probs(logits_sequence, input_ids, attention_mask)
        
        # 2. Wavelet Transform
        wavelet_spectrum = self.wavelet(token_probs, attention_mask[:, :-1])  # (B, 1, num_scales, T)
        wavelet_spectrum = wavelet_spectrum.repeat(1, 3, 1, 1)

        # print()
        # print(f"wavelet_spectrum: {wavelet_spectrum}")
        # print(f"wavelet_spectrum size: {wavelet_spectrum.size()}")

        # 3. CNN Backbone
        cnn_out = self.cnn(wavelet_spectrum, return_dict=True)
        pooled = cnn_out.last_hidden_state.mean(dim=[2, 3])
        pooled = pooled.to(self.classifier.weight.dtype)

        # 4. Classification
        logits = self.classifier(pooled)

        # 5. Losses
        loss, ce_loss, contrast_loss = None, None, None
        if labels is not None:
            loss = weighted_loss(logits=logits, labels=labels, config=self.config)
        
        if not return_dict:
            return loss, logits, pooled

        return WaveDetectOutput(
            loss=loss,
            ce_loss=ce_loss,
            contrastive_loss=contrast_loss,
            cnn_logits=logits,
            cnn_embed=pooled,
            base_model_logits=logits_sequence,
            token_probs=token_probs,
            wavelet_spectrum=wavelet_spectrum,
        )
