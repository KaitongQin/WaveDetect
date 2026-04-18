from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, Qwen2Config, ResNetConfig
from .wavedetect import WaveDetectConfig, WaveDetect
from tqdm import tqdm


class WaveletCNN_api:
    def __init__(self, device: str = "cuda:1", ckpt_name=None):
        self.device = torch.device(device)
        self.ckpt_path = f'checkpoints/waveletcnn/{ckpt_name}/stage2_model.bin' # your wavedetect model path
        self.max_seq_len = 256
        self.base_model_name = 'model/qwen2_5-0_5b-base' # your Qwen-2.5-0.5b path
        self.cnn_config_path = 'model/config/resnet18_config.json' # your CNN config path

        # ===== Tokenizer & Config =====
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        base_cfg = Qwen2Config.from_pretrained(self.base_model_name)
        cnn_cfg = ResNetConfig.from_pretrained(self.cnn_config_path)

        cfg = WaveDetectConfig(
            base_model_name=self.base_model_name,
            base_model_config=base_cfg,
            cnn_config=cnn_cfg,
            num_scales=16
        )
        
        # ===== Model =====
        model = WaveDetect(cfg).to(self.device)
        model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        model.eval()
        self.model = model

    @torch.no_grad() 
    def inference(self, texts: list, batch_size: int = 32) -> list:
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Infer"):
                batch_texts = texts[i:i + batch_size]

                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )

                probs = torch.softmax(outputs.cnn_logits, dim=-1)
                pos_scores = probs[:, 1]

                predictions.extend(pos_scores.cpu().tolist())

        return predictions