import os
import torch
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from accelerate import Accelerator
from transformers import Qwen2Tokenizer, Qwen2Config, ResNetConfig, get_scheduler
from wavedetect import WaveDetect, WaveDetectConfig
from dataset import TextLabelDataset, CustomDataCollator, BalancedBatchSampler
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== Logging =====
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, "w", "utf-8"), logging.StreamHandler()],
    )
    return log_file


# ===== Train one stage =====
def train_one_stage(model, train_dataloader, val_dataloader, optimizer, scheduler, accelerator, cfg, stage_name, writer):
    logging.info(f"🚀 Start {stage_name}")
    model.train()

    global_step = 0
    for epoch in range(cfg.epochs_stage1 if stage_name == "stage1" else cfg.epochs_stage2):
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"[{stage_name}] Epoch {epoch+1}")):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # TensorBoard log each step's loss (only main process)
            if accelerator.is_main_process:
                global_step = epoch * len(train_dataloader) + step
                writer.add_scalar(f"{stage_name}/train_loss", loss.item(), global_step)
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar(f"{stage_name}/lr", lr, global_step)

            global_step += 1

        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1} | Avg train loss: {avg_train_loss:.4f}")

        # ---- Eval ----
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(**batch)
                if outputs.loss is not None:
                    eval_loss += outputs.loss.item()
        eval_loss /= len(val_dataloader)
        logging.info(f"Eval loss: {eval_loss:.4f}")

        # TensorBoard: each epoch log eval loss
        if accelerator.is_main_process:
            writer.add_scalar(f"{stage_name}/eval_loss", eval_loss, epoch + 1)

        model.train()

    if accelerator.is_main_process and stage_name == 'stage2':
        torch.save(accelerator.get_state_dict(model), os.path.join(cfg.save_path, f"{stage_name}_model.bin"))
    logging.info(f"✅ {stage_name} done.")


# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="Train WaveDetect with Accelerate")
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--cnn_config_path", type=str, default="model/config/resnet18_config.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs_stage1", type=int, default=2)
    parser.add_argument("--epochs_stage2", type=int, default=3)
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=1e-5)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--num_scales", type=int, default=16)
    parser.add_argument("--wavelet_type", type=str, default="morl")
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--pos_ratio", type=float, default=None, 
                        help="Ratio of positive samples in each batch. "
                             "If None (default), use standard shuffling (original ratio).")
    args = parser.parse_args()
    set_seed(42)
    log_file = setup_logger(args.save_path)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
    device = accelerator.device
    logging.info("⚡ Using accelerate for distributed training")

    if accelerator.is_main_process:
        tb_dir = os.path.join(args.save_path, "tensorboard_logs")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
    else:
        writer = None

    tokenizer = Qwen2Tokenizer.from_pretrained(args.base_model_name)
    base_config = Qwen2Config.from_pretrained(args.base_model_name)
    cnn_config = ResNetConfig.from_pretrained(args.cnn_config_path)

    model_cfg = WaveDetectConfig(
        base_model_name=args.base_model_name,
        base_model_config=base_config,
        cnn_config=cnn_config,
        n_labels=args.n_labels,
        wavelet=args.wavelet_type,
        num_scales=args.num_scales,
        device=device,
    )
    model = WaveDetect(model_cfg)

    # ===== Dataset =====
    train_dataset = TextLabelDataset(args.train_path, tokenizer, args.max_seq_len)
    val_dataset = TextLabelDataset(args.val_path, tokenizer, args.max_seq_len)
    data_collator = CustomDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_seq_len,
    )

    # Conditional Dataloader/Sampler creation
    train_sampler_len = 0 # To store the length for step calculation
    
    if args.pos_ratio is not None:
        # User specified a ratio, use BalancedBatchSampler
        logging.info(f"Using BalancedBatchSampler with batch_size={args.batch_size} and pos_ratio={args.pos_ratio}")
        balanced_sampler = BalancedBatchSampler(
            dataset=train_dataset,
            batch_size=args.batch_size,
            pos_ratio=args.pos_ratio
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
            collate_fn=data_collator,
            num_workers=4,
            worker_init_fn=lambda wid: np.random.seed(42 + wid) 
        )
        train_sampler_len = len(balanced_sampler)
    else:
        # Default behavior: use standard DataLoader with shuffling
        logging.info(f"Using standard DataLoader with shuffle=True (original ratio)")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=data_collator,
            num_workers=4,
            worker_init_fn=lambda wid: np.random.seed(42 + wid)
        )
        train_sampler_len = len(train_dataloader)
    
    # Validation dataloader remains standard
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=data_collator
    )

    # ===== Stage 1 =====
    for p in model.qwen_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage1)
    
    # Use the dynamic length
    num_training_steps = train_sampler_len * args.epochs_stage1
    logging.info(f"[Stage 1] Total training steps: {num_training_steps}")
    
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    train_one_stage(model, train_dataloader, val_dataloader, optimizer, scheduler, accelerator, args, "stage1", writer)

    # ===== Stage 2 =====
    model = accelerator.unwrap_model(model)
    for p in model.qwen_model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2)
    
    # <--- CHANGED: Use the dynamic length
    num_training_steps = train_sampler_len * args.epochs_stage2
    logging.info(f"[Stage 2] Total training steps: {num_training_steps}")

    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    train_one_stage(model, train_dataloader, val_dataloader, optimizer, scheduler, accelerator, args, "stage2", writer)

    if accelerator.is_main_process:
        writer.close()
        logging.info(f"TensorBoard logs saved at: {tb_dir}")
    logging.info("🎯 All stages completed.")
    logging.info(f"Logs saved at: {log_file}")


if __name__ == "__main__":
    main()
