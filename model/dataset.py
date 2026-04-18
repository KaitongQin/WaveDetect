import torch
from torch.utils.data import Dataset
import json


class TextLabelDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = int(sample.get("label", 0))
        text = sample.get("text", "")

        self.tokenizer.padding_side = "right"
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0]          
        attention_mask = encoding["attention_mask"][0]

        return {
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask.to(torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

import torch
from torch.nn.utils.rnn import pad_sequence

class CustomDataCollator:
    def __init__(self, pad_token_id: int, max_length: int = None):
        """
        Args:
            pad_token_id: tokenizer.pad_token_id
            max_length: 限制最终 batch 的最大长度（可选）
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        # 拆分字段
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        # 截断到 max_length（如果指定）
        if self.max_length is not None:
            input_ids = [x[:self.max_length] for x in input_ids]
            attention_masks = [x[:self.max_length] for x in attention_masks]

        # 右侧 padding
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        # 再次强制截断（防止异常长度）
        if self.max_length is not None and input_ids_padded.size(1) > self.max_length:
            input_ids_padded = input_ids_padded[:, :self.max_length]
            attention_mask_padded = attention_mask_padded[:, :self.max_length]

        # import pdb;pdb.set_trace()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels,
        }

import torch
from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    """
    A BatchSampler designed for Supervised Contrastive (SupCon) learning.

    It ensures that each batch contains a fixed number of positive (label=1)
    and negative (label=0) samples.

    It uses an "under-sampling" strategy, where the total number of
    batches per epoch is determined by the class with the fewer samples.
    """
    def __init__(self, dataset, batch_size, pos_ratio=0.5, seed=42):
        """
        Args:
            dataset (Dataset): Your TextLabelDataset instance.
                               The sampler needs access to the dataset's underlying
                               data to check labels.
            batch_size (int): The total batch size.
            pos_ratio (float): The desired ratio of positive samples (label=1)
                               in each batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        # Calculate the number of positive and negative samples per batch
        self.pos_per_batch = int(batch_size * pos_ratio)
        self.neg_per_batch = batch_size - self.pos_per_batch
        
        if self.pos_per_batch == 0 or self.neg_per_batch == 0:
            raise ValueError(
                f"batch_size {batch_size} and pos_ratio {pos_ratio} "
                f"results in {self.pos_per_batch} positive samples and "
                f"{self.neg_per_batch} negative samples. Both must be > 0."
            )

        self.indices_label_0 = []
        self.indices_label_1 = []

        # Iterate through the dataset once to separate indices by label
        print("Initializing BalancedBatchSampler: sorting indices by label...")
        # We access dataset.data directly, assuming it was populated in __init__
        for i, item in enumerate(dataset.data):
            label = int(item.get("label", 0))
            if label == 1:
                self.indices_label_1.append(i)
            else:
                self.indices_label_0.append(i)
        
        print(f"Found {len(self.indices_label_1)} positive (1) and {len(self.indices_label_0)} negative (0) samples.")

        # Calculate the total number of batches we can create (under-sampling)
        # The epoch length is limited by the minority class
        self.num_neg_batches = len(self.indices_label_0) // self.neg_per_batch
        self.num_pos_batches = len(self.indices_label_1) // self.pos_per_batch
        self.num_batches = min(self.num_neg_batches, self.num_pos_batches)
        
        if self.num_batches == 0:
             raise ValueError(
                f"Cannot create any batches. "
                f"Need {self.pos_per_batch} positive and {self.neg_per_batch} negative samples per batch, "
                f"but only found {len(self.indices_label_1)} positive and {len(self.indices_label_0)} negative samples."
            )

        print(f"BalancedBatchSampler will produce {self.num_batches} batches per epoch.")

    def __iter__(self):
        """
        Yields a list of indices for one batch.
        """
        # Shuffle indices at the start of each epoch
        indices_0 = self.rng.permutation(self.indices_label_0)
        indices_1 = self.rng.permutation(self.indices_label_1)

        # Create iterators
        iter_0 = iter(indices_0)
        iter_1 = iter(indices_1)

        # Yield batches
        for _ in range(self.num_batches):
            batch = []
            
            # 1. Add positive samples
            for _ in range(self.pos_per_batch):
                batch.append(next(iter_1))
                
            # 2. Add negative samples
            for _ in range(self.neg_per_batch):
                batch.append(next(iter_0))
                
            # 3. Shuffle the batch internally
            # This prevents the model from always seeing positives first
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        """
        Returns the total number of batches per epoch.
        """
        return self.num_batches