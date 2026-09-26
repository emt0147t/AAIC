"""Whisper Data Collator for Sequence-to-Sequence Training.

Handles:
- Robust padding of input Mel features (standard 80x3000 or variable time-dimension)
- Dynamic padding of tokenized text label sequences
- Masking of padding token IDs in labels with -100 so CrossEntropyLoss ignores them
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor


@dataclass
class WhisperDataCollatorWithPadding:
    """Data collator for Whisper sequence-to-sequence fine-tuning.

    Args:
        processor: Pretrained WhisperProcessor (feature_extractor + tokenizer).
        decoder_start_token_id: Optional start token ID (defaults to model's decoder_start_token_id).
    """

    processor: WhisperProcessor
    decoder_start_token_id: int = 50258

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
        # 1. Collate and pad input_features along time dimension
        raw_feats = [feature["input_features"] for feature in features]
        tensors = [torch.as_tensor(f, dtype=torch.float32) for f in raw_feats]

        max_time = max(t.shape[-1] for t in tensors)
        padded_feats = []
        for t in tensors:
            if t.shape[-1] < max_time:
                pad_width = max_time - t.shape[-1]
                t_padded = F.pad(t, (0, pad_width), mode="constant", value=0.0)
                padded_feats.append(t_padded)
            else:
                padded_feats.append(t)

        batch_input_features = torch.stack(padded_feats, dim=0)

        # 2. Collate and pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3. Replace pad token id with -100 to ignore loss on padding tokens
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        # 4. If bos/start token is appended at beginning of labels, strip it
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {
            "input_features": batch_input_features,
            "labels": labels,
        }
