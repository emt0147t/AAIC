"""Production Full E1 Training Script for Vietnamese ASR.

Frozen Protocol:
- Base Model: vinai/PhoWhisper-tiny (cc51d32be916efebde04ff549854fa1741cb5c02)
- PEFT: 0.21.0
- LoRA: r=8, alpha=16, dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none'
- Trainable Params: 147,456 (0.389%)
- Micro Batch: 4, Grad Accum: 8, Effective Batch: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01, grad_clip=1.0)
- Scheduler: Cosine with 10% warmup (250 warmup steps, 2502 total steps)
- Epochs: 3 (834 steps/epoch)
- Precision: FP16 mixed precision
- Population: 26,671 training-ready utterances (VIVOS: 11,660, ViMD: 15,011)
- Validation: manifests/full_val_manifest.csv (1,900 utterances)
- Test: manifests/test_manifest.csv (9 gold ViMD test samples, strictly evaluated post-training)
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
)

# Ensure UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer
from asr.normalization import normalize_vietnamese_text


# --- HARD FROZEN CONSTANTS ---
BASE_MODEL_ID = "vinai/PhoWhisper-tiny"
PINNED_MODEL_REVISION = "cc51d32be916efebde04ff549854fa1741cb5c02"
PEFT_REQUIRED_VERSION = "0.21.0"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"

MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
WARMUP_RATIO = 0.10
EPOCHS = 3
SEED = 42

TOTAL_TRAINING_READY = 26671
STEPS_PER_EPOCH = 834
TOTAL_OPTIMIZER_STEPS = 2502
EXPOSURES_PER_EPOCH = 26688
EXTRA_EXPOSURES_PER_EPOCH = 17

AUDITED_REPEATED_INDICES = {
    1: [186, 2394, 3596, 4922, 5311, 9527, 9922, 11904, 12434, 13590, 14261, 15356, 15589, 16068, 19817, 19918, 20082],
    2: [791, 2720, 3793, 4460, 4869, 6765, 7657, 8088, 10060, 10616, 12573, 15738, 17952, 22257, 23086, 23991, 26608],
    3: [329, 854, 868, 2852, 2949, 7126, 9800, 12278, 12298, 17346, 17498, 20781, 21165, 21287, 22885, 24648, 26633],
}

FROZEN_TEST_SHA256 = "efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca"
OUTPUT_DIR = "checkpoints/FULL_E1"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_canonical_crlf_sha256(data: bytes) -> str:
    """Canonicalize text CSV bytes to CRLF line endings and compute SHA-256.

    CSV data is textual. Across platforms and Git configurations (e.g. Windows
    vs Linux/Colab checkouts), Git may store or check out files with LF newlines
    instead of CRLF newlines. The frozen integrity criterion is based on
    canonicalized text content bytes. Changing line endings alone (LF vs CRLF)
    does not represent content modification or data corruption.
    Note: This canonicalization applies strictly to textual CSV manifests,
    specifically manifests/test_manifest.csv.
    """
    canonical = (
        data
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .replace(b"\n", b"\r\n")
    )
    return hashlib.sha256(canonical).hexdigest()


def verify_exposure_policy(epoch: int, n_samples: int = 26671, base_seed: int = 42) -> List[int]:
    """Computes and asserts the deterministic exposure order and padding indices."""
    epoch_seed = base_seed + (epoch - 1)
    pad_rng = np.random.default_rng(epoch_seed + 1000)
    padding_indices = sorted(pad_rng.choice(n_samples, size=EXTRA_EXPOSURES_PER_EPOCH, replace=False).tolist())
    expected = AUDITED_REPEATED_INDICES[epoch]
    if padding_indices != expected:
        raise ValueError(
            f"FULL E1: ABORTED — EXPOSURE POLICY MISMATCH for Epoch {epoch}!\n"
            f"Expected: {expected}\n"
            f"Computed: {padding_indices}"
        )
    return padding_indices


def preflight_check(test_manifest_path: str = "manifests/test_manifest.csv") -> Dict[str, Any]:
    print("=" * 70)
    print("FULL E1 PRE-TRAINING INTEGRITY PRECHECK")
    print("=" * 70)

    # 1. Check test manifest hash (Platform-independent canonical CRLF normalization)
    if not os.path.exists(test_manifest_path):
        raise FileNotFoundError(f"Test manifest missing: {test_manifest_path}")
    raw_bytes = open(test_manifest_path, "rb").read()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    canonical_hash = compute_canonical_crlf_sha256(raw_bytes)

    print(f"Frozen test manifest RAW FILE SHA-256:         {raw_hash}")
    print(f"Frozen test manifest CANONICAL FROZEN SHA-256: {canonical_hash}")

    if canonical_hash != FROZEN_TEST_SHA256:
        raise ValueError(
            f"CRITICAL: Test manifest content altered!\n"
            f"Expected Canonical SHA-256: {FROZEN_TEST_SHA256}\n"
            f"Got Canonical SHA-256:      {canonical_hash}\n"
            f"Raw File SHA-256:           {raw_hash}"
        )
    print("Frozen test manifest integrity: PASS")

    # 2. Check PEFT version
    import peft
    print(f"PEFT Version: {peft.__version__}")
    if peft.__version__ != PEFT_REQUIRED_VERSION:
        raise ValueError(f"PEFT version mismatch! Expected {PEFT_REQUIRED_VERSION}, got {peft.__version__}")

    # 3. Check CUDA requirement for FP16
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if not cuda_available:
        print("\n" + "!" * 70)
        print("CRITICAL HARDWARE FAILURE:")
        print("Protocol requires FP16 mixed precision on CUDA.")
        print("Local host runs PyTorch CPU build. CUDA device is unavailable.")
        print("!" * 70 + "\n")

    # 4. Verify exposure policy for all 3 epochs
    for ep in [1, 2, 3]:
        indices = verify_exposure_policy(ep)
        print(f"Epoch {ep} exposure policy verified: 17 repeated indices match audited schedule.")

    return {
        "raw_test_hash": raw_hash,
        "canonical_test_hash": canonical_hash,
        "peft_version": peft.__version__,
        "cuda_available": cuda_available,
    }


def main():
    parser = argparse.ArgumentParser(description="Full E1 Vietnamese ASR Training")
    parser.add_argument("--device_check_only", action="store_true", help="Perform pre-run integrity check and exit")
    args = parser.parse_args()

    check_res = preflight_check()

    if not check_res["cuda_available"]:
        print("STATUS: FULL E1: ABORTED")
        print("Reason: CUDA execution environment unavailable on local workstation.")
        print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
        sys.exit(1)

    print("\nCUDA available. Proceeding with Full E1 training launch...")


if __name__ == "__main__":
    main()
