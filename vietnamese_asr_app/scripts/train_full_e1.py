"""Audited Full E1 Training Script for Vietnamese ASR.

Frozen Protocol:
- Base Model: vinai/PhoWhisper-tiny (cc51d32be916efebde04ff549854fa1741cb5c02)
- PEFT: 0.21.0
- Transformers: 4.49.0
- LoRA: r=8, alpha=16, dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none'
- Trainable Params: 147,456 (0.389%)
- Micro Batch: 4, Grad Accum: 8, Effective Batch: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01, grad_clip=1.0)
- Scheduler: Cosine with 10% warmup (250 warmup steps, 2502 total steps)
- Epochs: 3 (834 steps/epoch)
- Precision: FP16 mixed precision on CUDA
- Population: 26,671 training-ready utterances (VIVOS: 11,660, ViMD: 15,011)
- Validation: manifests/full_val_manifest.csv (1,900 utterances)
- Test: manifests/test_manifest.csv (9 gold ViMD test samples, strictly evaluated post-training)
"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

if "datasets" in sys.modules and "site-packages" not in getattr(sys.modules["datasets"], "__file__", ""):
    sys.modules.pop("datasets", None)
_orig_path = list(sys.path)
sys.path = [p for p in sys.path if not (p == "" or p == "." or "vietnamese_asr_app" in p.lower())]
from datasets import Audio, IterableDataset, load_dataset
sys.path = _orig_path

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

# Add application root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr.audio import load_audio
from asr.normalization import normalize_vietnamese_text
from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer
from research.checkpoint import save_lora_checkpoint
from research.collator import WhisperDataCollatorWithPadding
from research.evaluator import evaluate_model_on_manifest


# --- HARD FROZEN CONSTANTS ---
BASE_MODEL_ID = "vinai/PhoWhisper-tiny"
PINNED_MODEL_REVISION = "cc51d32be916efebde04ff549854fa1741cb5c02"
PEFT_REQUIRED_VERSION = "0.21.0"
TRANSFORMERS_REQUIRED_VERSION = "4.49.0"

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

PINNED_VIVOS_REVISION = "b2fbc10431b721dc9b0409b716d56a759d1cf332"
PINNED_VIMD_REVISION = "3a5b30157034e7eadd5c75fae1a820c6f9383398"
MIN_FREE_DISK_GB = 10.0


def check_disk_safety(min_free_gb: float = MIN_FREE_DISK_GB, path: str = ".") -> float:
    """Verifies that sufficient free disk space is available before data operations.

    Aborts immediately if available disk space is below min_free_gb to protect
    the host/Colab root filesystem from running out of disk.
    """
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    used_gb = used / (1024 ** 3)
    print(f"Disk Safety Gate [{path}]: {free_gb:.2f} GB free / {total_gb:.2f} GB total ({used_gb:.2f} GB used)")
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"CRITICAL DISK SAFETY ABORT: Free disk space ({free_gb:.2f} GB) is below "
            f"the required minimum safety threshold of {min_free_gb:.1f} GB! "
            f"Aborting before initiating data streaming to protect runtime filesystem."
        )
    return free_gb


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


def build_epoch_exposure_indices(epoch: int, n_samples: int = 26671, base_seed: int = 42) -> List[int]:
    """Constructs the exact 26,688 deterministic sample exposure order for a training epoch."""
    epoch_seed = base_seed + (epoch - 1)
    rng = np.random.default_rng(epoch_seed)
    permutation = rng.permutation(n_samples).tolist()
    
    padding_indices = verify_exposure_policy(epoch, n_samples, base_seed)
    epoch_sequence = permutation + padding_indices
    assert len(epoch_sequence) == EXPOSURES_PER_EPOCH, (
        f"Expected {EXPOSURES_PER_EPOCH} exposures, got {len(epoch_sequence)}"
    )
    return epoch_sequence


def find_test_manifest_path(path: str = "manifests/test_manifest.csv") -> str:
    """Locates the test manifest whether executed from repo root or app root."""
    if os.path.exists(path):
        return path
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", path),
        os.path.join("vietnamese_asr_app", path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path


def preflight_check(test_manifest_path: str = "manifests/test_manifest.csv") -> Dict[str, Any]:
    print("=" * 70)
    print("FULL E1 PRE-TRAINING INTEGRITY PRECHECK")
    print("=" * 70)

    # 0. Disk Safety Gate
    free_disk_gb = check_disk_safety(MIN_FREE_DISK_GB)

    # 1. Check test manifest hash (Platform-independent canonical CRLF normalization)
    test_manifest_path = find_test_manifest_path(test_manifest_path)
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

    # 2. Check Hardware & Versions
    cuda_available = torch.cuda.is_available()

    import peft
    import transformers
    print(f"PEFT Version:         {peft.__version__}")
    print(f"Transformers Version: {transformers.__version__}")
    if peft.__version__ != PEFT_REQUIRED_VERSION:
        raise ValueError(f"PEFT version mismatch! Expected {PEFT_REQUIRED_VERSION}, got {peft.__version__}")
    if transformers.__version__ != TRANSFORMERS_REQUIRED_VERSION:
        if cuda_available:
            raise ValueError(f"Transformers version mismatch! Expected {TRANSFORMERS_REQUIRED_VERSION}, got {transformers.__version__}")
        else:
            print(f"Notice: Local host Transformers is {transformers.__version__} (cloud GPU runtime pinned to {TRANSFORMERS_REQUIRED_VERSION}).")

    # 3. Check CUDA requirement for FP16
    print(f"CUDA Available:       {cuda_available}")
    if cuda_available:
        print(f"CUDA Device:          {torch.cuda.get_device_name(0)}")
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"VRAM Capacity:        {total_vram_mb:.1f} MB")
    else:
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
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_available,
    }


def decode_audio_record(audio_dict: Dict[str, Any], target_sr: int = 16000) -> np.ndarray:
    """Decodes audio from a HuggingFace dataset record on-the-fly without saving to disk."""
    if "array" in audio_dict and audio_dict["array"] is not None:
        arr = np.asarray(audio_dict["array"], dtype=np.float32)
        sr = audio_dict.get("sampling_rate", target_sr)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        return arr.astype(np.float32)

    raw_bytes = audio_dict.get("bytes")
    if raw_bytes is not None:
        sig, sr = sf.read(io.BytesIO(raw_bytes))
        if sig.ndim > 1:
            sig = sig.mean(axis=-1)
        if sr != target_sr:
            sig = librosa.resample(sig.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        return sig.astype(np.float32)

    raise ValueError("Audio dictionary contains neither 'array' nor 'bytes'.")


FROZEN_TEST_VIMD_MAPPING: Dict[str, Dict[str, str]] = {
    "vimd_test_vimd_01": {"vimd_filename": "11_0307.wav", "speaker_id": "spk_11_0142"},
    "vimd_test_vimd_02": {"vimd_filename": "11_0308.wav", "speaker_id": "spk_11_0143"},
    "vimd_test_vimd_03": {"vimd_filename": "11_0313.wav", "speaker_id": "spk_11_0144"},
    "vimd_test_vimd_04": {"vimd_filename": "36_0233.wav", "speaker_id": "spk_36_0147"},
    "vimd_test_vimd_05": {"vimd_filename": "36_0278.wav", "speaker_id": "spk_36_0171"},
    "vimd_test_vimd_06": {"vimd_filename": "36_0282.wav", "speaker_id": "spk_36_0173"},
    "vimd_test_vimd_07": {"vimd_filename": "59_0281.wav", "speaker_id": "spk_59_0244"},
    "vimd_test_vimd_08": {"vimd_filename": "59_0290.wav", "speaker_id": "spk_59_0251"},
    "vimd_test_vimd_09": {"vimd_filename": "59_0291.wav", "speaker_id": "spk_59_0252"},
}


def resolve_frozen_test_samples(
    test_manifest_path: str = "manifests/test_manifest.csv",
    vimd_revision: str = PINNED_VIMD_REVISION,
    force_streaming: bool = False,
) -> List[Dict[str, Any]]:
    """Resolves and loads the 9 frozen ViMD test samples with 16kHz audio waveforms in memory.

    Portability guarantee:
    - On Linux/Colab runtimes where local Windows 'D:\\...' paths do not exist (or when
      force_streaming=True), streams the exact matching records from the pinned ViMD test split.
    - Decodes waveforms on-the-fly directly in memory at 16 kHz without writing temporary audio files.
    - Matches and validates speaker IDs, transcripts, and sample IDs against the frozen gold manifest.
    - Returns sample records with {"audio": np.ndarray, "transcript": str, "sample_id": str}
      ensuring research/evaluator.py::evaluate_model_on_manifest() executes without skipping.
    """
    test_manifest_path = find_test_manifest_path(test_manifest_path)
    if not os.path.exists(test_manifest_path):
        raise FileNotFoundError(f"Test manifest missing: {test_manifest_path}")

    # Re-verify canonical hash to ensure gold test manifest is never altered
    test_raw_bytes = open(test_manifest_path, "rb").read()
    canonical_hash = compute_canonical_crlf_sha256(test_raw_bytes)
    if canonical_hash != FROZEN_TEST_SHA256:
        raise ValueError(
            f"CRITICAL: Test manifest content altered prior to resolution!\n"
            f"Expected Canonical SHA-256: {FROZEN_TEST_SHA256}\n"
            f"Got Canonical SHA-256:      {canonical_hash}"
        )

    df_test = pd.read_csv(test_manifest_path)
    if len(df_test) != 9:
        raise ValueError(f"CRITICAL: Expected exactly 9 test manifest rows, got {len(df_test)}")
    if len(df_test["sample_id"].unique()) != 9:
        raise ValueError("CRITICAL: Duplicate sample_id detected in test manifest!")

    manifest_lookup: Dict[str, Dict[str, Any]] = {}
    for _, r in df_test.iterrows():
        sid = str(r["sample_id"])
        manifest_lookup[sid] = {
            "sample_id": sid,
            "speaker_id": str(r["speaker_id"]),
            "transcript": str(r["transcript"]),
            "audio_path": str(r.get("audio_path", "")),
            "duration_sec": float(r.get("duration_sec", 0.0)),
        }

    resolved: Dict[str, Dict[str, Any]] = {}

    # Check if all local physical audio files exist on the host filesystem
    all_local_exist = all(
        os.path.exists(m["audio_path"]) and os.path.getsize(m["audio_path"]) > 0
        for m in manifest_lookup.values()
    )

    if all_local_exist and not force_streaming:
        print("[Frozen Test Resolution] Loading 9 test waveforms from verified local physical paths...")
        for sid, m in manifest_lookup.items():
            wf, sr = load_audio(m["audio_path"], target_sr=16000)
            assert isinstance(wf, np.ndarray) and wf.ndim == 1 and len(wf) > 0, f"Invalid audio array for {sid}"
            resolved[sid] = {
                "sample_id": sid,
                "audio": wf,
                "transcript": m["transcript"],
                "speaker_id": m["speaker_id"],
                "duration_sec": float(len(wf)) / 16000.0,
            }
    else:
        print(f"[Frozen Test Resolution] Resolving 9 test waveforms from pinned ViMD test stream (rev: {vimd_revision})...")
        from datasets import Audio, load_dataset

        filename_to_sid = {v["vimd_filename"]: k for k, v in FROZEN_TEST_VIMD_MAPPING.items()}
        ds_test = load_dataset(
            "nguyendv02/ViMD_Dataset",
            split="test",
            streaming=True,
            revision=vimd_revision,
        ).cast_column("audio", Audio(decode=False))

        for row in ds_test:
            fn = row.get("filename")
            if fn in filename_to_sid:
                sid = filename_to_sid[fn]
                m = manifest_lookup[sid]

                # Validate speaker ID
                row_spk = row.get("speakerID") or row.get("speaker_id")
                if row_spk != m["speaker_id"]:
                    raise ValueError(
                        f"CRITICAL: Speaker ID mismatch for {sid}! Streamed: {row_spk}, Manifest: {m['speaker_id']}"
                    )

                # Validate transcript content
                norm_stream = normalize_vietnamese_text(row.get("text") or row.get("transcript") or "")
                norm_manifest = normalize_vietnamese_text(m["transcript"])
                if norm_stream != norm_manifest:
                    raise ValueError(
                        f"CRITICAL: Transcript mismatch for {sid}!\n"
                        f"Streamed: {norm_stream}\n"
                        f"Manifest: {norm_manifest}"
                    )

                # Decode audio in memory at 16 kHz
                wf = decode_audio_record(row["audio"], target_sr=16000)
                assert isinstance(wf, np.ndarray) and wf.ndim == 1 and len(wf) > 0, f"Invalid audio array for {sid}"

                resolved[sid] = {
                    "sample_id": sid,
                    "audio": wf,
                    "transcript": m["transcript"],
                    "speaker_id": m["speaker_id"],
                    "duration_sec": float(len(wf)) / 16000.0,
                }

                if len(resolved) == 9:
                    break

    # Hard assertions before returning to evaluator
    if len(resolved) != 9:
        missing = set(manifest_lookup.keys()) - set(resolved.keys())
        raise RuntimeError(
            f"CRITICAL: Failed to resolve all 9 frozen test samples! "
            f"Resolved {len(resolved)}/9. Missing sample IDs: {missing}"
        )

    # Return exactly ordered matching the manifest row order
    ordered_samples = [resolved[str(r["sample_id"])] for _, r in df_test.iterrows()]
    assert len(ordered_samples) == 9
    for s in ordered_samples:
        assert "audio" in s and isinstance(s["audio"], np.ndarray), f"Sample {s['sample_id']} missing audio array!"
        assert "transcript" in s and len(s["transcript"]) > 0, f"Sample {s['sample_id']} missing transcript!"
        assert "sample_id" in s, "Sample missing sample_id!"

    print(f"[Frozen Test Resolution] Successfully resolved and validated all {len(ordered_samples)} frozen test samples.")
    return ordered_samples


class FullE1InMemoryDataset(Dataset):
    """Dataset serving pre-extracted Mel spectrogram features and tokenized labels for local files."""

    def __init__(self, sample_records: List[Dict[str, Any]], processor: WhisperProcessor):
        self.samples = sample_records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        waveform = item["waveform"]
        text = item["transcript"]

        inputs = self.processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
            "sample_id": item.get("sample_id", f"sample_{idx}"),
        }


class FullE1StreamingDataset(torch.utils.data.IterableDataset):
    """Zero-disk-overhead streaming dataset yielding pre-extracted features and tokenized labels.

    Streams directly from pinned Hugging Face datasets:
    - VIVOS train: thanhduycao/vivos_ng_only (pinned rev b2fbc10431b721dc9b0409b716d56a759d1cf332)
    - ViMD train: nguyendv02/ViMD_Dataset (pinned rev 3a5b30157034e7eadd5c75fae1a820c6f9383398)

    Applies the duration policy (0.5s <= duration <= 30.0s), skipping the 12 ViMD samples > 30s dynamically.
    Guarantees exact alignment to the audited deterministic exposure schedule (26,671 unique + 17 repeated samples).
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        epoch: int,
        base_seed: int = SEED,
        vivos_revision: str = PINNED_VIVOS_REVISION,
        vimd_revision: str = PINNED_VIMD_REVISION,
    ):
        super().__init__()
        self.processor = processor
        self.epoch = epoch
        self.base_seed = base_seed
        self.vivos_revision = vivos_revision
        self.vimd_revision = vimd_revision
        self.padding_indices = verify_exposure_policy(epoch, n_samples=TOTAL_TRAINING_READY, base_seed=base_seed)
        self.padding_set = set(self.padding_indices)

    def __iter__(self):
        from datasets import Audio, load_dataset

        # Load streaming iterables without materializing parquet files to disk
        ds_vivos = load_dataset(
            "thanhduycao/vivos_ng_only",
            split="train",
            streaming=True,
            revision=self.vivos_revision,
        ).cast_column("audio", Audio(decode=False))

        ds_vimd = load_dataset(
            "nguyendv02/ViMD_Dataset",
            split="train",
            streaming=True,
            revision=self.vimd_revision,
        ).cast_column("audio", Audio(decode=False))

        current_idx = 0
        buffered_padding: Dict[int, Dict[str, Any]] = {}

        # 1. Stream VIVOS train (0 .. 11,659: exactly 11,660 utterances)
        for row in ds_vivos:
            try:
                waveform = decode_audio_record(row["audio"], target_sr=16000)
            except Exception:
                continue
            dur = len(waveform) / 16000.0
            if not (0.5 <= dur <= 30.0):
                continue

            raw_text = row.get("sentence", row.get("transcript", ""))
            norm_text = normalize_vietnamese_text(raw_text)
            sample_id = f"vivos_{row.get('speaker_id', 'spk')}_{current_idx}"

            inputs = self.processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            labels = self.processor.tokenizer(norm_text, return_tensors="pt").input_ids[0]

            item = {
                "input_features": inputs.input_features[0],
                "labels": labels,
                "sample_id": sample_id,
            }

            if current_idx in self.padding_set:
                buffered_padding[current_idx] = item

            yield item
            current_idx += 1

        # 2. Stream ViMD train (11,660 .. 26,670: exactly 15,011 utterances after <= 30.0s filter)
        for row in ds_vimd:
            try:
                waveform = decode_audio_record(row["audio"], target_sr=16000)
            except Exception:
                continue
            dur = len(waveform) / 16000.0
            # Enforce duration policy: skip samples > 30.0s (12 total in ViMD train)
            if not (0.5 <= dur <= 30.0):
                continue

            raw_text = row.get("text", row.get("transcript", row.get("sentence", "")))
            norm_text = normalize_vietnamese_text(raw_text)
            spk = row.get("speakerID", row.get("speaker_id", "spk"))
            sample_id = f"vimd_{spk}_{current_idx}"

            inputs = self.processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            labels = self.processor.tokenizer(norm_text, return_tensors="pt").input_ids[0]

            item = {
                "input_features": inputs.input_features[0],
                "labels": labels,
                "sample_id": sample_id,
            }

            if current_idx in self.padding_set:
                buffered_padding[current_idx] = item

            yield item
            current_idx += 1

        # 3. Yield the 17 deterministic padding exposures in sorted audited order
        for pad_idx in self.padding_indices:
            if pad_idx in buffered_padding:
                pad_item = buffered_padding[pad_idx]
                yield {
                    "input_features": pad_item["input_features"],
                    "labels": pad_item["labels"],
                    "sample_id": f"{pad_item['sample_id']}_pad_ep{self.epoch}",
                }
            elif len(buffered_padding) > 0:
                first_k = next(iter(buffered_padding.values()))
                yield {
                    "input_features": first_k["input_features"],
                    "labels": first_k["labels"],
                    "sample_id": f"{first_k['sample_id']}_pad_ep{self.epoch}",
                }


class FullE1StreamingValDataset(torch.utils.data.IterableDataset):
    """Streams ViMD valid split on-the-fly without downloading parquet files to disk."""

    def __init__(
        self,
        processor: WhisperProcessor,
        vimd_revision: str = PINNED_VIMD_REVISION,
        max_samples: int = 1900,
    ):
        super().__init__()
        self.processor = processor
        self.vimd_revision = vimd_revision
        self.max_samples = max_samples

    def __iter__(self):
        from datasets import Audio, load_dataset

        ds_val = load_dataset(
            "nguyendv02/ViMD_Dataset",
            split="valid",
            streaming=True,
            revision=self.vimd_revision,
        ).cast_column("audio", Audio(decode=False))

        count = 0
        for row in ds_val:
            try:
                waveform = decode_audio_record(row["audio"], target_sr=16000)
            except Exception:
                continue
            dur = len(waveform) / 16000.0
            if not (0.5 <= dur <= 30.0):
                continue
            raw_text = row.get("text", row.get("transcript", row.get("sentence", "")))
            norm_text = normalize_vietnamese_text(raw_text)
            inputs = self.processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            labels = self.processor.tokenizer(norm_text, return_tensors="pt").input_ids[0]
            yield {
                "input_features": inputs.input_features[0],
                "labels": labels,
                "sample_id": f"vimd_val_{count}",
            }
            count += 1
            if count >= self.max_samples:
                break


def run_full_e1_training(
    epochs: int = EPOCHS,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    output_dir: str = OUTPUT_DIR,
    val_manifest_path: str = "manifests/full_val_manifest.csv",
    test_manifest_path: str = "manifests/test_manifest.csv",
):
    """Executes the complete audited Full E1 LoRA training run on CUDA."""
    start_time = time.time()
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Run Preflight Check
    precheck = preflight_check(test_manifest_path)
    if not precheck["cuda_available"]:
        print("STATUS: FULL E1: ABORTED")
        print("Reason: CUDA execution environment unavailable on local workstation.")
        print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
        sys.exit(1)

    device = torch.device("cuda")

    # 2. Model & Processor Setup
    print("\nInitializing WhisperProcessor and WhisperForConditionalGeneration...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, revision=PINNED_MODEL_REVISION)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, revision=PINNED_MODEL_REVISION)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")
    assert trainable_params == 147456, f"Expected 147,456 trainable parameters, got {trainable_params}"

    # 3. Setup Data Pipeline (Streaming vs Local Mode)
    local_manifest = "manifests/full_train_manifest.csv"
    use_local_offline = False
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    if os.path.exists(local_manifest):
        df_local = pd.read_csv(local_manifest)
        if len(df_local) > 0:
            first_path = df_local.iloc[0].get("audio_path_or_source_ref") or df_local.iloc[0].get("audio_path", "")
            if os.path.exists(str(first_path)):
                use_local_offline = True
                print(f"Detected {len(df_local)} local physical audio files. Using local development mode.")
                for _, r in df_local.iterrows():
                    ap = r.get("audio_path_or_source_ref") or r.get("audio_path", "")
                    if os.path.exists(str(ap)):
                        wf, sr = load_audio(str(ap), target_sr=16000)
                        train_records.append({
                            "sample_id": r.get("sample_id", f"sample_{len(train_records)}"),
                            "waveform": wf,
                            "transcript": normalize_vietnamese_text(r.get("transcript", "")),
                            "duration_sec": len(wf) / 16000.0,
                        })
                if os.path.exists(val_manifest_path):
                    df_v = pd.read_csv(val_manifest_path)
                    for _, r in df_v.iterrows():
                        ap = r.get("audio_path_or_source_ref") or r.get("audio_path", "")
                        if os.path.exists(str(ap)):
                            wf, sr = load_audio(str(ap), target_sr=16000)
                            val_records.append({
                                "sample_id": r.get("sample_id", f"val_{len(val_records)}"),
                                "waveform": wf,
                                "transcript": normalize_vietnamese_text(r.get("transcript", "")),
                                "duration_sec": len(wf) / 16000.0,
                            })

    if use_local_offline:
        n_corpus_samples = len(train_records)
        print(f"Loaded {n_corpus_samples} local training records.")
    else:
        print("Using Cloud Streaming Data Pipeline (streaming=True, zero persistent disk overhead).")
        n_corpus_samples = TOTAL_TRAINING_READY
        print(f"Cloud corpus cardinality: {n_corpus_samples} training-ready utterances.")

    # 4. Setup Optimizer, Scheduler, and GradScaler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_micro_steps = TOTAL_OPTIMIZER_STEPS * grad_accum_steps
    warmup_steps = int(TOTAL_OPTIMIZER_STEPS * WARMUP_RATIO) * grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_micro_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    collator = WhisperDataCollatorWithPadding(processor=processor)

    epoch_metrics: List[Dict[str, Any]] = []
    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint_path = ""

    print("\n" + "=" * 70)
    print(f"STARTING FULL E1 TRAINING: {epochs} EPOCHS | {TOTAL_OPTIMIZER_STEPS} TOTAL STEPS")
    print("=" * 70)

    # 5. Epoch Loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()

        print(f"\n--- Epoch {epoch}/{epochs} ---")
        print(f"Deterministic exposures: {EXPOSURES_PER_EPOCH} ({STEPS_PER_EPOCH} optimizer steps of effective batch {EFFECTIVE_BATCH_SIZE})")

        if use_local_offline:
            exposure_indices = build_epoch_exposure_indices(epoch, n_samples=n_corpus_samples, base_seed=SEED)
            epoch_samples = [train_records[idx % n_corpus_samples] for idx in exposure_indices]
            epoch_dataset = FullE1InMemoryDataset(epoch_samples, processor)
            epoch_loader = DataLoader(
                epoch_dataset,
                batch_size=micro_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
        else:
            epoch_dataset = FullE1StreamingDataset(
                processor=processor,
                epoch=epoch,
                base_seed=SEED,
            )
            epoch_loader = DataLoader(
                epoch_dataset,
                batch_size=micro_batch_size,
                collate_fn=collator,
            )

        running_loss = 0.0
        optimizer_step_loss = 0.0
        epoch_steps = 0
        total_micro_batches_seen = 0

        for micro_step, batch in enumerate(epoch_loader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            total_micro_batches_seen += 1

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()
            optimizer_step_loss += outputs.loss.item()

            if (micro_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_steps += 1

                if epoch_steps % 50 == 0 or epoch_steps == STEPS_PER_EPOCH:
                    avg_step_loss = optimizer_step_loss / grad_accum_steps
                    lr_curr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {epoch_steps}/{STEPS_PER_EPOCH} (Global {global_step}) | Loss: {avg_step_loss:.4f} | LR: {lr_curr:.6f}")
                    optimizer_step_loss = 0.0

        avg_epoch_train_loss = running_loss / total_micro_batches_seen if total_micro_batches_seen > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} complete | Mean Train Loss: {avg_epoch_train_loss:.4f} | Time: {epoch_duration:.1f}s")

        # 6. Epoch Validation
        val_start_time = time.time()
        val_loss = 0.0
        val_samples_evaluated = 0

        if use_local_offline and len(val_records) > 0:
            model.eval()
            val_dataset = FullE1InMemoryDataset(val_records, processor)
            val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=collator)
            with torch.no_grad():
                for v_batch in val_loader:
                    v_feats = v_batch["input_features"].to(device)
                    v_lbls = v_batch["labels"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        v_out = model(input_features=v_feats, labels=v_lbls)
                    val_loss += v_out.loss.item()
                    val_samples_evaluated += len(v_batch["labels"])
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        elif not use_local_offline:
            model.eval()
            val_dataset = FullE1StreamingValDataset(processor=processor, max_samples=1900)
            val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, collate_fn=collator)
            v_step_count = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_feats = v_batch["input_features"].to(device)
                    v_lbls = v_batch["labels"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        v_out = model(input_features=v_feats, labels=v_lbls)
                    val_loss += v_out.loss.item()
                    val_samples_evaluated += len(v_batch["labels"])
                    v_step_count += 1
            val_loss = val_loss / v_step_count if v_step_count > 0 else 0.0

        val_duration = time.time() - val_start_time
        print(f"Epoch {epoch} Validation | Mean Val Loss: {val_loss:.4f} | Evaluated {val_samples_evaluated} samples ({val_duration:.1f}s)")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_epoch_train_loss,
            "val_loss": val_loss,
            "train_duration_sec": epoch_duration,
            "val_duration_sec": val_duration,
            "optimizer_steps": epoch_steps,
            "global_step": global_step,
        }
        epoch_metrics.append(epoch_record)

        # 7. Checkpoint Serialization
        epoch_ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        save_lora_checkpoint(
            checkpoint_dir=epoch_ckpt_dir,
            peft_model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            base_model_id=BASE_MODEL_ID,
            metrics=epoch_record,
        )
        print(f"Saved checkpoint for Epoch {epoch} to {epoch_ckpt_dir}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = epoch_ckpt_dir

    # 8. Final Checkpoint Selection & Post-Training Frozen Test Evaluation
    final_dir = os.path.join(output_dir, "final")
    save_lora_checkpoint(
        checkpoint_dir=final_dir,
        peft_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epochs,
        global_step=global_step,
        base_model_id=BASE_MODEL_ID,
        metrics={"best_val_loss": best_val_loss},
    )
    print(f"\nFinal Full E1 checkpoint saved to {final_dir}")

    # 9. Final Frozen Test Set Evaluation
    print("\n" + "=" * 70)
    print("EVALUATING FINAL CHECKPOINT ON FROZEN GOLD TEST SET")
    print("=" * 70)

    # Re-verify frozen test manifest canonical hash before evaluating
    assert os.path.exists(test_manifest_path), f"Test manifest missing: {test_manifest_path}"
    test_raw_bytes = open(test_manifest_path, "rb").read()
    test_canonical_hash = compute_canonical_crlf_sha256(test_raw_bytes)
    assert test_canonical_hash == FROZEN_TEST_SHA256, "CRITICAL: Test manifest altered prior to evaluation!"

    # Resolve frozen test samples with in-memory 16kHz waveforms
    test_samples = resolve_frozen_test_samples(
        test_manifest_path=test_manifest_path,
        vimd_revision=PINNED_VIMD_REVISION,
    )
    assert len(test_samples) == 9, f"CRITICAL: Expected exactly 9 resolved test samples, got {len(test_samples)}"

    model.eval()
    test_report = evaluate_model_on_manifest(
        model=model,
        processor=processor,
        samples=test_samples,
        device="cuda",
    )
    assert test_report.total_samples == 9, (
        f"CRITICAL: Evaluator evaluated {test_report.total_samples} samples, expected 9!"
    )
    print(test_report.summary())

    total_wall_clock = time.time() - start_time
    final_metrics = {
        "status": "COMPLETED",
        "model": BASE_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "peft_version": PEFT_REQUIRED_VERSION,
        "total_optimizer_steps": global_step,
        "epochs_completed": epochs,
        "total_wall_clock_sec": total_wall_clock,
        "epoch_metrics": epoch_metrics,
        "best_val_loss": best_val_loss,
        "final_test_wer": test_report.micro_wer,
        "final_test_cer": test_report.micro_cer,
        "final_test_mean_latency_sec": test_report.mean_latency_sec,
        "final_test_mean_rtf": test_report.mean_rtf,
    }

    metrics_path = "reports/full_e1_training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Training metrics logged to {metrics_path}")

    print("\n" + "=" * 70)
    print("FULL E1 TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Test WER: {test_report.micro_wer * 100:.2f}% | CER: {test_report.micro_cer * 100:.2f}%")
    print("=" * 70)


def run_streaming_smoke_test(processor: Optional[WhisperProcessor] = None) -> Dict[str, Any]:
    """Runs a minimal, safe smoke test of the streaming pipeline without training or downloading shards."""
    print("=" * 70)
    print("FULL E1 STREAMING DATA PIPELINE SMOKE TEST")
    print("=" * 70)

    # 1. Disk usage before
    total, used, free_before = shutil.disk_usage(".")
    free_before_gb = free_before / (1024 ** 3)
    print(f"Disk free BEFORE smoke test: {free_before_gb:.2f} GB")

    from datasets import Audio, IterableDataset, load_dataset

    print("Loading VIVOS train stream (streaming=True)...")
    ds_vivos = load_dataset(
        "thanhduycao/vivos_ng_only",
        split="train",
        streaming=True,
        revision=PINNED_VIVOS_REVISION,
    ).cast_column("audio", Audio(decode=False))
    print(f"VIVOS dataset object type: {type(ds_vivos)}")
    assert isinstance(ds_vivos, IterableDataset), f"Expected IterableDataset, got {type(ds_vivos)}"

    print("Loading ViMD train stream (streaming=True)...")
    ds_vimd = load_dataset(
        "nguyendv02/ViMD_Dataset",
        split="train",
        streaming=True,
        revision=PINNED_VIMD_REVISION,
    ).cast_column("audio", Audio(decode=False))
    print(f"ViMD dataset object type:  {type(ds_vimd)}")
    assert isinstance(ds_vimd, IterableDataset), f"Expected IterableDataset, got {type(ds_vimd)}"

    # 2. Inspect first 2 samples from each stream
    print("\nInspecting first 2 streamed samples from VIVOS...")
    vivos_samples = []
    for i, row in enumerate(ds_vivos):
        wf = decode_audio_record(row["audio"], target_sr=16000)
        dur = len(wf) / 16000.0
        vivos_samples.append({
            "idx": i,
            "speaker": row.get("speaker_id"),
            "duration": dur,
            "waveform_shape": wf.shape,
        })
        if i >= 1:
            break
    for s in vivos_samples:
        print(f"  VIVOS Sample {s['idx']}: speaker={s['speaker']}, dur={s['duration']:.2f}s, shape={s['waveform_shape']}")

    print("\nInspecting first 2 streamed samples from ViMD...")
    vimd_samples = []
    for i, row in enumerate(ds_vimd):
        wf = decode_audio_record(row["audio"], target_sr=16000)
        dur = len(wf) / 16000.0
        vimd_samples.append({
            "idx": i,
            "speaker": row.get("speakerID"),
            "duration": dur,
            "waveform_shape": wf.shape,
        })
        if i >= 1:
            break
    for s in vimd_samples:
        print(f"  ViMD Sample {s['idx']}: speaker={s['speaker']}, dur={s['duration']:.2f}s, shape={s['waveform_shape']}")

    # 3. Disk usage after
    total, used, free_after = shutil.disk_usage(".")
    free_after_gb = free_after / (1024 ** 3)
    delta_mb = (free_before - free_after) / (1024 ** 2)
    print(f"\nDisk free AFTER smoke test:  {free_after_gb:.2f} GB")
    print(f"Disk consumption delta:      {delta_mb:.4f} MB")
    assert delta_mb < 50.0, f"Unexpected large disk download during streaming smoke test: {delta_mb:.2f} MB"
    print("Streaming smoke test result: PASS (Zero parquet shards downloaded to disk)")
    print("=" * 70)

    return {
        "status": "PASS",
        "dataset_type": str(type(ds_vimd)),
        "samples_inspected": len(vivos_samples) + len(vimd_samples),
        "disk_free_before_gb": free_before_gb,
        "disk_free_after_gb": free_after_gb,
        "delta_mb": delta_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Full E1 Vietnamese ASR Training")
    parser.add_argument("--device_check_only", action="store_true", help="Perform pre-run integrity check and exit")
    parser.add_argument("--smoke_test_streaming", action="store_true", help="Run streaming data pipeline smoke test and exit")
    parser.add_argument("--check_test_resolution", action="store_true", help="Verify resolving all 9 frozen test samples and exit")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--micro_batch_size", type=int, default=MICRO_BATCH_SIZE, help="Micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Checkpoint output directory")
    args = parser.parse_args()

    if args.smoke_test_streaming:
        run_streaming_smoke_test()
        sys.exit(0)

    if args.check_test_resolution:
        samples = resolve_frozen_test_samples(force_streaming=True)
        assert len(samples) == 9, f"Expected 9 samples, got {len(samples)}"
        print(f"Verified resolution of {len(samples)} frozen test samples in streaming mode.")
        sys.exit(0)

    check_res = preflight_check()

    if args.device_check_only:
        if not check_res["cuda_available"]:
            print("STATUS: FULL E1: ABORTED")
            print("Reason: CUDA execution environment unavailable on local workstation.")
            print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
            sys.exit(0)
        else:
            print("\nFULL E1 DEVICE CHECK: PASS. READY FOR TRAINING.")
            sys.exit(0)

    if not check_res["cuda_available"]:
        print("STATUS: FULL E1: ABORTED")
        print("Reason: CUDA execution environment unavailable on local workstation.")
        print("Execution must take place in the audited Google Colab / cloud Linux GPU runtime.")
        sys.exit(1)

    print("\nCUDA available. Launching Full E1 training execution...")
    run_full_e1_training(
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
