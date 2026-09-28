"""Corpus Evaluation Module for PhoWhisper / LoRA Experiments.

Computes exact corpus-level micro WER, micro CER, mean latency, and mean RTF.
Reuses existing evaluation and normalization utilities directly.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import WhisperProcessor

try:
    from asr.audio import load_audio
    from asr.normalization import normalize_vietnamese_text
    from evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer
except (ImportError, ValueError):
    from ..asr.audio import load_audio
    from ..asr.normalization import normalize_vietnamese_text
    from ..evaluation.metrics import compute_levenshtein_cer, compute_levenshtein_wer


@dataclass
class CorpusEvaluationReport:
    model_identifier: str
    total_samples: int
    micro_wer: float
    micro_cer: float
    total_reference_words: int
    total_reference_chars: int
    mean_latency_sec: float
    mean_rtf: float
    per_utterance_results: List[Dict[str, Any]]

    def summary(self) -> str:
        return (
            f"Corpus Evaluation Summary [{self.model_identifier}]\n"
            f"  Samples:     {self.total_samples}\n"
            f"  Micro WER:   {self.micro_wer * 100:.2f}%\n"
            f"  Micro CER:   {self.micro_cer * 100:.2f}%\n"
            f"  Mean Latency:{self.mean_latency_sec:.3f} s\n"
            f"  Mean RTF:    {self.mean_rtf:.3f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_model_on_manifest(
    model: nn.Module,
    processor: WhisperProcessor,
    samples: List[Dict[str, Any]],
    device: str = "cpu",
    language: str = "vi",
    task: str = "transcribe",
    max_new_tokens: int = 192,
) -> CorpusEvaluationReport:
    """Evaluate base Whisper model or PeftModel on a list of speech samples.

    Args:
        model: WhisperForConditionalGeneration or PeftModel.
        processor: WhisperProcessor.
        samples: List of dicts with 'audio' (np.ndarray or filepath) and 'reference' (str).
        device: 'cpu' or 'cuda'.
        language: Language code ('vi').
        task: 'transcribe'.
        max_new_tokens: Generation token limit.

    Returns:
        CorpusEvaluationReport instance.
    """
    model.eval()
    model.to(device)

    total_ref_words = 0
    total_word_errors = 0
    total_ref_chars = 0
    total_char_errors = 0
    latencies = []
    rtfs = []
    per_utt = []

    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    with torch.inference_mode():
        for idx, sample in enumerate(samples):
            audio_data = sample.get("audio")
            if audio_data is None:
                audio_data = sample.get("audio_path")
            ref_raw = str(sample.get("reference") or sample.get("transcript") or "")

            if isinstance(audio_data, (str, bytes)):
                audio_16k, sr = load_audio(audio_data, target_sr=16000)
            elif isinstance(audio_data, np.ndarray):
                audio_16k = audio_data
            else:
                continue

            dur_sec = float(len(audio_16k)) / 16000.0
            if dur_sec < 0.2:
                continue

            t0 = time.perf_counter()

            # Process input features
            inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device, dtype=target_dtype)

            # Generate
            pred_ids = model.generate(
                input_features,
                language=language,
                task=task,
                max_new_tokens=max_new_tokens,
                forced_decoder_ids=None,
            )
            hyp_raw = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            t1 = time.perf_counter()
            latency = t1 - t0
            rtf = latency / max(dur_sec, 1e-4)

            # Calculate exact Levenshtein errors using existing metrics module
            wer_m = compute_levenshtein_wer(ref_raw, hyp_raw)
            cer_m = compute_levenshtein_cer(ref_raw, hyp_raw)

            total_ref_words += wer_m.total_reference_tokens
            total_word_errors += (wer_m.substitutions + wer_m.deletions + wer_m.insertions)

            total_ref_chars += cer_m.total_reference_tokens
            total_char_errors += (cer_m.substitutions + cer_m.deletions + cer_m.insertions)

            latencies.append(latency)
            rtfs.append(rtf)

            per_utt.append({
                "sample_id": sample.get("sample_id", f"sample_{idx}"),
                "reference": ref_raw,
                "hypothesis": hyp_raw,
                "wer": wer_m.error_rate,
                "cer": cer_m.error_rate,
                "latency_sec": round(latency, 3),
                "rtf": round(rtf, 3),
                "duration_sec": round(dur_sec, 3),
            })

    micro_wer = (total_word_errors / max(1, total_ref_words)) if total_ref_words > 0 else 0.0
    micro_cer = (total_char_errors / max(1, total_ref_chars)) if total_ref_chars > 0 else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    mean_rtf = float(np.mean(rtfs)) if rtfs else 0.0

    return CorpusEvaluationReport(
        model_identifier=getattr(model, "name_or_path", "model"),
        total_samples=len(per_utt),
        micro_wer=round(micro_wer, 4),
        micro_cer=round(micro_cer, 4),
        total_reference_words=total_ref_words,
        total_reference_chars=total_ref_chars,
        mean_latency_sec=round(mean_lat, 4),
        mean_rtf=round(mean_rtf, 4),
        per_utterance_results=per_utt,
    )
