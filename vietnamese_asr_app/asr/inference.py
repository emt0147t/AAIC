"""Vietnamese ASR Inference Engine.

Handles audio preprocessing, model forward pass, beam search / greedy decoding,
real-time factor (RTF) calculation, post-processing normalization, and safe
architectural token clamping against Whisper's max_target_positions limit.
"""

from dataclasses import dataclass
import logging
import time
from typing import List, Optional
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .model import ASRModelManager, HardwareDiagnostics
from .normalization import normalize_vietnamese_text

logger = logging.getLogger(__name__)


class DecodingConfigurationError(ValueError):
    """Raised when decoding parameters violate architectural bounds or token limits."""
    pass


@dataclass
class TranscriptionResult:
    raw_text: str
    normalized_text: str
    audio_duration_sec: float
    latency_sec: float
    rtf: float
    device: str
    model_id: str
    decoding_strategy: str
    num_beams: int
    requested_max_new_tokens: int
    effective_max_new_tokens: int
    total_possible_decoder_length: int
    chunks_processed: int = 1


class ASRInferenceEngine:
    """Wrapper for PhoWhisper / Whisper transcription pipelines."""

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-small",
        preferred_device: str = "auto",
        default_max_new_tokens: int = 192,
    ):
        self.model_id = model_id
        self.preferred_device = preferred_device
        self.default_max_new_tokens = default_max_new_tokens
        self._model: Optional[WhisperForConditionalGeneration] = None
        self._processor: Optional[WhisperProcessor] = None
        self._hw: Optional[HardwareDiagnostics] = None

    def _ensure_loaded(self) -> None:
        if self._model is None or self._processor is None:
            self._model, self._processor, self._hw = ASRModelManager.get_model_and_processor(
                model_id=self.model_id,
                preferred_device=self.preferred_device,
            )

    @property
    def hardware_info(self) -> HardwareDiagnostics:
        self._ensure_loaded()
        assert self._hw is not None
        return self._hw

    @staticmethod
    def compute_safe_max_new_tokens(
        requested_max_new_tokens: int,
        prompt_len: int,
        max_target_positions: int = 448,
        safety_margin: int = 1,
    ) -> int:
        """Compute safe upper-bound for max_new_tokens ensuring:
        prompt_len + max_new_tokens <= max_target_positions.

        Args:
            requested_max_new_tokens: User or configuration requested token limit.
            prompt_len: Number of tokens in the initial decoder prompt sequence.
            max_target_positions: Architectural maximum sequence length of the Whisper decoder.
            safety_margin: Safety token buffer (default 1).

        Returns:
            Clamped safe integer token count >= 1.
        """
        max_allowed = max_target_positions - prompt_len - safety_margin
        safe_val = max(1, min(requested_max_new_tokens, max_allowed))
        return safe_val

    def transcribe(
        self,
        audio_16k: np.ndarray,
        strategy: str = "greedy",
        num_beams: int = 4,
        language: str = "vi",
        task: str = "transcribe",
        max_new_tokens: Optional[int] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ) -> TranscriptionResult:
        """Run speech recognition on 16kHz mono audio with safe decoding parameters.

        Args:
            audio_16k: 1D numpy array at 16000 Hz.
            strategy: 'greedy' or 'beam_search'.
            num_beams: Number of beams when strategy == 'beam_search'.
            language: Language token code (default 'vi').
            task: 'transcribe' or 'translate'.
            max_new_tokens: Requested maximum new tokens (defaults to 192).
            decoder_input_ids: Optional custom decoder input prompt tensor.

        Returns:
            TranscriptionResult object with timing, text, and generation audit details.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._hw is not None

        audio_len = len(audio_16k)
        duration_sec = float(audio_len) / 16000.0
        start_time = time.perf_counter()

        requested_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens

        # 1. Determine actual model architectural limit
        max_target_positions = getattr(self._model.config, "max_target_positions", 448)

        # 2. Determine initial decoder prompt tokens dynamically
        if decoder_input_ids is None:
            start_token = getattr(self._model.config, "decoder_start_token_id", 50258)
            prompt_ids = self._processor.get_decoder_prompt_ids(language=language, task=task)
            token_list = [start_token] + [token_id for _, token_id in prompt_ids]
            decoder_input_ids = torch.tensor([token_list], device=self._hw.device, dtype=torch.long)
        else:
            decoder_input_ids = decoder_input_ids.to(self._hw.device)

        prompt_len = int(decoder_input_ids.shape[-1])

        # 3. Compute safe generation limit with safety margin = 1
        safety_margin = 1
        effective_max_new_tokens = self.compute_safe_max_new_tokens(
            requested_max_new_tokens=requested_tokens,
            prompt_len=prompt_len,
            max_target_positions=max_target_positions,
            safety_margin=safety_margin,
        )

        total_possible_decoder_length = prompt_len + effective_max_new_tokens

        # 4. Explicit audit log and assertion before generation
        audit_msg = (
            f"[Generation Config Audit]\n"
            f"  max_target_positions:          {max_target_positions}\n"
            f"  decoder_prompt_length:         {prompt_len}\n"
            f"  requested_max_new_tokens:      {requested_tokens}\n"
            f"  effective_max_new_tokens:      {effective_max_new_tokens}\n"
            f"  total_possible_decoder_length: {total_possible_decoder_length}"
        )
        logger.info(audit_msg)
        print(audit_msg)

        if total_possible_decoder_length > max_target_positions:
            raise DecodingConfigurationError(
                f"DECODING CONFIGURATION ERROR: Combined decoder length "
                f"({prompt_len} + {effective_max_new_tokens} = {total_possible_decoder_length}) "
                f"exceeds model architectural limit max_target_positions={max_target_positions}."
            )

        # 5. Handle audio chunking if duration > 30.0s (480,000 samples at 16kHz)
        max_chunk_samples = 480000
        chunks: List[np.ndarray] = []
        if audio_len <= max_chunk_samples:
            chunks = [audio_16k]
        else:
            # Segment sequentially into 30.0s chunks without silent truncation
            num_chunks = int(np.ceil(audio_len / max_chunk_samples))
            for i in range(num_chunks):
                c_start = i * max_chunk_samples
                c_end = min((i + 1) * max_chunk_samples, audio_len)
                chunks.append(audio_16k[c_start:c_end])

        target_dtype = torch.float16 if self._hw.device.startswith("cuda") else torch.float32
        chunk_transcriptions: List[str] = []

        # 6. Generation kwargs preserving language/task and avoiding duplicate prompt injection
        gen_kwargs = {
            "language": language,
            "task": task,
            "decoder_input_ids": decoder_input_ids,
            "max_new_tokens": effective_max_new_tokens,
            "forced_decoder_ids": None,  # Prevent duplicate prompt injection or deprecation warning
        }

        if strategy == "beam_search" and num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["num_beams"] = 1
            gen_kwargs["do_sample"] = False

        # Process each chunk under inference_mode
        with torch.inference_mode():
            for chunk_data in chunks:
                if len(chunk_data) == 0:
                    continue
                feat_dict = self._processor.feature_extractor(
                    chunk_data,
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_features = feat_dict.input_features.to(self._hw.device, dtype=target_dtype)
                attention_mask = feat_dict.attention_mask.to(self._hw.device) if "attention_mask" in feat_dict else None

                gen_call_kwargs = dict(gen_kwargs)
                if attention_mask is not None:
                    gen_call_kwargs["attention_mask"] = attention_mask

                predicted_ids = self._model.generate(input_features, **gen_call_kwargs)
                text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                chunk_transcriptions.append(text.strip())

        end_time = time.perf_counter()
        latency_sec = end_time - start_time
        rtf = latency_sec / max(duration_sec, 1e-4)

        raw_text = " ".join(chunk_transcriptions).strip()
        norm_text = normalize_vietnamese_text(raw_text)

        return TranscriptionResult(
            raw_text=raw_text,
            normalized_text=norm_text,
            audio_duration_sec=duration_sec,
            latency_sec=latency_sec,
            rtf=rtf,
            device=self._hw.device,
            model_id=self.model_id,
            decoding_strategy=strategy,
            num_beams=num_beams if strategy == "beam_search" else 1,
            requested_max_new_tokens=requested_tokens,
            effective_max_new_tokens=effective_max_new_tokens,
            total_possible_decoder_length=total_possible_decoder_length,
            chunks_processed=len(chunks),
        )
