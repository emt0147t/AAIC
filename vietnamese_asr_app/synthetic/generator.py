"""Synthetic Vietnamese Speech Generator using Gwen-TTS 0.6B.

Standardizes outputs to 16 kHz mono PCM WAV suitable for PhoWhisper / Whisper ASR.
Adheres strictly to the voice policy: built-in reference speakers with synthetic IDs
(spk_synth_01 .. spk_synth_09), ensuring no cloning of real training/test speakers.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


# Authoritative mapping of synthetic speaker IDs to Gwen-TTS built-in voices
SYNTHETIC_VOICE_MAP = {
    "spk_synth_01": {"ref_key": "yen_nhi", "name": "Yến Nhi", "gender": "female"},
    "spk_synth_02": {"ref_key": "my_van", "name": "Mỹ Vân", "gender": "female"},
    "spk_synth_03": {"ref_key": "ai_vy", "name": "Ái Vy", "gender": "female"},
    "spk_synth_04": {"ref_key": "an_nhi", "name": "An Nhi", "gender": "female"},
    "spk_synth_05": {"ref_key": "dieu_linh", "name": "Diệu Linh", "gender": "female"},
    "spk_synth_06": {"ref_key": "khanh_toan", "name": "Khánh Toàn", "gender": "male"},
    "spk_synth_07": {"ref_key": "tran_lam", "name": "Trần Lâm", "gender": "male"},
    "spk_synth_08": {"ref_key": "nsnd_ha_phuong", "name": "NSND Hà Phương", "gender": "male"},
    "spk_synth_09": {"ref_key": "nsnd_kim_cuc", "name": "NSND Kim Cúc", "gender": "female"},
}


class SyntheticSpeechGenerator:
    """Manages model loading, voice selection, synthesis, and audio standardization."""

    def __init__(
        self,
        model_id: str = "g-group-ai-lab/gwen-tts-0.6B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if dtype is None:
            self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        else:
            self.dtype = dtype

        # 1. Download snapshot and parse built-in reference information
        print(f"[Synthetic TTS] Resolving snapshot for {model_id}...")
        self.snapshot_dir = snapshot_download(model_id)
        ref_info_path = os.path.join(self.snapshot_dir, "data", "ref_info.json")
        with open(ref_info_path, "r", encoding="utf-8") as f:
            self.ref_info = json.load(f)

        # 2. Extract commit / revision hash if available
        self.model_revision = os.path.basename(self.snapshot_dir)

        # 3. Load model
        print(f"[Synthetic TTS] Loading Qwen3TTSModel on {self.device} ({self.dtype})...")
        t0 = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            self.snapshot_dir,
            device_map=self.device,
            dtype=self.dtype,
        )
        print(f"[Synthetic TTS] Model loaded in {time.time() - t0:.2f} seconds.")

        # Default generation parameters
        self.default_generation_config = {
            "temperature": 0.3,
            "top_k": 20,
            "top_p": 0.9,
            "max_new_tokens": 1024,
            "repetition_penalty": 2.0,
            "subtalker_do_sample": True,
            "subtalker_temperature": 0.1,
            "subtalker_top_k": 20,
            "subtalker_top_p": 1.0,
        }

    def get_voice_info(self, synth_speaker_id: str) -> Dict[str, Any]:
        """Get voice profile metadata for a synthetic speaker ID."""
        if synth_speaker_id not in SYNTHETIC_VOICE_MAP:
            raise KeyError(f"Unknown synthetic speaker: {synth_speaker_id}. Valid: {list(SYNTHETIC_VOICE_MAP.keys())}")
        vinfo = SYNTHETIC_VOICE_MAP[synth_speaker_id]
        ref_key = vinfo["ref_key"]
        ref_data = self.ref_info[ref_key]
        return {
            "synthetic_speaker_id": synth_speaker_id,
            "ref_key": ref_key,
            "ref_name": vinfo["name"],
            "gender": vinfo["gender"],
            "ref_audio_rel": ref_data["audio_path"],
            "ref_audio_abs": os.path.join(self.snapshot_dir, ref_data["audio_path"]),
            "ref_text": ref_data["text"],
        }

    def generate(
        self,
        text: str,
        synth_speaker_id: str,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int, float, Dict[str, Any]]:
        """Generate raw synthetic speech using specified synthetic speaker voice.

        Returns:
            (raw_waveform, raw_sample_rate, generation_time_sec, voice_info)
        """
        voice_info = self.get_voice_info(synth_speaker_id)
        gen_kwargs = {**self.default_generation_config, **(generation_config or {})}

        t0 = time.time()
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="Vietnamese",
            ref_audio=voice_info["ref_audio_abs"],
            ref_text=voice_info["ref_text"],
            **gen_kwargs,
        )
        gen_time = time.time() - t0

        raw_audio = wavs[0]
        return raw_audio, sr, gen_time, voice_info

    @staticmethod
    def standardize_audio(
        raw_audio: np.ndarray,
        raw_sr: int,
        target_sr: int = 16000,
    ) -> np.ndarray:
        """Standardize audio: float32, mono, resampled to 16,000 Hz, finite values."""
        # Convert to float32
        audio = raw_audio.astype(np.float32)

        # Convert to mono if multi-channel
        if audio.ndim > 1:
            if audio.shape[0] == 1:
                audio = audio[0]
            elif audio.shape[1] == 1:
                audio = audio[:, 0]
            else:
                audio = np.mean(audio, axis=1 if audio.shape[1] < audio.shape[0] else 0)

        # Ensure finite
        if not np.all(np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Resample to target sample rate (16000 Hz)
        if raw_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=raw_sr, target_sr=target_sr)

        # Prevent digital clipping: peak normalize if clipping occurs
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if peak > 0.99:
            audio = audio * (0.95 / peak)

        return audio.astype(np.float32)

    @staticmethod
    def save_pcm16_wav(audio_16k: np.ndarray, output_path: str, sr: int = 16000):
        """Save standardized 16-bit PCM WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, audio_16k, sr, subtype="PCM_16")
