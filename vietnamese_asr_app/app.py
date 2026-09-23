"""Vietnamese Speech Recognition and Waveform Augmentation Studio.

Minimal, professional, text-first engineering interface.
Reuses all underlying ASR, augmentation, validation, evaluation, and export modules.
"""

import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from asr.audio import load_audio, save_wav_pcm16, validate_audio
from asr.inference import ASRInferenceEngine, DecodingConfigurationError
from asr.model import ASRModelManager, detect_hardware
from asr.normalization import normalize_vietnamese_text
from augmentation.pipeline import AugmentedAudioSample, AugmentationStudioEngine
from augmentation.qc import run_10_point_qc
from datasets.mixture import DatasetMixtureBuilder
from datasets.provenance import QualityTier
from datasets.registry import DATASET_REGISTRY, list_all_provenance
from evaluation.metrics import (
    compute_levenshtein_cer,
    compute_levenshtein_wer,
    evaluate_augmentation_consistency,
)
from export.exporter import AugmentationSessionExporter

DEFAULT_MODEL_ID = "vinai/PhoWhisper-small"
asr_engine = ASRInferenceEngine(model_id=DEFAULT_MODEL_ID, default_max_new_tokens=192)
aug_engine = AugmentationStudioEngine(sample_rate=16000)
session_exporter = AugmentationSessionExporter(base_export_dir="exports")

# In-memory session store
current_session_state: Dict[str, Any] = {
    "original_audio": None,
    "original_transcript": "",
    "reference_transcript": "",
    "augmented_samples": [],
    "augmented_transcripts": [],
    "model_id": DEFAULT_MODEL_ID,
    "last_export": None,
}


def get_compact_hardware_info() -> str:
    """Format compact hardware status line without decorative elements."""
    hw = detect_hardware()
    vram_part = f" | VRAM: {hw.vram_allocated_gb:.2f}GB / {hw.vram_total_gb:.2f}GB" if hw.cuda_available else ""
    return f"Model: {asr_engine.model_id} | Device: {hw.device} ({hw.gpu_name}){vram_part} | Precision: {hw.recommended_precision} | PyTorch: {hw.torch_version}"


def format_7point_details(audio_16k: np.ndarray, sr: int, val_res) -> str:
    """Format detailed text for the 7-point validation gate."""
    num_samples = len(audio_16k) if audio_16k is not None else 0
    dur = float(num_samples) / float(sr) if sr > 0 else 0.0
    finite_ok = np.all(np.isfinite(audio_16k)) if num_samples > 0 else False
    mono_ok = (audio_16k.ndim == 1) if audio_16k is not None else False
    lower_dur_ok = dur >= 0.2
    upper_dur_ok = dur <= 30.0
    rms_ok = val_res.rms >= 1e-5
    peak_ok = val_res.peak_abs <= 1.2

    lines = [
        f"1. Signal existence:    {'PASS' if num_samples > 0 else 'FAIL'} ({num_samples:,} samples)",
        f"2. Finite values:        {'PASS' if finite_ok else 'FAIL'} (all values finite real numbers)",
        f"3. Mono channel:         {'PASS' if mono_ok else 'FAIL'} (1 channel)",
        f"4. Lower duration bound: {'PASS' if lower_dur_ok else 'FAIL'} ({dur:.2f} s >= 0.20 s)",
        f"5. Upper duration bound: {'PASS' if upper_dur_ok else 'PASS (Chunking supported)'} ({dur:.2f} s <= 30.00 s)",
        f"6. RMS signal level:     {'PASS' if rms_ok else 'FAIL'} ({val_res.rms:.4f} >= 0.00001)",
        f"7. Dynamic range:        {'PASS' if peak_ok else 'FAIL'} (Peak {val_res.peak_abs:.2f} <= 1.20)",
    ]
    return "\n".join(lines)


# ==============================================================================
# WORKFLOW: TRANSCRIPTION & VALIDATION
# ==============================================================================
def handle_transcription(
    audio_file,
    model_id: str,
    strategy: str,
    num_beams: int,
    max_new_tokens: int,
    reference_text: str,
    progress=gr.Progress(),
):
    if audio_file is None:
        return (
            "No audio selected.",
            "Duration: - | Latency: - | RTF: -",
            "Audio status: No audio provided.",
            "No audio data to validate.",
            "",
            pd.DataFrame(),
        )

    try:
        progress(0.1, desc="Ingesting and resampling audio to 16kHz...")
        audio_16k, sr = load_audio(audio_file, target_sr=16000)

        progress(0.2, desc="Evaluating 7-point validation gate...")
        val_res = validate_audio(audio_16k, sample_rate=sr, allow_chunking=True)
        validation_details = format_7point_details(audio_16k, sr, val_res)

        if not val_res.is_valid:
            status_line = f"Audio validation: FAIL - {val_res.error_message}"
            return (
                "Transcription blocked by validation gate.",
                "Duration: - | Latency: - | RTF: -",
                status_line,
                validation_details,
                "",
                pd.DataFrame(),
            )

        status_line = f"Audio validation: PASS (Duration: {val_res.duration_sec:.2f} s, RMS: {val_res.rms:.4f}, Peak: {val_res.peak_abs:.2f})"

        progress(0.4, desc=f"Running inference with {model_id}...")
        if asr_engine.model_id != model_id:
            asr_engine.model_id = model_id
            asr_engine._model = None
            asr_engine._processor = None

        tokens_req = int(max_new_tokens) if max_new_tokens else 192
        t_res = asr_engine.transcribe(
            audio_16k,
            strategy=strategy,
            num_beams=int(num_beams),
            max_new_tokens=tokens_req,
        )

        progress(0.85, desc="Computing metrics...")
        metrics_line = (
            f"Duration: {t_res.audio_duration_sec:.2f} s | "
            f"Latency: {t_res.latency_sec:.2f} s | "
            f"RTF: {t_res.rtf:.3f} | "
            f"Tokens: {t_res.effective_max_new_tokens} (clamped from {t_res.requested_max_new_tokens}) | "
            f"Chunks: {t_res.chunks_processed}"
        )

        eval_line = ""
        wer_val = None
        cer_val = None
        if reference_text and reference_text.strip():
            wer_res = compute_levenshtein_wer(reference_text, t_res.raw_text)
            cer_res = compute_levenshtein_cer(reference_text, t_res.raw_text)
            wer_val = f"{wer_res.error_rate * 100:.1f}%"
            cer_val = f"{cer_res.error_rate * 100:.1f}%"
            eval_line = (
                f"Evaluation vs Reference: WER = {wer_val} "
                f"(Substitutions={wer_res.substitutions}, Deletions={wer_res.deletions}, Insertions={wer_res.insertions}) | "
                f"CER = {cer_val}"
            )

        current_session_state["original_audio"] = audio_16k
        current_session_state["original_transcript"] = t_res.raw_text
        current_session_state["reference_transcript"] = reference_text
        current_session_state["model_id"] = model_id
        current_session_state["augmented_samples"] = []
        current_session_state["augmented_transcripts"] = []

        # Initialize results table with the original audio row
        initial_rows = [{
            "Variant": "Original",
            "Transformation": "None (Source)",
            "Duration (s)": round(t_res.audio_duration_sec, 2),
            "Transcript": t_res.raw_text,
            "QC Status": "PASS",
            "Drift WER": "0.0%",
            "Drift CER": "0.0%",
            "WER vs Ref": wer_val if wer_val else "-",
        }]
        df_results = pd.DataFrame(initial_rows)

        full_metrics = f"{metrics_line}\n{eval_line}".strip()
        return t_res.raw_text, full_metrics, status_line, validation_details, t_res.normalized_text, df_results

    except Exception as e:
        err_str = str(e)
        if (
            isinstance(e, DecodingConfigurationError)
            or "max_target_positions" in err_str
            or "combined length" in err_str
            or "DECODING CONFIGURATION ERROR" in err_str
        ):
            status_line = "Audio validation: FAIL - DECODING CONFIGURATION ERROR"
            diag_str = f"DECODING CONFIGURATION ERROR: {err_str}"
        else:
            status_line = f"Audio validation: FAIL - {err_str}"
            diag_str = f"Transcription error: {err_str}"

        return "Error occurred during transcription.", diag_str, status_line, str(e), "", pd.DataFrame()


# ==============================================================================
# WORKFLOW: AUGMENTATION GENERATION & RESULTS
# ==============================================================================
def handle_augmentation_generation(
    mode: str,
    k_val: int,
    base_seed: int,
    enabled_transforms: List[str],
    plan_gain_db: float,
    plan_noise_snr: float,
    plan_shift_sec: float,
    plan_stretch_rate: float,
    plan_pitch_semitones: float,
    plan_reverb_decay: float,
    plan_filter_low: float,
    plan_filter_high: float,
    progress=gr.Progress(),
):
    orig_audio = current_session_state.get("original_audio")
    orig_tx = current_session_state.get("original_transcript", "")
    ref_tx = current_session_state.get("reference_transcript", "")

    if orig_audio is None:
        return (
            "No audio found in current session. Please transcribe an audio input first.",
            None, None, None, None, None,
            pd.DataFrame(),
        )

    try:
        progress(0.1, desc="Preparing augmentation pipeline...")
        aug_samples: List[AugmentedAudioSample] = []

        if mode == "Random":
            progress(0.3, desc=f"Generating {k_val} randomized waveform variants...")
            aug_samples = aug_engine.generate_random_k(
                orig_audio,
                k=int(k_val),
                base_seed=int(base_seed),
                enabled_transforms=enabled_transforms,
            )
        else:
            progress(0.3, desc="Generating exact planned waveform variant...")
            exact_plan = [
                {"op": "gain", "gain_db": plan_gain_db},
                {"op": "additive_noise", "snr_db": plan_noise_snr},
                {"op": "time_shift", "shift_sec": plan_shift_sec},
                {"op": "time_stretch", "rate": plan_stretch_rate},
                {"op": "pitch_shift", "n_steps": plan_pitch_semitones},
                {"op": "synthetic_reverb", "decay_sec": plan_reverb_decay, "wet_mix": 0.25},
                {"op": "bandpass_filter", "low_cut": plan_filter_low, "high_cut": plan_filter_high},
            ]
            active_plan = [
                op for op in exact_plan
                if not (
                    (op["op"] == "gain" and abs(op["gain_db"]) < 0.01)
                    or (op["op"] == "time_shift" and abs(op["shift_sec"]) < 0.001)
                    or (op["op"] == "time_stretch" and abs(op["rate"] - 1.0) < 0.001)
                    or (op["op"] == "pitch_shift" and abs(op["n_steps"]) < 0.01)
                )
            ]
            sample = aug_engine.generate_exact_plan(orig_audio, active_plan, aug_index=1, seed=int(base_seed))
            aug_samples = [sample]

        current_session_state["augmented_samples"] = aug_samples
        current_session_state["augmented_transcripts"] = []

        # Prepare audio preview files (first 5 variants)
        preview_paths = [None] * 5
        for idx in range(min(5, len(aug_samples))):
            tmp_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            save_wav_pcm16(aug_samples[idx].audio_16k, tmp_f.name, sample_rate=16000)
            preview_paths[idx] = tmp_f.name

        # Transcribe augmented samples immediately for a unified table
        progress(0.6, desc="Transcribing augmented variants...")
        results_rows = []

        # Row 0: Original
        ref_wer_str = "-"
        if ref_tx and orig_tx:
            wer_0 = compute_levenshtein_wer(ref_tx, orig_tx)
            ref_wer_str = f"{wer_0.error_rate * 100:.1f}%"

        orig_dur = round(float(len(orig_audio)) / 16000.0, 2)
        results_rows.append({
            "Variant": "Original",
            "Transformation": "None (Source)",
            "Duration (s)": orig_dur,
            "Transcript": orig_tx if orig_tx else "(Transcribe to view)",
            "QC Status": "PASS",
            "Drift WER": "0.0%",
            "Drift CER": "0.0%",
            "WER vs Ref": ref_wer_str,
        })

        for idx, sample in enumerate(aug_samples):
            progress(0.6 + 0.35 * (idx / len(aug_samples)), desc=f"Transcribing {sample.aug_id}...")
            t_res = asr_engine.transcribe(sample.audio_16k)
            current_session_state["augmented_transcripts"].append(t_res.raw_text)

            eval_res = evaluate_augmentation_consistency(
                original_hyp=orig_tx,
                augmented_hyp=t_res.raw_text,
                reference_text=ref_tx if ref_tx else None,
            )

            transforms_str = ", ".join([op["op"] for op in sample.transform_chain]) or "None"
            wer_ref_cell = f"{(eval_res.wer_vs_ref or 0.0) * 100:.1f}%" if ref_tx else "-"

            results_rows.append({
                "Variant": sample.aug_id,
                "Transformation": transforms_str,
                "Duration (s)": round(sample.duration_sec, 2),
                "Transcript": t_res.raw_text,
                "QC Status": sample.qc_result.status,
                "Drift WER": f"{eval_res.consistency_drift_wer * 100:.1f}%",
                "Drift CER": f"{eval_res.consistency_drift_cer * 100:.1f}%",
                "WER vs Ref": wer_ref_cell,
            })

        df_results = pd.DataFrame(results_rows)
        status_msg = f"Generated and transcribed {len(aug_samples)} augmented variants. All 10-point QC checks evaluated."

        return (
            status_msg,
            preview_paths[0],
            preview_paths[1],
            preview_paths[2],
            preview_paths[3],
            preview_paths[4],
            df_results,
        )

    except Exception as e:
        return (
            f"Augmentation generation error: {str(e)}",
            None, None, None, None, None,
            pd.DataFrame(),
        )


# ==============================================================================
# WORKFLOW: EXPORT
# ==============================================================================
def handle_export():
    orig_audio = current_session_state.get("original_audio")
    orig_tx = current_session_state.get("original_transcript", "")
    ref_tx = current_session_state.get("reference_transcript", "")
    aug_samples = current_session_state.get("augmented_samples", [])
    aug_txs = current_session_state.get("augmented_transcripts", [])
    model_id = current_session_state.get("model_id", DEFAULT_MODEL_ID)

    if orig_audio is None:
        return "No active session data to export. Please transcribe audio first.", None

    session_id = f"asr_session_{int(time.time())}"
    res = session_exporter.export_session(
        session_id=session_id,
        original_audio_16k=orig_audio,
        original_transcript=orig_tx,
        augmented_samples=aug_samples,
        augmented_transcripts=aug_txs,
        reference_transcript=ref_tx if ref_tx else None,
        model_id=model_id,
    )

    zip_size_bytes = os.path.getsize(res["zip_path"]) if os.path.exists(res["zip_path"]) else 0
    msg = (
        f"Session export completed successfully.\n"
        f"Session ID: {session_id}\n"
        f"Directory: {res['session_dir']}\n"
        f"ZIP package: {res['zip_path']} ({zip_size_bytes:,} bytes)\n"
        f"Manifest CSV: {res['manifest_csv']}\n"
        f"Results JSON: {res['results_json']}"
    )
    return msg, res["zip_path"]


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================
def handle_batch_processing(files, strategy: str, num_beams: int, progress=gr.Progress()):
    if not files:
        return "No files uploaded.", pd.DataFrame(), None

    try:
        rows = []
        total = len(files)
        progress(0.05, desc=f"Queuing {total} files for sequential streaming...")

        for idx, file_obj in enumerate(files):
            progress((idx + 1) / total, desc=f"Processing file {idx + 1} of {total}...")
            fpath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            fname = os.path.basename(fpath)

            try:
                audio_16k, sr = load_audio(fpath, target_sr=16000)
                val_res = validate_audio(audio_16k, sample_rate=sr, allow_chunking=True)

                if not val_res.is_valid:
                    rows.append({
                        "Filename": fname,
                        "Duration (s)": round(val_res.duration_sec, 2),
                        "Status": f"FAIL: {val_res.error_message}",
                        "Raw Transcript": "",
                        "Normalized Transcript": "",
                        "Latency (s)": 0.0,
                        "RTF": 0.0,
                    })
                    continue

                t_res = asr_engine.transcribe(audio_16k, strategy=strategy, num_beams=int(num_beams))
                rows.append({
                    "Filename": fname,
                    "Duration (s)": round(t_res.audio_duration_sec, 2),
                    "Status": "PASS",
                    "Raw Transcript": t_res.raw_text,
                    "Normalized Transcript": t_res.normalized_text,
                    "Latency (s)": round(t_res.latency_sec, 3),
                    "RTF": round(t_res.rtf, 3),
                })
            except Exception as fe:
                rows.append({
                    "Filename": fname,
                    "Duration (s)": 0.0,
                    "Status": f"ERROR: {str(fe)}",
                    "Raw Transcript": "",
                    "Normalized Transcript": "",
                    "Latency (s)": 0.0,
                    "RTF": 0.0,
                })

        df_batch = pd.DataFrame(rows)
        tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df_batch.to_csv(tmp_csv.name, index=False, encoding="utf-8")

        status_msg = f"Batch processing completed: {len(rows)} files processed."
        return status_msg, df_batch, tmp_csv.name

    except Exception as e:
        return f"Batch processing error: {str(e)}", pd.DataFrame(), None


# ==============================================================================
# DATASET PROVENANCE
# ==============================================================================
def get_clean_provenance_table() -> pd.DataFrame:
    items = list_all_provenance()
    rows = []
    for prov in items:
        rows.append({
            "Dataset": prov.name,
            "HF Repository": prov.hf_repo_id,
            "Quality Tier": prov.quality_tier.value,
            "Domain": prov.domain.value,
            "License": prov.license,
            "Commercial OK": "Yes" if prov.commercial_use_allowed else "No",
            "Redistribute OK": "Yes" if prov.redistribution_allowed else "No",
            "Weight": prov.recommended_train_weight,
        })
    return pd.DataFrame(rows)


def handle_mixture_simulation(val_ratio: float, enforce_speaker_disjoint: bool):
    builder = DatasetMixtureBuilder(
        val_ratio=val_ratio,
        enforce_speaker_disjoint=enforce_speaker_disjoint,
    )
    mock_samples = []
    for d_key in ["vivos", "vimd", "bud500", "vietmed", "vietsuperspeech"]:
        for spk_i in range(10):
            spk_id = f"{d_key}_spk_{spk_i:02d}"
            for u in range(4):
                mock_samples.append({
                    "id": f"{spk_id}_u{u}",
                    "dataset": d_key,
                    "speaker_id": spk_id,
                    "duration_sec": 3.5,
                    "transcript": "thu nghiem mo phong",
                })

    train_rows, val_rows, audit = builder.partition_samples(mock_samples)
    lines = [
        "Mixture Partition Simulation:",
        f"- Total input utterances:       {audit['total_input_samples']}",
        f"- Train split utterances:       {audit['train_samples_count']}",
        f"- Validation split utterances:  {audit['val_samples_count']}",
        f"- Speaker-disjoint separation:  {'VERIFIED (0 speaker overlap)' if audit['speaker_disjoint_verified'] else 'NOT ENFORCED'}",
        f"- Zero test contamination:      {'VERIFIED' if audit['zero_test_contamination_verified'] else 'FAILED'}",
    ]
    return "\n".join(lines)


# ==============================================================================
# SETTINGS & MEMORY
# ==============================================================================
def handle_clear_cache():
    ASRModelManager.clear_cache()
    hw = detect_hardware()
    return f"Model cache cleared. Memory released. Current device: {hw.device}"


# ==============================================================================
# GRADIO APPLICATION BUILDER
# ==============================================================================
def build_app():
    minimal_css = """
    body, .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #111827;
        background-color: #fafafa;
        max-width: 1040px !important;
        margin: 0 auto !important;
        padding-top: 16px !important;
    }
    .header-box {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .header-title {
        font-size: 18px;
        font-weight: 600;
        color: #111827;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }
    .header-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin: 0;
    }
    .status-text-bar {
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #374151;
        background: #f3f4f6;
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
    }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 6px;
    }
    button.primary {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500 !important;
    }
    button.secondary {
        background-color: #f3f4f6 !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
    }
    """

    with gr.Blocks(title="Vietnamese ASR Studio", css=minimal_css, theme=gr.themes.Base()) as demo:
        # Header
        gr.HTML(
            """
            <div class="header-box">
                <h1 class="header-title">Vietnamese ASR</h1>
                <p class="header-subtitle">Speech recognition and audio augmentation studio</p>
            </div>
            """
        )

        # Compact Hardware & Model Status
        hw_status_display = gr.HTML(
            value=f"<div class='status-text-bar'>{get_compact_hardware_info()}</div>"
        )

        with gr.Tabs():
            # ==================================================================
            # TAB 1: STUDIO (PRIMARY UNIFIED WORKFLOW)
            # ==================================================================
            with gr.TabItem("Studio"):
                # STEP 1: AUDIO INPUT & VALIDATION
                gr.Markdown("#### Audio Input")
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio source (WAV, MP3, FLAC, or microphone)",
                    )
                with gr.Row():
                    val_status_short = gr.Textbox(
                        value="Audio status: No audio selected.",
                        label="Audio Validation Status",
                        interactive=False,
                        max_lines=1,
                    )

                with gr.Accordion("Validation details (7-Point Gate)", open=False):
                    val_details_box = gr.Textbox(
                        value="1. Signal existence:    PENDING\n2. Finite values:        PENDING\n3. Mono channel:         PENDING\n4. Lower duration bound: PENDING\n5. Upper duration bound: PENDING\n6. RMS signal level:     PENDING\n7. Dynamic range:        PENDING",
                        interactive=False,
                        lines=7,
                        max_lines=8,
                        label="7-Point Gate Breakdown",
                    )

                with gr.Accordion("Reference transcript (Optional for WER/CER benchmarking)", open=False):
                    ref_input = gr.Textbox(
                        label="Ground-truth reference text",
                        placeholder="Enter reference transcript to compute exact Levenshtein WER and CER...",
                        lines=2,
                    )

                # STEP 2: TRANSCRIPTION ACTION
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

                # STEP 3: TRANSCRIPT & METRICS
                gr.Markdown("#### Transcript")
                raw_transcript_box = gr.Textbox(
                    value="No transcription yet.",
                    label="Recognized Text",
                    lines=3,
                    interactive=False,
                )
                with gr.Accordion("Normalized text (NFC, lowercased, punctuation stripped)", open=False):
                    norm_transcript_box = gr.Textbox(
                        label="Normalized Text",
                        lines=2,
                        interactive=False,
                    )
                metrics_box = gr.Textbox(
                    value="Duration: - | Latency: - | RTF: -",
                    label="Performance & Evaluation Metrics",
                    lines=2,
                    interactive=False,
                )

                gr.Markdown("---")

                # STEP 4: AUGMENTATION (COLLAPSIBLE / CLEAN)
                with gr.Accordion("Augmentation Studio (Generate waveform variants)", open=False):
                    gr.Markdown("Generate physically authentic 1D waveform variants validated by the 10-check QC pipeline.")
                    with gr.Row():
                        aug_mode_selector = gr.Radio(
                            choices=["Random", "Exact plan"],
                            value="Random",
                            label="Mode",
                        )
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                            label="Number of variants (K)",
                        )
                        master_seed_input = gr.Number(
                            value=42,
                            label="Master seed",
                            precision=0,
                        )

                    with gr.Group(visible=True) as random_group:
                        transforms_pool = gr.CheckboxGroup(
                            choices=[
                                ("Gain", "gain"),
                                ("Noise (Calibrated SNR)", "additive_noise"),
                                ("Time shift", "time_shift"),
                                ("Time stretch", "time_stretch"),
                                ("Pitch shift", "pitch_shift"),
                                ("Synthetic reverb", "synthetic_reverb"),
                                ("Bandpass filter (300-3400Hz)", "bandpass_filter"),
                            ],
                            value=["gain", "additive_noise", "time_shift", "synthetic_reverb"],
                            label="Transform families",
                        )

                    with gr.Group(visible=False) as exact_plan_group:
                        with gr.Row():
                            plan_gain = gr.Slider(minimum=-12.0, maximum=12.0, value=0.0, step=0.5, label="Gain (dB)")
                            plan_noise = gr.Slider(minimum=10.0, maximum=35.0, value=25.0, step=1.0, label="Noise SNR (dB)")
                            plan_shift = gr.Slider(minimum=-0.5, maximum=0.5, value=0.0, step=0.05, label="Time shift (s)")
                        with gr.Row():
                            plan_stretch = gr.Slider(minimum=0.85, maximum=1.15, value=1.0, step=0.05, label="Time stretch rate")
                            plan_pitch = gr.Slider(minimum=-2.0, maximum=2.0, value=0.0, step=0.5, label="Pitch shift (semitones)")
                            plan_reverb = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="Reverb decay (s)")
                        with gr.Row():
                            plan_flow = gr.Slider(minimum=100.0, maximum=500.0, value=300.0, step=50.0, label="Bandpass low (Hz)")
                            plan_fhigh = gr.Slider(minimum=2500.0, maximum=4500.0, value=3400.0, step=100.0, label="Bandpass high (Hz)")

                    def switch_aug_mode(m):
                        return gr.update(visible=m == "Random"), gr.update(visible=m == "Exact plan")

                    aug_mode_selector.change(switch_aug_mode, inputs=[aug_mode_selector], outputs=[random_group, exact_plan_group])

                    generate_aug_btn = gr.Button("Generate augmentations", variant="primary")
                    aug_status_text = gr.Textbox(value="No augmented audio generated.", label="Augmentation Status", interactive=False)

                    with gr.Accordion("Audio players for generated variants (first 5 variants)", open=False):
                        with gr.Row():
                            prev_1 = gr.Audio(label="Variant 1", type="filepath", interactive=False)
                            prev_2 = gr.Audio(label="Variant 2", type="filepath", interactive=False)
                        with gr.Row():
                            prev_3 = gr.Audio(label="Variant 3", type="filepath", interactive=False)
                            prev_4 = gr.Audio(label="Variant 4", type="filepath", interactive=False)
                        prev_5 = gr.Audio(label="Variant 5", type="filepath", interactive=False)

                gr.Markdown("---")

                # STEP 5: RESULTS & CONSISTENCY TABLE
                gr.Markdown("#### Results & Consistency Diagnostics")
                results_table = gr.DataFrame(
                    headers=["Variant", "Transformation", "Duration (s)", "Transcript", "QC Status", "Drift WER", "Drift CER", "WER vs Ref"],
                    label="Evaluation Comparison",
                    interactive=False,
                )

                gr.Markdown("---")

                # STEP 6: EXPORT
                gr.Markdown("#### Export")
                with gr.Row():
                    export_btn = gr.Button("Export session", variant="secondary")
                export_status_box = gr.Textbox(value="No export bundle generated yet.", label="Export Summary", lines=4, interactive=False)
                download_zip_file = gr.File(label="Download ZIP bundle")

                # Studio Event Bindings
                transcribe_btn.click(
                    fn=handle_transcription,
                    inputs=[
                        audio_input,
                        gr.State(DEFAULT_MODEL_ID),  # Uses configured model
                        gr.State("greedy"),          # Default greedy
                        gr.State(4),                 # Default beams
                        gr.State(192),               # Default safe token limit
                        ref_input,
                    ],
                    outputs=[
                        raw_transcript_box,
                        metrics_box,
                        val_status_short,
                        val_details_box,
                        norm_transcript_box,
                        results_table,
                    ],
                )

                generate_aug_btn.click(
                    fn=handle_augmentation_generation,
                    inputs=[
                        aug_mode_selector,
                        k_slider,
                        master_seed_input,
                        transforms_pool,
                        plan_gain,
                        plan_noise,
                        plan_shift,
                        plan_stretch,
                        plan_pitch,
                        plan_reverb,
                        plan_flow,
                        plan_fhigh,
                    ],
                    outputs=[
                        aug_status_text,
                        prev_1,
                        prev_2,
                        prev_3,
                        prev_4,
                        prev_5,
                        results_table,
                    ],
                )

                export_btn.click(
                    fn=handle_export,
                    outputs=[export_status_box, download_zip_file],
                )

            # ==================================================================
            # TAB 2: BATCH PROCESSING
            # ==================================================================
            with gr.TabItem("Batch Processing"):
                gr.Markdown("#### Batch Audio Transcription")
                gr.Markdown("Upload multiple audio files. Audio files are streamed sequentially without bulk memory allocation.")

                with gr.Row():
                    batch_files_input = gr.File(file_count="multiple", label="Audio files", file_types=["audio"])
                    with gr.Column():
                        batch_strategy = gr.Radio(choices=["greedy", "beam_search"], value="greedy", label="Decoding strategy")
                        batch_beams = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Beam size")
                        batch_run_btn = gr.Button("Run batch transcription", variant="primary")

                batch_status = gr.Textbox(value="No batch files processed.", label="Batch Status", interactive=False)
                batch_table = gr.DataFrame(label="Batch Results Summary", interactive=False)
                batch_download_csv = gr.File(label="Download batch CSV")

                batch_run_btn.click(
                    fn=handle_batch_processing,
                    inputs=[batch_files_input, batch_strategy, batch_beams],
                    outputs=[batch_status, batch_table, batch_download_csv],
                )

            # ==================================================================
            # TAB 3: DATASET PROVENANCE
            # ==================================================================
            with gr.TabItem("Dataset Provenance"):
                gr.Markdown("#### Vietnamese Speech Corpora Registry")
                gr.Markdown("Authoritative specifications, quality tiers, and licensing constraints for candidate training and evaluation corpora.")
                prov_df = gr.DataFrame(value=get_clean_provenance_table, interactive=False)

                gr.Markdown("#### Mixture Partition Simulator")
                with gr.Row():
                    sim_val_slider = gr.Slider(minimum=0.05, maximum=0.30, value=0.10, step=0.05, label="Validation split ratio")
                    sim_spk_disjoint = gr.Checkbox(value=True, label="Enforce speaker-disjoint partition")
                    sim_btn = gr.Button("Simulate partition", variant="secondary")

                sim_output = gr.Textbox(label="Partition Guarantee Summary", lines=6, interactive=False)
                sim_btn.click(
                    fn=handle_mixture_simulation,
                    inputs=[sim_val_slider, sim_spk_disjoint],
                    outputs=[sim_output],
                )

            # ==================================================================
            # TAB 4: SETTINGS & DIAGNOSTICS
            # ==================================================================
            with gr.TabItem("Settings & Diagnostics"):
                gr.Markdown("#### Model Configuration")
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=[
                            "vinai/PhoWhisper-tiny",
                            "vinai/PhoWhisper-base",
                            "vinai/PhoWhisper-small",
                            "vinai/PhoWhisper-medium",
                        ],
                        value=DEFAULT_MODEL_ID,
                        label="Model variant",
                    )
                    dec_strategy_sel = gr.Radio(
                        choices=["greedy", "beam_search"],
                        value="greedy",
                        label="Decoding strategy",
                    )
                    beam_size_sel = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Beam size",
                    )
                    max_tokens_sel = gr.Slider(
                        minimum=32,
                        maximum=448,
                        value=192,
                        step=16,
                        label="Requested max new tokens (safe clamping applied)",
                    )

                gr.Markdown("#### Compute Resources & Memory")
                hw_box = gr.Textbox(value=detect_hardware().summary(), label="Hardware Diagnostics", interactive=False)
                clear_mem_btn = gr.Button("Clear memory cache", variant="secondary")
                clear_mem_status = gr.Textbox(label="Cache Status", interactive=False)

                clear_mem_btn.click(
                    fn=handle_clear_cache,
                    outputs=[clear_mem_status],
                )

                def update_model_choice(m_id):
                    asr_engine.model_id = m_id
                    asr_engine._model = None
                    asr_engine._processor = None
                    current_session_state["model_id"] = m_id
                    return f"<div class='status-text-bar'>{get_compact_hardware_info()}</div>", detect_hardware().summary()

                model_selector.change(
                    update_model_choice,
                    inputs=[model_selector],
                    outputs=[hw_status_display, hw_box],
                )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
