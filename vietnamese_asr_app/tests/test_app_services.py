"""Test Suite for Application Services Layer.

Tests:
1. Model switching and adapter attachment/detachment.
2. Dynamic parameter calculation vs hardcoded values.
3. Frozen Full E1 read-only protocol and CPU safety execution gate.
4. Synthetic audio generation, 10-point QC evaluation, and provenance logging.
5. Reference transcript requirement for accuracy vs PREDICTION DRIFT / CONSISTENCY.
6. Neutral descriptive delta comparisons (no promotional winner claims).
7. Section 10 machine-readable JSON object formatting.
8. Section 12 Markdown report generation.
9. Session and experiment export packaging (CSV, JSON, Markdown, ZIP).
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from services.asr_service import ASRService
from services.evaluation_service import EvaluationService
from services.experiment_service import ExperimentService
from services.export_service import ExportService
from services.lora_service import FROZEN_FULL_E1_PROTOCOL, LoRAService
from services.synthetic_service import SyntheticService


@pytest.fixture
def temp_export_dir():
    d = tempfile.mkdtemp(prefix="test_app_export_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_lora_dynamic_parameters():
    """Verify that dynamic parameter calculation produces exact numbers without hardcoding."""
    lora_service = LoRAService()
    stats = lora_service.compute_dynamic_parameters(model_id="vinai/PhoWhisper-tiny")
    assert stats.total_params == 37_908_096
    assert stats.trainable_params == 147_456
    assert stats.frozen_params == 37_760_640
    assert abs(stats.trainable_ratio_pct - 0.3890) < 0.005


def test_frozen_full_e1_protocol_and_safety_gate():
    """Verify that Full E1 protocol is frozen and read-only, and safety gate blocks CPU execution."""
    lora_service = LoRAService()
    proto = lora_service.get_frozen_protocol()
    assert proto["experiment_id"] == "FULL_E1"
    assert proto["total_optimizer_steps"] == 2502
    assert proto["training_samples"] == 26671
    assert proto["expected_trainable_parameters"] == 147456
    assert proto["frozen_test_manifest_sha256"] == "efe54856d12613f109cebc9482a6f224ca4e2370d0ff3f8a2fe4ea0a013b60ca"

    # Execution on CPU host must be rejected by safety gate
    is_passed, summary_msg, reasons = lora_service.validate_full_e1_execution_gate(
        test_manifest_path="manifests/frozen_test_manifest.csv"
    )
    # Since test runner is on CPU, it must be BLOCKED
    if not torch.cuda.is_available():
        assert not is_passed
        assert "BLOCKED" in summary_msg
        assert any("CUDA is required" in r for r in reasons)


def test_evaluation_reference_enforcement():
    """Verify reference transcript requirement: accuracy when present, drift when missing."""
    eval_service = EvaluationService()

    # Case A: Reference provided -> Accuracy mode
    res_acc = eval_service.evaluate_prediction(
        hypothesis="xin chào việt nam",
        reference="xin chào việt nam",
    )
    assert res_acc.mode == "ACCURACY"
    assert res_acc.wer == 0.0
    assert res_acc.cer == 0.0
    assert res_acc.exact_match is True

    # Case B: Reference missing, baseline provided -> Prediction Drift mode
    res_drift = eval_service.evaluate_prediction(
        hypothesis="xin chào việt nam mới",
        reference="",
        baseline_hypothesis="xin chào việt nam",
    )
    assert res_drift.mode == "PREDICTION_DRIFT_CONSISTENCY"
    assert res_drift.wer is not None
    assert "PREDICTION DRIFT / CONSISTENCY" in res_drift.summary_text
    assert "Not accuracy" in res_drift.summary_text

    # Case C: Both missing -> Unavailable mode
    res_unavail = eval_service.evaluate_prediction(
        hypothesis="xin chào",
        reference="",
        baseline_hypothesis="",
    )
    assert res_unavail.mode == "UNAVAILABLE"
    assert res_unavail.wer is None


def test_neutral_comparison_deltas():
    """Verify neutral descriptive deltas (absolute and relative) without winner claims."""
    eval_service = EvaluationService()
    comp = eval_service.compute_neutral_comparison(
        base_wer=18.93,
        adapted_wer=19.21,
        base_cer=11.07,
        adapted_cer=12.16,
        label_a="E0 Baseline",
        label_b="E1 LoRA Real",
    )
    assert comp["delta_wer_absolute"] == 0.28
    assert abs(comp["delta_wer_relative_pct"] - 1.48) < 0.05
    assert comp["delta_cer_absolute"] == 1.09
    assert "Delta: +0.28 percentage points" in comp["narrative"]
    assert "winner" not in comp["narrative"].lower()
    assert "better" not in comp["narrative"].lower()


def test_synthetic_service_generation_qc_and_provenance(temp_export_dir):
    """Verify synthetic speech generation, 10-point QC evaluation, and provenance lineage."""
    synth_service = SyntheticService(export_dir=temp_export_dir)
    res = synth_service.generate_single_utterance(
        transcript="thử nghiệm tổng hợp giọng nói việt nam",
        voice_id="spk_synth_01",
        source_dataset="vivos_pilot",
        source_text_id="txt_test_001",
    )

    assert os.path.exists(res["audio_path"])
    assert res["duration_sec"] > 0.2
    assert res["qc_status"] in ["PASS", "PASSED", "FAIL", "FAILED"]
    assert len(res["audio_sha256"]) == 64

    # Provenance fields check
    prov = res["provenance"]
    assert prov["synthetic_id"].startswith("synth_spk_synth_01")
    assert prov["source_text_id"] == "txt_test_001"
    assert prov["source_dataset"] == "vivos_pilot"
    assert prov["tts_model"] == "g-group-ai-lab/gwen-tts-0.6B"
    assert prov["voice_id"] == "spk_synth_01"
    assert prov["built_in_voice"] == "Yến Nhi"
    assert prov["qc_status"] == res["qc_status"]

    # QC metrics summary check
    metrics = synth_service.get_qc_summary_metrics()
    assert metrics["generated_count"] >= 1
    assert metrics["total_duration_sec"] > 0.0


def test_experiment_service_section10_json():
    """Verify Section 10 machine-readable JSON object adheres to schema."""
    exp_service = ExperimentService()
    json_obj = exp_service.generate_section10_json("FULL_E1")

    assert json_obj["experiment_id"] == "FULL_E1"
    assert json_obj["model"]["name"] == "vinai/PhoWhisper-tiny"
    assert json_obj["model"]["revision"] == "cc51d32be916efebde04ff549854fa1741cb5c02"
    assert json_obj["adapter"]["enabled"] is True
    assert json_obj["data"]["sample_count"] == 26671
    # For FULL_E1, metrics must be null because experiment is pending cloud execution
    assert json_obj["metrics"]["wer"] is None
    assert json_obj["metrics"]["cer"] is None
    assert json_obj["training"]["optimizer_steps"] == 2502
    assert json_obj["training"]["trainable_parameters"] == 147456
    assert json_obj["training"]["peak_vram_mb"] == 726.16  # Measured from T4 preflight


def test_experiment_service_section12_markdown():
    """Verify Section 12 markdown report distinguishes MEASURED, ESTIMATED, and N/A."""
    exp_service = ExperimentService()
    md_report = exp_service.generate_section12_markdown_report("FULL_E1")

    assert "# EXPERIMENT SUMMARY" in md_report
    assert "FULL E1 RESULT: NOT YET AVAILABLE" in md_report
    assert "MEASURED" in md_report
    assert "ESTIMATED" in md_report
    assert "N/A" in md_report
    assert "726.16 MB" in md_report  # Measured T4 peak VRAM


def test_export_service_bundle(temp_export_dir):
    """Verify multi-format export bundle generates CSV, JSON, Markdown, and ZIP."""
    exp_service = ExperimentService()
    export_service = ExportService(base_export_dir=temp_export_dir)

    df_results = exp_service.get_results_dataframe()
    json_obj = exp_service.generate_section10_json("FULL_E1")
    md_report = exp_service.generate_section12_markdown_report("FULL_E1")

    bundle = export_service.export_experiment_bundle(
        experiment_id="FULL_E1",
        results_df=df_results,
        json_obj=json_obj,
        markdown_report=md_report,
    )

    assert os.path.exists(bundle["csv_path"])
    assert os.path.exists(bundle["json_path"])
    assert os.path.exists(bundle["markdown_path"])
    assert os.path.exists(bundle["zip_path"])
    assert os.path.getsize(bundle["zip_path"]) > 0


def test_asr_service_lifecycle_and_adapter():
    """Verify ASRService base model switching, adapter attachment, and status reporting."""
    asr = ASRService(default_model_id="vinai/PhoWhisper-tiny")
    assert asr.model_id == "vinai/PhoWhisper-tiny"
    assert asr.model_revision == "cc51d32be916efebde04ff549854fa1741cb5c02"
    assert not asr.is_adapter_active

    # Model switching
    asr.set_base_model("vinai/PhoWhisper-small")
    assert asr.model_id == "vinai/PhoWhisper-small"
    assert not asr.is_adapter_active

    # Switch back to tiny and test adapter loading
    asr.set_base_model("vinai/PhoWhisper-tiny")
    adapter_path = "checkpoints/E1_lora_real/checkpoint_epoch_3"
    if os.path.exists(adapter_path):
        load_res = asr.load_adapter(adapter_path, adapter_name="E1_Real_Ep3")
        assert asr.is_adapter_active
        assert asr.active_adapter_name == "E1_Real_Ep3"
        assert load_res["status"] == "LOADED"

        # Unload adapter
        asr.unload_adapter()
        assert not asr.is_adapter_active
        assert asr.active_adapter_name == "None (Base Model)"

