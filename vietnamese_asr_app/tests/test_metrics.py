"""Unit Tests for ASR Evaluation Metrics."""

import pytest
from evaluation.metrics import (
    compute_levenshtein_cer,
    compute_levenshtein_wer,
    evaluate_augmentation_consistency,
)


def test_wer_identical_strings():
    ref = "cộng hòa xã hội chủ nghĩa việt nam"
    hyp = "cộng hòa xã hội chủ nghĩa việt nam"
    res = compute_levenshtein_wer(ref, hyp)
    assert res.error_rate == 0.0
    assert res.substitutions == 0
    assert res.deletions == 0
    assert res.insertions == 0
    assert res.hits == 8


def test_wer_with_errors():
    ref = "hà nội mùa thu"
    hyp = "hà nội mùa đông lạnh"  # 1 sub (thu->đông), 1 ins (lạnh)
    res = compute_levenshtein_wer(ref, hyp)
    assert res.substitutions == 1
    assert res.insertions == 1
    assert res.total_reference_tokens == 4


def test_cer_identical_strings():
    ref = "xin chào"
    hyp = "xin chào"
    res = compute_levenshtein_cer(ref, hyp)
    assert res.error_rate == 0.0


def test_augmentation_consistency():
    orig_hyp = "xin chào các bạn"
    aug_hyp = "xin chào tất cả các bạn"
    eval_res = evaluate_augmentation_consistency(orig_hyp, aug_hyp)
    assert eval_res.consistency_drift_wer > 0.0
    assert eval_res.consistency_drift_cer > 0.0
