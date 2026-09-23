# 05 — Test Set Candidate Audit (Gate 3)

> No test set is finalized. This is a candidate analysis only.
> A dataset with a good official split is NOT automatically a clean test for every pretrained ASR model.

---

## Test Candidate Table

| Candidate | Human Transcript | Speaker-Disjoint | Official Split | Source-Disjoint | Model Contamination Risk | Access | Suitability Concerns |
|-----------|:---:|:---:|:---:|:---:|---|---|---|
| **VIVOS official test** (760 utt) | ✅ Gold | ✅ 19 speakers (test-only) | ✅ Official test | ✅ Single source | **HIGH**: PhoWhisper trained on VIVOS; Zipformer trained on VIVOS | globally_public | Contaminated for PhoWhisper and Zipformer; read speech only; small (760 utt) |
| **ViMD official test** | ✅ Gold | ✅ Speaker IDs available | ✅ Paper-defined | ✅ Independent project | **LOW**: NO_EVIDENCE_FOUND for PhoWhisper, Whisper, MMS, Zipformer | globally_public | Spontaneous multi-dialect speech; may be challenging; dialect imbalance possible |
| **VietMed test** | ✅ Gold | UNKNOWN | ✅ Paper-defined | ✅ Independent project | **MODERATE**: Zipformer and leduckhai models trained on VietMed; PhoWhisper/Whisper/MMS clean | public_original_source | Medical domain only; domain-specific vocabulary; small labeled set |
| **FOSD custom split** | ✅ Gold | ⚠️ Difficult (weak meta) | ❌ No official split | ✅ FPT source | **MODERATE**: Zipformer and khanhld models trained on FOSD; PhoWhisper/Whisper/MMS clean | public_original_source | No official split; speaker-disjoint split difficult; read speech |
| **VietSuperSpeech dev** (6,749 utt HF) | ❌ Silver (pseudo) | ❌ No speaker IDs | ⚠️ Combined dev/test | ⚠️ Only 4 channels | **CRITICAL for Zipformer**: teacher model created the labels; PhoWhisper/Whisper/MMS clean | globally_public | VERSION_UNRESOLVED; pseudo-labels; no speaker control; circular eval for Zipformer |
| **VLSP2021 official test** | ✅ Gold | UNKNOWN | ✅ Competition test | UNKNOWN | **HIGH**: Zipformer trained on VLSP2021; PhoWhisper used VLSP2020 not 2021 | registration_required | Access not obtained; competition dataset |

---

## Per-Model Test Set Contamination Summary

### Which test candidates are clean per model?

| Test Candidate | PhoWhisper | Whisper (orig) | MMS | Zipformer | wav2vec2-vi-250h | wav2vec2-vi-160h |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| VIVOS test | **CONTAMINATED** | NEF | NEF | **CONTAMINATED** | **CONTAMINATED** | **CONTAMINATED** |
| ViMD test | NEF | NEF | NEF | NEF | NEF | NEF |
| VietMed test | NEF | NEF | NEF | **CONTAMINATED** | NEF | NEF |
| FOSD split | NEF | NEF | NEF | **CONTAMINATED** | NEF | **CONTAMINATED** |
| VietSuperSpeech | NEF | NEF | NEF | **CONTAMINATED** (teacher) | NEF | NEF |
| VLSP2021 test | NEF* | NEF | NEF | **CONTAMINATED** | NEF | NEF |

*NEF = NO_EVIDENCE_FOUND (not "clean")*
*PhoWhisper used VLSP2020, not VLSP2021 — but these may share speakers/sources*

---

## Key Observations

1. **ViMD official test is the strongest cross-model test candidate**: NEF across ALL examined models. Published EMNLP 2024, postdates training of PhoWhisper, Zipformer, MMS, Whisper. Multi-dialect spontaneous speech provides genuine out-of-domain evaluation.

2. **VIVOS test is contaminated for the most practical baseline models** (PhoWhisper, Zipformer). It can only be used as a "known ceiling" reference, not a true generalization test.

3. **VietSuperSpeech cannot serve as any test set**: VERSION_UNRESOLVED, pseudo-labeled, no speaker control, and circular evaluation for Zipformer.

4. **FOSD** could work for PhoWhisper/Whisper/MMS evaluation but lacks official splits and speaker metadata.

5. **VietMed test** is clean for PhoWhisper/Whisper/MMS but domain-specific (medical).
