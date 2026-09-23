# 06 — Model Contamination Audit (Gate 3)

> Classification: CONFIRMED_CONTAMINATED | PROBABLE | UNKNOWN | NO_EVIDENCE_FOUND
> "NO_EVIDENCE_FOUND" is NOT the same as "clean" — it means no documentation of overlap was found

---

## Master Contamination Matrix

| Model \ Dataset | VIVOS | FOSD | VLSP2020 | VLSP2021 | ViMD | VietMed | VietSuperSpeech |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PhoWhisper** (all sizes) | CONFIRMED | NEF | CONFIRMED | NEF | NEF | NEF | NEF |
| **OpenAI Whisper** (base) | NEF | NEF | NEF | NEF | NEF | NEF | NEF |
| **XLS-R** (Meta base) | NEF | NEF | NEF | NEF | NEF | NEF | NEF |
| **wav2vec2-vi-250h** | CONFIRMED (eval) | NEF | CONFIRMED (train) | NEF | NEF | NEF | NEF |
| **wav2vec2-vi-160h** (khanhld) | CONFIRMED | CONFIRMED | CONFIRMED | NEF | NEF | NEF | NEF |
| **w2v2-Viet** (leduckhai) | NEF | NEF | NEF | NEF | NEF | CONFIRMED | NEF |
| **Zipformer-30M-6000h** | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | NEF | CONFIRMED | CONFIRMED* |
| **MMS** (mms-1b-all) | NEF | NEF | NEF | NEF | NEF | NEF | NEF |

*CONFIRMED = CONFIRMED_CONTAMINATED, NEF = NO_EVIDENCE_FOUND*
*Zipformer ↔ VietSuperSpeech: circular evaluation — Zipformer is the teacher model that created VietSuperSpeech pseudo-labels*

---

## Detailed Evidence per Model

### PhoWhisper (vinai) — All sizes [V]
- **Paper**: arXiv:2406.02555, ICLR 2024 Tiny Papers
- **Training data** (843.79h total, Table 1):
  - VIVOS: 13.94h train, 0.98h valid, 0.75h test — **CONFIRMED in training**
  - VLSP 2020: Task-1 + Task-2 (~243.8h) — **CONFIRMED in training**
  - Common Voice Vi 14: 14.0h
  - VinAI private data: 586.0h (26,000 speakers, 63 provinces)
- **Evaluation**: VIVOS test, VLSP 2020 Task-1 test, VLSP 2020 Task-2 test
- **License**: Apache 2.0

### OpenAI Whisper (base multilingual) [V]
- **Paper**: arXiv:2212.04356
- **Training**: 680,000h weakly supervised internet audio; Vietnamese ~1,700h web-scraped
- **No documented overlap** with any Vietnamese academic benchmark
- **Caveat**: Proprietary web scrape — incidental overlap cannot be completely disproven

### Zipformer-30M-RNNT-6000h (hynt) [V]
- **HuggingFace**: hynt/Zipformer-30M-RNNT-6000h
- **Training pool (~6,000h)** explicitly lists:
  - VLSP2020, VLSP2021, VLSP2023
  - FPT (= FOSD)
  - VIVOS
  - VietMed_Labeled
  - FLEURS, VIET_BUD500, VietSpeech, ViVoice, Sub-GigaSpeech2-Vi, Sub-PhoAudioBook
- **Critical**: This model is the teacher that generated VietSuperSpeech pseudo-labels
- **ViMD**: NOT in training pool → NEF

### Meta MMS (mms-1b-all) [V]
- **Paper**: arXiv:2305.13516
- **Training**: MMS-lab (New Testament readings, 1,107 langs) + FLEURS + Common Voice
- **Vietnamese**: Only from Bible recordings + FLEURS/CV
- **No documented overlap** with VIVOS, FOSD, VLSP, ViMD, VietMed, VietSuperSpeech

---

## Critical Implications

1. **PhoWhisper was trained on VIVOS**: Any PhoWhisper WER on VIVOS test measures in-domain performance, not generalization.

2. **Zipformer trained on 5 of our 7 candidates**: VIVOS, FOSD, VLSP2020, VLSP2021, VietMed. Only ViMD and VietSuperSpeech were not in its training pool — but VietSuperSpeech has circular contamination (teacher model).

3. **ViMD is the only dataset with NEF across ALL models examined**: Published EMNLP 2024, postdates training runs of PhoWhisper (2024), Zipformer, MMS (2023), Whisper (2022).

4. **Models with least contamination across our datasets**: OpenAI Whisper (base), MMS — both have NEF for all candidates. But both are general multilingual models, not Vietnamese-specialized.
