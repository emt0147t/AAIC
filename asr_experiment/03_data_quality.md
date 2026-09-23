# Gate 2 — Data Quality, Provenance & Accessibility Audit

> Classification: **[V]** = verified from original source, **[R]** = from report, **[U]** = unverified/uncertain, **[D]** = discrepancy found

---

## Critical Findings

### VIVOS License Correction
The authoritative license from Zenodo record 7068130 is **CC BY-NC-SA 4.0** (NonCommercial). The HuggingFace card also confirms `cc-by-nc-sa-4.0`. Gate 1 initially reported CC BY-SA 4.0 from a community mirror — this was incorrect.

### VietSuperSpeech Size Discrepancy
The HuggingFace dataset card states **32,267 samples / 103.18 hours**, not the 52,023 / 267.39h claimed in the report. The repo has 118,264 files. Either the report refers to a different version, or there are multiple dataset configurations.

### Access Barriers
ViMD, VietMed, and FOSD (HF mirror) all return HTTP 401. Only FOSD's Mendeley source remains publicly accessible.

---

## Detailed Quality Profiles

### 1. VIVOS

| Dimension | Value | Source |
|-----------|-------|--------|
| Transcript provenance | Human: scripted read speech | [V] |
| Transcript quality | High — uppercase text from prompts | [V] |
| Human vs pseudo | Human | [V] |
| Speaker metadata | speaker_id field, 65 speakers (46 train / 19 test) | [V] |
| Source metadata | AILAB VNUHCM single source | [V] |
| Official splits | train=11,660 / test=760 / NO validation | [V] |
| Duplicate risk | Low | [V] |
| Speaker leakage risk | Controllable — speaker IDs available | [V] |
| Source leakage risk | Low | [V] |
| Audio format | WAV PCM 16-bit mono, 16kHz | [V] |
| Domain | Academic / General Vietnamese | [V] |
| Speech style | Read speech (scripted) | [V] |
| Dialect coverage | UNKNOWN | [U] |
| Access | Public HuggingFace, not gated | [V] |
| License | CC BY-NC-SA 4.0 (Zenodo) | [V] |
| Download size | ~1.5 GB | [V] |

### 2. FOSD

| Dimension | Value | Source |
|-----------|-------|--------|
| Transcript provenance | Human: FPT staff recordings | [V] |
| Transcript quality | High — read speech | [V] |
| Human vs pseudo | Human | [V] |
| Speaker metadata | LIMITED — inconsistent IDs | [V] |
| Official splits | NONE | [V] |
| Speaker leakage risk | DIFFICULT to control | [V] |
| Audio format | WAV PCM 16-bit mono, 16kHz | [V] |
| Domain | General Vietnamese | [V] |
| Speech style | Read speech | [V] |
| Access — Mendeley | Public | [V] |
| Access — HuggingFace | 401 (gated/deleted) | [V] |
| License | CC BY 4.0 (Mendeley) | [V] |
| Download size | ~3.7 GB | [V] |

### 3. ViMD

| Dimension | Value | Source |
|-----------|-------|--------|
| Transcript provenance | Human — ASR baselines in paper | [V] |
| Speaker metadata | Yes — speaker IDs + province/dialect | [V] |
| Official splits | Yes — paper-defined | [V] |
| Speech style | Spontaneous | [V] |
| Dialect coverage | 63 provincial dialects | [V] |
| Access | HuggingFace 401 (GATED) | [V] |
| License | UNKNOWN | [U] |
| All other fields | UNKNOWN — cannot inspect (gated) | [U] |

### 4. VietMed (labeled)

| Dimension | Value | Source |
|-----------|-------|--------|
| Transcript provenance | Human medical transcripts | [V] |
| Domain | Medical (doctor-patient) | [V] |
| Official splits | Yes — train/val/test | [V] |
| Access | HuggingFace 401 (GATED) | [V] |
| License | UNKNOWN | [U] |
| Additional | Also has sentiment labels | [V] |
| All other fields | UNKNOWN — cannot inspect (gated) | [U] |

### 5. VietSuperSpeech

| Dimension | Value | Source |
|-----------|-------|--------|
| Transcript provenance | Pseudo-labeled by Zipformer-30M-RNNT-6000h | [V] |
| Transcript quality | SILVER — not human-verified | [V] |
| Human vs pseudo | Pseudo-labels only | [V] |
| Speaker metadata | NONE | [V] |
| Source metadata | 3 YouTube channels | [V] |
| Official splits | train=29,041 / dev=3,226 / NO test | [V] |
| Speaker leakage risk | UNCONTROLLABLE | [V] |
| Audio format | WAV PCM mono, 16kHz, 3-30s | [V] |
| Domain | News/Community/Vlog (overseas Vietnamese) | [V] |
| Speech style | Spontaneous conversational | [V] |
| Access | Public HuggingFace, not gated | [V] |
| License | MIT (README) / NOT SET (HF field) | [D] |
| Size | 103.18h / 32,267 (HF) vs 267.39h / 52,023 (paper) | [D] |

### 6. VLSP2021 (transcribed 280h + untranscribed 400h)

| Dimension | Transcribed | Untranscribed |
|-----------|-------------|---------------|
| Transcript provenance | Human (competition) | NONE |
| Access | Registration required | Registration required |
| License | VLSP data agreement | VLSP data agreement |
| Status | UNAVAILABLE | UNAVAILABLE |
| All other fields | UNKNOWN | UNKNOWN |

---

## Accessibility Summary

| Dataset | Status | Source |
|---------|--------|--------|
| VIVOS | ACCESSIBLE | HuggingFace |
| FOSD | ACCESSIBLE (Mendeley only) | Mendeley |
| ViMD | GATED — needs HF access request | HuggingFace |
| VietMed | GATED — needs HF access request | HuggingFace |
| VietSuperSpeech | ACCESSIBLE (data integrity unclear) | HuggingFace |
| VLSP2021 | UNAVAILABLE — registration pending | VLSP.org |

---

## Compact Comparison Table

| Dimension | VIVOS | FOSD | ViMD | VietMed labeled | VietSuperSpeech | VLSP2021 transcribed | VLSP2021 untranscribed |
|-----------|-------|------|------|-----------------|-----------------|---------------------|----------------------|
| Size | 15.67h | ~30h | 102.56h | ~16h | 103h (HF) | ~280h | ~400h |
| Utterances | 12,420 | 25,921 | ~19,000 | UNKNOWN | 32,267 (HF) | UNKNOWN | UNKNOWN |
| Transcripts | Human | Human | Human | Human | Pseudo | Human | NONE |
| Speaker meta | Yes (65) | LIMITED | Yes | UNKNOWN | NONE | UNKNOWN | UNKNOWN |
| Splits | train/test | NONE | train/val/test | train/val/test | train/dev | train/dev/test | N/A |
| Leakage ctrl | Yes | Difficult | Yes | UNKNOWN | Impossible | UNKNOWN | N/A |
| Format | WAV 16k | WAV 16k | WAV ?k | UNKNOWN | WAV 16k | WAV 16k (exp) | WAV 16k (exp) |
| Domain | General | General | Multi-dialect | Medical | News/Vlog | General | In-domain |
| Style | Read | Read | Spontaneous | Medical conv | Spont. conv | UNKNOWN | UNKNOWN |
| Dialects | UNKNOWN | UNKNOWN | 63 provinces | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Access | PUBLIC | MENDELEY | **PUBLIC** [G4] | **PUBLIC** [G4] | PUBLIC | UNAVAIL | UNAVAIL |
| License | CC BY-NC-SA 4.0 | **FPT Public License** [G4] | **CC BY-NC-ND 4.0** [G4] | **MIT** [G4] | MIT / NOT SET | VLSP | VLSP |
| Download | 1.5 GB | 3.7 GB | **~74 GB (parquet)** [G4] | ~2-3 GB | ~62.6 GB (?) | ~30-50 GB | ~40-70 GB |

---

## Gate 4 Corrections

- **ViMD**: Access corrected from GATED to PUBLIC. Canonical repo is `nguyendv02/ViMD_Dataset` (not the gated community mirror). License: CC BY-NC-ND 4.0.
- **VietMed**: Access corrected from GATED to PUBLIC. Canonical repo is `leduckhai/VietMed`. Also available via Google Drive.
- **FOSD**: License corrected from "CC BY 4.0" to **FPT Public License** (custom permissive license, commercial OK, requires copyright notice).
- **VietSuperSpeech**: VERSION_UNRESOLVED — excluded from frozen configuration.

