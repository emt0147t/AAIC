# 04 — Data Selection Analysis (Gate 3 + Gate 4 Updates)

> Classification: **[V]** = verified, **[U]** = unverified, **[D]** = discrepancy

---

## 1. VietSuperSpeech Version Resolution

### Finding: THREE distinct dataset stages exist

| Metric | Stage 1: HF README | Stage 2: arXiv Paper | Stage 3: Current HF Manifests |
|--------|--------------------|--------------------|------------------------------|
| Samples | 32,267 | 52,023 | **67,405** |
| Hours | 103.18h | 267.39h | ~300+h |
| Train | 29,041 | 46,822 | **60,656** |
| Dev | 3,226 | 5,201 | **6,749** |
| Test | None | None | None |
| Sources | 3 channels | 4 channels | 4 channels |
| Split ratio | 90/10 | 90/10 | 90/10 |

### Explanation [V]

- **Stage 1**: Initial HF upload (3 sources: nguoivietdailynews, nguyenkhangofficial, trinhlieu)
- **Stage 2**: Paper snapshot with stricter quality filtering + 4th source (vietcetera)
- **Stage 3**: Latest HF commit `cbf624a` (March 2026) — 60,656 train + 6,749 dev, filtered from 118,259 raw WAV files
- The README was **never updated** after Stage 1
- Over **1,323 commits** on `main` branch, mostly automated `upload-large-folder` batches

### 4 YouTube Source Channels [V]

1. `asr_dataset_nguoivietdailynews` — Overseas Vietnamese news/community
2. `asr_dataset_nguyenkhangofficial` — Entertainment vlogs
3. `asr_dataset_trinhlieu` — Personal vlog, informal conversation
4. `vietcetera` (dir: `audio/asr_segments_vietcetera_part00/`) — Podcasts (HaveASip)

### Resolution Status

**VIETSUPERSPEECH = VERSION_UNRESOLVED**

- README says 32k; paper says 52k; manifest says 67k
- No frozen, tagged release exists
- Cannot be used as a frozen benchmark or final test set
- If used as silver training data, must pin to specific manifest commit

---

## 2. Corrected Access Status

| Dataset | Access Classification | Canonical Source | Notes |
|---------|----------------------|-----------------|-------|
| VIVOS | globally_public | `AILAB-VNUHCM/vivos` (HF) + Zenodo 7068130 | [V] |
| FOSD | public_original_source | Mendeley `k9sxg2twv4/4` | HF mirror `linhtran92/fpt_fosd` returns 401 [V] |
| ViMD | globally_public | `nguyendv02/ViMD_Dataset` (HF canonical) | Gate 2 checked wrong repo `thenghia2206/ViMD` [V] |
| VietMed | public_original_source | `leduckhai/VietMed` (HF) + Google Drive | `doof-ferb/VietMed` is a community mirror [V] |
| VietSuperSpeech | globally_public | `thanhnew2001/VietSuperSpeech` (HF) | VERSION_UNRESOLVED [V] |
| VLSP2021 | registration_required | vlsp.org.vn | Access not obtained [V] |

> [!IMPORTANT]
> Gate 2 incorrectly reported ViMD and VietMed as "gated" because it checked community mirrors, not canonical author repositories.

---

## 3. Gold / Silver / Unlabeled Separation

### GOLD — Human-verified transcripts

| Dataset | Hours | Utterances | Provenance |
|---------|-------|------------|-----------|
| VIVOS | 15.67h | 12,420 | Human scripted read speech |
| FOSD | ~30h | 25,921 | Human (FPT staff recordings) |
| ViMD | 102.56h | ~19,000 | Human transcripts (EMNLP 2024 paper) |
| VietMed labeled | ~16h | UNKNOWN | Human medical transcripts |
| VLSP2021 transcribed | ~280h | UNKNOWN | Human (competition-grade) |

### SILVER — Pseudo-labeled transcripts

| Dataset | Hours | Utterances | Pseudo-label Source |
|---------|-------|------------|-------------------|
| VietSuperSpeech | ~300h+ (HF) | 67,405 (HF) | Zipformer-30M-RNNT-6000h |

### UNLABELED — No transcripts

| Dataset | Hours | Source |
|---------|-------|--------|
| VLSP2021 untranscribed | ~400h | VLSP shared task |
| VietMed unlabeled medical | ~1,000h | Medical domain |
| VietMed unlabeled general | ~1,200h | General domain |

---

## 4. Candidate Data Mix Configurations

### CONFIG A — Small / Reproducible (GOLD only)

| Component | Dataset | Hours | Transcript | Role |
|-----------|---------|-------|-----------|------|
| Primary training | VIVOS train (carved) | ~14h | Gold | Core ASR data |
| Supplement | FOSD (subset) | ~15-30h | Gold | Additional read speech |
| **Total labeled** | | **~29-44h** | **Gold** | |

- **Speech style**: Read speech only
- **Dialect diversity**: Low (UNKNOWN for both)
- **Domain diversity**: Low (general academic)
- **Speaker diversity**: 65 speakers (VIVOS) + UNKNOWN (FOSD)
- **Storage**: ~5.2 GB
- **Leakage risk**: Low (VIVOS has speaker IDs; FOSD lacks them → moderate risk)
- **Pseudo-label risk**: None
- **Preprocessing complexity**: Low (both 16kHz WAV)
- **When appropriate**: Quick reproducible baseline; minimal compute; clean gold-only experiment; when read speech suffices

### CONFIG B — General + Dialect (GOLD, multi-style)

| Component | Dataset | Hours | Transcript | Role |
|-----------|---------|-------|-----------|------|
| Read speech | VIVOS train (carved) | ~14h | Gold | Baseline |
| Supplementary read | FOSD (subset) | ~15-30h | Gold | Coverage |
| Spontaneous/dialect | ViMD (subset) | up to 102h | Gold | Dialect diversity |
| **Total labeled** | | **~29-146h** | **Gold** | |

- **Speech style**: Read + Spontaneous
- **Dialect diversity**: High (63 provincial dialects from ViMD)
- **Domain diversity**: Moderate (academic + multi-dialect)
- **Speaker diversity**: High (ViMD has speaker metadata)
- **Storage**: ~9-24 GB
- **Leakage risk**: Low (ViMD has speaker IDs and official splits)
- **Pseudo-label risk**: None
- **Preprocessing complexity**: Moderate (ViMD sample rate needs verification; domain mismatch between read and spontaneous speech)
- **When appropriate**: When dialect robustness matters; when studying read vs spontaneous domain gap; multi-style ASR research

### CONFIG C — General + Conversational (GOLD + SILVER)

| Component | Dataset | Hours | Transcript | Role |
|-----------|---------|-------|-----------|------|
| Read speech | VIVOS train (carved) | ~14h | Gold | Baseline |
| Conversational | VietSuperSpeech (subset) | up to 300h | **Silver** | Conversational diversity |
| **Total** | | **~14h gold + up to 300h silver** | **Mixed** | |

- **Speech style**: Read + Spontaneous conversational
- **Dialect diversity**: UNKNOWN (VietSuperSpeech dialect distribution not documented)
- **Domain diversity**: Moderate (academic + news/vlog/podcast)
- **Speaker diversity**: UNKNOWN (VietSuperSpeech has no speaker IDs)
- **Storage**: ~1.5 GB (VIVOS) + ~62.6 GB (VietSuperSpeech) = ~64 GB
- **Leakage risk**: High (VietSuperSpeech has no speaker IDs, only 4 source channels)
- **Pseudo-label risk**: **HIGH** — VietSuperSpeech is pseudo-labeled by Zipformer
- **Preprocessing complexity**: Low (both 16kHz WAV)
- **When appropriate**: When large-scale conversational data is needed; SSL experiments needing unlabeled/silver pool; when silver label noise is acceptable; when studying label quality impact

---

## 5. Download Strategy

| Dataset | Full Size | Min Metadata | Min Audio Samples | Streaming? | Action Before Freeze |
|---------|-----------|-------------|-------------------|-----------|---------------------|
| VIVOS | 1.5 GB | HF API (done) | 3-5 samples | Yes | Download full (small) |
| FOSD | 3.7 GB | Mendeley page (done) | 3-5 samples | No (Mendeley) | Download after split design |
| ViMD | ~10-20 GB | HF API (`nguyendv02/ViMD_Dataset`) | 3-5 samples | Yes | Stream samples first |
| VietMed labeled | ~2-3 GB | HF API (`leduckhai/VietMed`) | 3-5 samples | Yes | Stream samples first |
| VietSuperSpeech | ~62.6 GB | HF API + train.json header | 3-5 samples | Yes | Stream ONLY until version resolved |
| VLSP2021 | ~70-120 GB | N/A | N/A | No | Blocked: registration pending |

---

## Gate 4 Updates — Sample Verification Results

### ViMD Schema Verified [V]
- **Columns**: `audio`, `filename`, `gender`, `province_code`, `province_name`, `region`, `speakerID`, `text`
- **License**: CC BY-NC-ND 4.0 (NonCommercial, NoDerivatives)
- **Splits**: train=15,023 / valid=1,900 / test=2,026 (total 18,949)
- **Speakers**: train=10,291 / valid=1,320 / test=1,344 (total 12,955)
- **Duration**: train=81.43h / valid=10.26h / test=10.87h (total 102.56h)
- **Speaker-disjoint**: Confirmed — different speakerIDs across splits
- **Transcript**: Vietnamese with diacritics, lowercase, no punctuation issues observed
- **Audio filename**: `{province_code}_{sequence}.wav` — WAV embedded in parquet

### FOSD License Correction [V]
- **NOT CC BY 4.0** — actual license is **FPT Public License**
- Permissive: free, non-exclusive, worldwide, irrevocable, commercial use allowed
- Requires copyright notice inclusion
- Patent and trademark rights NOT licensed
- Audio format: **MP3** (not WAV — requires conversion for ASR)

### VietMed Schema Verified [V]
- **Columns**: `accent`, `audio`, `audio_name`, `duration`, `gender`, `icd10_code`, `rec_condition`, `role`, `seq_name`, `speaker_name`, `text`, `utterance_id`
- **Audio format**: OGG (not WAV)
- **Duration**: 5-7s per utterance in inspected samples
- **Domain**: Medical (ICD-10 codes present)

### VietSuperSpeech
- **STATUS**: VERSION_UNRESOLVED → EXCLUDED FROM FROZEN CONFIGURATION
- CONFIG C marked: NOT READY TO FREEZE

### VIVOS
- HF mirror returns 401 for file downloads — must download from Zenodo
- Schema verified via HF API only (no streaming samples available without auth)

### Provisional Selection: CONFIG B (VIVOS + ViMD)

