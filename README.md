# 🎙️ Hybrid ASR-NLP Disfluency Restoration System

> An end-to-end pipeline combining Whisper ASR, rule-based insertion modeling, bigram language scoring, and TF-IDF retrieval to restore disfluencies in spontaneous Hindi speech transcripts.

---

## 📌 Overview

Spontaneous speech is messy — speakers use fillers, repetitions, and hesitation markers (*disfluencies*) that get dropped during transcription. This project tackles **automatic disfluency restoration**: given a cleaned transcript and its corresponding audio, reinsert the disfluencies that were originally spoken.

Built for a competitive exam held by **IIT Madras**, this system achieved a **Word Error Rate (WER) of 0.205** — placing **27th out of 129 participants (top 21%)** using only `whisper-tiny` and algorithmic signal fusion, without any fine-tuned Hindi ASR or large language models.

---

## 🧠 Problem

Given:
- A **cleaned transcript** (disfluencies already removed)
- The **original audio file** of the spoken utterance (Hindi / Devanagari)

Predict the **original transcript** with disfluencies restored.

Challenges:
- Hindi spontaneous speech with dialectal variation and code-switching
- Noisy audio with short, ambiguous filler tokens
- 29 known disfluency types, but insertions must be placed precisely
- ASR output is imperfect — cannot blindly trust it

---

## 🏗️ System Architecture

```
Audio (.wav)  ──► Whisper ASR (tiny) ──────────────────────────┐
                                                                ▼
Clean Transcript ──────────────────────► Multi-Signal Insertion Engine
                                                                │
                        ┌───────────────┬────────────────┐      │
                        ▼               ▼                ▼      │
                   Known Disf.    Train Frequency   Short Deva  │
                   Lexicon        + Edit Distance   Filler Rule │
                        └───────────────┴────────────────┘      │
                                        ▼                       │
                              Bigram LM Verification            │
                                        ▼                       │
                              TF-IDF Retrieval Fallback ◄───────┘
                                        │
                                        ▼
                              Restored Transcript
```

---

## ⚙️ Methodology

### Step 1 — ASR Transcription (Whisper)
- Audio files transcribed using `openai-whisper` (`tiny` model) with Hindi (`hi`) language setting
- Robust fallback decoding (`temperature=0.0`) on failure
- ASR output used purely as **evidence** for where disfluencies occurred — not as the final transcript

### Step 2 — Train-derived Insertion Patterns
- Extract all disfluency insertion sequences from the training set by aligning original vs. cleaned transcripts using a token-level diff
- Build frequency counts (`aug_seq_counts`) and context counts (`aug_ctx_counts`) — tracking what comes *before* and *after* each disfluency sequence
- Build a **bigram language model** (Laplace-smoothed) over training transcripts for fluency scoring

### Step 3 — Multi-Signal Acceptance Engine
Candidate insertions from ASR alignment are accepted only if they pass at least one rule:

| Rule | Condition | Rationale |
|---|---|---|
| **A — Known Disfluency** | Token exists in the 29-item disfluency lexicon | Direct lexicon match |
| **B — Train Frequency + Edit Distance** | Seen in training AND reduces token edit distance to ASR | Empirically grounded insertion |
| **C — Short Devanagari Filler** | ≤2 tokens, all Devanagari, length ≤3 chars | Catches short Hindi fillers ASR picks up |

Accepted insertions are ranked by priority (A > B > C), with a maximum of 2 insertions per utterance to avoid over-generation.

### Step 4 — Postprocessing
- Collapse triple-repeat tokens (artifact suppression)
- Drop long non-Devanagari tokens that sneak through ASR noise

### Step 5 — TF-IDF Retrieval Fallback
For test samples where ASR provides no useful signal:
- Retrieve the most similar training example using TF-IDF cosine similarity
- Transfer validated disfluency patterns from the retrieved example if context matches

---

## 📊 Results

| Metric | Value |
|---|---|
| Word Error Rate (WER) | **0.205** |
| Top leaderboard WER | 0.168 |
| Competition Rank | **27 / 129 (Top 21%)** |

> Achieved using `whisper-tiny` — the smallest Whisper variant — with no fine-tuning and no large language models.

---

## 📂 Project Structure

```
├── notebook.ipynb              # Full pipeline (Kaggle)
├── submission.csv              # Final predictions
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Downloading the Dataset

#### Prerequisites
- A Kaggle account
- Kaggle API credentials (`kaggle.json`) configured 👉 https://www.kaggle.com/docs/api

#### Step 1: Download via Kaggle CLI
```bash
kaggle competitions download -c nppe-2-automatic-disfluency-restoration
```

#### Step 2: Extract
```bash
unzip nppe-2-automatic-disfluency-restoration.zip
```

### Running the Notebook

Open `notebook.ipynb` in a Kaggle GPU environment or any environment with CUDA support. All dependencies are installed in the first cell.

**Requirements:**
```
openai-whisper
librosa
soundfile
torch
scikit-learn
pandas
numpy
ffmpeg  # system dependency
```

---

## 📝 Key Takeaways

- **ASR as evidence, not ground truth**: Whisper's output is noisy, especially on short Hindi fillers. Treating it as a signal source rather than a direct answer was the key design decision.
- **Edit distance verification prevents hallucination**: Rule B only accepts an insertion if it measurably closes the gap between the clean transcript and ASR output — this filters out spurious matches.
- **Retrieval fallback adds robustness**: When audio quality is poor and ASR fails entirely, TF-IDF retrieval from training data provides a principled fallback rather than leaving the transcript unchanged.
- **`whisper-tiny` is surprisingly competitive**: Despite being the smallest Whisper model, careful post-processing closed most of the gap to top performers who likely used larger models or fine-tuned ASR.
