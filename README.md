# Hybrid ASR–NLP Disfluency Restoration System

This project addresses the task of **Automatic Disfluency Restoration**: reconstructing the original *spoken*, disfluent transcript (with fillers, hesitations, and repetitions) from a **clean transcript and its corresponding speech audio**.

The work focuses on building a **robust, compute-efficient hybrid ASR–NLP pipeline** suitable for Kaggle notebook environments.

---

## Problem Overview

Human speech naturally contains disfluencies such as:

* fillers (e.g., *uh, um, hmm*)
* repetitions (e.g., *I… I think*)
* hesitations and false starts

While many speech and NLP systems aim to *remove* these artifacts, this task requires doing the **reverse**:

> **Restore disfluencies that were originally spoken but removed from clean transcripts.**

---

## Dataset Description 

The dataset used in this project contains both text and audio modalities.

### Training Data

* **Disfluent transcripts** (text)
* **Unique disfluency token list**
* Used to automatically derive clean transcripts and learn disfluency insertion patterns

### Test Data

* **Clean transcripts** (text)
* **Corresponding speech audio** (`.wav`)
* Goal: predict the original disfluent transcript

---

## Importing the Dataset in Kaggle

This project is designed to run **inside a Kaggle Notebook**.

### Step 1: Upload Dataset to Kaggle

1. Go to **Kaggle → Datasets**
2. Create a new dataset and upload the dataset files

---

### Step 2: Attach Dataset to Notebook

In your Kaggle Notebook:

1. Click **“Add Data”**
2. Attach the dataset you uploaded

Kaggle will automatically mount the dataset at:

```bash
/kaggle/input/<dataset-name>/
```

---

### Step 3: Expected Directory Structure

Ensure the dataset follows this structure:

```
/kaggle/input/nppe-2-automatic-disfluency-restoration/
├── train.csv
├── test.csv
├── unique_disfluencies.csv
└── downloaded_audios/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

---

### Step 4: Dataset Path Used in Code

The notebook assumes the following paths:

```python
INPUT_DIR = '/kaggle/input/nppe-2-automatic-disfluency-restoration'
AUDIO_DIR = os.path.join(INPUT_DIR, 'downloaded_audios')
```

No additional configuration is required.

---

## Approach Summary

Instead of training a large end-to-end generation model, this project adopts a **hybrid, evidence-driven approach** that balances accuracy and compute efficiency.

### 1. Data Preprocessing

* Programmatically derive **clean transcripts** from disfluent training text using known disfluency tokens.
* Align clean and disfluent transcripts to learn **what disfluencies occur and where they are inserted**.

### 2. Audio Evidence via ASR

* Use **Whisper ASR (tiny)** to transcribe test audio.
* Treat ASR output as *noisy spoken evidence*, not ground truth.

### 3. Controlled Disfluency Insertion

* Compare ASR output with clean transcripts to identify candidate insertions.
* Insert disfluencies only when supported by:

  * Known disfluency tokens
  * Previously observed insertion patterns from training data
  * Reduction in edit distance to ASR output

### 4. Retrieval-Based Fallback

* Apply **TF-IDF similarity** over training data when ASR evidence is weak.
* Borrow disfluency patterns from similar sentences conservatively.

This design prioritizes **precision over hallucination**, ensuring natural and realistic outputs.

---

## Technologies Used

* **Python**
* **Whisper ASR** (OpenAI / HuggingFace)
* **PyTorch**
* **Librosa & SoundFile** (audio preprocessing)
* **Scikit-learn** (TF-IDF retrieval)
* **Pandas / NumPy** (data handling)

---

## Key Design Choices & Rationale

| Design Choice                   | Rationale                                     |
| ------------------------------- | --------------------------------------------- |
| Whisper Tiny                    | Lightweight and fast under Kaggle constraints |
| Rule-based insertion acceptance | Prevents hallucinated disfluencies            |
| No end-to-end training          | Limited data and compute                      |
| Hybrid ASR–NLP system           | Combines audio evidence with text structure   |

---

## How to Run

1. Open the notebook in a Kaggle environment.
2. Attach the dataset as described above.
3. Run all notebook cells sequentially.
4. The final predictions will be written to:

```bash
submission.csv
```

---

## Possible Extensions

* Fine-tuning Whisper on disfluent speech
* Using forced alignment (CTC-based) for precise insertion timing
* Replacing TF-IDF retrieval with semantic embeddings
* Learning probabilistic insertion positions with lightweight seq2seq models

---

## Outcome

This project demonstrates:

* Strong **data preprocessing and alignment skills**
* Practical **multi-modal reasoning** using audio and text
* Robust ML system design under real-world constraints

---

## License

This project is intended for academic and educational use only.
* Decide what files to include in `.gitignore`
* Create a **pipeline diagram** for the README
