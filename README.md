# Hybrid ASR–NLP Disfluency Restoration System

This project addresses the task of **Automatic Disfluency Restoration**: reconstructing the original *spoken*, disfluent transcript (with fillers, hesitations, and repetitions) from a **clean transcript and its corresponding speech audio**.

The work focuses on building a **robust, compute-efficient hybrid ASR–NLP pipeline** designed to run within **Kaggle notebook constraints**.

---

## Problem Overview

Human speech naturally contains disfluencies such as:

* fillers (e.g., *uh, um, hmm*)
* repetitions (e.g., *I… I think*)
* hesitations and false starts

While many speech and NLP systems aim to *remove* these artifacts, this task requires doing the **reverse**:

> **Restore disfluencies that were originally spoken but removed from clean transcripts.**

The challenge lies in reintroducing disfluencies **accurately and conservatively**, using both **audio evidence** and **learned text patterns**, without hallucination.

---

## Dataset Description

The dataset used in this project is the official **NPPE-2: Automatic Disfluency Restoration** dataset provided via Kaggle Competitions.

Due to its large size (~860 MB), the dataset is **not included in this GitHub repository**. Instead, it must be downloaded directly from Kaggle.

---

## Downloading the Dataset (Required)

### Prerequisites

* A Kaggle account
* Kaggle API credentials (`kaggle.json`) configured
  👉 [https://www.kaggle.com/docs/api](https://www.kaggle.com/docs/api)

---

### Step 1: Download Dataset Using Kaggle CLI

Run the following command:

```bash
kaggle competitions download -c nppe-2-automatic-disfluency-restoration
```

This will download a ZIP file containing the full dataset.

---

### Step 2: Extract the Dataset

Unzip the downloaded file:

```bash
unzip nppe-2-automatic-disfluency-restoration.zip
```

This will extract the dataset files and audio folders.

---

## Running the Notebook on Kaggle

This project is intended to be executed **inside a Kaggle Notebook**.

### Step 1: Upload Files to Kaggle Notebook

1. Open **Kaggle → Notebooks**
2. Create a new notebook
3. Upload:

   * The notebook file from this repository
   * The extracted dataset folder via **“Add Data → Upload”**

Kaggle will mount the uploaded dataset under:

```bash
/kaggle/input/
```

---

### Step 2: Expected Dataset Directory Structure

After upload, the dataset must appear **exactly as follows** inside the Kaggle environment:

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

**Do not rename the folder or change the internal structure**, as the notebook relies on fixed paths.

---

### Step 3: Dataset Path Used in Code

The notebook assumes the following paths:

```python
INPUT_DIR = '/kaggle/input/nppe-2-automatic-disfluency-restoration'
AUDIO_DIR = os.path.join(INPUT_DIR, 'downloaded_audios')
```

No further configuration is required.

---

## Approach Summary

Rather than training a large end-to-end generation model, this project adopts a **hybrid, evidence-driven approach** that balances accuracy, interpretability, and computational efficiency.

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

1. Download the dataset using the Kaggle CLI.
2. Extract the dataset ZIP.
3. Open a Kaggle Notebook.
4. Upload the notebook file and extracted dataset folder.
5. Run all notebook cells sequentially.
6. The final predictions will be written to:

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

## 🏁 Outcome

This project demonstrates:

* Strong **data preprocessing and alignment skills**
* Practical **multi-modal reasoning** using audio and text
* Robust ML system design under real-world constraints

---

## 📄 License

This project is intended for **academic and educational use only**.
Dataset ownership and licensing remain with the original Kaggle competition organizers. of this project

* Decide what files to include in `.gitignore`
* Create a **pipeline diagram** for the README
