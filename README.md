# Naive Bayes Spam Classifier

A spam classifier that categorizes emails as **spam** or **ham** (not spam), trained and evaluated on the **TREC06 Public Spam Corpus**.

---

## Requirements

```
python >= 3.8
matplotlib
pandas
tabulate
```

```bash
pip install matplotlib pandas tabulate
```

No other external libraries are required beyond the Python standard library (`email`, `re`, `unicodedata`, `math`, `csv`, `collections`, `os`, `copy`).

---

## File Structure

```
naivebayes/
├── main.py            # Entry point — runs the full pipeline
├── config.py          # Dataset source path
├── dataset.py         # Label loading, TrainingSplit, test.csv persistence
├── preprocessing.py   # parse() and clean() email functions
├── vocabulary.py      # VocabularyExtractor, save_vocabs(), load_vocabs()
├── classifier.py      # priors(), denom(), classify(), Scores, topwords()
├── evaluate.py        # Lambda sweep, topwords sweep, tablegraph(), plotgraph()
└── storage.py         # save_model(), save_results()
```

---

## Usage

### 1. Set the dataset path

Open `config.py` and set `source` to the path of your local TREC06 corpus:

```python
source = 'trec06p-ai201'
```

If the dataset is in the same directory as the scripts, the default value works as-is.

### 2. Run the full pipeline

```bash
python main.py
```

This runs all steps in order — loading the dataset, building the vocabulary, sweeping lambda values, running the topwords improvement, and saving all outputs.

### 3. Outputs

After a successful run, four files will be written to the working directory:

| File | Contents |
|------|----------|
| `test.csv` | The 70/30 folder split (cached so reruns use the same split) |
| `vocabs.csv` | Ham and spam word frequency tables |
| `model.csv` | Full trained model: word counts, vocabulary flags, and metadata |
| `results.csv` | Evaluation metrics across all lambda and top-n configurations |

### 4. Reuse individual modules

Each module can be imported independently for reuse in other scripts.

**Classify a single raw email string:**

```python
from vocabulary import VocabularyExtractor, load_vocabs
from classifier import denom, classify
from config import source

vocabs = VocabularyExtractor(source, trainingset=None)
load_vocabs(vocabs)  # load from vocabs.csv

LAMBDA_VAL = 0.005
dham, dspam = denom(vocabs, LAMBDA_VAL)

label, pspam, pham, _ = classify(raw_email_text, vocabs, LAMBDA_VAL, dham, dspam)
print(label)  # 'spam' or 'ham'
```

**Rebuild vocabulary from scratch (skips cache):**

```python
from dataset import load_labels, load_or_create_split
from vocabulary import VocabularyExtractor, save_vocabs
from config import source
import dataset

dataset.load_labels()
tests = sorted(dataset.tests)
trec06 = load_or_create_split(tests)

vocabs = VocabularyExtractor(source, trec06)
vocabs.build_vocabs(trec06)
save_vocabs(vocabs)
```

**Run evaluation with custom lambda values:**

```python
from evaluate import run_lambda_sweep, tablegraph, plotgraph

results = run_lambda_sweep(vocabs, LAMBDA_VALUES=[0, 0.1, 0.005])
df = tablegraph(results)
plotgraph(df)
```

**Apply topwords filtering before classifying:**

```python
from classifier import topwords, denom, classify

filtered_vocabs, removed = topwords(100, vocabs)
dham, dspam = denom(filtered_vocabs, 0.005)
label, pspam, pham, _ = classify(raw_email_text, filtered_vocabs, 0.005, dham, dspam)
```

---

## Overview

This project implements a **Naive Bayes classifier** that categorizes emails as **spam** or **ham** (not spam). The model is trained and evaluated on a subset of the **TREC06 Public Spam Corpus**, a standard benchmark for email spam filtering research.

---

## Dataset

**TREC06 Public Spam Corpus** — a chronologically ordered collection of emails (primarily English, with some Chinese) used in the TREC 2006 Spam Track.

### Directory Structure

```
trec06p-ai201/
├── data/
│   ├── 000/
│   │   ├── 000
│   │   ├── 001
│   │   └── ...
│   └── 126/
└── labels
```

The `labels` file maps each email to a class using the format:

```
ham ../data/000/000
spam ../data/000/001
```

---

## Methodology

### Naive Bayes Formula

The classifier uses the Naive Bayes posterior:

$$P(C_k \mid X) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$

All computations are performed in **log space** for numerical stability.

### Laplace (Lambda) Smoothing

To handle words absent from the training vocabulary, lambda smoothing is applied:

$$P(x_i \mid C_k) = \frac{N_{ik} + \lambda}{N_k + \lambda V}$$

Setting `λ = 0` disables smoothing; any `λ > 0` prevents zero-probability collapse.

### Decision Rule

$$\hat{y} = \begin{cases} \text{spam} & \text{if Score}_\text{spam} > \text{Score}_\text{ham} \\ \text{ham} & \text{otherwise} \end{cases}$$

---

## Pipeline

| Step | Description |
|------|-------------|
| **1. Load Data** | Parse `labels` file; separate ham/spam file pairs |
| **2. Split** | 70/30 train/validation split across email folders |
| **3. Parse & Clean** | Extract headers and body; normalize text (lowercase, URLs, emails, digits, Chinese characters) |
| **4. Build Vocabulary** | Extract word frequencies from training set for both classes |
| **5. Classify** | Score each test email using log-probability Naive Bayes |
| **6. Evaluate** | Compute Precision, Recall, and F1 score |
| **7. Optimize** | Tune `λ` and `top_n` (stopword removal) for best F1 |

---

## Email Preprocessing

The `parse()` and `clean()` functions handle the following:

- **Headers used:** Subject, From, Reply-To, Received, Content-Type
- **Attachments:** Flagged with an `attachment` token
- **URLs** → `url`
- **Email addresses** → `emailaddr`
- **Repeated punctuation** (`!!`, `??`) → `multiexclaim`
- **Prices** (`$100`) → `price`
- **Digits** → `num`
- **Chinese characters** → split into individual character tokens
- Text is lowercased; extra whitespace is removed

---

## Generated Files

| File | Description |
|------|-------------|
| `test.csv` | Folder index split (full / training / testing sets) |
| `vocabs.csv` | Ham and spam word frequency tables |
| `model.csv` | Full model: word counts, vocabulary membership, and metadata (priors, denominators, vocab size) |
| `results.csv` | Evaluation results across all lambda and top-n configurations |

---

## Results

### Preliminary Results (varying λ, no stopword removal)

| λ | Accuracy | Precision | Recall | F1 |
|---|----------|-----------|--------|----|
| 0 | 0.3908 | 1.0000 | 0.0925 | 0.1693 |
| 2.0 | 0.8267 | 0.9958 | 0.7449 | 0.8523 |
| 1.0 | 0.8382 | 0.9961 | 0.7620 | 0.8634 |
| 0.5 | 0.8473 | 0.9961 | 0.7755 | 0.8721 |
| 0.1 | 0.8644 | 0.9966 | 0.8007 | 0.8880 |
| **0.005** | **0.8835** | **0.9964** | **0.8294** | **0.9053** |

### Final Results (λ = 0.005, with top-n stopword removal)

| top-n | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| 200 | 0.9716 | 0.9835 | 0.9740 | 0.9788 |
| **100** | **0.9721** | **0.9830** | **0.9752** | **0.9791** |

**Best configuration: λ = 0.005, top-n = 100 → F1 = 0.9791**

---

## Conclusion

The best Precision-Recall balance is achieved at **λ = 0.005** with **top-n = 100** (removing the 100 most frequent non-discriminative words such as stopwords, HTML tags, SMTP headers, and single letters). Higher Precision is preferred to minimize legitimate emails being marked as spam (false positives).

Disabling smoothing (λ = 0) results in perfect Precision but very poor Recall (F1 = 0.1692), as the classifier defaults conservatively to `ham` for any unseen word.
