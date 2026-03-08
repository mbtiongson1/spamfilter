# Naive Bayes Spam Classifier

**Author:** Marcus Rafael B. Tiongson  

---

## Overview

This notebook implements a **Naive Bayes classifier** that categorizes emails as **spam** or **ham** (not spam). The model is trained and evaluated on a subset of the **TREC06 Public Spam Corpus**, a standard benchmark for email spam filtering research.

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

Set the `source` variable in the notebook to your local path before running:

```python
source = 'trec06p-ai201'
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

---

## Requirements

```
python >= 3.8
matplotlib
pandas
tabulate
```

Install dependencies:

```bash
pip install matplotlib pandas tabulate
```

No other external libraries are required beyond the Python standard library (`email`, `re`, `unicodedata`, `math`, `csv`, `collections`, `os`, `copy`).
