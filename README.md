# Clinical Note Router: Team Zero, MA6700 Assignment 3

Routes a clinical note to one of nine specialist departments. Model 1 is TF-IDF with logistic regression.
Model 2 is a fine-tuned PubMedBERT. Both use the MTSamples transcriptions (Kaggle, CC0).

Team: Immanuel James Wong Hon Leung (leader), Mevin Ramjeet, Jeffrey Li Chiu Lim, Deevesh Rughoo.

| Deliverable | File |
|---|---|
| 1. Code | `run_pipeline.py` and the `clinical_router/` package (below) |
| 2. Report | `Team_Zero_Report.pdf` (to submit) and `Team_Zero_Report.docx` (the same report, editable in Word) |
| 3. Presentation | `Team_Zero_Presentation.pptx` (12 slides, speaker notes name the presenter and give the script) |

## Run it

Python 3.12, from this folder:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv\Scripts\python run_pipeline.py
```

- **`run_pipeline.py` is the only entry point.** It runs every stage, from the raw CSV to every number and figure in the report.
- It prints the metrics and saves the same log to `outputs/results.txt`. Figures go to `outputs/figures/`, and the split to `outputs/split.csv`.
- The default run loads the nine saved fine-tuning runs from `outputs/runs/`. It needs no GPU and takes a few minutes.
- The first run downloads three public tokenizers from Hugging Face, so it needs internet access. No login or token is used.
- `python run_pipeline.py --retrain` fine-tunes all nine transformer runs again. This needs an NVIDIA GPU and the CUDA build of torch (see `requirements.txt`), and takes about 45 minutes on a 6 GB RTX 4050. GPU training is not bit-for-bit repeatable, so new runs can differ in the third decimal.
- **Optional: Model 2 privacy checks.** The redaction and membership-inference checks for Model 2 need the saved weights (419 MB) in `models/model2/`. They are not in this submission because of their size. Without them, the pipeline skips those two Model 2 checks, says so, and runs everything else.

## Structure

| File | What it does | Report section |
|---|---|---|
| `run_pipeline.py` | Entry point. Runs the stages below in report order. | all |
| `clinical_router/config.py` | Paths, the seed (6700), the fixed choices and the reasons for them | |
| `clinical_router/data.py` | Load and audit the CSV; one label per note; near-duplicates; class rule; note length; **the split, created once** | 2 |
| `clinical_router/preprocessing.py` | Format repair, TF-IDF cleaning, NegEx-style negation, before/after examples, settings test | 3 |
| `clinical_router/baseline.py` | Keyword rules (why not rules?) and Model 1 (TF-IDF + logistic regression) | 1, 4 |
| `clinical_router/transformer.py` | Model 2: fine-tuning, the long-note, checkpoint and seed experiments | 4 |
| `clinical_router/evaluation.py` | Test metrics with bootstrap intervals, per-class F1, McNemar, confusions, errors on real notes | 5 |
| `clinical_router/responsible_ai.py` | Accuracy by sex and age, personal data, redaction, membership inference | 6, 7 |
| `clinical_router/deployment.py` | Calibration, fast-track thresholds, out-of-scope notes, error costs, inference cost | 7 |
| `data/mtsamples.csv` | The dataset (CC0, 4,999 rows) | 2 |
| `outputs/runs/*.npz` | Saved probabilities and training logs of the nine fine-tuning runs | 4 |

## Reproducibility

- One seed, `SEED = 6700`, for Python, NumPy, PyTorch, the split and the bootstraps. It is set in `config.set_seed()`.
- The 60 / 20 / 20 stratified split is made once, in `data.make_split()`. Every model uses it.
- The script checks that no text is in two parts of the split, and prints an MD5 fingerprint of the test split (`1604267baf04`).
- Every model and setting choice uses the validation split. The test split is used only from Section 5 on.
- No credentials, API keys or tokens are in the code. The dataset and the tokenizers are public downloads.

## Sources of adapted code and methods

| Where | Source | What we changed |
|---|---|---|
| `preprocessing.clean_tfidf` | NegEx: Chapman et al. (2001), *J. Biomed. Inform.* 34(5) | Simplified: 13 cues, a fixed six-word window and clause punctuation as the scope end. No pseudo-negation list. |
| `transformer.make_pieces` (head + tail) | Sun et al. (2019), *How to fine-tune BERT for text classification?* | We apply it to 510 text tokens (128 + 382) and compare it with head-only and with overlapping chunks. |
| `transformer.fine_tune` | Hugging Face Transformers tutorial "Fine-tune a pretrained model", native PyTorch loop | We added class-weighted loss, bfloat16 autocast, length-sorted batching of pre-cut pieces, best-epoch restore on validation macro F1, and averaging of piece probabilities per note. |
| `deployment.ece` | Guo et al. (2017), *On calibration of modern neural networks* | Standard 10-bin ECE, written from the formula. |
| `responsible_ai.membership_inference` | Yeom et al. (2018), *Privacy risk in machine learning* | The simplest confidence-threshold attack, measured as ROC AUC. |
| `evaluation.significance` | McNemar (1947); paired bootstrap (Efron and Tibshirani, 1993) | The exact binomial form of McNemar's test, because only 19 notes disagree. |

The libraries (scikit-learn, PyTorch, Transformers, pandas, matplotlib, SciPy) are used through their public APIs.

## Use of an AI assistant

We used an AI assistant (Claude, from Anthropic) to write and restructure this code from our analysis notebook. It also ran the code and drafted the comments.

**[Team: before submission, replace this sentence with what each member checked, changed and can explain.]**
