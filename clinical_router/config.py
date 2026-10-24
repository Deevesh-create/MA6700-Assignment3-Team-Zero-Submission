"""Settings shared by every stage of the pipeline: paths, the seed, the device and the fixed choices.

Every value that a reader might want to change lives here, so the other modules contain only logic.
The values are the ones the team fixed before looking at the test split; the comments say why.
"""
import os

# Must be set before torch starts CUDA: it makes cuBLAS matrix products repeatable on the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # a script writes figures to files; it needs no window
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "mtsamples.csv"
# Public CC0 dataset; Kaggle serves it without a login, so no credentials are needed or stored.
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/tboyle10/medicaltranscriptions"
OUT_DIR = ROOT / "outputs"
RUN_DIR = OUT_DIR / "runs"          # one .npz per fine-tuning run, so the pipeline can run without a GPU
FIG_DIR = OUT_DIR / "figures"
MODEL2_COPY = ROOT / "models" / "model2"   # optional: saved weights of Model 2, only for the privacy checks

SEED = 6700  # the course code; one seed for Python, NumPy, PyTorch and every split

# Section 2.3: labels that name a document format, not a department, so they cannot be a routing target.
DOC_TYPES = {"Consult - History and Phy.", "SOAP / Chart / Progress Notes", "Discharge Summary",
             "Emergency Room Reports", "Office Notes", "Letters", "IME-QME-Work Comp etc."}
# Website pages that collect all operations or all imaging; the specialty page is the real department.
UMBRELLA = {"Surgery", "Radiology"}
MIN_WORDS = 20        # shorter "notes" are headings only (for example "CHIEF COMPLAINT:")
NEAR_DUP = 0.9        # TF-IDF cosine at or above this is the same template; keeps copies out of two splits
MIN_NOTES = 75        # 75 gives every class >= 15 test notes with a 20% test split (Section 2.4 of the report)

# PubMedBERT reads 512 positions; [CLS] and [SEP] take two, so 510 are left for text.
BODY = 510
HEAD = 128            # head + tail keeps the first 128 and the last 382 tokens (Sun et al., 2019)
STRIDE = 256          # the chunk option overlaps windows by half, so no sentence is cut at every border
PUBMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
CHECKPOINTS = {"PubMedBERT": PUBMEDBERT,
               "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
               # the legacy id "bert-base-uncased" fails to download (xet 404); this is the same model
               "BERT-base": "google-bert/bert-base-uncased"}

N_BOOT = 2000         # bootstrap resamples for every confidence interval
TARGET_ACCURACY = 0.95  # a suggestion goes on the fast track only where validation accuracy reaches this

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch, and ask PyTorch for repeatable GPU kernels where they exist."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # warn_only: a few GPU kernels have no deterministic version; we accept them and say so in the report.
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_dirs():
    """Create the output folders if they are missing."""
    for d in (RUN_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name):
    """Write a figure to outputs/figures and close it, because a script has no screen to show it on."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
