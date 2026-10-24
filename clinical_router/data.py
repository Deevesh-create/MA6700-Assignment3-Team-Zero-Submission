"""From the raw CSV to one clean train / validation / test split, created once.

Steps: load and audit the file; collapse the rows into unique notes (the website lists one note on several
pages); give each note one department or remove it; remove near-duplicate templates; keep departments with
enough notes; measure note length in PubMedBERT tokens; split 60 / 20 / 20, stratified, with the fixed seed.
"""
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from . import config
from .preprocessing import repair_format


@dataclass
class Split:
    """The one split that every model uses, plus the out-of-scope notes and the class names."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    oos: pd.DataFrame          # notes of the departments removed in Section 2.4, never used for training
    classes: list

    @property
    def k(self):
        """Number of departments."""
        return len(self.classes)

    @property
    def y_train(self):
        """Training labels as integers."""
        return self.train["y"].to_numpy()

    @property
    def y_val(self):
        """Validation labels as integers."""
        return self.val["y"].to_numpy()

    @property
    def y_test(self):
        """Test labels as integers."""
        return self.test["y"].to_numpy()


def load_raw():
    """Load the MTSamples CSV, downloading the public Kaggle copy first if it is missing."""
    if not config.DATA_FILE.exists():
        config.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        archive = config.DATA_FILE.parent / "medicaltranscriptions.zip"
        urllib.request.urlretrieve(config.DATA_URL, archive)
        zipfile.ZipFile(archive).extractall(config.DATA_FILE.parent)
    raw = pd.read_csv(config.DATA_FILE, index_col=0)
    raw["medical_specialty"] = raw["medical_specialty"].str.strip()
    return raw


def audit(raw):
    """Print the size, the missing values, and the leakage check that decides which column we use."""
    print("--- 2.1 The file ---")
    print(f"rows {len(raw):,} | columns {list(raw.columns)} | labels {raw['medical_specialty'].nunique()}")
    print("missing values:", raw.isna().sum().to_dict())
    kw = raw.dropna(subset=["keywords"])
    starts = pd.Series([k.lower().strip().startswith(s.lower())
                        for k, s in zip(kw["keywords"], kw["medical_specialty"])])
    # If the keyword string begins with the label, a model would learn to read the answer, not the note.
    print(f"rows whose keywords start with the label: {starts.sum():,} of {len(kw):,} ({starts.mean():.1%})"
          " -> we use only the transcription")


def resolve(labels):
    """One department for a set of website labels, or the reason the note is removed."""
    specialties = labels - config.DOC_TYPES
    if not specialties:
        return "REMOVE: document type only"
    specific = specialties - config.UMBRELLA
    if len(specific) == 1:
        return next(iter(specific))
    if len(specific) == 0 and len(specialties) == 1:
        return next(iter(specialties))          # only "Surgery" or only "Radiology": that page is the department
    # Two real specialties (for example a spine operation under Neurosurgery and Orthopedic): a single-label
    # model cannot be right for both, so the note is removed rather than given an arbitrary label.
    return "REMOVE: two or more specialties"


def unique_notes(raw):
    """Collapse the rows into unique transcriptions with their label sets, and apply the one-label rule."""
    notes = raw.dropna(subset=["transcription"]).copy()
    notes["raw_text"] = notes["transcription"].str.strip()
    notes = notes[notes["raw_text"] != ""]
    # Group by the exact text: the same note on two website pages must never be split into two rows,
    # or the same text could land in training and in test.
    label_sets = notes.groupby("raw_text")["medical_specialty"].agg(frozenset)
    n_labels = label_sets.map(len)
    print("\n--- 2.2 One note, several labels ---")
    print(f"rows with a transcription {len(notes):,} | unique transcriptions {len(label_sets):,} | "
          f"with 2+ labels {(n_labels > 1).sum():,} ({(n_labels > 1).mean():.1%})")
    print("labels per unique note:", n_labels.value_counts().sort_index().to_dict())

    first_row = notes.drop_duplicates("raw_text").set_index("raw_text")
    uniq = pd.DataFrame({"label_set": label_sets})
    uniq["label"] = uniq["label_set"].map(resolve)
    uniq["sample_name"] = first_row.loc[uniq.index, "sample_name"].str.strip()
    uniq["text"] = [repair_format(t) for t in uniq.index]
    uniq["n_words"] = uniq["text"].str.split().str.len()
    uniq.loc[uniq["n_words"] < config.MIN_WORDS, "label"] = f"REMOVE: fewer than {config.MIN_WORDS} words"
    uniq = uniq.reset_index(drop=True)

    print("\n--- 2.3 One label for each note ---")
    outcome = uniq["label"].where(uniq["label"].str.startswith("REMOVE"), "kept: one department")
    print(outcome.value_counts().to_string())
    two = uniq["label_set"][uniq["label"] == "REMOVE: two or more specialties"]
    pairs = two.map(lambda s: " + ".join(sorted(s - config.DOC_TYPES - config.UMBRELLA))).value_counts()
    print("most frequent removed pairs:", pairs.head(5).to_dict())
    return uniq, len(label_sets)


def remove_near_duplicates(uniq):
    """Drop the second note of every pair with TF-IDF cosine >= 0.9 (same template, a few words changed)."""
    single = uniq[~uniq["label"].str.startswith("REMOVE")].reset_index(drop=True)
    S = cosine_similarity(TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(single["text"]))
    np.fill_diagonal(S, 0)
    drop = set()
    for i, j in zip(*np.where(np.triu(S) >= config.NEAR_DUP)):
        if i not in drop:
            drop.add(j)
    pairs = np.argwhere(np.triu(S) >= config.NEAR_DUP)
    same = sum(single["label"][i] == single["label"][j] for i, j in pairs)
    single = single.drop(index=sorted(drop)).reset_index(drop=True)
    print(f"near-duplicate pairs {len(pairs)} ({same} with the same label) | removed {len(drop)} | "
          f"single-department notes {len(single):,}")
    return single


def choose_classes(single, n_unique):
    """Keep departments with >= MIN_NOTES notes; the rest become the out-of-scope test set."""
    counts = single["label"].value_counts()
    print("\n--- 2.4 Class reduction rule ---")
    rows = []
    for min_n in (50, 75, 100):
        c = counts[counts >= min_n]
        rows.append({"rule (notes >=)": min_n, "classes": len(c), "notes": int(c.sum()),
                     "smallest class": f"{c.idxmin()} ({c.min()})",
                     "its test notes at 20%": int(c.min() * 0.20)})
    print(pd.DataFrame(rows).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["tab:blue" if n >= config.MIN_NOTES else "lightgray" for n in counts]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.axvline(config.MIN_NOTES, color="tab:red", ls="--", label=f"rule: {config.MIN_NOTES} notes")
    ax.axvline(50, color="gray", ls=":", label="proposal: 50 notes")
    ax.set_xlabel("single-department notes (after duplicates removed)")
    ax.set_title("Notes for each department")
    ax.legend()
    config.save_figure(fig, "fig_class_counts.png")

    classes = list(counts[counts >= config.MIN_NOTES].index)     # largest class first
    data = single[single["label"].isin(classes)].reset_index(drop=True)
    data["y"] = data["label"].map({c: i for i, c in enumerate(classes)})
    oos = single[~single["label"].isin(classes)].reset_index(drop=True)
    print(f"classes kept {len(classes)} | notes kept {len(data):,} ({len(data) / n_unique:.1%} of unique notes) | "
          f"largest / smallest {counts[classes].max()} / {counts[classes].min()} | "
          f"out-of-scope notes {len(oos)} from {oos['label'].nunique()} departments")
    return data, oos, classes


def measure_length(data, classes):
    """Count PubMedBERT tokens per note: this decides how much text a 512-token model loses."""
    tok = AutoTokenizer.from_pretrained(config.PUBMEDBERT)
    data["n_tokens"] = [len(x) for x in tok(list(data["text"]), add_special_tokens=False)["input_ids"]]
    n = data["n_tokens"]
    over = n > config.BODY
    lost = (n - config.BODY).clip(lower=0)
    print("\n--- 2.5 Note length ---")
    print(f"words per note: median {data['n_words'].median():.0f} | tokens per note: median {n.median():.0f}, "
          f"max {n.max()} | tokens per word {n.sum() / data['n_words'].sum():.2f}")
    print(f"notes longer than {config.BODY} tokens: {over.sum():,} of {len(n):,} ({over.mean():.1%}) | "
          f"text lost by first-512, median long note: {(lost[over] / n[over]).median():.1%}, "
          f"p90: {(lost[over] / n[over]).quantile(.9):.1%}")
    print("long notes by class:", data.assign(o=over).groupby("label")["o"].mean().loc[classes]
          .map("{:.0%}".format).to_dict())

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(n.clip(upper=3000), bins=60, color="tab:blue")
    ax.axvline(config.BODY, color="tab:red", ls="--", label="510 text tokens")
    ax.set_xlabel("PubMedBERT tokens per note (notes above 3,000 shown at 3,000)")
    ax.set_ylabel("notes")
    ax.legend()
    config.save_figure(fig, "fig_token_lengths.png")


def make_split(data, oos, classes):
    """Create the 60 / 20 / 20 stratified split once, check it for leakage and save it to outputs/split.csv."""
    # 20% test (not 15%) so the smallest class has 15 test notes: one note then moves its recall by <= 6.7 points.
    train, rest = train_test_split(data, test_size=0.40, stratify=data["y"], random_state=config.SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["y"], random_state=config.SEED)
    train, val, test = (d.reset_index(drop=True) for d in (train, val, test))
    assert not (set(train.text) & set(val.text) or set(train.text) & set(test.text)
                or set(val.text) & set(test.text)), "a text is in two splits"
    test_md5 = hashlib.md5("".join(sorted(test.text)).encode()).hexdigest()

    table = pd.DataFrame({name: d["label"].value_counts() for name, d in
                          [("train", train), ("validation", val), ("test", test)]}).loc[classes]
    table.loc["TOTAL"] = table.sum()
    print("\n--- 2.6 The split (created once) ---")
    print(f"train {len(train)} | validation {len(val)} | test {len(test)} | test MD5 {test_md5[:12]}")
    print(table.to_string())
    pd.concat([d.assign(split=name)[["split", "label", "sample_name"]]
               for name, d in [("train", train), ("validation", val), ("test", test)]]) \
        .to_csv(config.OUT_DIR / "split.csv", index=False)
    return Split(train, val, test, oos, classes)

