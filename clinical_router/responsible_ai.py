"""Responsible-AI measurements: performance by patient group, personal data, redaction and memorisation.

The dataset has no sex or age column, so we read them from the note text ("a 64-year-old female").
This is imperfect and the groups are small; the report states both limits next to the numbers.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from . import config
from .evaluation import macro_f1, short
from .transformer import predict_notes

SEX_WORDS = {"female": "female", "woman": "female", "lady": "female", "girl": "female",
             "male": "male", "man": "male", "gentleman": "male", "boy": "male"}

# Direct identifiers replaced before a model reads a note (GDPR Art. 5(1)(c) and 25: data minimisation).
# Ages stay because they are clinically relevant. (kind, pattern, replacement, regex flags)
REDACTIONS = [
    ("e-mail address", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]", 0),
    ("phone number", r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)", "[PHONE]", 0),
    ("date", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? "
             r"\d{1,2}(?:st|nd|rd|th)?,? \d{4}\b", "[DATE]", 0),
    ("record number", r"\b(?:MRN|medical record(?: number)?|record number|account number|SSN|NHS number)\s*[:#]?\s*"
                      r"[A-Z0-9-]{4,}\b|\b\d{7,}\b", "[ID]", re.I),
    ("name after a title", r"\b(Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?", r"\1. [NAME]", 0),
]


def patient_sex(text):
    """Sex from "NN-year-old <word>", else from the majority of she/her against he/him; else unknown."""
    m = re.search(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\s+(?:[a-z]+\s+){0,2}?"
                  r"(female|woman|lady|girl|male|man|gentleman|boy)\b", text, flags=re.I)
    if m:
        return SEX_WORDS[m.group(2).lower()]
    t = text.lower()
    she = len(re.findall(r"\b(she|her|hers)\b", t))
    he = len(re.findall(r"\b(he|him|his)\b", t))
    return "female" if she > he else "male" if he > she else "unknown"


def patient_age(text):
    """Age band from the first "NN-year-old"; the bands match child, working-age adult and older adult."""
    m = re.search(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", text, flags=re.I)
    if not m:
        return "unknown"
    a = int(m.group(1))
    return "0-17" if a < 18 else "18-64" if a < 65 else "65+"


def subgroups(split, scores):
    """Accuracy with a 95% bootstrap interval for each sex and age group, and the department mix by sex."""
    groups = pd.DataFrame({"sex": [patient_sex(t) for t in split.test["text"]],
                           "age": [patient_age(t) for t in split.test["text"]]})
    y = split.y_test

    def acc_ci(mask, pred):
        """Accuracy of one group with a 1,000-resample bootstrap interval."""
        idx = np.where(mask)[0]
        rng = np.random.default_rng(config.SEED)
        boots = [accuracy_score(y[s], pred[s]) for s in (rng.choice(idx, len(idx)) for _ in range(1000))]
        return f"{accuracy_score(y[idx], pred[idx]):.3f} ({np.percentile(boots, 2.5):.2f}-{np.percentile(boots, 97.5):.2f})"

    rows = []
    for col in ("sex", "age"):
        for g in sorted(groups[col].unique()):
            mask = (groups[col] == g).to_numpy()
            rows.append({"group": f"{col}: {g}", "test notes": int(mask.sum()),
                         **{s["name"].split(":")[0]: acc_ci(mask, s["pred"]) for s in scores}})
    print("\n--- 6.1 Accuracy by patient group (95% bootstrap interval) ---")
    print(pd.DataFrame(rows).set_index("group").to_string())
    # Groups differ in department mix (all Ob/Gyn notes are female), which can explain a gap by itself.
    print("department mix by sex:")
    print(pd.crosstab(split.test["label"].map(short), groups["sex"]).to_string())


def personal_data(split):
    """Count notes that still contain quasi-identifiers; the website removed names, real notes would not."""
    data = pd.concat([split.train, split.val, split.test])
    checks = {"an age": r"\b\d{1,3}[- ](?:year|yr)s?[- ]old\b", "a date": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
              "a doctor's name or placeholder": r"\bDr\.\s*[A-Z]", "a placeholder name (XYZ, ABC)": r"\b(XYZ|ABC)\b"}
    print("\n--- 6.2 Personal data still in the kept notes ---")
    for k, v in checks.items():
        n = int(data["text"].str.contains(v, regex=True).sum())
        print(f"notes with {k}: {n} ({n / len(data):.1%})")


def redact(text):
    """The text with direct identifiers replaced, and how many of each kind were found."""
    counts = {}
    for kind, pattern, replacement, flags in REDACTIONS:
        text, n = re.subn(pattern, replacement, text, flags=flags)
        if n:
            counts[kind] = n
    return text, counts


def copy_predictor(copy, split):
    """A text -> probabilities function for the saved Model 2 copy, or None when the copy is absent."""
    if copy is None:
        return None
    model, tok = copy
    return lambda texts: predict_notes(model, tok, list(texts), "head_tail", split.k)


def redaction_check(split, model1, model2):
    """Macro F1 with and without redaction: data minimisation must not cost routing accuracy."""
    found, touched = Counter(), 0
    redacted = []
    for t in split.test["text"]:
        r, counts = redact(t)
        redacted.append(r)
        found.update(counts)
        touched += bool(counts)
    print("\n--- 7.4 Redaction of direct identifiers ---")
    print(f"test notes with a direct identifier: {touched} of {len(split.test)} | replaced: {dict(found)}")
    for name, predict in [("Model 1", model1["predict"]), ("Model 2 (saved copy)", model2)]:
        if predict is None:
            print(f"{name}: skipped (put the saved weights in models/model2 to run it)")
            continue
        raw, red = predict(split.test["text"]).argmax(1), predict(redacted).argmax(1)
        print(f"{name}: macro F1 {macro_f1(split.y_test, raw, split.k):.4f} without redaction, "
              f"{macro_f1(split.y_test, red, split.k):.4f} with | predictions that change {int((raw != red).sum())}")


def membership_inference(split, model1, model2):
    """Simplest membership attack (Yeom et al., 2018): does true-class confidence separate training notes?"""
    print("\n--- 7.4 Membership inference (AUC 0.5 = cannot tell training notes from test notes) ---")
    for name, predict in [("Model 1", model1["predict"]), ("Model 2 (saved copy)", model2)]:
        if predict is None:
            print(f"{name}: skipped (put the saved weights in models/model2 to run it)")
            continue
        tr = predict(split.train["text"])[np.arange(len(split.train)), split.y_train]
        te = predict(split.test["text"])[np.arange(len(split.test)), split.y_test]
        auc = roc_auc_score(np.r_[np.ones(len(tr)), np.zeros(len(te))], np.r_[tr, te])
        print(f"{name}: mean true-class confidence, training {tr.mean():.4f} | test {te.mean():.4f} | AUC {auc:.4f}")
