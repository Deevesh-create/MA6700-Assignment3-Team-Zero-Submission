"""Test-split evaluation: metrics with bootstrap intervals, per-class F1, significance, errors on real notes.

Macro F1 is the primary metric: each department has the same right to its notes, so a model that is good
only on the large departments (Cardiovascular has 4.3 times the notes of Ophthalmology) must score low.
The test split is used here for the first time; every choice before this point used the validation split.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import config


def macro_f1(y_true, y_pred, k):
    """Macro F1 over all k classes; a class with no prediction counts as 0, not as missing."""
    return f1_score(y_true, y_pred, average="macro", labels=list(range(k)), zero_division=0)


def bootstrap_indices(n):
    """The same 2,000 resamples of the test notes for every model, so two models can be compared pairwise."""
    return np.random.default_rng(config.SEED).integers(0, n, size=(config.N_BOOT, n))


def score(name, probs, split, boot_idx, verbose=True):
    """Score one model on the test split; return its predictions, bootstrap scores and headline metrics."""
    probs = np.asarray(probs)
    assert probs.shape == (len(split.test), split.k), "these are not test-split probabilities"
    y, pred = split.y_test, probs.argmax(1)
    boot = np.array([macro_f1(y[i], pred[i], split.k) for i in boot_idx])
    s = {"name": name, "probs": probs, "pred": pred, "boot": boot,
         "macro_f1": macro_f1(y, pred, split.k), "ci": (np.percentile(boot, 2.5), np.percentile(boot, 97.5)),
         "accuracy": accuracy_score(y, pred), "weighted_f1": f1_score(y, pred, average="weighted")}
    if verbose:
        print(f"\n--- {name}: test split ---")
        print(classification_report(y, pred, target_names=split.classes, digits=3, zero_division=0))
    print(f"{name}: macro F1 {s['macro_f1']:.4f} (95% CI {s['ci'][0]:.3f} to {s['ci'][1]:.3f}) | "
          f"accuracy {s['accuracy']:.4f} | weighted F1 {s['weighted_f1']:.4f}")
    return s


def comparison_table(scores, costs):
    """One row per model: quality and cost side by side."""
    rows = [{"model": s["name"], "macro F1": round(s["macro_f1"], 4),
             "95% CI": f"{s['ci'][0]:.3f}-{s['ci'][1]:.3f}", "accuracy": round(s["accuracy"], 4),
             **costs.get(s["name"], {})} for s in scores]
    print("\n--- 5.1 Comparison on the test split ---")
    print(pd.DataFrame(rows).set_index("model").fillna("").to_string())


def short(name):
    """Short department names for plot axes."""
    for long, s in [("Cardiovascular / Pulmonary", "Cardio/Pulm"), ("Obstetrics / Gynecology", "Ob/Gyn"),
                    ("ENT - Otolaryngology", "ENT"), ("Hematology - Oncology", "Hem/Onc"),
                    ("General Medicine", "Gen Med"), ("Gastroenterology", "Gastro")]:
        name = name.replace(long, s)
    return name


def per_class(split, s1, s2):
    """F1 per department for both models, with the test support, as a table and a figure."""
    k = list(range(split.k))
    table = pd.DataFrame({"test notes": np.bincount(split.y_test, minlength=split.k),
                          "Model 1 F1": f1_score(split.y_test, s1["pred"], average=None, labels=k),
                          "Model 2 F1": f1_score(split.y_test, s2["pred"], average=None, labels=k)},
                         index=split.classes)
    table["difference"] = table["Model 2 F1"] - table["Model 1 F1"]
    print("\n--- 5.2 F1 per department ---")
    print(table.round(3).to_string())
    ax = table[["Model 1 F1", "Model 2 F1"]].iloc[::-1].plot.barh(figsize=(8, 5.5), color=["tab:gray", "tab:blue"])
    ax.set_xlabel("F1 on the test split"); ax.set_xlim(0, 1); ax.set_title("F1 for each department")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    config.save_figure(ax.figure, "fig_per_class_f1.png")


def significance(split, s1, s2):
    """Is Model 2 really better? A paired bootstrap of the macro F1 gap, and McNemar's exact test."""
    diff = s2["boot"] - s1["boot"]
    print("\n--- 5.3 Is the difference real? ---")
    print(f"macro F1 difference {s2['macro_f1'] - s1['macro_f1']:+.4f} | 95% paired bootstrap "
          f"{np.percentile(diff, 2.5):+.4f} to {np.percentile(diff, 97.5):+.4f} | "
          f"Model 2 better in {(diff > 0).mean():.1%} of resamples")
    c1, c2 = s1["pred"] == split.y_test, s2["pred"] == split.y_test
    only1, only2 = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    # McNemar uses only the notes where the models disagree on correctness; exact binomial because n is small.
    p = binomtest(only2, only1 + only2, 0.5).pvalue
    print(f"both correct {int((c1 & c2).sum())} | only Model 1 {only1} | only Model 2 {only2} | "
          f"both wrong {int((~c1 & ~c2).sum())} | McNemar exact p = {p:.4f}")


def confusion(split, scores):
    """Confusion matrices for both models (a figure) and their most frequent confusions (printed)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, s in zip(axes, scores):
        cm = confusion_matrix(split.y_test, s["pred"], labels=list(range(split.k)))
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(split.k):
            for j in range(split.k):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        names = [short(c) for c in split.classes]
        ax.set_xticks(range(split.k)); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(split.k)); ax.set_yticklabels(names)
        ax.set_xlabel("predicted department"); ax.set_ylabel("true department")
        ax.set_title(f"{s['name']}\nmacro F1 {s['macro_f1']:.3f}")
    config.save_figure(fig, "fig_confusion.png")
    print("\n--- 5.4 Most frequent confusions (true -> predicted: notes) ---")
    for s in scores:
        pairs = Counter((split.classes[t], split.classes[p]) for t, p in zip(split.y_test, s["pred"]) if t != p)
        print(s["name"].split(":")[0], "|", "; ".join(f"{short(t)} -> {short(p)}: {c}" for (t, p), c in pairs.most_common(5)))


def misrouted(split, s1, s2, n=6):
    """The test notes Model 2 got wrong with the highest confidence, with their website labels."""
    wrong = np.where(s2["pred"] != split.y_test)[0]
    wrong = wrong[np.argsort(-s2["probs"][wrong].max(1))]
    print(f"\n--- 5.5 The {n} most confident errors of Model 2 ---")
    for i in wrong[:n]:
        print(f"* true {split.classes[split.y_test[i]]} | Model 2 {split.classes[s2['pred'][i]]} "
              f"({s2['probs'][i].max():.3f}) | Model 1 {split.classes[s1['pred'][i]]} | "
              f"title '{split.test['sample_name'][i]}' | website labels "
              f"{' + '.join(sorted(split.test['label_set'][i]))}")
        print("  ", split.test["text"][i][:200], "...")


def length_effect(split, scores):
    """Accuracy on short and long notes: does head + tail truncation cost Model 2 more than Model 1?"""
    long_ = (split.test["n_tokens"] > config.BODY).to_numpy()
    print(f"\n--- 5.5 Accuracy by note length ({(~long_).sum()} short, {long_.sum()} long test notes) ---")
    for s in scores:
        ok = s["pred"] == split.y_test
        print(f"{s['name']}: short {ok[~long_].mean():.3f} | long {ok[long_].mean():.3f}")


def negation_effect(split, pairs):
    """Remove every negation cue and predict again; a model that uses negation changes its answers."""
    print("\n--- 5.6 Do the models use negation? (cues removed from the test notes) ---")
    idx = np.arange(len(split.y_test))
    for s, noneg in pairs:
        pred = noneg.argmax(1)
        print(f"{s['name']}: predictions that change {(pred != s['pred']).mean():.1%} | macro F1 "
              f"{s['macro_f1']:.4f} -> {macro_f1(split.y_test, pred, split.k):.4f} | mean change in true-class "
              f"probability {np.mean(noneg[idx, split.y_test] - s['probs'][idx, split.y_test]):+.4f}")
