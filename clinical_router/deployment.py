"""Deployment analysis: calibration, the fast-track threshold, notes from unknown departments, and cost.

The worst error for a hospital is a confident wrong route that nobody checks: the note sits in the wrong
queue and the patient waits. A note sent to a person costs a few seconds of staff time. So every threshold
here is chosen to keep fast-track suggestions at least 95% correct on validation, and the report counts
both error types per 1,000 notes. Under GDPR Article 22 a person still confirms every note; the fast track
only decides which notes arrive pre-filled and which need a full review.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .evaluation import macro_f1, short


def ece(probs, y, bins=10):
    """Expected calibration error: the mean gap between confidence and accuracy over 10 confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def calibration(split, scores):
    """Print the ECE of each model and draw the reliability diagram."""
    print("\n--- 7.1 Calibration on the test split ---")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for s, color in zip(scores, ("tab:gray", "tab:blue")):
        conf, ok = s["probs"].max(1), s["pred"] == split.y_test
        xs, ys = [], []
        edges = np.linspace(0, 1, 11)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (conf > lo) & (conf <= hi)
            if sel.sum() >= 5:   # fewer than 5 notes in a bin gives a meaningless point
                xs.append(conf[sel].mean()); ys.append(ok[sel].mean())
        e = ece(s["probs"], split.y_test)
        ax.plot(xs, ys, "o-", color=color, label=f"{s['name'].split(':')[0]} (ECE {e:.3f})")
        print(f"{s['name']}: ECE {e:.4f} | mean confidence {conf.mean():.3f} | accuracy {ok.mean():.3f}")
    ax.plot([0, 1], [0, 1], "k:", label="perfect calibration")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability on the test split")
    ax.legend()
    config.save_figure(fig, "fig_calibration.png")


def pick_threshold(probs, y, target=config.TARGET_ACCURACY):
    """Lowest confidence threshold whose suggestions are >= target accurate (on >= 10 notes)."""
    conf, ok = probs.max(1), probs.argmax(1) == y
    for t in np.round(np.arange(0.30, 1.00, 0.01), 2):
        if (conf >= t).sum() >= 10 and ok[conf >= t].mean() >= target:
            return t
    return 1.0


def fast_track(split, s1, s2, val1, val2, oos1, oos2):
    """Thresholds from validation, then the fast-track share, accuracy and error costs per rule."""
    t1, t2 = pick_threshold(val1, split.y_val), pick_threshold(val2, split.y_val)
    print(f"\n--- 7.2 Thresholds chosen on validation (target {config.TARGET_ACCURACY:.0%}): "
          f"Model 1 {t1}, Model 2 {t2} ---")

    def keys(p1, p2):
        """Which notes each rule puts on the fast track."""
        a1, a2 = p1.max(1) >= t1, p2.max(1) >= t2
        # Two keys: two models with different features rarely make the same confident error.
        return {"Model 1 alone": a1, "Model 2 alone": a2, "Two keys": a1 & a2 & (p1.argmax(1) == p2.argmax(1))}

    test, oos = keys(s1["probs"], s2["probs"]), keys(oos1, oos2)
    rows = []
    for rule in test:
        ok = (s1["pred"] if rule == "Model 1 alone" else s2["pred"]) == split.y_test
        a = test[rule]
        rows.append({"rule": rule, "fast track": f"{a.mean():.1%}", "accuracy on fast track": round(ok[a].mean(), 3),
                     "accuracy of the rest": round(ok[~a].mean(), 3),
                     "out-of-scope on fast track (always wrong)": f"{oos[rule].mean():.1%}",
                     # error costs per 1,000 incoming notes, in-scope and out-of-scope counted separately
                     "wrong fast-track per 1,000 in-scope": round(1000 * (a & ~ok).mean()),
                     "full reviews per 1,000 in-scope": round(1000 * (~a).mean())})
    print(pd.DataFrame(rows).set_index("rule").to_string())
    for s in (s1, s2):
        top2 = (np.argsort(-s["probs"], 1)[:, :2] == split.y_test[:, None]).any(1)
        print(f"{s['name']}: top-2 accuracy {top2.mean():.3f} (a reviewer who sees two suggestions)")
    print(f"mean confidence on out-of-scope notes: Model 1 {oos1.max(1).mean():.3f} | Model 2 {oos2.max(1).mean():.3f}")

    a2 = oos["Model 2 alone"]
    tab = pd.crosstab(split.oos["label"].to_numpy()[a2], np.array(split.classes)[oos2[a2].argmax(1)],
                      rownames=["true department"], colnames=["sent to"])
    print(f"where Model 2 alone sends the {a2.sum()} out-of-scope notes it is sure of (top 6 departments):")
    print(tab.loc[tab.sum(1).sort_values(ascending=False).index[:6]].rename(columns=short).to_string())
    coverage_figure(split, (s1, s2))


def coverage_figure(split, scores):
    """Accuracy of the suggestions as more notes go on the fast track, most confident first."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s, color in zip(scores, ("tab:gray", "tab:blue")):
        order = np.argsort(-s["probs"].max(1))
        ok = (s["pred"] == split.y_test)[order]
        n = np.arange(1, len(order) + 1)
        ax.plot(n / len(order), np.cumsum(ok) / n, color=color, label=s["name"].split(":")[0])
    ax.axhline(config.TARGET_ACCURACY, color="tab:red", ls="--", lw=1, label="95% target")
    ax.set_xlabel("share of test notes on the fast track (most confident first)")
    ax.set_ylabel("accuracy of fast-track suggestions"); ax.set_ylim(0.85, 1.01); ax.legend(loc="lower left")
    ax.set_title("Coverage and accuracy")
    config.save_figure(fig, "fig_coverage.png")


def cost(split, model1, run2):
    """Training and inference cost of both models, the numbers a hospital IT team asks for first."""
    print("\n--- 7.3 Cost ---")
    print(f"Model 1: {model1['n_params']:,} parameters | train {model1['train_s']:.1f} s on a CPU | "
          f"{model1['infer_ms']:.2f} ms per note on a CPU")
    print(f"Model 2: {run2['n_params']:,} parameters | train {run2['train_s'] / 60:.1f} min on the GPU | "
          f"{1000 * run2['infer_s'] / len(split.test):.1f} ms per note on the GPU, {run2['cpu_ms']:.0f} ms on a CPU "
          f"(about {3600_000 / run2['cpu_ms']:,.0f} notes per CPU hour)")


def copy_check(split, predict):
    """Score the saved Model 2 copy: a re-training of Model 2, so it needs its own threshold."""
    print("\n--- 7.5 The saved Model 2 copy (GPU training is not bit-for-bit repeatable) ---")
    if predict is None:
        print("skipped (put the saved weights in models/model2 to run it)")
        return
    val, test = predict(split.val["text"]), predict(split.test["text"])
    print(f"validation macro F1 {macro_f1(split.y_val, val.argmax(1), split.k):.4f} | "
          f"test macro F1 {macro_f1(split.y_test, test.argmax(1), split.k):.4f} | "
          f"its own threshold {pick_threshold(val, split.y_val)}")
