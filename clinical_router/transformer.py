"""Model 2: fine-tuning a domain-specific transformer, and the three experiments that chose it.

Experiments, all scored on the validation split: (1) how to handle notes longer than 512 tokens (head,
head + tail, chunks); (2) which checkpoint (PubMedBERT, Bio_ClinicalBERT, BERT-base); (3) how much the seed
moves the result. Each run saves its probabilities to outputs/runs/, so the pipeline re-runs without a GPU;
--retrain trains every run again (about 45 minutes on a 6 GB laptop GPU).

The training loop follows the native PyTorch loop of the Hugging Face tutorial "Fine-tune a pretrained
model" (huggingface.co/docs/transformers/training). We changed it: class-weighted loss, bfloat16 autocast,
our own length-sorted batching of pre-cut token pieces, per-epoch validation macro F1 with best-epoch
restore, and probability averaging over the pieces of a note.
"""
import gc
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import config
from .evaluation import macro_f1
from .preprocessing import strip_negation

# Chosen once, not tuned per checkpoint, so the checkpoint comparison is fair. 3e-5 and batch 8 are in the
# range recommended by Devlin et al. (2019) and fit a 6 GB GPU; 6 epochs with best-epoch restore lets
# each run stop where validation F1 peaks (epoch 3 to 6 in our runs).
EPOCHS, LR, BATCH, WEIGHT_DECAY, WARMUP = 6, 3e-5, 8, 0.01, 0.1


def make_pieces(tok, texts, strategy):
    """Cut each note into the token pieces the model reads, and record which note owns each piece."""
    ids = tok(list(texts), add_special_tokens=False)["input_ids"]
    pieces, owner = [], []
    for i, x in enumerate(ids):
        if len(x) <= config.BODY or strategy == "head":
            parts = [x[:config.BODY]]
        elif strategy == "head_tail":
            # The start gives the reason for the visit; the end gives the plan and the follow-up department.
            parts = [x[:config.HEAD] + x[-(config.BODY - config.HEAD):]]
        elif strategy == "chunks":
            starts = list(range(0, len(x) - config.BODY, config.STRIDE)) + [len(x) - config.BODY]
            parts = [x[s:s + config.BODY] for s in starts]
        for p in parts:
            pieces.append([tok.cls_token_id] + p + [tok.sep_token_id])
            owner.append(i)
    return pieces, np.array(owner)


def batches(pieces, order, bs, pad_id, device):
    """Yield padded batches; padding only to the longest piece in the batch saves GPU time."""
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(pieces[i]) for i in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(pieces[i])] = torch.tensor(pieces[i])
            mask[r, :len(pieces[i])] = 1
        yield idx, ids.to(device), mask.to(device)


@torch.no_grad()
def predict_notes(model, tok, texts, strategy, k, device=config.DEVICE, bs=32):
    """Probabilities for each note: the mean of its pieces' probabilities (one piece unless chunks)."""
    model.eval()
    pieces, owner = make_pieces(tok, texts, strategy)
    order = np.argsort([len(p) for p in pieces])   # similar lengths together: less padding
    probs = np.zeros((len(pieces), k), dtype=np.float32)
    for idx, ids, mask in batches(pieces, order, bs, tok.pad_token_id, device):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    note_probs = np.zeros((len(texts), k))
    np.add.at(note_probs, owner, probs)
    return note_probs / np.bincount(owner, minlength=len(texts))[:, None]


def show_log(name, r):
    """Print the epoch log of one run."""
    for e in r["log"]:
        print(f"  epoch {e['epoch']}: train loss {e['train_loss']:.4f} | validation macro F1 {e['val_f1']:.4f}")
    print(f"{name}: best epoch {r['best_epoch']} | validation macro F1 {r['val_f1']:.4f} | "
          f"train {r['train_s'] / 60:.1f} min | {r['n_pieces']:,} training pieces")


def fine_tune(split, ckpt, strategy, seed=config.SEED, retrain=False):
    """Fine-tune one checkpoint with one long-note option and seed, or load its saved run.

    Returns validation, test, test-without-negation and out-of-scope probabilities plus the training log.
    Only validation macro F1 selects the epoch; test probabilities are saved but not looked at here.
    """
    name = f"{ckpt}_{strategy}_s{seed}"
    path = config.RUN_DIR / f"{name}.npz"
    if path.exists() and not retrain:
        z = np.load(path, allow_pickle=True)
        r = {key: z[key] for key in z.files}
        r.update(json.loads(str(z["meta"])))
        print(f"{name}: loaded from outputs/runs/{path.name}")
        show_log(name, r)
        return r
    if config.DEVICE.type != "cuda":
        raise SystemExit(f"{name}: no saved run and no GPU. Put the .npz files in outputs/runs/ or use a GPU.")

    config.set_seed(seed)
    k, device = split.k, config.DEVICE
    tok = AutoTokenizer.from_pretrained(config.CHECKPOINTS[ckpt])
    model = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINTS[ckpt], num_labels=k).to(device)
    tr_pieces, tr_owner = make_pieces(tok, split.train["text"], strategy)
    y_pieces = split.y_train[tr_owner]
    steps = EPOCHS * math.ceil(len(tr_pieces) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    # Balanced class weights: without them the loss is dominated by the four largest departments.
    weights = len(split.y_train) / (k * np.bincount(split.y_train, minlength=k))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    rng = np.random.default_rng(seed)

    log, best_f1, best_state, best_epoch = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for idx, ids, mask in batches(tr_pieces, rng.permutation(len(tr_pieces)), BATCH, tok.pad_token_id, device):
            # bfloat16 halves memory with the float32 exponent range, so no loss scaling is needed.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), torch.tensor(y_pieces[idx], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # one long batch cannot blow up an update
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            total += loss.item() * len(idx)
        f1 = macro_f1(split.y_val, predict_notes(model, tok, split.val["text"], strategy, k).argmax(1), k)
        log.append({"epoch": epoch, "train_loss": total / len(tr_pieces), "val_f1": f1})
        if f1 > best_f1:   # keep the best epoch's weights: 871 notes over-fit quickly
            best_f1, best_epoch = f1, epoch
            best_state = {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}
    train_s = time.time() - t0

    model.load_state_dict(best_state)
    val_probs = predict_notes(model, tok, split.val["text"], strategy, k)
    torch.cuda.synchronize()
    t0 = time.time()
    test_probs = predict_notes(model, tok, split.test["text"], strategy, k)
    torch.cuda.synchronize()
    infer_s = time.time() - t0
    test_noneg_probs = predict_notes(model, tok, [strip_negation(t) for t in split.test["text"]], strategy, k)
    oos_probs = predict_notes(model, tok, split.oos["text"], strategy, k)

    model.to("cpu").float()   # a hospital may have no GPU: measure the CPU cost on 20 notes
    t0 = time.time()
    predict_notes(model, tok, split.test["text"][:20], strategy, k, device=torch.device("cpu"), bs=4)
    cpu_ms = 1000 * (time.time() - t0) / 20

    meta = {"log": log, "best_epoch": best_epoch, "val_f1": best_f1, "train_s": train_s, "infer_s": infer_s,
            "cpu_ms": cpu_ms, "n_params": sum(p.numel() for p in model.parameters()), "n_pieces": len(tr_pieces)}
    np.savez(path, val_probs=val_probs, test_probs=test_probs, test_noneg_probs=test_noneg_probs,
             oos_probs=oos_probs, meta=json.dumps(meta))
    del model, opt, best_state
    gc.collect(); torch.cuda.empty_cache()
    r = {"val_probs": val_probs, "test_probs": test_probs, "test_noneg_probs": test_noneg_probs,
         "oos_probs": oos_probs, **meta}
    show_log(name, r)
    return r


def compare_tokenizers(split):
    """Tokens per word for each checkpoint: a vocabulary that splits clinical words loses more text at 512."""
    rows = []
    for name, ckpt in config.CHECKPOINTS.items():
        tok = AutoTokenizer.from_pretrained(ckpt)
        lens = np.array([len(x) for x in tok(list(split.train["text"]), add_special_tokens=False)["input_ids"]])
        rows.append({"checkpoint": name, "tokens per word": round(lens.sum() / split.train["n_words"].sum(), 2),
                     "notes > 510 tokens": f"{(lens > config.BODY).mean():.1%}",
                     "tokens lost by first-512": f"{(lens - config.BODY).clip(min=0).sum() / lens.sum():.1%}",
                     "'cholecystectomy'": " ".join(tok.tokenize("cholecystectomy"))})
    print("\n--- 4.2 Tokenizers (training notes) ---")
    print(pd.DataFrame(rows).set_index("checkpoint").to_string())


def run_experiments(split, retrain=False):
    """Run the three validation experiments in order; return all runs and the selected Model 2 key."""
    runs = {}
    long_val = (split.val["n_tokens"] > config.BODY).to_numpy()
    print("\n--- 4.3 Test 1: the long-note option (PubMedBERT) ---")
    rows = []
    for strategy in ("head", "head_tail", "chunks"):
        r = runs["PubMedBERT", strategy, config.SEED] = fine_tune(split, "PubMedBERT", strategy, retrain=retrain)
        pred = r["val_probs"].argmax(1)
        rows.append({"option": strategy, "validation macro F1": r["val_f1"],
                     "accuracy short": accuracy_score(split.y_val[~long_val], pred[~long_val]),
                     "accuracy long": accuracy_score(split.y_val[long_val], pred[long_val]),
                     "train min": r["train_s"] / 60, "pieces": r["n_pieces"]})
    table = pd.DataFrame(rows).set_index("option")
    strategy = table["validation macro F1"].idxmax()
    print(table.round(4).to_string())
    print(f"validation notes {(~long_val).sum()} short, {long_val.sum()} long | selected: {strategy}")

    print("\n--- 4.4 Test 2: the checkpoint ---")
    rows = []
    for ck in config.CHECKPOINTS:
        if (ck, strategy, config.SEED) not in runs:
            runs[ck, strategy, config.SEED] = fine_tune(split, ck, strategy, retrain=retrain)
        r = runs[ck, strategy, config.SEED]
        rows.append({"checkpoint": ck, "validation macro F1": r["val_f1"], "best epoch": r["best_epoch"]})
    table = pd.DataFrame(rows).set_index("checkpoint")
    ckpt = table["validation macro F1"].idxmax()
    print(table.round(4).to_string())
    print(f"selected: {ckpt}")

    print("\n--- 4.5 Test 3: the seed (the selected design, five seeds) ---")
    seeds = [config.SEED + i for i in range(5)]
    for s in seeds:
        if (ckpt, strategy, s) not in runs:
            runs[ckpt, strategy, s] = fine_tune(split, ckpt, strategy, seed=s, retrain=retrain)
    # Test F1 of the other seeds is printed only to show the spread; Model 2 is seed 6700, chosen on validation.
    seed_table = pd.DataFrame([{"seed": s, "validation macro F1": runs[ckpt, strategy, s]["val_f1"],
                                "test macro F1": macro_f1(split.y_test, runs[ckpt, strategy, s]["test_probs"].argmax(1), split.k)}
                               for s in seeds]).set_index("seed")
    print(seed_table.round(4).to_string())
    print(f"mean validation {seed_table.iloc[:, 0].mean():.4f} (sd {seed_table.iloc[:, 0].std():.4f}) | "
          f"mean test {seed_table.iloc[:, 1].mean():.4f} (sd {seed_table.iloc[:, 1].std():.4f})")
    plot_curves(runs, strategy)
    return runs, (ckpt, strategy, config.SEED)


def plot_curves(runs, strategy):
    """Validation macro F1 per epoch for the long-note options and the checkpoints (one figure, two panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, keys, title in [(axes[0], [("PubMedBERT", s) for s in ("head", "head_tail", "chunks")], "Long-note option (PubMedBERT)"),
                            (axes[1], [(c, strategy) for c in config.CHECKPOINTS], f"Checkpoint ({strategy})")]:
        for (ck, st), grey in zip(keys, ("0.35", "0.6", "0.8")):
            log = runs[ck, st, config.SEED]["log"]
            chosen = (ck, st) == ("PubMedBERT", strategy)   # the selected run is blue in both panels
            ax.plot([e["epoch"] for e in log], [e["val_f1"] for e in log], "o-",
                    color="tab:blue" if chosen else grey, lw=2.5 if chosen else 1.5,
                    label=ck if ax is axes[1] else st)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    axes[0].set_ylabel("validation macro F1")
    config.save_figure(fig, "fig_training_curves.png")


def load_copy():
    """Load the saved Model 2 weights (models/model2) if they are present; None otherwise."""
    if not (config.MODEL2_COPY / "model.safetensors").exists():
        return None
    tok = AutoTokenizer.from_pretrained(config.MODEL2_COPY)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL2_COPY).to(config.DEVICE)
    return model, tok
