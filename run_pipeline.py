"""The one entry point: run the whole pipeline from the raw CSV to every number and figure in the report.

    python run_pipeline.py            # loads the saved fine-tuning runs (CPU is enough, about 3 minutes)
    python run_pipeline.py --retrain  # fine-tunes all 9 transformer runs again (NVIDIA GPU, about 45 minutes)

Metrics are printed and also saved to outputs/results.txt; figures go to outputs/figures/.
The order follows the report: data (2), preprocessing (3), modelling (4), evaluation (5),
responsible AI (6) and deployment (7).
"""
import argparse
import sys
import warnings

from clinical_router import config  # noqa: F401  (first: sets the environment and the plot backend)
from clinical_router import baseline, data, deployment, evaluation, preprocessing, responsible_ai, transformer
from transformers import logging as hf_logging


class Tee:
    """Write everything printed to the screen and to outputs/results.txt, so the marker has both."""

    def __init__(self, *streams):
        """Keep the streams to copy to."""
        self.streams = streams

    def write(self, text):
        """Write to every stream."""
        for s in self.streams:
            s.write(text)

    def flush(self):
        """Flush every stream."""
        for s in self.streams:
            s.flush()


def main():
    """Run every stage in report order."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain", action="store_true", help="fine-tune every transformer run again (GPU)")
    args = parser.parse_args()

    config.make_dirs()
    sys.stdout = Tee(sys.__stdout__, open(config.OUT_DIR / "results.txt", "w", encoding="utf8", buffering=1))
    warnings.filterwarnings("ignore")   # library deprecation notices only; they hide the results
    hf_logging.set_verbosity_error()
    config.set_seed()
    print(f"seed {config.SEED} | device {config.DEVICE}")

    # 2. Data: audit, one label per note, near-duplicates, class rule, length, and the split (once).
    raw = data.load_raw()
    data.audit(raw)
    uniq, n_unique = data.unique_notes(raw)
    single = data.remove_near_duplicates(uniq)
    kept, oos, classes = data.choose_classes(single, n_unique)
    data.measure_length(kept, classes)
    split = data.make_split(kept, oos, classes)

    # 3. Preprocessing: before / after on a real note, then the settings test on validation.
    preprocessing.show_examples(split, raw)
    settings = preprocessing.compare_settings(split)

    # 4. Modelling: Model 1, then the three Model 2 experiments.
    model1 = baseline.train_model1(split, settings)
    transformer.compare_tokenizers(split)
    runs, key2 = transformer.run_experiments(split, retrain=args.retrain)
    run2 = runs[key2]

    # 5. Evaluation: the test split is used from here on.
    boot = evaluation.bootstrap_indices(len(split.test))
    rule_probs, routed = baseline.rules_baseline(split)
    s1 = evaluation.score("Model 1: TF-IDF + LR", model1["test_probs"], split, boot)
    s2 = evaluation.score(f"Model 2: {key2[0]} ({key2[1]})", run2["test_probs"], split, boot)
    s0 = evaluation.score("Keyword rules", rule_probs, split, boot, verbose=False)
    baseline.rule_failure(split, routed, s2)
    evaluation.comparison_table([s0, s1, s2], {
        s0["name"]: {"parameters": f"{sum(map(len, baseline.RULES.values()))} hand-written words"},
        s1["name"]: {"parameters": f"{model1['n_params']:,}", "CPU ms per note": round(model1["infer_ms"], 2)},
        s2["name"]: {"parameters": f"{run2['n_params']:,}", "CPU ms per note": round(run2["cpu_ms"])}})
    evaluation.per_class(split, s1, s2)
    evaluation.significance(split, s1, s2)
    evaluation.confusion(split, [s1, s2])
    evaluation.misrouted(split, s1, s2)
    evaluation.length_effect(split, [s1, s2])
    noneg1 = model1["predict"]([preprocessing.strip_negation(t) for t in split.test["text"]])
    evaluation.negation_effect(split, [(s1, noneg1), (s2, run2["test_noneg_probs"])])

    # 6. Responsible AI: patient groups and personal data.
    responsible_ai.subgroups(split, [s1, s2])
    responsible_ai.personal_data(split)

    # 7. Deployment: calibration, the fast track, unknown departments, privacy checks and cost.
    deployment.calibration(split, [s1, s2])
    oos1 = model1["predict"](split.oos["text"])
    deployment.fast_track(split, s1, s2, model1["val_probs"], run2["val_probs"], oos1, run2["oos_probs"])
    deployment.cost(split, model1, run2)
    model2_copy = responsible_ai.copy_predictor(transformer.load_copy(), split)
    responsible_ai.redaction_check(split, model1, model2_copy)
    responsible_ai.membership_inference(split, model1, model2_copy)
    deployment.copy_check(split, model2_copy)
    print(f"\nDone. Figures in {config.FIG_DIR}; this log in {config.OUT_DIR / 'results.txt'}")


if __name__ == "__main__":
    main()
