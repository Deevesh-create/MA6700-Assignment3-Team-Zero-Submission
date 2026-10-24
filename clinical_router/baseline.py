"""Model 1, the traditional baseline (TF-IDF + logistic regression), and a hand-written keyword rule system.

The rule system answers the question "why not just rules?" with a number and a real failing note.
Model 1 is the learned baseline that the deep model must beat to justify its cost.
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .evaluation import macro_f1
from .preprocessing import clean_tfidf, make_tfidf

# Written from anatomy and procedure vocabulary, not tuned on the data: this is what a hospital
# administrator could write without machine learning. General Medicine has no organ of its own,
# so it is the fallback when no other list matches.
RULES = {
    "Cardiovascular / Pulmonary": ["heart", "cardiac", "coronary", "chest", "lung", "lungs", "pulmonary", "artery",
                                   "echocardiogram", "atrial", "ventricular", "bronchoscopy"],
    "Orthopedic": ["fracture", "knee", "hip", "shoulder", "bone", "joint", "tendon", "ligament", "arthroplasty",
                   "arthroscopy", "spine", "wrist"],
    "General Medicine": ["fever", "cough", "diabetes", "hypertension", "fatigue", "physical"],
    "Gastroenterology": ["abdominal", "colonoscopy", "endoscopy", "stomach", "colon", "bowel", "liver",
                         "esophagus", "gallbladder", "duodenum", "rectal", "hernia"],
    "Obstetrics / Gynecology": ["pregnancy", "pregnant", "uterus", "uterine", "vaginal", "cervix", "ovary",
                                "ovarian", "gestation", "delivery", "cesarean", "menstrual"],
    "Urology": ["bladder", "prostate", "kidney", "urine", "urinary", "ureter", "cystoscopy", "scrotum",
                "scrotal", "testicle", "urethra", "renal"],
    "Neurology": ["brain", "seizure", "seizures", "headache", "stroke", "neurologic", "nerve", "numbness",
                  "dementia", "cerebral", "weakness", "tremor"],
    "ENT - Otolaryngology": ["ear", "ears", "nose", "throat", "sinus", "tonsil", "tonsils", "nasal", "hearing",
                             "larynx", "adenoid", "tympanic"],
    "Ophthalmology": ["eye", "eyes", "vision", "cataract", "retina", "cornea", "lens", "glaucoma", "visual",
                      "intraocular", "eyelid", "pupil"],
}
FALLBACK = "General Medicine"


def rule_route(text, classes):
    """Count the keyword hits of each department; the most hits wins, no hit falls back to General Medicine."""
    words = re.findall(r"[a-z]+", text.lower())
    hits = {c: sum(words.count(w) for w in RULES[c]) for c in classes}
    best = max(classes, key=lambda c: hits[c])     # ties go to the larger department (classes is largest first)
    return (best if hits[best] else FALLBACK), hits


def rules_baseline(split):
    """Score the keyword rules on the test split; return one-hot "probabilities" for the comparison table."""
    routed = [rule_route(t, split.classes) for t in split.test["text"]]
    pred = np.array([split.classes.index(r[0]) for r in routed])
    probs = np.eye(split.k)[pred]
    print("\n--- 1. Why not rules? A keyword rule system on the test split ---")
    print(f"keyword rules: macro F1 {macro_f1(split.y_test, pred, split.k):.4f} | "
          f"accuracy {(pred == split.y_test).mean():.4f}")
    return probs, routed


def rule_failure(split, routed, s2):
    """Print the real test note where the rules are most sure (most hits) and wrong, while Model 2 is right."""
    wrong = [i for i, (dept, hits) in enumerate(routed)
             if split.classes.index(dept) != split.y_test[i] and s2["pred"][i] == split.y_test[i]]
    i = max(wrong, key=lambda i: max(routed[i][1].values()))
    dept, hits = routed[i]
    true = split.classes[split.y_test[i]]
    print(f"rules wrong but Model 2 right on {len(wrong)} test notes. Example: '{split.test['sample_name'][i]}'")
    print(f"  true {true} | rules {dept} ({hits[dept]} hits: "
          f"{', '.join(w for w in RULES[dept] if re.search(rf'\b{w}\b', split.test['text'][i].lower()))}) | "
          f"true department's hits {hits[true]} | Model 2 {split.classes[s2['pred'][i]]} "
          f"({s2['probs'][i].max():.2f})")
    print("  ", split.test["text"][i][:300], "...")


def train_model1(split, settings):
    """Select C and the class weights on validation, train Model 1 on the training split and time it."""
    best = settings.iloc[0]
    negation, ngrams = bool(best["negation markers"]), tuple(best["n-grams"])
    stop = None if best["stop words"] == "keep all" else "english"
    text = {name: [clean_tfidf(t, negation) for t in d["text"]] for name, d in
            [("train", split.train), ("val", split.val)]}

    vec = make_tfidf(ngrams, stop)
    Xtr, Xva = vec.fit_transform(text["train"]), vec.transform(text["val"])
    rows = []
    # C from strong to weak regularisation; balanced weights because the largest class is 4.3x the smallest.
    for C in (0.3, 1, 3, 10, 30, 100):
        for cw in (None, "balanced"):
            clf = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(Xtr, split.y_train)
            rows.append({"C": C, "class_weight": cw or "none",
                         "validation macro F1": macro_f1(split.y_val, clf.predict(Xva), split.k)})
    grid = pd.DataFrame(rows).sort_values("validation macro F1", ascending=False, kind="stable").reset_index(drop=True)
    print("\n--- 4.1 Model 1: C and class weights, validation split (top 5) ---")
    print(grid.head(5).round(4).to_string(index=False))
    C, cw = float(grid.loc[0, "C"]), None if grid.loc[0, "class_weight"] == "none" else "balanced"

    t0 = time.time()
    vec = make_tfidf(ngrams, stop)
    model = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(
        vec.fit_transform([clean_tfidf(t, negation) for t in split.train["text"]]), split.y_train)
    train_s = time.time() - t0

    def predict(texts):
        """Class probabilities for raw note texts: the full path from text to department."""
        return model.predict_proba(vec.transform([clean_tfidf(t, negation) for t in texts]))

    t0 = time.time()
    test_probs = predict(split.test["text"])
    infer_s = time.time() - t0
    val_probs = predict(split.val["text"])
    n_params = model.coef_.size + model.intercept_.size
    print(f"Model 1: negation={negation}, stop words={stop}, n-grams={ngrams}, C={C}, class_weight={cw} | "
          f"features {len(vec.vocabulary_):,} | parameters {n_params:,} | train {train_s:.1f} s | "
          f"validation macro F1 {macro_f1(split.y_val, val_probs.argmax(1), split.k):.4f}")

    terms = np.array(vec.get_feature_names_out())
    print("strongest terms per department:")
    for k, c in enumerate(split.classes):
        print(f"  {c}: {', '.join(terms[np.argsort(model.coef_[k])[::-1][:8]])}")
    return {"predict": predict, "val_probs": val_probs, "test_probs": test_probs, "train_s": train_s,
            "infer_ms": 1000 * infer_s / len(split.test), "n_params": n_params}
