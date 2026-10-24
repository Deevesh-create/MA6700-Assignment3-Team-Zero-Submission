"""Traditional text preprocessing: format repair, cleaning for TF-IDF, negation scope, and the settings test.

Model 1 (TF-IDF) has no word order, so its input is lower-cased words with optional negation markers.
Model 2 (PubMedBERT) reads word order with its own tokenizer, so it gets the repaired text unchanged:
removing stop words would remove "no" and "not", which the transformer can use in context.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config
from .evaluation import macro_f1

# Negation cues and scope ends: a simplified NegEx (Chapman et al., 2001). Adapted: we use a fixed six-word
# window and clause punctuation instead of NegEx's full trigger list and pseudo-negation phrases.
NEG_CUES = {"no", "not", "denies", "denied", "deny", "denying", "without", "negative", "never", "nor",
            "absent", "absence", "unremarkable"}
SCOPE_END = {"but", "however", "although", "though", "except", "apart", "aside", "which", "who"}
NEG_WINDOW = 6  # NegEx uses a five-word window; six covers lists such as "chest pain, shortness of breath"

NEG_DEMO = "REVIEW OF SYSTEMS: The patient denies chest pain, shortness of breath or fever. No nausea but has knee pain."


def repair_format(text):
    """Undo the scrape's ", " after each heading, so headings and sentences read as in the original note."""
    t = re.sub(r":\s*,", ": ", text)           # "DIAGNOSIS: , Morbid" -> "DIAGNOSIS: Morbid"
    t = re.sub(r"\.\s*,", ". ", t)             # "obesity.,PROCEDURE" -> "obesity. PROCEDURE"
    t = re.sub(r",\s*(?=[A-Z][A-Z /&-]{3,}:)", " ", t)   # ",HISTORY:" -> " HISTORY:"
    return re.sub(r"\s+", " ", t).strip()


def tokens(text):
    """Lower-case words that start with a letter; numbers alone carry no department."""
    return re.findall(r"[a-z][a-z0-9]+", text.lower())


def clean_tfidf(text, negation=True):
    """The input of Model 1: lower-case words, and optionally the prefix neg_ on words inside a negation scope."""
    out = []
    for clause in re.split(r"[.;:!?]", text):   # a negation never crosses a clause border
        scope = 0
        for word in tokens(clause):
            if word in SCOPE_END:
                scope = 0
            if word in NEG_CUES:
                out.append(word)
                scope = NEG_WINDOW if negation else 0
            elif scope > 0:
                out.append("neg_" + word)       # "chest" in "denies chest pain" becomes its own feature
                scope -= 1
            else:
                out.append(word)
    return " ".join(out)


def strip_negation(text):
    """Remove every negation cue, so "denies chest pain" reads "chest pain" (the test of Section 5.6)."""
    return re.sub(r"\b(" + "|".join(NEG_CUES) + r")\b", " ", text, flags=re.I)


def make_tfidf(ngrams, stop):
    """A TF-IDF vectoriser with the fixed settings and the two settings under test.

    min_df=2 drops words seen in one note only (typing errors, names); max_df=0.9 drops words in almost
    every note; sublinear_tf stops a word repeated 20 times in a long note from dominating it.
    """
    return TfidfVectorizer(token_pattern=r"[a-z][a-z0-9_]+", ngram_range=ngrams, stop_words=stop,
                           min_df=2, max_df=0.9, sublinear_tf=True)


def show_examples(split, raw):
    """Print one real note before and after each preprocessing step, and the negation statistics."""
    # A training note whose raw text shows the scrape artefact and a negation (training data only).
    note = next(t for t in split.train["text"] if " denies " in t and len(t) < 1500)
    raw_text = next(r for r in raw["transcription"].dropna() if repair_format(r.strip()) == note)
    print("\n--- Before and after: one training note ---")
    print("raw (as scraped)      :", raw_text[:260], "...")
    print("1. format repaired    :", note[:260], "...")
    print("2. Model 1 words      :", clean_tfidf(note, negation=False)[:260], "...")
    stop = make_tfidf((1, 1), "english").get_stop_words()
    print("3. stop words removed :", " ".join(w for w in tokens(note) if w not in stop)[:260], "...")
    i = note.find(" denies ")
    print("4. negation markers   :", clean_tfidf(note[i:i + 160]))
    print("   (sentence used     :", note[i:i + 160].strip(), ")")
    print("\nnegation example:", NEG_DEMO)
    print("   with markers      :", clean_tfidf(NEG_DEMO))

    marked = [clean_tfidf(t, True) for t in split.train["text"]]
    has_cue = np.array([bool(NEG_CUES & set(tokens(t))) for t in split.train["text"]])
    negated = Counter(w for t in marked for w in set(t.split()) if w.startswith("neg_"))
    print(f"\ntraining notes with a negation cue: {has_cue.sum()} of {len(has_cue)} ({has_cue.mean():.1%})")
    print("most frequent negated words (training notes):",
          ", ".join(f"{w} ({c})" for w, c in negated.most_common(12)))


def compare_settings(split):
    """Test 8 TF-IDF settings on the validation split and return them, best first.

    The options are the ones a reader would question: negation markers, the stop word list, and word
    pairs. A fixed C=10 with balanced weights isolates the effect of the text input.
    """
    texts = {(name, neg): [clean_tfidf(t, neg) for t in d["text"]]
             for name, d in [("train", split.train), ("val", split.val)] for neg in (False, True)}
    rows = []
    for negation in (False, True):
        for stop in (None, "english"):
            for ngrams in ((1, 1), (1, 2)):
                vec = make_tfidf(ngrams, stop)
                Xtr = vec.fit_transform(texts["train", negation])
                clf = LogisticRegression(C=10, class_weight="balanced", max_iter=3000).fit(Xtr, split.y_train)
                f1 = macro_f1(split.y_val, clf.predict(vec.transform(texts["val", negation])), split.k)
                rows.append({"negation markers": negation, "stop words": stop or "keep all",
                             "n-grams": ngrams, "features": Xtr.shape[1], "validation macro F1": round(f1, 4)})
    settings = pd.DataFrame(rows).sort_values("validation macro F1", ascending=False, kind="stable")
    settings = settings.reset_index(drop=True)
    print("\n--- TF-IDF settings, validation split (C = 10, balanced) ---")
    print(settings.to_string())
    return settings
