"""
Dual preprocessing for Twitter sentiment analysis.

Two cleaning levels solve different problems:

1. **clean_text** (classical ML: BoW, TF-IDF, Word2Vec)
   - Removes non-alpha characters (keeps @, #)
   - Removes stopwords EXCEPT sentiment-relevant words (negations, intensifiers, contrast)
   - Lemmatizes tokens
   - Rationale: classical models need sparse, discriminative tokens. But stripping negation
     words ("not", "don't") destroys sentiment signal — 15.5% of classification errors
     involve negation. We preserve 29 sentiment-critical words from the NLTK stopword list.

2. **light_clean** (transformers: BERT, RoBERTa, sentence-transformers)
   - Lowercase + collapse whitespace only
   - Rationale: sentence transformers are pretrained on natural language. Aggressive
     tokenization strips grammar and function words they rely on, producing degenerate
     embeddings (e.g., clusters of just "know", "guess" instead of meaningful topics).

See preprocessing/feature_analysis.ipynb for the empirical analysis behind these choices.
"""

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _res in ("corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"):
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res.split("/")[1], quiet=True)

_lemmatizer = WordNetLemmatizer()

_CLEAN_RE = re.compile(r"[^a-z\s@#]")
_WHITESPACE_RE = re.compile(r"\s+")

# ── Sentiment-aware stopword list ──────────────────────────────────────────────
# Standard NLTK English stopwords = 179 words. We remove 29 that carry sentiment
# signal, leaving 150 words to filter.
#
# These 29 words fall into three categories:
#   - Negation: "not happy" vs "happy" flips sentiment entirely
#   - Intensifiers: "very bad" vs "bad" — degree matters for ordinal sentiment
#   - Contrast: "good but expensive" — BUT signals sentiment shift
#
# Empirical result: preserving these words improved BoW+LR F1 by +0.006
# (0.6938 → 0.6997), the single largest classical preprocessing improvement.

SENTIMENT_KEEP = {
    # Negation words (15.5% of errors involve negation — critical to preserve)
    "not", "no", "nor", "don", "aren", "isn", "wasn", "weren",
    "hasn", "haven", "hadn", "couldn", "wouldn", "shouldn",
    "didn", "won", "doesn", "mustn", "needn",
    # Intensifiers
    "very", "too", "most", "more", "just", "only",
    # Contrast (supports BUT-clause analysis)
    "but", "however",
    # Other sentiment-relevant
    "against", "few", "own",
}

_BASE_STOPWORDS = set(stopwords.words("english"))
STOP_WORDS = _BASE_STOPWORDS - SENTIMENT_KEEP


# ── Cleaning functions ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Aggressive cleaning for classical ML models.

    Steps: lowercase → keep only alpha + @ + # → remove stopwords (sentiment-aware)
    → lemmatize → drop single-char tokens.

    Keeps @mentions and #hashtags because URLs/mentions carry statistically
    significant sentiment signal (chi2=57.38, p<1e-12 for URLs).
    """
    text = str(text).lower()
    text = _CLEAN_RE.sub("", text)
    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 1
    ]
    return " ".join(tokens)


def light_clean(text: str) -> str:
    """Minimal cleaning for transformer models and sentence embeddings.

    Only lowercases and normalizes whitespace. Preserves punctuation, function
    words, and sentence structure that pretrained transformers depend on.
    """
    return _WHITESPACE_RE.sub(" ", str(text).lower()).strip()


def preprocess_df(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """Add clean_text and light_text columns to a DataFrame.

    Returns a copy — does not mutate the input.

    Parameters
    ----------
    df : DataFrame
        Must contain a column named *text_col*.
    text_col : str
        Name of the raw text column (default "text").

    Returns
    -------
    DataFrame with two new columns: ``clean_text`` and ``light_text``.
    """
    out = df.copy()
    out["clean_text"] = out[text_col].apply(clean_text)
    out["light_text"] = out[text_col].apply(light_clean)
    return out
