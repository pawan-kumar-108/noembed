"""
Tokenizer — no nltk, no spacy. Lowercase, strip punctuation, split on
whitespace, filter a small hardcoded stopword list.

See STDLIB.md: this replaces nltk's tokenizer + stopword corpus.
"""

import string

# Small, hand-picked stopword list. Not exhaustive by design — the goal is
# to strip the highest-frequency noise words that would otherwise dominate
# every document's vector, not to replicate a linguistics corpus.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "for", "with", "by", "from", "up", "down", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "as", "not", "no", "so", "than", "too", "very", "can",
    "will", "just", "do", "does", "did", "has", "have", "had", "i", "you",
    "he", "she", "we", "they", "them", "his", "her", "their", "our", "your",
})

# Translation table mapping every ASCII punctuation character to a space.
# Using str.translate is the stdlib-idiomatic way to strip punctuation in
# bulk without a regex substitution per character.
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Turn raw text into a list of lowercase tokens.

    - Lowercases the text.
    - Strips ASCII punctuation (replaced with whitespace, so
      "hello,world" splits into two tokens rather than merging).
    - Splits on any whitespace run.
    - Drops empty tokens and, by default, stopwords.

    Non-ASCII text is preserved as-is aside from lowercasing and ASCII
    punctuation stripping — this is a known, documented limitation (see
    README "Honest limits"), not an oversight.
    """
    if not text:
        return []

    lowered = text.lower()
    stripped = lowered.translate(_PUNCT_TABLE)
    raw_tokens = stripped.split()

    if remove_stopwords:
        return [t for t in raw_tokens if t and t not in STOPWORDS]
    return [t for t in raw_tokens if t]