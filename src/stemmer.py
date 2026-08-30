"""
Stemmer — a small, hand-written suffix-stripper. No nltk, no spacy.

See STDLIB.md: this replaces nltk.stem (e.g. PorterStemmer / SnowballStemmer).

This is deliberately a simplified, original rule set — a handful of common
English suffix patterns stripped with plain string checks — not a
line-for-line reimplementation of any published stemming algorithm's code.
It exists to solve one concrete problem: "cook", "cooking", and "cooked"
were three unrelated tokens to the tokenizer/vectorizer, so a search for
"cook" found nothing even though "cooked" was right there in a document.
Collapsing common inflections to a shared root fixes that.

This is intentionally conservative. A stemmer that's too aggressive merges
unrelated words (e.g. stripping "-s" from "gas" would wrongly produce "ga"),
which hurts search quality more than it helps. Every rule below has a
minimum stem length guard and a short, explicit rationale.
"""

# Ordered longest-suffix-first so e.g. "-ing" is tried before a shorter,
# coincidentally-matching suffix would be. Each entry is
# (suffix, minimum_remaining_stem_length, replacement).
_SUFFIX_RULES: list[tuple[str, int, str]] = [
    ("ies", 3, "y"),      # "companies" -> "company"
    ("ied", 3, "y"),      # "tried" -> "try"
    ("ying", 3, "y"),     # "trying" -> "try" (must precede "-ing" below)
    ("ing", 3, ""),       # "cooking" -> "cook"
    ("edly", 3, ""),      # "reportedly" -> "report"
    ("ed", 3, ""),        # "cooked" -> "cook"
    ("es", 3, ""),        # "matches" -> "match"
    ("s", 3, ""),         # "cats" -> "cat"  (kept last: shortest, most
                           #  aggressive rule)
]

# Endings where a trailing "s" is very rarely a plural marker: "-ss"
# (class, glass), "-is" (this, axis), "-us" (bus, status), "-os" (chaos).
# Stripping "s" off these produces nonsense stems ("this" -> "thi"), so the
# "-s" rule is skipped whenever the word ends in one of these two-letter
# sequences.
_S_RULE_EXCEPTIONS: tuple[str, ...] = ("ss", "is", "us", "os")


def stem(word: str) -> str:
    """
    Strip a common inflectional suffix from a single lowercase token,
    returning its approximate root form. Words shorter than or equal to
    the rule's minimum stem length are left untouched, so short words
    (e.g. "as", "is", "gas") are never mangled. Words ending in "-ss",
    "-is", "-us", or "-os" are also exempted from the "-s" rule
    specifically, since a trailing "s" there is almost never a plural
    marker (see _S_RULE_EXCEPTIONS above).

    This only handles suffixes -- prefixes and irregular forms ("ran" ->
    "run", "mice" -> "mouse") are out of scope, same as most lightweight
    stemmers.
    """
    for suffix, min_len, replacement in _SUFFIX_RULES:
        if suffix == "s" and word.endswith(_S_RULE_EXCEPTIONS):
            continue
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            return word[: -len(suffix)] + replacement
    return word


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply stem() across a list of already-tokenized, lowercase words."""
    return [stem(t) for t in tokens]