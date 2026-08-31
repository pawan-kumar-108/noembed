# noembed

[![tests](https://img.shields.io/badge/tests-75%20passing-brightgreen)](#tests)
[![dependencies](https://img.shields.io/badge/dependencies-0-blue)](#zero-dependency-proof)
[![stdlib only](https://img.shields.io/badge/python-stdlib%20only-yellow)](#how-it-works)

A zero-dependency semantic search engine. Hand-rolled TF-IDF vectorization,
cosine similarity, and an inverted index, persisted to disk with crash-safe
atomic writes. No numpy, no scikit-learn, no faiss — Python standard library
only.

## Demo

[![noembed demo](https://img.youtube.com/vi/TXFwBJZLFv0/maxresdefault.jpg)](https://youtu.be/TXFwBJZLFv0)

**5 minutes, unedited.** Live search over a real corpus, the crash-recovery
test running to completion, and a stemmer bug caught mid-build — shown, not
just claimed.

## What it does

Point it at a folder of text files and it builds a searchable index. Queries
return ranked results by TF-IDF + cosine-similarity relevance — not exact
keyword matching, so "loyal animal that plays with its owner" correctly
surfaces a document about dogs even without those exact words overlapping
much. Verified end-to-end against a real four-document corpus during
development; see the demo video for the live run.

## How to run it

```
make run          # shows CLI help

python3 -m src.cli index <folder> [--out PATH]           # build an index
python3 -m src.cli search "<query>" [-k N] [--explain]   # search it
python3 -m src.cli stats [--out PATH]                     # index stats
python3 -m src.cli inspect [--out PATH] [--doc NAME] [--top N]  # inspect the index
```

`--out` defaults to `.noembed_index.json` in the current directory if
omitted. `--explain` on search shows which shared terms drove each result's
score, sorted by contribution. `inspect` shows the most distinctive terms
across the corpus, or within a single document, by rarity — a legible way
to see what the index actually learned.

Only files with these extensions are indexed: `.txt .md .markdown .rst .log
.csv .json`. Everything else in the folder is skipped silently (not an
error — real folders have binaries, images, etc. in them).

## How it works

Tokenizer strips punctuation and a small hardcoded stopword list, then
lowercases and splits on whitespace. TF-IDF vectors are built with the
standard smoothed formula (`tf * (ln((1+N)/(1+df)) + 1)`) over plain Python
dicts — no arrays. An inverted index (term → document IDs) means a search
only scores documents that actually share a term with the query, not the
whole corpus. Ranking is cosine similarity over the sparse TF-IDF vectors.
A hand-written suffix-stripping stemmer normalizes word forms (e.g. "cook"
matches "cooked"). See `STDLIB.md` for every package this replaced and why.

## Durability guarantee

Every `save()` writes the full index to a temp file in the same directory,
`fsync`s it, then swaps it into place with `os.replace()` — atomic on both
POSIX and Windows. This means a save either lands completely, or the
previous valid index is left completely untouched; there is no window where
a half-written file could be loaded as valid. This is verified by an
automated test (`tests/test_index_crash.py`) that simulates a crash during
the replace step and confirms the original index survives byte-for-byte and
still loads and searches correctly.

This protects against process crashes and power loss between writes. It
does **not** protect against disk-level bit rot or a corrupted underlying
filesystem — that's out of scope here.

## Honest limits

- This is TF-IDF / keyword-adjacent search, **not neural embeddings**. It
  will not catch true synonyms or deep conceptual similarity the way an
  embedding model would — it works because related documents tend to share
  *some* vocabulary, weighted by rarity, not because it understands meaning.
- Exact search only — no approximate-nearest-neighbor indexing. The
  inverted index narrows candidates by shared terms, but scoring within
  that candidate set is a full, exact cosine-similarity computation.
- A query term never seen during indexing contributes nothing to the query
  vector (its IDF is undefined) — it's silently dropped rather than
  penalizing the query.
- Non-ASCII text is lowercased and whitespace-split but not otherwise
  linguistically processed (no locale-aware casing).
- The stemmer is a hand-written suffix-rule approach and can over-stem
  proper nouns (e.g. "James" → "jam") — a documented trade-off against
  better recall on regular word forms, not a hidden bug.
- The stopword list is small and hand-picked, not an exhaustive corpus.

## Zero-dependency proof

See `deps-proof.txt` and `STDLIB.md`. `requirements.txt` is empty.

## Tests

```
python3 -m unittest discover -s tests -v
```

75 tests, covering the tokenizer, TF-IDF math (hand-verified against known
formulas), cosine similarity properties, the stemmer (including regression
tests for caught bugs), end-to-end build/search/persist round-trips, the
colored display module, the `inspect` command, and the crash-recovery
guarantee above.

## License

MIT — see LICENSE.