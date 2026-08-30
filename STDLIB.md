# STDLIB.md

Every place this project would normally reach for a third-party package,
and what standard-library code replaced it instead. Filled in continuously
during the build, not at the end — see PLAN.md.

| Instead of installing... | We used | Why |
|---|---|---|
| `scikit-learn` (`TfidfVectorizer`) | Hand-rolled TF-IDF in `src/tfidf.py` | Term frequency, inverse document frequency, and sparse vector construction implemented from the definitions directly — no vectorization library. |
| `faiss` / `chromadb` | Hand-rolled inverted index in `src/index.py` | Term → document-ID mapping built and queried with plain dicts; no vector database. |
| `numpy` | `math` + list/dict comprehensions | Dot product, magnitude, and cosine similarity computed with plain arithmetic over sparse dict vectors — no arrays, no vectorized ops. |
| `nltk` | Hand-rolled tokenizer + hardcoded stopword list in `src/tokenizer.py` | No NLP toolkit; tokenization is lowercase + punctuation strip + whitespace split. |
| `click` / `typer` | `argparse` | Subcommands (`index`, `search`, `stats`) built on stdlib argument parsing. |
| `pytest` | `unittest` | Python ships `unittest` in the standard library, so this project does not qualify for the hackathon's "no test framework" dev-dependency exception. |
| `orjson` / `ujson` | `json` | Index serialization uses the stdlib JSON encoder/decoder. |
| SQLite / leveldb-style embedded DB | Hand-rolled JSON index file + atomic `os.replace()` write | Durability comes from write-to-temp + `fsync` + atomic rename, not from an embedded database engine. |
| `pathlib.Path.rglob`-adjacent tools (e.g. `glob2`, third-party file walkers) | `pathlib.Path.rglob("*")` | Recursive folder walking for `index` uses stdlib `pathlib`, no third-party file-walking helper. |
| `python-magic` / extension-sniffing libraries | A hardcoded `TEXT_EXTENSIONS` set in `src/cli.py` | File-type filtering during indexing is a plain set membership check, not a content-sniffing library. |
| `unittest.mock` third-party alternatives (e.g. older standalone `mock` package) | `unittest.mock` | Simulating a crash mid-write in `tests/test_index_crash.py` uses the mocking library that ships inside `unittest` itself, not the standalone PyPI `mock` package it was folded from. |

## Package Killer target

**`scikit-learn`** (`sklearn.feature_extraction.text.TfidfVectorizer` +
`sklearn.metrics.pairwise.cosine_similarity`) — one of the most-installed
packages in the Python data/ML ecosystem. `noembed` reimplements the exact
piece of it this project needed (fit/transform TF-IDF vectorization, cosine
similarity ranking) in `src/tfidf.py` and `src/similarity.py`, using only
`math` and `collections.Counter`. It does not reimplement all of
scikit-learn — just the documented slice that would otherwise have pulled
in the whole package for two functions.
