"""
CLI entry point — argparse only, no click/typer.

See STDLIB.md: click/typer -> argparse.

Subcommands:
  noembed index <folder> [--out PATH]
  noembed search <query> [--out PATH] [-k N] [--explain]
  noembed stats [--out PATH]

Exit codes: 0 on success, 1 on user/input errors (bad path, empty corpus,
missing index), 2 on unexpected internal errors. stdout carries results;
stderr carries errors and diagnostics, kept separate on purpose so output
is pipeable.
"""

import argparse
import sys
from pathlib import Path

from src.display import bold, contribution_bar, dim, score_bar, score_label
from src.index import NoEmbedIndex

DEFAULT_INDEX_PATH = ".noembed_index.json"

# Files with these extensions are treated as text and indexed. Anything
# else is skipped silently (not an error — a real folder has binaries,
# images, etc. in it).
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json"}


def _iter_text_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def cmd_index(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: '{folder}' is not a directory", file=sys.stderr)
        return 1

    documents: dict[str, str] = {}
    for path in _iter_text_files(folder):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"warning: skipping '{path}': {e}", file=sys.stderr)
            continue
        doc_id = str(path.relative_to(folder))
        documents[doc_id] = text

    if not documents:
        print(f"error: no text files found under '{folder}'", file=sys.stderr)
        return 1

    index = NoEmbedIndex()
    index.build(documents)
    index.save(args.out)

    stats = index.stats()
    print(
        f"indexed {stats['documents']} document(s), "
        f"{stats['vocabulary_size']} unique term(s) -> {args.out}"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    if not NoEmbedIndex.exists(args.out):
        print(f"error: no index found at '{args.out}' — run 'noembed index' first", file=sys.stderr)
        return 1

    index = NoEmbedIndex()
    try:
        index.load(args.out)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: could not load index at '{args.out}': {e}", file=sys.stderr)
        return 1

    if not args.query.strip():
        print("error: query must not be empty", file=sys.stderr)
        return 1

    results = index.search(args.query, k=args.k)
    if not results:
        print("no matches")
        return 0

    print(dim(f"query: \"{args.query}\"  ({len(results)} result(s))"))
    print()

    for rank, (doc_id, score, query_vector) in enumerate(results, start=1):
        print(f"{bold(f'{rank}. {doc_id}')}   {score_bar(score)}   {score_label(score)}")
        if args.explain:
            shared = index.explain(doc_id, query_vector)[:5]
            if shared:
                max_contribution = shared[0][1]
                for term, contribution in shared:
                    bar = contribution_bar(contribution, max_contribution)
                    print(f"     {term:<15} {bar}  {dim(f'{contribution:.4f}')}")
        print()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    if not NoEmbedIndex.exists(args.out):
        print(f"error: no index found at '{args.out}' — run 'noembed index' first", file=sys.stderr)
        return 1

    index = NoEmbedIndex()
    try:
        index.load(args.out)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: could not load index at '{args.out}': {e}", file=sys.stderr)
        return 1

    stats = index.stats()
    index_size = Path(args.out).stat().st_size
    print(bold("noembed index"))
    print(f"  {'documents':<16} {stats['documents']}")
    print(f"  {'vocabulary size':<16} {stats['vocabulary_size']}")
    print(f"  {'index file size':<16} {index_size:,} bytes")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """
    A human-readable view of what's actually inside an index file — the
    intended alternative to opening the raw JSON in an editor, which shows
    implementation detail (idf tables, inverted-index internals) rather
    than anything meaningful to look at directly.
    """
    if not NoEmbedIndex.exists(args.out):
        print(f"error: no index found at '{args.out}' — run 'noembed index' first", file=sys.stderr)
        return 1

    index = NoEmbedIndex()
    try:
        index.load(args.out)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: could not load index at '{args.out}': {e}", file=sys.stderr)
        return 1

    if args.doc:
        if args.doc not in index.doc_vectors:
            print(f"error: '{args.doc}' is not in this index", file=sys.stderr)
            print("available documents:", file=sys.stderr)
            for doc_id in index.document_ids():
                print(f"  {doc_id}", file=sys.stderr)
            return 1

        top = index.top_terms_for_doc(args.doc, n=args.top)
        print(bold(f"{args.doc}"))
        if not top:
            print(dim("  (no terms — empty document)"))
            return 0
        max_weight = top[0][1]
        for term, weight in top:
            bar = contribution_bar(weight, max_weight)
            print(f"  {term:<15} {bar}  {dim(f'{weight:.4f}')}")
        return 0

    stats = index.stats()
    print(bold("noembed index overview"))
    print(f"  {stats['documents']} document(s), {stats['vocabulary_size']} unique term(s)")
    print()

    print(bold(f"most distinctive terms (top {args.top} by rarity across the corpus)"))
    top = index.top_terms(n=args.top)
    if top:
        max_idf = top[0][1]
        for term, idf in top:
            bar = contribution_bar(idf, max_idf)
            print(f"  {term:<15} {bar}  {dim(f'idf={idf:.4f}')}")
    print()

    print(bold("documents"))
    for doc_id in index.document_ids():
        term_count = len(index.doc_vectors[doc_id])
        print(f"  {doc_id:<30} {dim(f'{term_count} unique term(s)')}")
    print()
    print(dim(f"tip: noembed inspect --out {args.out} --doc <name> to see one document's top terms"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noembed",
        description="Zero-dependency semantic search: TF-IDF + cosine similarity, stdlib only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Build an index from a folder of text files")
    p_index.add_argument("folder", help="Folder to index (recursively)")
    p_index.add_argument("--out", default=DEFAULT_INDEX_PATH, help="Index file path")
    p_index.set_defaults(func=cmd_index)

    p_search = subparsers.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=5, help="Number of results to return")
    p_search.add_argument("--out", default=DEFAULT_INDEX_PATH, help="Index file path")
    p_search.add_argument(
        "--explain", action="store_true", help="Show which shared terms drove each result's score"
    )
    p_search.set_defaults(func=cmd_search)

    p_stats = subparsers.add_parser("stats", help="Show index statistics")
    p_stats.add_argument("--out", default=DEFAULT_INDEX_PATH, help="Index file path")
    p_stats.set_defaults(func=cmd_stats)

    p_inspect = subparsers.add_parser(
        "inspect", help="Human-readable view of an index's contents (alternative to opening the raw JSON)"
    )
    p_inspect.add_argument("--out", default=DEFAULT_INDEX_PATH, help="Index file path")
    p_inspect.add_argument("--doc", default=None, help="Show top terms for one specific document")
    p_inspect.add_argument("--top", type=int, default=10, help="Number of terms to show")
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # last-resort guard so the CLI never stack-traces at a user
        print(f"internal error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())