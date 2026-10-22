"""Build the FAISS index with intfloat/e5-large-v2. Skips work when the index is current unless --force."""
import argparse
import sys

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.rag.embedding import E5Embedder
from app.rag.indexer import build_index, is_stale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the index is up to date")
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level)
    if not args.force and not is_stale(s):
        print(f"index up to date: {s.index_dir}")
        return 0
    try:
        store = build_index(s, E5Embedder(s))
    except AppError as exc:
        print(f"ERROR: {exc.detail}", file=sys.stderr)
        return 2
    print(f"index built: {store.manifest['num_chunks']} chunks, dim={store.manifest['dim']} -> {s.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
