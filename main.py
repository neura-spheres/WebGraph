"""
NeuraSearch — CLI entry point.

Usage
-----
# Start API server (default)
python main.py serve

# Start a crawl from seed URLs
python main.py crawl https://en.wikipedia.org

# Run indexer on all un-indexed pages
python main.py index

# Run full PageRank pipeline (Panda + Penguin + PageRank)
python main.py pagerank

# Do everything in sequence: crawl → index → pagerank → serve
python main.py all --seeds https://en.wikipedia.org

# First-time setup (NLTK data download)
python main.py setup
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neurasearch")


def cmd_setup(_args):
    from setup_nltk import packages
    import nltk
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=False)
        except Exception as e:
            logger.warning(f"Could not download {pkg}: {e}")
    logger.info("Setup complete.")


def cmd_serve(args):
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(
        "api.app:app",
        host=API_HOST,
        port=getattr(args, "port", API_PORT),
        reload=getattr(args, "reload", False),
        log_level="info",
    )


def cmd_crawl(args):
    from database.db import Database
    from crawler.crawler import Crawler
    db = Database()
    seeds = args.seeds or ["https://en.wikipedia.org/wiki/Python_(programming_language)"]
    logger.info(f"Crawling {len(seeds)} seed(s)…")
    crawler = Crawler(
        db,
        seed_urls   = seeds,
        crawl_limit = args.limit,
        max_workers = args.workers,
        max_depth   = args.depth,
    )
    crawler.run()


def cmd_index(_args):
    from database.db import Database
    from indexer.indexer import Indexer
    db = Database()
    idx = Indexer(db)
    n = idx.run(refresh_bm25=True)
    logger.info(f"Indexed {n} pages.")


def cmd_pagerank(_args):
    from database.db import Database
    from pagerank.pagerank import PageRankEngine
    from indexer.indexer import Indexer
    db = Database()
    engine = PageRankEngine(db)
    engine.run(run_panda=True, run_penguin=True)
    # Refresh BM25 after authority update
    Indexer(db).index.refresh_all_bm25()


def cmd_export(_args):
    from database.db import Database
    from utils.json_exporter import export_all
    db = Database()
    paths = export_all(db)
    logger.info(f"Exported {len(paths)} file(s).")


def cmd_all(args):
    cmd_crawl(args)
    cmd_index(args)
    cmd_pagerank(args)
    cmd_export(args)
    cmd_serve(args)


# ── Parser ────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(prog="neurasearch", description="NeuraSearch engine CLI")
    sub = parser.add_subparsers(dest="command")

    # setup
    sub.add_parser("setup", help="Download NLTK data (run once)")

    # serve
    sp = sub.add_parser("serve", help="Start the API server")
    sp.add_argument("--port",   type=int, default=8000)
    sp.add_argument("--reload", action="store_true")

    # crawl
    cp = sub.add_parser("crawl", help="Crawl from seed URLs")
    cp.add_argument("seeds",    nargs="*", default=[])
    cp.add_argument("--limit",   type=int, default=50_000)
    cp.add_argument("--workers", type=int, default=10)
    cp.add_argument("--depth",   type=int, default=6)

    # index
    sub.add_parser("index", help="Index all un-indexed pages")

    # pagerank
    sub.add_parser("pagerank", help="Run PageRank + Panda + Penguin pipeline")

    # export
    sub.add_parser("export", help="Export all data to JSON files")

    # all
    ap = sub.add_parser("all", help="Crawl → Index → PageRank → Serve")
    ap.add_argument("--seeds",   nargs="*", default=[])
    ap.add_argument("--limit",   type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--depth",   type=int, default=6)
    ap.add_argument("--port",    type=int, default=8000)
    ap.add_argument("--reload",  action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "setup":    cmd_setup,
        "serve":    cmd_serve,
        "crawl":    cmd_crawl,
        "index":    cmd_index,
        "pagerank": cmd_pagerank,
        "export":   cmd_export,
        "all":      cmd_all,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
