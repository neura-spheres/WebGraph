"""
FastAPI REST API

Endpoints
---------
GET  /             → health check
GET  /stats        → index statistics
GET  /search       → main search  ?q=...&page=1&limit=10&expand=false
GET  /suggest      → autocomplete ?q=...&limit=8
GET  /entity       → entity search ?name=...
POST /crawl        → start a crawl session
POST /index        → run indexer on un-indexed pages
POST /pagerank     → run full PageRank pipeline
POST /export       → export all JSON data snapshots
GET  /tasks        → background task status
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))
from config import CORS_ORIGINS, DEFAULT_RESULTS_PER_PAGE, MAX_RESULTS
from database.db import Database
from indexer.indexer import Indexer
from indexer.text_processor import process_text, build_snippet
from pagerank.pagerank import PageRankEngine
from crawler.crawler import Crawler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neurasearch.api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuraSearch API",
    description="Search engine backend — crawl, index, rank, and search the web.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared singletons ─────────────────────────────────────────────────────────
db      = Database()
indexer = Indexer(db)
pr_eng  = PageRankEngine(db)

_active_tasks: dict = {}
_task_lock = threading.Lock()


# ── Request / Response models ─────────────────────────────────────────────────
class CrawlRequest(BaseModel):
    seeds:       list
    limit:       int = 10_000
    max_depth:   int = 6
    max_workers: int = 10


class SearchResult(BaseModel):
    url:         str
    title:       str
    description: str
    snippet:     str        # context window around query term hits
    score:       float
    pagerank:    float
    panda:       float
    penguin:     float
    matched_terms: list     # stemmed query terms that matched this result


class SearchResponse(BaseModel):
    query:          str
    processed_terms: list   # after lemmatize + stem
    expanded_terms:  list   # after WordNet expansion (if enabled)
    total:          int
    page:           int
    limit:          int
    results:        list
    took_ms:        float


# ── Query expansion via WordNet ───────────────────────────────────────────────
def _expand_query(stems: list, max_synonyms: int = 2) -> list:
    """
    Expand a list of stemmed query terms with WordNet synonyms.
    Synonyms are processed through the same pipeline (lemmatize+stem).

    Returns the expanded list (original terms first, then synonyms).
    """
    try:
        from nltk.corpus import wordnet
        expanded = list(stems)
        seen = set(stems)
        for stem in stems[:5]:   # expand first 5 terms only
            synsets = wordnet.synsets(stem)[:2]
            for synset in synsets:
                for lemma in synset.lemmas()[:max_synonyms]:
                    syn_raw = lemma.name().replace("_", " ").lower()
                    syn_stems = process_text(syn_raw)
                    for s in syn_stems:
                        if s not in seen:
                            seen.add(s)
                            expanded.append(s)
        return expanded
    except Exception:
        return stems


# ── Helpers ───────────────────────────────────────────────────────────────────
def _run_background(name: str, fn):
    def wrapper():
        with _task_lock:
            _active_tasks[name] = "running"
        try:
            fn()
            with _task_lock:
                _active_tasks[name] = "done"
        except Exception as exc:
            logger.exception(f"Background task '{name}' failed: {exc}")
            with _task_lock:
                _active_tasks[name] = "error"

    t = threading.Thread(target=wrapper, daemon=True, name=name)
    t.start()
    return t


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "service": "NeuraSearch API", "version": "2.0.0"}


@app.get("/stats", tags=["info"])
def stats():
    return db.get_stats()


@app.get("/tasks", tags=["info"])
def task_status():
    with _task_lock:
        return dict(_active_tasks)


# ── Search ────────────────────────────────────────────────────────────────────
@app.get("/search", tags=["search"])
def search(
    q:      str = Query(..., min_length=1, max_length=500),
    page:   int = Query(1, ge=1),
    limit:  int = Query(DEFAULT_RESULTS_PER_PAGE, ge=1, le=50),
    expand: bool = Query(False, description="Expand query with WordNet synonyms"),
):
    t0 = time.perf_counter()

    # Process query through same NLP pipeline as indexer
    query_terms  = process_text(q)
    search_terms = query_terms

    expanded_terms = []
    if expand and query_terms:
        search_terms   = _expand_query(query_terms)
        expanded_terms = [t for t in search_terms if t not in query_terms]

    offset  = (page - 1) * limit
    raw     = indexer.index.search(search_terms, limit=min(limit, MAX_RESULTS), offset=offset)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for r in raw:
        # Generate snippet from stored content
        content = r.get("content", "") or ""
        snippet = build_snippet(content, search_terms, window=40) if content else ""

        results.append({
            "url":           r["url"],
            "title":         r.get("title") or "No Title",
            "description":   r.get("description") or "",
            "snippet":       snippet,
            "score":         r.get("score", 0.0),
            "pagerank":      r.get("pagerank", 0.0),
            "panda":         r.get("panda", 1.0),
            "penguin":       r.get("penguin", 1.0),
            "matched_terms": list(set(r.get("matched_terms", []))),
        })

    return {
        "query":           q,
        "processed_terms": query_terms,
        "expanded_terms":  expanded_terms,
        "total":           len(results),
        "page":            page,
        "limit":           limit,
        "results":         results,
        "took_ms":         round(elapsed_ms, 2),
    }


@app.get("/suggest", tags=["search"])
def suggest(
    q:     str = Query(..., min_length=2, max_length=100),
    limit: int = Query(8, ge=1, le=20),
):
    terms = indexer.get_suggestion_terms(q, limit=limit)
    return {"query": q, "suggestions": terms}


@app.get("/entity", tags=["search"])
def entity_search(
    name:  str = Query(..., min_length=2, max_length=200, description="Entity name to look up"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Find all pages that contain a given named entity.
    e.g. /entity?name=Python+Software+Foundation
    """
    pages = db.get_pages_by_entity(name)[:limit]
    return {
        "entity":     name,
        "page_count": len(pages),
        "pages":      pages,
    }


# ── Admin: crawl ──────────────────────────────────────────────────────────────
@app.post("/crawl", tags=["admin"])
def start_crawl(req: CrawlRequest):
    with _task_lock:
        if _active_tasks.get("crawl") == "running":
            raise HTTPException(409, "A crawl is already running.")

    def _do_crawl():
        crawler = Crawler(
            db,
            seed_urls   = req.seeds,
            crawl_limit = req.limit,
            max_workers = req.max_workers,
            max_depth   = req.max_depth,
        )
        crawler.run()

    _run_background("crawl", _do_crawl)
    return {"message": "Crawl started in background.", "seeds": req.seeds}


# ── Admin: index ──────────────────────────────────────────────────────────────
@app.post("/index", tags=["admin"])
def run_indexer():
    with _task_lock:
        if _active_tasks.get("index") == "running":
            raise HTTPException(409, "Indexer is already running.")

    _run_background("index", lambda: indexer.run(refresh_bm25=True))
    return {"message": "Indexer started in background."}


# ── Admin: pagerank ───────────────────────────────────────────────────────────
@app.post("/pagerank", tags=["admin"])
def run_pagerank():
    with _task_lock:
        if _active_tasks.get("pagerank") == "running":
            raise HTTPException(409, "PageRank is already running.")

    def _do_pr():
        pr_eng.run(run_panda=True, run_penguin=True)
        indexer.index.refresh_all_bm25()

    _run_background("pagerank", _do_pr)
    return {"message": "PageRank pipeline started in background."}


# ── Admin: export ──────────────────────────────────────────────────────────────
@app.post("/export", tags=["admin"])
def run_export():
    with _task_lock:
        if _active_tasks.get("export") == "running":
            raise HTTPException(409, "Export is already running.")

    def _do_export():
        from utils.json_exporter import export_all
        paths = export_all(db)
        logger.info(f"[Export] Wrote {len(paths)} files.")

    _run_background("export", _do_export)
    return {"message": "Full export started in background."}
