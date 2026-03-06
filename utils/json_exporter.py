"""
Writes human-readable JSON snapshots of all engine data, useful for:
  • Inspecting the state of the search engine
  • AI/ML training datasets
  • Debugging and research

Output structure inside data/
├── crawled/
│   ├── pages.json          ← all crawled pages + rich metadata + scores
│   └── links.json          ← full link graph  {src: [dst, ...]}
├── pagerank/
│   └── scores.json         ← per-page authority scores (sorted by final score)
├── index/
│   ├── vocabulary.json     ← terms with FULL provenance per source page
│   └── indexed_pages.json  ← all indexed pages with metadata
└── entities/
    ├── named_entities.json ← all named entities + their source pages
    └── entity_types.json   ← entity type distribution summary
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR

logger = logging.getLogger("neurasearch.exporter")

# ── Output directories ────────────────────────────────────────────────────────
CRAWLED_DIR  = DATA_DIR / "crawled"
PAGERANK_DIR = DATA_DIR / "pagerank"
INDEX_DIR    = DATA_DIR / "index"
ENTITIES_DIR = DATA_DIR / "entities"

for _d in (CRAWLED_DIR, PAGERANK_DIR, INDEX_DIR, ENTITIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Low-level writer ──────────────────────────────────────────────────────────
def _write_json(path: Path, data: object, indent: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    tmp.replace(path)   # atomic rename
    logger.info(f"[Export] {path.relative_to(DATA_DIR.parent)}")
    return path


# ── Crawled pages ─────────────────────────────────────────────────────────────
def export_crawled_pages(db, include_content: bool = False) -> Path:
    """
    Save all crawled pages to data/crawled/pages.json.

    Each entry contains the URL, title, description, headings, meta_keywords,
    canonical_url, entities, scores, and optional truncated content.
    """
    pages = db.get_all_pages()
    rows = []
    for p in pages:
        # Parse stored JSON fields
        headings_parsed = None
        try:
            if p.get("headings"):
                headings_parsed = json.loads(p["headings"]) if isinstance(p["headings"], str) else p["headings"]
        except Exception:
            pass

        entities_parsed = None
        try:
            if p.get("entities_json"):
                entities_parsed = json.loads(p["entities_json"]) if isinstance(p["entities_json"], str) else p["entities_json"]
        except Exception:
            pass

        row = {
            "url":           p["url"],
            "title":         p["title"] or "",
            "description":   p["description"] or "",
            "meta_keywords": p.get("meta_keywords") or "",
            "canonical_url": p.get("canonical_url") or "",
            "headings":      headings_parsed or {},
            "entities":      entities_parsed or [],
            "language":      p.get("language") or "en",
            "word_count":    p["word_count"],
            "depth":         p["depth"],
            "status_code":   p["status_code"],
            "crawled_at":    p["crawled_at"],
            "in_link_count": p["in_link_count"],
            "out_link_count":p["out_link_count"],
            "scores": {
                "pagerank":   p["pagerank_score"],
                "panda":      p["panda_score"],
                "penguin":    p["penguin_score"],
                "final":      p["final_score"],
            },
            "is_indexed": bool(p["is_indexed"]),
        }
        if include_content:
            row["content_preview"] = (p["content"] or "")[:2_000]
        rows.append(row)

    rows.sort(key=lambda x: -(x["scores"]["final"] or 0))

    return _write_json(CRAWLED_DIR / "pages.json", {
        "exported_at":  datetime.utcnow().isoformat() + "Z",
        "total_pages":  len(rows),
        "pages":        rows,
    })


# ── Link graph ────────────────────────────────────────────────────────────────
def export_link_graph(db) -> Path:
    graph = db.get_link_graph()
    return _write_json(CRAWLED_DIR / "links.json", {
        "exported_at":   datetime.utcnow().isoformat() + "Z",
        "total_sources": len(graph),
        "total_links":   sum(len(v) for v in graph.values()),
        "graph":         graph,
    })


# ── PageRank / authority scores ───────────────────────────────────────────────
def export_pagerank_scores(db) -> Path:
    pages = db.get_all_pages()
    entries = []
    for p in pages:
        entries.append({
            "rank":          0,
            "url":           p["url"],
            "title":         p["title"] or "",
            "pagerank":      p["pagerank_score"],
            "panda":         p["panda_score"],
            "penguin":       p["penguin_score"],
            "final_score":   p["final_score"],
            "in_link_count": p["in_link_count"],
        })

    entries.sort(key=lambda x: -(x["final_score"] or 0))
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    return _write_json(PAGERANK_DIR / "scores.json", {
        "exported_at":  datetime.utcnow().isoformat() + "Z",
        "total_pages":  len(entries),
        "top_10":       entries[:10],
        "all_scores":   entries,
    })


# ── Vocabulary with full provenance ──────────────────────────────────────────
def export_index_vocabulary(db, limit: int = 20_000) -> Path:
    """
    Save the index vocabulary to data/index/vocabulary.json.

    Each term entry now includes:
      - term          : stemmed form (search key)
      - lemma         : lemmatized intermediate form (human-readable)
      - doc_freq      : number of documents containing this term
      - sources       : list of source pages, each with:
          - url, title, doc_id
          - field_flags : {in_title, in_description, in_url, in_anchor}
          - frequency   : raw frequency in this document
          - positions   : word positions in cleaned token stream
          - bm25        : BM25 score for this (term, doc) pair
          - original_forms : list of raw word forms seen in this doc
          - pos_tag     : dominant POS tag for this term in this doc
          - page_scores : {pagerank, panda} for ranking context
    """
    logger.info("[Export] Building vocabulary with full provenance…")
    term_rows = db.get_terms_with_doc_freq(limit=limit)

    vocabulary = []
    for row in term_rows:
        term     = row["term"]
        lemma    = row.get("lemma") or term
        doc_freq = row["doc_freq"]

        # Fetch provenance for this term (capped at 100 sources per term)
        provenance = db.get_term_provenance(term, limit=100)

        sources = []
        for p in provenance:
            # Parse JSON fields
            positions = []
            try:
                positions = json.loads(p["positions"]) if p.get("positions") else []
            except Exception:
                pass

            originals = []
            try:
                originals = json.loads(p["original_forms"]) if p.get("original_forms") else []
            except Exception:
                pass

            sources.append({
                "doc_id":   p["doc_id"],
                "url":      p["url"],
                "title":    p.get("title") or "",
                "field_flags": {
                    "in_title":       bool(p.get("in_title", 0)),
                    "in_description": bool(p.get("in_description", 0)),
                    "in_url":         bool(p.get("in_url", 0)),
                    "in_anchor":      bool(p.get("in_anchor", 0)),
                },
                "frequency":      p["frequency"],
                "positions":      positions[:50],   # cap for readability
                "bm25":           round(p["bm25"], 6),
                "original_forms": originals,
                "pos_tag":        p.get("pos_tag") or "",
                "page_scores": {
                    "pagerank": p.get("pagerank_score", 0.0),
                    "panda":    p.get("panda_score", 1.0),
                },
            })

        vocabulary.append({
            "term":     term,
            "lemma":    lemma,
            "doc_freq": doc_freq,
            "sources":  sources,
        })

    return _write_json(INDEX_DIR / "vocabulary.json", {
        "exported_at":  datetime.utcnow().isoformat() + "Z",
        "unique_terms": len(vocabulary),
        "shown":        len(vocabulary),
        "note": (
            "Each term includes its lemma, document frequency, and full provenance "
            "per source page (field flags, positions, BM25, original word forms)."
        ),
        "terms": vocabulary,
    })


# ── Indexed pages ─────────────────────────────────────────────────────────────
def export_indexed_pages(db) -> Path:
    pages = db.get_all_pages()
    indexed = []
    for p in pages:
        if not p.get("is_indexed"):
            continue

        headings_parsed = None
        try:
            if p.get("headings"):
                headings_parsed = json.loads(p["headings"]) if isinstance(p["headings"], str) else p["headings"]
        except Exception:
            pass

        indexed.append({
            "id":            p["id"],
            "url":           p["url"],
            "title":         p["title"] or "",
            "description":   (p["description"] or "")[:300],
            "meta_keywords": p.get("meta_keywords") or "",
            "headings":      headings_parsed or {},
            "word_count":    p["word_count"],
            "is_indexed":    True,
            "crawled_at":    p["crawled_at"],
        })

    return _write_json(INDEX_DIR / "indexed_pages.json", {
        "exported_at":   datetime.utcnow().isoformat() + "Z",
        "total_indexed": len(indexed),
        "pages":         indexed,
    })


# ── Named entities ────────────────────────────────────────────────────────────
def export_named_entities(db, limit: int = 10_000) -> Path:
    """
    Export all named entities to data/entities/named_entities.json.

    Format: list of unique entities, each with:
      - entity       : entity surface form
      - entity_type  : PERSON, ORGANIZATION, GPE, LOCATION, etc.
      - total_freq   : total occurrences across all pages
      - pages        : list of {url, title, frequency} where this entity appears
    """
    logger.info("[Export] Exporting named entities…")
    with db.get_conn() as conn:
        rows = conn.execute("""
            SELECT n.entity, n.entity_type,
                   SUM(n.frequency) AS total_freq,
                   COUNT(DISTINCT n.doc_id) AS doc_count
            FROM named_entities n
            GROUP BY n.entity, n.entity_type
            ORDER BY total_freq DESC
            LIMIT ?
        """, (limit,)).fetchall()

    entities = []
    for row in rows:
        pages_using = db.get_pages_by_entity(row["entity"])
        entities.append({
            "entity":      row["entity"],
            "entity_type": row["entity_type"],
            "total_freq":  row["total_freq"],
            "doc_count":   row["doc_count"],
            "pages": [
                {
                    "url":       p["url"],
                    "title":     p.get("title") or "",
                    "frequency": p["frequency"],
                    "pagerank":  p.get("pagerank_score", 0.0),
                }
                for p in pages_using[:20]   # cap per entity
            ],
        })

    return _write_json(ENTITIES_DIR / "named_entities.json", {
        "exported_at":    datetime.utcnow().isoformat() + "Z",
        "unique_entities": len(entities),
        "entities":        entities,
    })


def export_entity_types(db) -> Path:
    """Export entity type distribution to data/entities/entity_types.json."""
    summary = db.get_entity_types_summary()
    return _write_json(ENTITIES_DIR / "entity_types.json", {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "distribution": [dict(r) for r in summary],
    })


# ── Convenience: export everything ───────────────────────────────────────────
def export_all(db) -> list:
    """Run all exports. Returns list of paths written."""
    fns = [
        export_crawled_pages,
        export_link_graph,
        export_pagerank_scores,
        export_index_vocabulary,
        export_indexed_pages,
        export_named_entities,
        export_entity_types,
    ]
    paths = []
    for fn in fns:
        try:
            paths.append(fn(db))
        except Exception as exc:
            logger.warning(f"[Export] {fn.__name__} failed: {exc}")
    return paths
