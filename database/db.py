"""
Central SQLite database layer.

All threads create their own connection (SQLite is safe when each
thread uses its own connection and WAL mode is enabled).
"""

import sqlite3
import json
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH

_local = threading.local()


class Database:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self._create_tables()
        self._migrate_schema()

    # ── Connection management ────────────────────────────────────────────────
    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")   # 256 MB
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema ───────────────────────────────────────────────────────────────
    def _create_tables(self):
        with self.get_conn() as conn:
            conn.executescript("""
                -- ── Crawled pages ──────────────────────────────────────────
                CREATE TABLE IF NOT EXISTS pages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    url             TEXT    UNIQUE NOT NULL,
                    title           TEXT,
                    description     TEXT,
                    content         TEXT,
                    content_hash    TEXT,
                    word_count      INTEGER DEFAULT 0,
                    language        TEXT    DEFAULT 'en',
                    crawled_at      TEXT,
                    last_modified   TEXT,
                    status_code     INTEGER DEFAULT 200,
                    depth           INTEGER DEFAULT 0,
                    in_link_count   INTEGER DEFAULT 0,
                    out_link_count  INTEGER DEFAULT 0,
                    pagerank_score  REAL    DEFAULT 0.0,
                    panda_score     REAL    DEFAULT 1.0,
                    penguin_score   REAL    DEFAULT 1.0,
                    final_score     REAL    DEFAULT 0.0,
                    is_indexed      INTEGER DEFAULT 0,
                    -- Rich content fields (added in v2)
                    headings        TEXT,           -- JSON: {"h1":["..."],"h2":[...]}
                    entities_json   TEXT,           -- JSON: [{"text":"...","label":"..."}]
                    meta_keywords   TEXT,           -- raw meta keywords string
                    canonical_url   TEXT            -- canonical URL if differs from url
                );

                -- ── Link graph ─────────────────────────────────────────────
                CREATE TABLE IF NOT EXISTS links (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    src_url     TEXT NOT NULL,
                    dst_url     TEXT NOT NULL,
                    anchor_text TEXT,
                    rel         TEXT,
                    UNIQUE(src_url, dst_url)
                );

                -- ── URL frontier / crawl queue ─────────────────────────────
                CREATE TABLE IF NOT EXISTS crawl_queue (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    url      TEXT UNIQUE NOT NULL,
                    priority REAL    DEFAULT 0.0,
                    depth    INTEGER DEFAULT 0,
                    added_at TEXT,
                    status   TEXT    DEFAULT 'pending'
                );

                -- ── Inverted index ─────────────────────────────────────────
                -- One row per (term, document) pair.
                -- term        : Porter-stemmed form  (search key)
                -- lemma       : WordNet-lemmatized form before stemming
                -- original_forms: JSON list of raw word forms seen in this doc
                -- in_title / in_description / in_url / in_anchor : field flags
                -- pos_tag     : dominant NLTK POS tag for this term in this doc
                CREATE TABLE IF NOT EXISTS inverted_index (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    term           TEXT    NOT NULL,
                    doc_id         INTEGER NOT NULL,
                    frequency      INTEGER DEFAULT 0,
                    positions      TEXT,            -- JSON list of word positions
                    tf_idf         REAL    DEFAULT 0.0,
                    bm25           REAL    DEFAULT 0.0,
                    in_title       INTEGER DEFAULT 0,
                    in_description INTEGER DEFAULT 0,
                    lemma          TEXT,            -- lemmatized intermediate form
                    original_forms TEXT,            -- JSON list of distinct raw forms
                    in_url         INTEGER DEFAULT 0,
                    in_anchor      INTEGER DEFAULT 0,
                    pos_tag        TEXT,            -- e.g. "NN", "VB", "JJ"
                    FOREIGN KEY (doc_id) REFERENCES pages(id) ON DELETE CASCADE,
                    UNIQUE(term, doc_id)
                );

                -- ── Named entities ─────────────────────────────────────────
                -- One row per (document, entity, entity_type) triple.
                CREATE TABLE IF NOT EXISTS named_entities (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id      INTEGER NOT NULL,
                    entity      TEXT    NOT NULL,
                    entity_type TEXT    NOT NULL,    -- PERSON, ORGANIZATION, GPE …
                    frequency   INTEGER DEFAULT 1,
                    FOREIGN KEY (doc_id) REFERENCES pages(id) ON DELETE CASCADE,
                    UNIQUE(doc_id, entity, entity_type)
                );

                -- ── Anchor text index ──────────────────────────────────────
                -- Tracks stemmed anchor-text terms pointing at destination pages.
                CREATE TABLE IF NOT EXISTS anchor_index (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    term      TEXT NOT NULL,
                    dst_url   TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    UNIQUE(term, dst_url)
                );

                -- ── Per-domain crawl metadata ──────────────────────────────
                CREATE TABLE IF NOT EXISTS domain_stats (
                    domain       TEXT PRIMARY KEY,
                    page_count   INTEGER DEFAULT 0,
                    avg_pagerank REAL    DEFAULT 0.0,
                    last_crawled TEXT,
                    crawl_delay  REAL    DEFAULT 1.0,
                    robots_txt   TEXT,
                    is_blocked   INTEGER DEFAULT 0
                );

                -- ── Indices ────────────────────────────────────────────────
                CREATE INDEX IF NOT EXISTS idx_inv_term      ON inverted_index(term);
                CREATE INDEX IF NOT EXISTS idx_inv_doc       ON inverted_index(doc_id);
                CREATE INDEX IF NOT EXISTS idx_links_src     ON links(src_url);
                CREATE INDEX IF NOT EXISTS idx_links_dst     ON links(dst_url);
                CREATE INDEX IF NOT EXISTS idx_pages_score   ON pages(final_score DESC);
                CREATE INDEX IF NOT EXISTS idx_queue_pri     ON crawl_queue(priority DESC, status);
                CREATE INDEX IF NOT EXISTS idx_anchor_term   ON anchor_index(term);
            """)

    def _migrate_schema(self):
        """
        Safely add new columns / tables to an existing database.
        SQLite does not support IF NOT EXISTS on ALTER TABLE, so we
        use try/except to skip columns that already exist.
        """
        inverted_index_cols = [
            "ALTER TABLE inverted_index ADD COLUMN lemma TEXT",
            "ALTER TABLE inverted_index ADD COLUMN original_forms TEXT",
            "ALTER TABLE inverted_index ADD COLUMN in_url INTEGER DEFAULT 0",
            "ALTER TABLE inverted_index ADD COLUMN in_anchor INTEGER DEFAULT 0",
            "ALTER TABLE inverted_index ADD COLUMN pos_tag TEXT",
        ]
        pages_cols = [
            "ALTER TABLE pages ADD COLUMN headings TEXT",
            "ALTER TABLE pages ADD COLUMN entities_json TEXT",
            "ALTER TABLE pages ADD COLUMN meta_keywords TEXT",
            "ALTER TABLE pages ADD COLUMN canonical_url TEXT",
        ]
        with self.get_conn() as conn:
            for sql in inverted_index_cols + pages_cols:
                try:
                    conn.execute(sql)
                except Exception:
                    pass  # Column already exists — that's fine

            # Indices that depend on columns added by migration
            post_migration_indices = [
                "CREATE INDEX IF NOT EXISTS idx_inv_lemma ON inverted_index(lemma)",
                "CREATE INDEX IF NOT EXISTS idx_ne_entity ON named_entities(entity)",
                "CREATE INDEX IF NOT EXISTS idx_ne_doc    ON named_entities(doc_id)",
                "CREATE INDEX IF NOT EXISTS idx_ne_type   ON named_entities(entity_type)",
            ]
            for sql in post_migration_indices:
                try:
                    conn.execute(sql)
                except Exception:
                    pass

    # ── Pages ────────────────────────────────────────────────────────────────
    def upsert_page(self, url, *, title=None, description=None, content=None,
                    content_hash=None, word_count=0, status_code=200, depth=0,
                    headings=None, entities_json=None,
                    meta_keywords=None, canonical_url=None):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO pages
                    (url, title, description, content, content_hash,
                     word_count, status_code, depth, crawled_at,
                     headings, entities_json, meta_keywords, canonical_url)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(url) DO UPDATE SET
                    title         = excluded.title,
                    description   = excluded.description,
                    content       = excluded.content,
                    content_hash  = excluded.content_hash,
                    word_count    = excluded.word_count,
                    status_code   = excluded.status_code,
                    depth         = excluded.depth,
                    crawled_at    = excluded.crawled_at,
                    headings      = excluded.headings,
                    entities_json = excluded.entities_json,
                    meta_keywords = excluded.meta_keywords,
                    canonical_url = excluded.canonical_url
            """, (url, title, description, content, content_hash,
                  word_count, status_code, depth,
                  datetime.utcnow().isoformat(),
                  json.dumps(headings) if headings is not None else None,
                  json.dumps(entities_json) if entities_json is not None else None,
                  meta_keywords, canonical_url))

    def get_page(self, url):
        with self.get_conn() as conn:
            row = conn.execute("SELECT * FROM pages WHERE url=?", (url,)).fetchone()
            return dict(row) if row else None

    def get_page_by_id(self, doc_id):
        with self.get_conn() as conn:
            row = conn.execute("SELECT * FROM pages WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def get_all_pages(self):
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE status_code=200"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pages_not_indexed(self):
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE is_indexed=0 AND status_code=200"
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_page_indexed(self, url):
        with self.get_conn() as conn:
            conn.execute("UPDATE pages SET is_indexed=1 WHERE url=?", (url,))

    def update_page_scores(self, url, pagerank_score=None,
                           panda_score=None, penguin_score=None):
        with self.get_conn() as conn:
            if pagerank_score is not None:
                conn.execute(
                    "UPDATE pages SET pagerank_score=? WHERE url=?",
                    (pagerank_score, url))
            if panda_score is not None:
                conn.execute(
                    "UPDATE pages SET panda_score=? WHERE url=?",
                    (panda_score, url))
            if penguin_score is not None:
                conn.execute(
                    "UPDATE pages SET penguin_score=? WHERE url=?",
                    (penguin_score, url))
            conn.execute("""
                UPDATE pages
                SET final_score = pagerank_score * panda_score * penguin_score
                WHERE url=?
            """, (url,))

    def bulk_update_pagerank(self, scores: dict):
        """scores: {url: pagerank_float}"""
        with self.get_conn() as conn:
            conn.executemany(
                "UPDATE pages SET pagerank_score=? WHERE url=?",
                [(v, k) for k, v in scores.items()])
            conn.execute("""
                UPDATE pages
                SET final_score = pagerank_score * panda_score * penguin_score
            """)

    def get_page_count(self):
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM pages WHERE status_code=200"
            ).fetchone()[0]

    def get_avg_word_count(self):
        with self.get_conn() as conn:
            result = conn.execute(
                "SELECT AVG(word_count) FROM pages WHERE status_code=200 AND word_count>0"
            ).fetchone()[0]
            return result or 0.0

    def url_exists(self, url):
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT 1 FROM pages WHERE url=?", (url,)
            ).fetchone() is not None

    # ── Links ────────────────────────────────────────────────────────────────
    def add_link(self, src_url, dst_url, anchor_text=None, rel=None):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO links (src_url, dst_url, anchor_text, rel)
                VALUES (?,?,?,?)
            """, (src_url, dst_url, anchor_text, rel))

    def add_links_bulk(self, rows):
        """rows: list of (src_url, dst_url, anchor_text, rel)"""
        with self.get_conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO links (src_url, dst_url, anchor_text, rel)
                VALUES (?,?,?,?)
            """, rows)

    def get_link_graph(self):
        """Returns {src_url: [dst_url, ...]}"""
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT src_url, dst_url FROM links"
            ).fetchall()
        graph = {}
        for row in rows:
            graph.setdefault(row["src_url"], []).append(row["dst_url"])
        return graph

    def get_full_link_graph(self):
        """Returns {src_url: [{dst_url, anchor_text}]}"""
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT src_url, dst_url, anchor_text FROM links"
            ).fetchall()
        graph = {}
        for row in rows:
            graph.setdefault(row["src_url"], []).append({
                "dst_url": row["dst_url"],
                "anchor_text": row["anchor_text"] or "",
            })
        return graph

    def get_inbound_links(self, url):
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT src_url, anchor_text FROM links WHERE dst_url=?", (url,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_link_counts(self):
        with self.get_conn() as conn:
            conn.execute("""
                UPDATE pages SET out_link_count=(
                    SELECT COUNT(*) FROM links WHERE src_url=pages.url)
            """)
            conn.execute("""
                UPDATE pages SET in_link_count=(
                    SELECT COUNT(*) FROM links WHERE dst_url=pages.url)
            """)

    # ── Crawl queue ──────────────────────────────────────────────────────────
    def enqueue(self, url, priority=0.0, depth=0):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO crawl_queue
                    (url, priority, depth, added_at, status)
                VALUES (?,?,?,?,'pending')
            """, (url, priority, depth, datetime.utcnow().isoformat()))

    def enqueue_bulk(self, rows):
        """rows: list of (url, priority, depth)"""
        now = datetime.utcnow().isoformat()
        with self.get_conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO crawl_queue
                    (url, priority, depth, added_at, status)
                VALUES (?,?,?,?,'pending')
            """, [(url, pri, depth, now) for (url, pri, depth) in rows])

    def dequeue_batch(self, batch_size=10):
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM crawl_queue
                WHERE status='pending'
                ORDER BY priority DESC
                LIMIT ?
            """, (batch_size,)).fetchall()
            if rows:
                ids = [r["id"] for r in rows]
                conn.execute(
                    f"UPDATE crawl_queue SET status='processing' "
                    f"WHERE id IN ({','.join('?' * len(ids))})",
                    ids)
            return [dict(r) for r in rows]

    def mark_queue_done(self, url):
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE crawl_queue SET status='done' WHERE url=?", (url,))

    def mark_queue_failed(self, url):
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE crawl_queue SET status='failed' WHERE url=?", (url,))

    def reset_stuck_queue(self) -> int:
        with self.get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM crawl_queue WHERE status='processing'"
            ).fetchone()[0]
            if count > 0:
                conn.execute(
                    "UPDATE crawl_queue SET status='pending' WHERE status='processing'"
                )
        return count

    def get_pending_urls(self, limit: int = 100_000) -> list:
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT url, priority, depth FROM crawl_queue
                WHERE status='pending'
                ORDER BY priority DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def pending_count(self):
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM crawl_queue WHERE status='pending'"
            ).fetchone()[0]

    def queue_url_exists(self, url):
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT 1 FROM crawl_queue WHERE url=?", (url,)
            ).fetchone() is not None

    # ── Inverted index ───────────────────────────────────────────────────────
    def upsert_term(self, term, doc_id, frequency, positions,
                    tf_idf=0.0, bm25=0.0, in_title=False, in_description=False,
                    lemma=None, original_forms=None,
                    in_url=False, in_anchor=False, pos_tag=None):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO inverted_index
                    (term, doc_id, frequency, positions, tf_idf, bm25,
                     in_title, in_description,
                     lemma, original_forms, in_url, in_anchor, pos_tag)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(term, doc_id) DO UPDATE SET
                    frequency      = excluded.frequency,
                    positions      = excluded.positions,
                    tf_idf         = excluded.tf_idf,
                    bm25           = excluded.bm25,
                    in_title       = excluded.in_title,
                    in_description = excluded.in_description,
                    lemma          = excluded.lemma,
                    original_forms = excluded.original_forms,
                    in_url         = excluded.in_url,
                    in_anchor      = excluded.in_anchor,
                    pos_tag        = excluded.pos_tag
            """, (term, doc_id, frequency, json.dumps(positions),
                  tf_idf, bm25, int(in_title), int(in_description),
                  lemma,
                  json.dumps(original_forms) if original_forms is not None else None,
                  int(in_url), int(in_anchor), pos_tag))

    def upsert_terms_bulk(self, rows):
        """
        rows: list of 13-tuples:
          (term, doc_id, freq, positions_json, tf_idf, bm25,
           in_title, in_desc, lemma, original_forms_json, in_url, in_anchor, pos_tag)
        """
        with self.get_conn() as conn:
            conn.executemany("""
                INSERT INTO inverted_index
                    (term, doc_id, frequency, positions, tf_idf, bm25,
                     in_title, in_description,
                     lemma, original_forms, in_url, in_anchor, pos_tag)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(term, doc_id) DO UPDATE SET
                    frequency      = excluded.frequency,
                    positions      = excluded.positions,
                    tf_idf         = excluded.tf_idf,
                    bm25           = excluded.bm25,
                    in_title       = excluded.in_title,
                    in_description = excluded.in_description,
                    lemma          = excluded.lemma,
                    original_forms = excluded.original_forms,
                    in_url         = excluded.in_url,
                    in_anchor      = excluded.in_anchor,
                    pos_tag        = excluded.pos_tag
            """, rows)

    def get_postings(self, term):
        """Return all postings for a term with page data (including rich fields)."""
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT i.doc_id, i.frequency, i.positions, i.tf_idf, i.bm25,
                       i.in_title, i.in_description, i.in_url, i.in_anchor,
                       i.lemma, i.original_forms, i.pos_tag,
                       p.url, p.title, p.description, p.content,
                       p.word_count, p.pagerank_score, p.panda_score, p.penguin_score,
                       p.final_score
                FROM inverted_index i
                JOIN pages p ON i.doc_id = p.id
                WHERE i.term=? AND p.status_code=200
            """, (term,)).fetchall()
            return [dict(r) for r in rows]

    def get_term_provenance(self, term, limit: int = 100):
        """
        Return full provenance for a term: all documents it appears in,
        with field info, lemma, original forms, scores.
        Used by the vocabulary exporter.
        """
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT i.doc_id, i.frequency, i.positions, i.bm25,
                       i.in_title, i.in_description, i.in_url, i.in_anchor,
                       i.lemma, i.original_forms, i.pos_tag,
                       p.url, p.title, p.pagerank_score, p.panda_score
                FROM inverted_index i
                JOIN pages p ON i.doc_id = p.id
                WHERE i.term=? AND p.status_code=200
                ORDER BY i.bm25 DESC
                LIMIT ?
            """, (term, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_doc_freq(self, term):
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM inverted_index WHERE term=?",
                (term,)
            ).fetchone()[0]

    def get_postings_for_doc(self, doc_id: int):
        """Return all index rows for a given document (for BM25 refresh)."""
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT term, frequency FROM inverted_index WHERE doc_id=?",
                (doc_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_terms(self):
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT term FROM inverted_index"
            ).fetchall()
            return [r[0] for r in rows]

    def get_terms_with_doc_freq(self, limit: int = 0):
        """Return list of (term, lemma, doc_freq) sorted by doc_freq desc."""
        sql = """
            SELECT i.term,
                   MAX(i.lemma) AS lemma,
                   COUNT(DISTINCT i.doc_id) AS doc_freq
            FROM inverted_index i
            GROUP BY i.term
            ORDER BY doc_freq DESC
        """
        if limit > 0:
            sql += f" LIMIT {limit}"
        with self.get_conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]

    def delete_doc_index(self, doc_id):
        with self.get_conn() as conn:
            conn.execute(
                "DELETE FROM inverted_index WHERE doc_id=?", (doc_id,))

    # ── Named entities ────────────────────────────────────────────────────────
    def upsert_named_entity(self, doc_id: int, entity: str,
                            entity_type: str, frequency: int = 1):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO named_entities (doc_id, entity, entity_type, frequency)
                VALUES (?,?,?,?)
                ON CONFLICT(doc_id, entity, entity_type)
                DO UPDATE SET frequency = frequency + excluded.frequency
            """, (doc_id, entity, entity_type, frequency))

    def upsert_named_entities_bulk(self, rows):
        """rows: list of (doc_id, entity, entity_type, frequency)"""
        with self.get_conn() as conn:
            conn.executemany("""
                INSERT INTO named_entities (doc_id, entity, entity_type, frequency)
                VALUES (?,?,?,?)
                ON CONFLICT(doc_id, entity, entity_type)
                DO UPDATE SET frequency = frequency + excluded.frequency
            """, rows)

    def get_named_entities_for_doc(self, doc_id: int) -> list:
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT entity, entity_type, frequency
                FROM named_entities WHERE doc_id=?
                ORDER BY frequency DESC
            """, (doc_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_pages_by_entity(self, entity: str) -> list:
        """Find all pages that contain a given named entity."""
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT p.url, p.title, n.entity_type, n.frequency,
                       p.pagerank_score, p.final_score
                FROM named_entities n
                JOIN pages p ON n.doc_id = p.id
                WHERE LOWER(n.entity) = LOWER(?)
                ORDER BY p.final_score DESC
            """, (entity,)).fetchall()
            return [dict(r) for r in rows]

    def get_entity_types_summary(self) -> list:
        """Return counts of named entities grouped by type."""
        with self.get_conn() as conn:
            rows = conn.execute("""
                SELECT entity_type, COUNT(DISTINCT entity) AS unique_entities,
                       COUNT(*) AS total_occurrences
                FROM named_entities
                GROUP BY entity_type
                ORDER BY total_occurrences DESC
            """).fetchall()
            return [dict(r) for r in rows]

    # ── Anchor index ─────────────────────────────────────────────────────────
    def add_anchor_term(self, term, dst_url):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO anchor_index (term, dst_url, frequency) VALUES (?,?,1)
                ON CONFLICT(term, dst_url) DO UPDATE SET frequency = frequency + 1
            """, (term, dst_url))

    def add_anchor_terms_bulk(self, rows):
        """rows: list of (term, dst_url)"""
        with self.get_conn() as conn:
            conn.executemany("""
                INSERT INTO anchor_index (term, dst_url, frequency) VALUES (?,?,1)
                ON CONFLICT(term, dst_url) DO UPDATE SET frequency = frequency + 1
            """, rows)

    def get_anchor_postings(self, term):
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT dst_url, frequency FROM anchor_index WHERE term=?", (term,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Domain stats ─────────────────────────────────────────────────────────
    def upsert_domain(self, domain, crawl_delay=1.0,
                      robots_txt=None, is_blocked=False):
        with self.get_conn() as conn:
            conn.execute("""
                INSERT INTO domain_stats
                    (domain, crawl_delay, robots_txt, is_blocked, last_crawled, page_count)
                VALUES (?,?,?,?,?,1)
                ON CONFLICT(domain) DO UPDATE SET
                    crawl_delay  = excluded.crawl_delay,
                    robots_txt   = excluded.robots_txt,
                    is_blocked   = excluded.is_blocked,
                    last_crawled = excluded.last_crawled,
                    page_count   = page_count + 1
            """, (domain, crawl_delay, robots_txt,
                  int(is_blocked), datetime.utcnow().isoformat()))

    def get_domain(self, domain):
        with self.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM domain_stats WHERE domain=?", (domain,)
            ).fetchone()
            return dict(row) if row else None

    # ── Stats ────────────────────────────────────────────────────────────────
    def get_stats(self):
        with self.get_conn() as conn:
            pages   = conn.execute("SELECT COUNT(*) FROM pages WHERE status_code=200").fetchone()[0]
            links   = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
            terms   = conn.execute("SELECT COUNT(DISTINCT term) FROM inverted_index").fetchone()[0]
            domains = conn.execute("SELECT COUNT(*) FROM domain_stats").fetchone()[0]
            entities = conn.execute("SELECT COUNT(DISTINCT entity) FROM named_entities").fetchone()[0]
            bigrams = conn.execute(
                "SELECT COUNT(*) FROM inverted_index WHERE term LIKE '% %'"
            ).fetchone()[0]
            return {
                "pages_indexed":    pages,
                "links_found":      links,
                "unique_terms":     terms,
                "unique_bigrams":   bigrams,
                "named_entities":   entities,
                "domains":          domains,
            }
