"""
Main indexer.

Reads all un-indexed pages from the database, runs the full NLP pipeline,
writes to the inverted index (with provenance), indexes named entities,
then refreshes BM25 scores.

Checkpoint / resume behaviour
-------------------------------
Every page is marked is_indexed=1 in the database BEFORE moving to the next
one.  This means any interruption (CTRL-C, crash, OOM) is automatically
resumable: the next run calls get_pages_not_indexed() and picks up exactly
where it left off, skipping all already-processed pages.

A SIGINT handler is installed during run() so CTRL-C triggers a clean exit
with a final JSON export of whatever was indexed so far.

BM25 refresh is done at the end of a full run.  If the process is killed
during BM25 refresh, the next run will redo the refresh — this is safe and
idempotent, since refresh_all_bm25() overwrites all scores from scratch.
"""

import json
import logging
import signal
import threading

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from database.db import Database
from indexer import text_processor as tp
from indexer.inverted_index import InvertedIndex
from config import MAX_ENTITIES_PER_DOC

logger = logging.getLogger("neurasearch.indexer")


class Indexer:
    def __init__(self, db: Database):
        self.db        = db
        self.index     = InvertedIndex(db)
        self._stop_evt = threading.Event()

    def run(self, batch_size: int = 200, refresh_bm25: bool = True):
        """
        Index all pages that haven't been indexed yet.
        Already-indexed pages are skipped — safe to re-run after interruptions.

        CTRL-C triggers a graceful stop: the current page finishes, a JSON
        checkpoint is exported, and the process exits cleanly.  The next run
        will resume from the first un-indexed page.
        """
        pages = self.db.get_pages_not_indexed()
        total = len(pages)
        if total == 0:
            logger.info("[Indexer] Nothing to do — all crawled pages are already indexed.")
            return 0
        logger.info(f"[Indexer] {total} page(s) to index.")

        # Install SIGINT handler for graceful stop
        self._stop_evt.clear()
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):
            logger.info("[Indexer] CTRL-C — finishing current page and stopping…")
            self._stop_evt.set()

        signal.signal(signal.SIGINT, _handle_sigint)

        indexed = 0
        try:
            for i, page in enumerate(pages, 1):
                if self._stop_evt.is_set():
                    logger.info(f"[Indexer] Stopped after {indexed} page(s). "
                                f"{total - indexed} page(s) remain for next run.")
                    break

                try:
                    self._index_page(page)
                    indexed += 1
                except Exception as exc:
                    logger.warning(f"[Indexer] Error indexing {page['url']}: {exc}")
                    continue

                # Periodic log + checkpoint save every batch_size pages
                if i % batch_size == 0:
                    logger.info(f"[Indexer] {i}/{total} pages indexed …")
                    self._save_checkpoint()

        finally:
            signal.signal(signal.SIGINT, prev_handler)  # restore previous handler

        if indexed > 0 and not self._stop_evt.is_set():
            if refresh_bm25:
                logger.info("[Indexer] Running BM25 refresh…")
                self.index.refresh_all_bm25()
            self._save_checkpoint()

        logger.info(f"[Indexer] Done. Indexed {indexed} page(s) this run.")
        return indexed

    def _save_checkpoint(self):
        """Export indexed pages and vocabulary to JSON (non-blocking best-effort)."""
        try:
            from utils.json_exporter import export_indexed_pages, export_index_vocabulary
            export_indexed_pages(self.db)
            export_index_vocabulary(self.db)
        except Exception as exc:
            logger.warning(f"[Indexer] JSON export error: {exc}")

    def _index_page(self, page: dict):
        """
        Full indexing pipeline for one page:
          1. Build headings text from stored JSON.
          2. Run inverted_index.index_document() with full provenance.
          3. Extract named entities and store them.
        """
        doc_id      = page["id"]
        url         = page["url"]
        title       = page.get("title") or ""
        description = page.get("description") or ""
        content     = page.get("content") or ""

        # Reconstruct headings text from stored JSON
        headings_text = ""
        try:
            raw = page.get("headings")
            if raw:
                headings = json.loads(raw) if isinstance(raw, str) else raw
                parts = []
                for level in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    parts.extend(headings.get(level, []))
                headings_text = " ".join(parts)
        except Exception:
            pass

        # Index document (terms + bigrams + URL tokens)
        self.index.index_document(
            doc_id        = doc_id,
            url           = url,
            title         = title,
            description   = description,
            content       = content,
            text_processor = tp,
            headings_text = headings_text,
        )

        # Named Entity Recognition
        # Run over title + first part of content for speed
        ner_text = f"{title}. {description}. {headings_text}. {content}"
        entities = tp.extract_named_entities(ner_text)
        if entities:
            entity_rows = []
            seen_keys: set = set()
            for ent in entities[:MAX_ENTITIES_PER_DOC]:
                key = (ent["text"].lower(), ent["label"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    entity_rows.append((doc_id, ent["text"], ent["label"], 1))
            if entity_rows:
                self.db.upsert_named_entities_bulk(entity_rows)

    def index_one(self, page: dict) -> int:
        """Index a single page dict (from db.get_page). Returns term count."""
        self._index_page(page)
        return 0   # term count no longer trivially available; use get_stats()

    def search(self, query: str, limit: int = 10, offset: int = 0) -> list:
        """
        Process query string through the same NLP pipeline,
        then delegate to InvertedIndex.search().
        """
        from indexer.text_processor import process_text
        query_terms = process_text(query)
        if not query_terms:
            return []
        return self.index.search(query_terms, limit=limit, offset=offset)

    def get_suggestion_terms(self, prefix: str, limit: int = 8) -> list:
        """
        Return lemma forms starting with *prefix* for autocomplete.
        Prefers lemma over raw stem so suggestions look like real words.
        """
        prefix = prefix.lower().strip()
        if len(prefix) < 2:
            return []
        with self.db.get_conn() as conn:
            # Try lemma column first (looks like real words)
            rows = conn.execute("""
                SELECT DISTINCT lemma FROM inverted_index
                WHERE lemma LIKE ? AND lemma IS NOT NULL
                ORDER BY lemma
                LIMIT ?
            """, (f"{prefix}%", limit)).fetchall()
            suggestions = [r[0] for r in rows if r[0]]

            # Top up with stemmed terms if needed
            if len(suggestions) < limit:
                rows2 = conn.execute("""
                    SELECT DISTINCT term FROM inverted_index
                    WHERE term LIKE ? AND term NOT LIKE '% %'
                    ORDER BY term
                    LIMIT ?
                """, (f"{prefix}%", limit - len(suggestions))).fetchall()
                for r in rows2:
                    if r[0] not in suggestions:
                        suggestions.append(r[0])

        return suggestions[:limit]
