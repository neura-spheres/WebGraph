"""
Priority-based URL frontier.

Each URL is stored with a float score (higher = crawl sooner).
Thread-safe. Backed by an in-memory heap with a seen-set and
periodically synced to the database for crash recovery.
"""

import heapq
import threading
import time
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from crawler.url_utils import url_hash


@dataclass(order=True)
class _Entry:
    """Heap entry. Negated priority for max-heap via heapq (min-heap)."""
    neg_priority: float
    url: str = field(compare=False)
    depth: int = field(compare=False)


class Frontier:
    """
    In-memory priority queue for URLs to crawl.

    Design goals:
      - O(log n) push / O(log n) pop.
      - Duplicate detection in O(1) via seen-hash set.
      - Domain-level per-domain last-crawl tracking so callers can
        implement politeness without a separate data structure.
    """

    def __init__(self, db=None):
        self._heap: list[_Entry] = []
        self._seen: set[str] = set()          # hashes of seen URLs
        self._lock = threading.Lock()
        self._domain_last: dict[str, float] = {}   # domain → last crawl time
        self._db = db

    # ── Public API ──────────────────────────────────────────────────────────
    def push(self, url: str, priority: float, depth: int = 0) -> bool:
        """
        Add *url* to the frontier.
        Returns True if added, False if already seen.
        """
        h = url_hash(url)
        with self._lock:
            if h in self._seen:
                return False
            self._seen.add(h)
            heapq.heappush(self._heap, _Entry(-priority, url, depth))
            if self._db:
                try:
                    self._db.enqueue(url, priority=priority, depth=depth)
                except Exception:
                    pass
            return True

    def push_many(self, items: list[tuple[str, float, int]]) -> int:
        """
        Bulk push. items = [(url, priority, depth), ...]
        Returns number of newly added URLs.
        """
        added = 0
        db_rows = []
        with self._lock:
            for url, priority, depth in items:
                h = url_hash(url)
                if h not in self._seen:
                    self._seen.add(h)
                    heapq.heappush(self._heap, _Entry(-priority, url, depth))
                    db_rows.append((url, priority, depth))
                    added += 1
        if self._db and db_rows:
            try:
                self._db.enqueue_bulk(db_rows)
            except Exception:
                pass
        return added

    def pop(self) -> tuple[str, int] | None:
        """
        Return (url, depth) of highest-priority URL, or None if empty.
        """
        with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)
                return entry.url, entry.depth
            return None

    def empty(self) -> bool:
        with self._lock:
            return len(self._heap) == 0

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def mark_crawled(self, domain: str):
        """Record that *domain* was just crawled (for rate-limiting checks)."""
        with self._lock:
            self._domain_last[domain] = time.time()

    def seconds_since_last(self, domain: str) -> float:
        """Seconds elapsed since we last crawled anything on *domain*."""
        with self._lock:
            last = self._domain_last.get(domain, 0.0)
            return time.time() - last

    def seen_count(self) -> int:
        with self._lock:
            return len(self._seen)

    def load_from_db(self, db) -> int:
        """
        Hydrate the frontier from pending URLs already stored in the DB.

        Unlike dequeue_batch(), this does NOT mark URLs as 'processing' —
        URLs stay 'pending' and will be updated normally when popped by a worker.
        Returns the number of URLs loaded.
        """
        rows = db.get_pending_urls(limit=100_000)
        added = 0
        with self._lock:
            for row in rows:
                h = url_hash(row["url"])
                if h not in self._seen:
                    self._seen.add(h)
                    heapq.heappush(
                        self._heap,
                        _Entry(-row["priority"], row["url"], row["depth"])
                    )
                    added += 1
        return added
