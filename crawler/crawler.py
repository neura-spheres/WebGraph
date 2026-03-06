"""
Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  Frontier (priority queue)                           │
  │  ↑ URLs scored by depth + TLD + in-link count        │
  └────────────────────┬─────────────────────────────────┘
                       │ pop()
              ┌────────▼────────┐
              │   Thread Pool   │  (MAX_WORKERS threads)
              │  _crawl_worker  │
              └────────┬────────┘
          ┌────────────┼───────────────┐
          ▼            ▼               ▼
     Robots.txt   HTTP fetch       Rate limit
     check        + retry          per domain
                       │
              ┌────────▼────────┐
              │  HTML parser    │  BeautifulSoup
              │  link extractor │
              │  heading/entity │  h1-h6, meta_keywords, canonical
              └────────┬────────┘
              ┌────────▼────────┐
              │   Database      │  pages + links + queue + entities
              └─────────────────┘

Smart features:
  • Priority queue: shallower + high-TLD + in-link boosted pages first.
  • Per-domain rate limiting using Crawl-Delay from robots.txt.
  • Duplicate detection via content MD5 hash.
  • noindex / nofollow respect.
  • Automatic retry with exponential back-off.
  • Graceful shutdown on CTRL-C.
  • Rich page metadata: h1-h6 headings, meta keywords, canonical URL.
  • NLP-processed anchor text for better anchor index quality.
"""

import hashlib
import json
import logging
import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
import sys

import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MAX_WORKERS, CRAWL_LIMIT, REQUEST_TIMEOUT,
    MIN_DELAY, MAX_DELAY, MAX_RETRIES, MAX_DEPTH,
    MAX_CONTENT_SIZE, USER_AGENT, CRAWL_LANGUAGES,
)
from database.db import Database
from crawler.url_utils import normalize_url, url_to_domain, score_url, extract_links
from crawler.robots_handler import RobotsHandler
from crawler.frontier import Frontier

logger = logging.getLogger("neurasearch.crawler")


def _detect_language(soup, content_text: str) -> str:
    """
    Detect page language.  Returns a 2-letter ISO 639-1 code (e.g. "en", "id")
    or an empty string if detection fails.

    Detection priority:
      1. <html lang="..."> attribute
      2. <meta http-equiv="Content-Language"> tag
      3. langdetect library (if installed) on first 2000 chars of content
    """
    # 1. HTML lang attribute — most reliable
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        raw = html_tag["lang"].strip().lower()
        code = raw.split("-")[0]   # "en-US" → "en"
        if len(code) == 2:
            return code

    # 2. Meta content-language
    meta_lang = soup.find("meta", attrs={"http-equiv": lambda v: v and v.lower() == "content-language"})
    if meta_lang and meta_lang.get("content"):
        raw = meta_lang["content"].strip().lower()
        code = raw.split("-")[0]
        if len(code) == 2:
            return code

    # 3. langdetect fallback
    try:
        from langdetect import detect
        return detect(content_text[:2000]) or ""
    except Exception:
        pass

    return ""


def _language_allowed(lang_code: str) -> bool:
    """
    Return True if the page's language is allowed by the CRAWL_LANGUAGES filter.
    Always returns True when CRAWL_LANGUAGES is empty (no filter).
    """
    if not CRAWL_LANGUAGES:
        return True
    if not lang_code:
        # Unknown language — if filter is set, skip by default
        return False
    return lang_code.lower() in {c.lower() for c in CRAWL_LANGUAGES}


def _process_anchor(anchor_text: str) -> list:
    """
    Run anchor text through the full NLP pipeline (lemmatize + stem).
    Falls back to simple split if text_processor is unavailable.
    Returns list of stemmed terms.
    """
    try:
        from indexer.text_processor import process_text
        return process_text(anchor_text)
    except Exception:
        return [w.lower() for w in anchor_text.split() if len(w) >= 2]


class Crawler:
    """
    Multi-threaded web crawler with smart prioritization.

    Usage:
        db = Database()
        crawler = Crawler(db, seed_urls=["https://en.wikipedia.org/wiki/Python"])
        crawler.run()
    """

    def __init__(
        self,
        db: Database,
        seed_urls: list,
        crawl_limit: int = CRAWL_LIMIT,
        max_workers: int = MAX_WORKERS,
        max_depth: int = MAX_DEPTH,
        on_page_crawled: Callable = None,
    ):
        self.db            = db
        self.crawl_limit   = crawl_limit
        self.max_workers   = max_workers
        self.max_depth     = max_depth
        self.on_page_crawled = on_page_crawled

        self.robots   = RobotsHandler()
        self.frontier = Frontier(db=db)

        self._crawl_count    = [0]
        self._filtered_count = [0]
        self._lock           = threading.Lock()
        self._stop_event     = threading.Event()
        self._last_json_save = 0

        # Resume interrupted session
        stuck = db.reset_stuck_queue()
        if stuck > 0:
            logger.info(f"[Crawler] Resuming: reset {stuck} interrupted URL(s) to pending.")
        loaded = self.frontier.load_from_db(db)
        if loaded > 0:
            logger.info(f"[Crawler] Resuming: loaded {loaded} pending URL(s).")

        # Add seed URLs
        new_seeds = 0
        for url in seed_urls:
            norm = normalize_url(url)
            if norm:
                if self.frontier.push(norm, priority=1.0, depth=0):
                    new_seeds += 1
        if new_seeds:
            logger.info(f"[Crawler] Added {new_seeds} new seed URL(s).")

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":      USER_AGENT,
            "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        })

    # ── Public interface ──────────────────────────────────────────────────────
    def run(self):
        """Start crawling. Blocks until crawl_limit or frontier is exhausted."""
        logger.info(
            f"[Crawler] Starting — limit={self.crawl_limit}, "
            f"workers={self.max_workers}, max_depth={self.max_depth}"
        )
        signal.signal(signal.SIGINT, self._handle_sigint)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            while not self._stop_event.is_set():
                with self._lock:
                    if self._crawl_count[0] >= self.crawl_limit:
                        logger.info("[Crawler] Crawl limit reached.")
                        break

                while len(futures) < self.max_workers * 2:
                    item = self.frontier.pop()
                    if item is None:
                        break
                    url, depth = item
                    if depth > self.max_depth:
                        continue
                    future = pool.submit(self._crawl_worker, url, depth)
                    futures[future] = url

                if not futures:
                    if self.frontier.empty():
                        logger.info("[Crawler] Frontier exhausted.")
                        break
                    time.sleep(0.5)
                    continue

                done = [f for f in futures if f.done()]
                for f in done:
                    futures.pop(f)
                    try:
                        f.result()
                    except Exception as exc:
                        logger.debug(f"Worker error: {exc}")

                time.sleep(0.05)

        logger.info(
            f"[Crawler] Done. Pages stored: {self._crawl_count[0]}  "
            f"Lang-filtered: {self._filtered_count[0]}  "
            f"URLs seen: {self.frontier.seen_count()}"
        )
        self._save_json_checkpoint()

    # ── Worker ────────────────────────────────────────────────────────────────
    def _crawl_worker(self, url: str, depth: int):
        domain = url_to_domain(url)

        # Per-domain rate limiting
        delay   = self.robots.get_crawl_delay(url, minimum=MIN_DELAY)
        delay   = min(delay, MAX_DELAY)
        elapsed = self.frontier.seconds_since_last(domain)
        if elapsed < delay:
            time.sleep(delay - elapsed + random.uniform(0, 0.3))
        self.frontier.mark_crawled(domain)

        # robots.txt
        if not self.robots.can_fetch(url):
            logger.debug(f"[robots] Blocked: {url}")
            self.db.mark_queue_done(url)
            return

        response = self._fetch_with_retry(url)
        if response is None:
            self.db.mark_queue_failed(url)
            return

        parsed = self._parse_html(response, url)
        soup            = parsed["soup"]
        content_text    = parsed["content_text"]
        title           = parsed["title"]
        description     = parsed["description"]
        headings        = parsed["headings"]
        meta_keywords   = parsed["meta_keywords"]
        canonical_url   = parsed["canonical_url"]

        # noindex check
        if self.robots.has_noindex(soup):
            logger.debug(f"[noindex] Skipping: {url}")
            self.db.mark_queue_done(url)
            return

        # Language filter — detect language then decide whether to store the page.
        # Even if the page is filtered out, we still follow its outbound links
        # (they might point to pages in the desired language).
        detected_lang = _detect_language(soup, content_text)
        if not _language_allowed(detected_lang):
            logger.debug(f"[lang-filter] Skipping {url} (lang={detected_lang!r})")
            # Still process links (fall through); don't store the page content.
            store_page = False
        else:
            store_page = True

        content_hash = hashlib.md5(content_text.encode()).hexdigest()
        words        = content_text.split()

        if store_page:
            self.db.upsert_page(
                url,
                title         = title,
                description   = description,
                content       = content_text[:50_000],
                content_hash  = content_hash,
                word_count    = len(words),
                status_code   = response.status_code,
                depth         = depth,
                headings      = headings,
                meta_keywords = meta_keywords,
                canonical_url = canonical_url,
            )

        # Extract and score outbound links (always, regardless of language filter)
        raw_links        = extract_links(soup, url)
        link_rows        = []
        frontier_items   = []
        anchor_term_rows = []

        for link in raw_links:
            dst    = link["url"]
            anchor = link["anchor_text"]
            rel    = link.get("rel", "")

            if "nofollow" in rel:
                continue

            if store_page:
                link_rows.append((url, dst, anchor, rel))

            if not self.db.url_exists(dst) and not self.db.queue_url_exists(dst):
                priority = score_url(dst, depth + 1)
                frontier_items.append((dst, priority, depth + 1))

            # NLP-processed anchor text
            if store_page and anchor.strip():
                stems = _process_anchor(anchor)
                for stem in stems:
                    anchor_term_rows.append((stem, dst))

        if link_rows:
            self.db.add_links_bulk(link_rows)
        if frontier_items:
            self.frontier.push_many(frontier_items)
        if anchor_term_rows:
            self.db.add_anchor_terms_bulk(anchor_term_rows)

        self.db.mark_queue_done(url)

        with self._lock:
            if store_page:
                self._crawl_count[0] += 1
            else:
                self._filtered_count[0] += 1
            count    = self._crawl_count[0]
            filtered = self._filtered_count[0]

        if store_page and (count <= 5 or count % 100 == 0):
            logger.info(
                f"[Crawler] {count} pages stored | "
                f"{filtered} filtered | frontier={self.frontier.size()} urls"
            )
        elif not store_page and filtered % 200 == 0:
            logger.info(
                f"[Crawler] {count} pages stored | "
                f"{filtered} lang-filtered | frontier={self.frontier.size()} urls"
            )

        if count % 500 == 0 and count != self._last_json_save:
            self._last_json_save = count
            threading.Thread(
                target=self._save_json_checkpoint, daemon=True,
                name=f"json-checkpoint-{count}"
            ).start()

        if self.on_page_crawled:
            try:
                self.on_page_crawled(url, title, depth)
            except Exception:
                pass

    # ── HTTP fetch ────────────────────────────────────────────────────────────
    def _fetch_with_retry(self, url: str):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    stream=True,
                    allow_redirects=True,
                )
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return None

                raw = b""
                for chunk in resp.iter_content(chunk_size=8192):
                    raw += chunk
                    if len(raw) > MAX_CONTENT_SIZE:
                        break
                resp._content = raw
                return resp

            except requests.exceptions.TooManyRedirects:
                return None
            except requests.exceptions.RequestException as exc:
                if attempt == MAX_RETRIES:
                    logger.debug(f"[fetch] Failed {url} after {MAX_RETRIES} tries: {exc}")
                    return None
                time.sleep(2 ** attempt + random.uniform(0, 1))
        return None

    # ── HTML parsing ──────────────────────────────────────────────────────────
    @staticmethod
    def _parse_html(response, url: str = "") -> dict:
        """
        Parse an HTML response and extract:
          - soup             : BeautifulSoup object (for further processing)
          - content_text     : clean plain-text body
          - title            : page <title>
          - description      : meta description (or first paragraph)
          - headings         : dict {"h1": [...], "h2": [...], ...}
          - meta_keywords    : <meta name="keywords"> content string
          - canonical_url    : <link rel="canonical"> href (or None)
        """
        try:
            soup = BeautifulSoup(response.content, "lxml")
        except Exception:
            soup = BeautifulSoup(response.content, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "noscript", "iframe"]):
            tag.decompose()

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True)[:300] if title_tag else "No Title"

        # Meta description
        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"][:500]
        else:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    description = text[:500]
                    break

        # Meta keywords
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        meta_keywords = None
        if meta_kw and meta_kw.get("content"):
            meta_keywords = meta_kw["content"][:300]

        # Canonical URL
        canonical_url = None
        link_canonical = soup.find("link", attrs={"rel": "canonical"})
        if link_canonical and link_canonical.get("href"):
            canonical_url = link_canonical["href"][:500]

        # Headings h1–h6
        from indexer.text_processor import extract_headings
        headings = extract_headings(soup)

        # Full plain-text content
        content_text = soup.get_text(separator=" ", strip=True)

        return {
            "soup":          soup,
            "content_text":  content_text,
            "title":         title,
            "description":   description,
            "headings":      headings,
            "meta_keywords": meta_keywords,
            "canonical_url": canonical_url,
        }

    # ── JSON checkpoint ───────────────────────────────────────────────────────
    def _save_json_checkpoint(self):
        try:
            from utils.json_exporter import export_crawled_pages, export_link_graph
            logger.info("[Crawler] Saving JSON checkpoint…")
            export_crawled_pages(self.db)
            export_link_graph(self.db)
        except Exception as exc:
            logger.warning(f"[Crawler] JSON checkpoint error: {exc}")

    # ── Signals ───────────────────────────────────────────────────────────────
    def _handle_sigint(self, signum, frame):
        logger.info("[Crawler] CTRL-C received — saving checkpoint and stopping gracefully…")
        self._stop_event.set()
        self._save_json_checkpoint()
