"""
Robust robots.txt handler with per-domain caching.
Respects Crawl-Delay and honours noindex meta tag.
"""

import time
import threading
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import USER_AGENT, REQUEST_TIMEOUT


class RobotsHandler:
    """
    Thread-safe robots.txt cache.

    Caches parsed RobotFileParser objects per domain.
    Automatically expires cache after TTL (default 6 hours).
    """

    DEFAULT_TTL = 6 * 3600   # 6 hours

    def __init__(self, ttl: int = DEFAULT_TTL):
        self._cache: dict[str, dict] = {}   # domain → {parser, crawl_delay, fetched_at}
        self._lock = threading.Lock()
        self.ttl = ttl

    def _fetch(self, domain: str, scheme: str) -> dict:
        """Fetch and parse robots.txt for *domain*. Returns cache entry."""
        robots_url = f"{scheme}://{domain}/robots.txt"
        parser = RobotFileParser(robots_url)
        crawl_delay = 1.0
        try:
            resp = requests.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            parser.parse(resp.text.splitlines())
            delay = parser.crawl_delay(USER_AGENT) or parser.crawl_delay("*")
            if delay:
                crawl_delay = float(delay)
        except Exception:
            # If we can't fetch robots.txt, be generous and allow crawling
            pass

        return {
            "parser": parser,
            "crawl_delay": crawl_delay,
            "fetched_at": time.time(),
        }

    def _get_or_fetch(self, url: str) -> dict | None:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        scheme = parsed.scheme.lower()

        with self._lock:
            entry = self._cache.get(domain)
            if entry is None or (time.time() - entry["fetched_at"]) > self.ttl:
                entry = self._fetch(domain, scheme)
                self._cache[domain] = entry
        return entry

    def can_fetch(self, url: str) -> bool:
        """Return True if NeuraSearchBot is allowed to fetch *url*."""
        try:
            entry = self._get_or_fetch(url)
            return entry["parser"].can_fetch(USER_AGENT, url)
        except Exception:
            return True   # default: allow

    def get_crawl_delay(self, url: str, minimum: float = 0.5) -> float:
        """Return the Crawl-Delay for *url*'s domain (clamped to >= minimum)."""
        try:
            entry = self._get_or_fetch(url)
            return max(minimum, entry["crawl_delay"])
        except Exception:
            return minimum

    def has_noindex(self, soup) -> bool:
        """
        Check for <meta name="robots" content="noindex"> in parsed page.
        Returns True if the page should NOT be indexed.
        """
        if soup is None:
            return False
        for tag in soup.find_all("meta", attrs={"name": re.compile(r"robots", re.I)}):
            content = tag.get("content", "").lower()
            if "noindex" in content:
                return True
        return False


# ── Lazy import to avoid circular issues ────────────────────────────────────
import re
