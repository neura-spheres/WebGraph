"""
URL normalization, scoring, and filtering utilities.
"""
import re
import hashlib
from urllib.parse import (
    urlparse, urlunparse, urljoin, urlencode, parse_qsl, quote
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import BLOCKED_EXTENSIONS

_STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "campaign",
    "_ga", "mc_cid", "mc_eid", "yclid",
}

# ── TLD boosts ───────────────────────────────────────────────────────────────
_TLD_BOOST = {".edu": 1.4, ".gov": 1.3, ".org": 1.1, ".com": 1.0}


def normalize_url(url: str, base_url: str = "") -> str | None:
    """
    Return a canonical form of *url* or None if it should be skipped.

    Steps:
      1. Resolve relative URLs against base_url.
      2. Lowercase scheme + netloc.
      3. Remove default ports.
      4. Strip tracking query params.
      5. Sort remaining query params for stability.
      6. Remove fragment.
      7. Reject non-http(s) schemes and blocked extensions.
    """
    try:
        if base_url:
            url = urljoin(base_url, url)
        parsed = urlparse(url)

        # Only http / https
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            return None

        netloc = parsed.netloc.lower()

        # Strip default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Skip blocked file extensions
        path = parsed.path
        ext = path.rsplit(".", 1)[-1].lower() if "." in path.split("/")[-1] else ""
        if f".{ext}" in BLOCKED_EXTENSIONS:
            return None

        # Clean query string
        params = [
            (k, v) for k, v in parse_qsl(parsed.query)
            if k.lower() not in _STRIP_PARAMS
        ]
        query = urlencode(sorted(params))

        canonical = urlunparse((scheme, netloc, path, "", query, ""))
        return canonical

    except Exception:
        return None


def url_to_domain(url: str) -> str:
    """Extract bare domain (host) from URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def score_url(url: str, depth: int, in_link_count: int = 0) -> float:
    """
    Compute an initial priority score for a URL.

    Higher score → crawled sooner.

    Factors:
      - Depth: shallower pages are more important.
      - TLD: .edu/.gov domains get a boost.
      - In-links: more referrers → more important.
      - URL length: shorter URLs tend to be more canonical.
      - No query string: prefer clean URLs.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Depth penalty: score halves every extra level
        score = 1.0 / (1.0 + depth)

        # TLD boost
        for tld, boost in _TLD_BOOST.items():
            if domain.endswith(tld):
                score *= boost
                break

        # In-link boost (logarithmic)
        if in_link_count > 0:
            import math
            score *= (1.0 + math.log1p(in_link_count) * 0.3)

        # Prefer shorter, cleaner URLs
        url_length_penalty = max(0.5, 1.0 - len(url) / 500)
        score *= url_length_penalty

        # No query string bonus
        if not parsed.query:
            score *= 1.1

        return round(score, 6)

    except Exception:
        return 0.0


def is_valid_url(url: str) -> bool:
    """Quick sanity check before spending time on normalization."""
    if not url or len(url) > 2048:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def extract_links(soup, base_url: str) -> list[dict]:
    """
    Extract all <a href> links from a BeautifulSoup object.

    Returns list of {url, anchor_text, rel}.
    """
    links = []
    seen = set()
    for tag in soup.find_all("a", href=True):
        raw_href = tag["href"].strip()
        if not raw_href or raw_href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        norm = normalize_url(raw_href, base_url)
        if norm and norm not in seen:
            seen.add(norm)
            rel = tag.get("rel", [])
            if isinstance(rel, list):
                rel = " ".join(rel)
            links.append({
                "url": norm,
                "anchor_text": tag.get_text(strip=True)[:200],
                "rel": rel,
            })
    return links
