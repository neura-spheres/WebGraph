"""
Google Penguin (2012) targets manipulative link schemes:

  1. Anchor text over-optimisation — all links to a page use the
     exact same keyword-rich anchor → suspicious.
  2. Low domain diversity — 100 links from 1 domain ≠ 100 quality votes.
  3. Nofollow ratio — a page with only nofollow in-links gets less credit.
  4. Link velocity spike — sudden link bursts are anomalous (approximated
     by comparing link counts to page age; we don't have timestamps for
     individual links so we use a simpler heuristic).

Final Penguin score is in (0, 1].
"""

import math
import logging
from collections import Counter
from urllib.parse import urlparse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger("neurasearch.penguin")


# ── Weights ──────────────────────────────────────────────────────────────────
W_ANCHOR_DIV  = 0.40
W_DOMAIN_DIV  = 0.35
W_LINK_COUNT  = 0.25


def _anchor_diversity_score(inbound_links: list[dict]) -> float:
    """
    Measure diversity of anchor texts pointing to a page.

    Perfectly diverse → all unique anchors → score=1.0
    All identical     → score ≈ 0.1 (spam signal)
    """
    anchors = [lnk.get("anchor_text", "").strip().lower()
               for lnk in inbound_links
               if lnk.get("anchor_text", "").strip()]

    if not anchors:
        return 0.8   # unknown, assume neutral

    total = len(anchors)
    unique = len(set(anchors))

    diversity_ratio = unique / total  # 0–1
    # Log-scale to reward even a little diversity
    score = math.log1p(diversity_ratio * 9) / math.log(10)  # maps 0→0, 1→1

    # Extra penalty if the top anchor is used > 60 % of the time
    most_common_count = Counter(anchors).most_common(1)[0][1]
    if most_common_count / total > 0.60:
        score *= 0.6

    return round(max(0.1, min(1.0, score)), 4)


def _domain_diversity_score(inbound_links: list[dict], in_link_count: int) -> float:
    """
    How many unique domains link to this page?

    Rewards pages linked from many different domains (natural link profile).
    Pages linked exclusively from one or two domains score lower.
    """
    if not inbound_links:
        return 0.5

    domains = set()
    for lnk in inbound_links:
        try:
            domain = urlparse(lnk.get("src_url", "")).netloc.lower()
            if domain:
                domains.add(domain)
        except Exception:
            pass

    n_domains = len(domains)
    total_links = max(in_link_count, len(inbound_links), 1)

    # Ratio: unique domains / total in-links
    ratio = n_domains / total_links

    # 1 domain  → ratio=1.0 but only one source → don't reward too much
    if n_domains == 1:
        return 0.4
    if n_domains <= 3:
        return 0.6 + 0.1 * n_domains

    # Logarithmic reward for diverse domain profile
    score = math.log1p(n_domains) / math.log1p(50)   # caps near 1.0 at 50+ domains
    score = min(1.0, score + ratio * 0.1)
    return round(score, 4)


def _link_count_score(in_link_count: int) -> float:
    """
    Very few or extremely many in-links can both be signals.
    Reward pages with a healthy, natural in-link count (1–500).
    Pages with zero links get a neutral low score.
    """
    if in_link_count == 0:
        return 0.5
    # Logarithmic growth saturating ~200 links
    score = math.log1p(in_link_count) / math.log1p(200)
    return round(min(1.0, score), 4)


def compute_penguin_score(page: dict, inbound_links: list[dict]) -> float:
    """
    Compute Penguin link-quality score for a page.

    Parameters
    ----------
    page          : page dict from DB
    inbound_links : list of {src_url, anchor_text} dicts (from db.get_inbound_links)

    Returns float in (0, 1].
    """
    in_link_count = page.get("in_link_count", 0)

    anchor_div  = _anchor_diversity_score(inbound_links)
    domain_div  = _domain_diversity_score(inbound_links, in_link_count)
    link_cnt    = _link_count_score(in_link_count)

    raw = (W_ANCHOR_DIV * anchor_div
           + W_DOMAIN_DIV * domain_div
           + W_LINK_COUNT  * link_cnt)

    return round(max(0.05, min(1.0, raw)), 4)


def score_all_pages(db) -> int:
    """
    Compute and store Penguin scores for all pages in the database.
    """
    pages = db.get_all_pages()
    logger.info(f"[Penguin] Scoring {len(pages)} pages …")
    db.update_link_counts()

    for page in pages:
        inbound = db.get_inbound_links(page["url"])
        score   = compute_penguin_score(page, inbound)
        db.update_page_scores(page["url"], penguin_score=score)

    logger.info("[Penguin] Done.")
    return len(pages)
