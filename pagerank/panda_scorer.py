"""
Google Panda (2011) targets low-quality, thin, or spammy content.
Our implementation scores pages on several measurable signals:

  1. Content length     — thin pages score lower
  2. Keyword stuffing   — suspiciously high keyword density is penalised
  3. Readability        — Flesch-Kincaid estimate (very readable = good)
  4. Content uniqueness — placeholder (SimHash requires all pages in memory)
  5. Title quality      — descriptive titles are rewarded

Final Panda score is in (0, 1]. All signals are combined as a weighted
harmonic-ish product so a zero in one axis heavily penalises the page.
"""

import re
import math
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from indexer.text_processor import (
    tokenize, process_text, estimated_reading_level, _STOP_WORDS
)

logger = logging.getLogger("neurasearch.panda")


# ── Tuneable weights ─────────────────────────────────────────────────────────
W_LENGTH      = 0.25
W_KEYWORD_DEN = 0.20
W_READABILITY = 0.20
W_TITLE       = 0.15
W_DIVERSITY   = 0.20


def _length_score(word_count: int) -> float:
    """
    Reward pages with more content, up to a soft ceiling of 1500 words.

    < 50 words   → ~0.05 (thin content)
      200 words  → ~0.65
      600+ words → ~0.95
    """
    if word_count <= 0:
        return 0.0
    # Logistic growth: saturates around 800 words
    score = 1.0 / (1.0 + math.exp(-0.008 * (word_count - 300)))
    return round(score, 4)


def _keyword_density_score(content: str) -> float:
    """
    Measures the density of *non-stop-word* tokens.
    Ideal: 2–5 %.  > 8 % signals keyword stuffing.
    """
    tokens = tokenize(content)
    if not tokens:
        return 0.5
    kw_tokens = [t for t in tokens if t not in _STOP_WORDS and t.isalpha()]
    density = len(kw_tokens) / len(tokens)

    # Penalty curve: peak at ~4 %, slopes down at extremes
    if density < 0.01:
        return 0.3   # almost no meaningful words
    if density <= 0.08:
        return 1.0   # healthy range
    # Keyword stuffing penalty
    penalty = max(0.0, 1.0 - (density - 0.08) * 5)
    return round(penalty, 4)


def _readability_score(reading_level: float) -> float:
    """
    Convert Flesch-Kincaid score (0–100) to 0–1.
    Very easy text (80+) or very difficult (<10) both lose a bit.
    Ideal: 40–70.
    """
    if reading_level <= 0:
        return 0.4
    # 60 is optimal; bell-curve around it
    z = (reading_level - 60.0) / 30.0
    score = math.exp(-0.5 * z * z)
    return round(score, 4)


def _title_score(title: str, content: str) -> float:
    """
    Reward descriptive titles that contain words also in the content.
    Penalise pages with no title or uninformative titles (<3 words).
    """
    if not title or title == "No Title":
        return 0.3
    title_words = set(process_text(title))
    if len(title_words) < 2:
        return 0.5
    content_words = set(process_text(content[:2000]))
    if not content_words:
        return 0.7
    overlap = len(title_words & content_words) / max(len(title_words), 1)
    return round(min(1.0, 0.5 + overlap), 4)


def _diversity_score(content: str) -> float:
    """
    Type-token ratio: unique clean words / total clean words.
    High ratio → rich, diverse vocabulary → good content quality.
    """
    tokens = process_text(content[:5000])
    if not tokens:
        return 0.5
    ttr = len(set(tokens)) / len(tokens)
    # Scale so TTR=0.5 → score=0.8 and TTR=1.0 → perfect
    return round(min(1.0, ttr * 1.2), 4)


def compute_panda_score(page: dict) -> float:
    """
    Compute Panda quality score for a page dict (fields from DB).

    Returns float in (0, 1].
    """
    content     = page.get("content") or ""
    title       = page.get("title") or "No Title"
    word_count  = page.get("word_count") or len(content.split())

    ls  = _length_score(word_count)
    kds = _keyword_density_score(content)
    rls = _readability_score(estimated_reading_level(content[:3000]))
    ts  = _title_score(title, content)
    ds  = _diversity_score(content)

    # Weighted average
    raw = (W_LENGTH * ls + W_KEYWORD_DEN * kds + W_READABILITY * rls
           + W_TITLE * ts + W_DIVERSITY * ds)

    # Hard floor at 0.05 so no page gets a zero
    return round(max(0.05, min(1.0, raw)), 4)


def score_all_pages(db) -> int:
    """
    Compute and store Panda scores for all pages in the database.
    Returns number of pages scored.
    """
    pages = db.get_all_pages()
    logger.info(f"[Panda] Scoring {len(pages)} pages …")
    for page in pages:
        score = compute_panda_score(page)
        db.update_page_scores(page["url"], panda_score=score)
    logger.info("[Panda] Done.")
    return len(pages)
