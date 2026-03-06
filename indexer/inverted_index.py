"""
BM25 inverted index builder and query engine.

Index model (per term × document pair)
---------------------------------------
term           : Porter-stemmed form  (search key)
lemma          : WordNet-lemmatized intermediate form
original_forms : all distinct raw word forms seen in this document
positions      : word positions in the cleaned token stream (phrase proximity)
in_title       : term appears in the page <title>
in_description : term appears in the meta description
in_url         : term appears in the URL path segments
in_anchor      : term appears in inbound anchor texts
pos_tag        : dominant NLTK POS tag for this term in this document

Bigrams are stored in the same table with term = "w1 w2" (stemmed forms).

Query-time scoring
------------------
final_score(q, D) =
    Σ_t [ BM25(t, D)
          × field_multiplier(t, D)   (title/desc/anchor boosts)
          + anchor_score(t, D) ]
    × pagerank_factor(D)
    × panda_score(D)
    × penguin_score(D)
    × coverage_bonus(q, D)
"""

import json
import math
import re
import logging
from urllib.parse import urlparse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BM25_K1, BM25_B,
    TITLE_BOOST, DESCRIPTION_BOOST, HEADING_BOOST, ANCHOR_BOOST, URL_BOOST,
    MAX_BIGRAMS_PER_DOC,
)

logger = logging.getLogger("neurasearch.index")


# ── BM25 ──────────────────────────────────────────────────────────────────────
def _bm25(tf: int, df: int, N: int, dl: int, avgdl: float,
          k1: float = BM25_K1, b: float = BM25_B) -> float:
    """Robertson-Walker BM25."""
    if N == 0 or df == 0:
        return 0.0
    idf     = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / max(avgdl, 1)))
    return idf * tf_norm


# ── URL token extraction ───────────────────────────────────────────────────────
def _url_terms(url: str, text_processor) -> set:
    """
    Extract meaningful tokens from the URL path.
    e.g. /wiki/Python_programming_language -> {"python", "program", "languag"}
    """
    try:
        path = urlparse(url).path
        # Split on non-alpha characters (/, _, -, .)
        raw = re.split(r"[^a-zA-Z]+", path)
        tokens = []
        for tok in raw:
            tok = tok.lower()
            if 2 <= len(tok) <= 50:
                tokens.append(tok)
        return set(text_processor.process_text(" ".join(tokens)))
    except Exception:
        return set()


class InvertedIndex:
    """
    Manages building and querying the inverted index stored in the database.
    """

    def __init__(self, db):
        self.db = db

    # ── Build ─────────────────────────────────────────────────────────────────
    def index_document(self, doc_id: int, url: str,
                       title: str, description: str, content: str,
                       text_processor,
                       headings_text: str = ""):
        """
        Index a single document with full provenance.

        Parameters
        ----------
        doc_id         : database page id
        url            : canonical page URL
        title          : page <title> text
        description    : meta description text
        content        : full plain-text body content
        text_processor : the text_processor module
        headings_text  : h1-h6 text concatenated (optional)

        What gets stored per (term, doc) pair:
          - stemmed term (search key)
          - lemmatized form (intermediate)
          - original raw word forms seen in this document
          - word positions in cleaned stream
          - field flags: in_title, in_description, in_url, in_anchor
          - dominant POS tag
          - adjusted frequency (includes field boost synthetic counts)
        """
        # ── Field provenance ─────────────────────────────────────────────────
        title_prov   = text_processor.process_with_provenance(title)
        desc_prov    = text_processor.process_with_provenance(description)
        heading_prov = text_processor.process_with_provenance(headings_text) if headings_text else {}
        body_prov    = text_processor.process_with_provenance(content)
        url_terms    = _url_terms(url, text_processor)

        title_stems   = set(title_prov.keys())
        desc_stems    = set(desc_prov.keys())
        heading_stems = set(heading_prov.keys())

        # ── Merge all terms ───────────────────────────────────────────────────
        # Start with body (has positions); overlay field flags
        all_terms: dict = {}

        for stem, data in body_prov.items():
            all_terms[stem] = {
                "lemma":     data["lemma"],
                "originals": list(data["originals"]),
                "positions": data["positions"],
                "pos":       data["pos"],
                "frequency": len(data["positions"]),
                "in_title":  False,
                "in_desc":   False,
                "in_head":   False,
                "in_url":    stem in url_terms,
                "in_anchor": False,
            }

        # Title terms not in body: add with empty positions
        for stem, data in title_prov.items():
            if stem not in all_terms:
                all_terms[stem] = {
                    "lemma":     data["lemma"],
                    "originals": list(data["originals"]),
                    "positions": [],
                    "pos":       data["pos"],
                    "frequency": 0,
                    "in_title":  True,
                    "in_desc":   False,
                    "in_head":   False,
                    "in_url":    stem in url_terms,
                    "in_anchor": False,
                }
            else:
                all_terms[stem]["in_title"] = True
                # Merge originals
                for orig in data["originals"]:
                    if orig not in all_terms[stem]["originals"]:
                        all_terms[stem]["originals"].append(orig)

        # Description terms
        for stem, data in desc_prov.items():
            if stem not in all_terms:
                all_terms[stem] = {
                    "lemma":     data["lemma"],
                    "originals": list(data["originals"]),
                    "positions": [],
                    "pos":       data["pos"],
                    "frequency": 0,
                    "in_title":  False,
                    "in_desc":   True,
                    "in_head":   False,
                    "in_url":    stem in url_terms,
                    "in_anchor": False,
                }
            else:
                all_terms[stem]["in_desc"] = True
                for orig in data["originals"]:
                    if orig not in all_terms[stem]["originals"]:
                        all_terms[stem]["originals"].append(orig)

        # Heading terms
        for stem, data in heading_prov.items():
            if stem not in all_terms:
                all_terms[stem] = {
                    "lemma":     data["lemma"],
                    "originals": list(data["originals"]),
                    "positions": [],
                    "pos":       data["pos"],
                    "frequency": 0,
                    "in_title":  False,
                    "in_desc":   False,
                    "in_head":   True,
                    "in_url":    stem in url_terms,
                    "in_anchor": False,
                }
            else:
                all_terms[stem]["in_head"] = True
                for orig in data["originals"]:
                    if orig not in all_terms[stem]["originals"]:
                        all_terms[stem]["originals"].append(orig)

        # URL-only terms (not found in any text field)
        for stem in url_terms:
            if stem not in all_terms:
                all_terms[stem] = {
                    "lemma":     stem,
                    "originals": [],
                    "positions": [],
                    "pos":       "NN",
                    "frequency": 0,
                    "in_title":  False,
                    "in_desc":   False,
                    "in_head":   False,
                    "in_url":    True,
                    "in_anchor": False,
                }

        # Apply field-boost synthetic frequency counts so BM25 reflects importance
        for stem, entry in all_terms.items():
            if entry["in_title"]:
                entry["frequency"] += int(TITLE_BOOST)
            if entry["in_desc"]:
                entry["frequency"] += int(DESCRIPTION_BOOST)
            if entry["in_head"]:
                entry["frequency"] += int(HEADING_BOOST)
            if entry["in_url"]:
                entry["frequency"] += int(URL_BOOST)

        # ── Bigrams ──────────────────────────────────────────────────────────
        body_stems = list(body_prov.keys())
        bigrams    = text_processor.extract_bigrams(body_stems, min_freq=1)
        bigrams    = bigrams[:MAX_BIGRAMS_PER_DOC]

        for bigram in bigrams:
            if bigram not in all_terms:
                all_terms[bigram] = {
                    "lemma":     bigram,
                    "originals": bigram.split(" "),  # stemmed forms
                    "positions": [],
                    "pos":       "BG",    # Bigram marker
                    "frequency": 1,
                    "in_title":  False,
                    "in_desc":   False,
                    "in_head":   False,
                    "in_url":    False,
                    "in_anchor": False,
                }
            else:
                all_terms[bigram]["frequency"] += 1

        # ── Write to DB ──────────────────────────────────────────────────────
        rows = []
        for term, entry in all_terms.items():
            rows.append((
                term,
                doc_id,
                entry["frequency"],
                json.dumps(entry["positions"][:200]),   # cap stored positions
                0.0,                                    # tf_idf (computed later)
                0.0,                                    # bm25   (computed later)
                int(entry["in_title"]),
                int(entry["in_desc"]),
                entry["lemma"],
                json.dumps(entry["originals"][:20]),    # cap originals list
                int(entry["in_url"]),
                int(entry["in_anchor"]),
                entry["pos"],
            ))

        self.db.upsert_terms_bulk(rows)
        self.db.mark_page_indexed(url)
        return len(rows)

    # ── Score refresh ─────────────────────────────────────────────────────────
    def refresh_bm25_for_doc(self, doc_id: int, doc_length: int):
        """Recompute BM25 for all terms in a single document."""
        N       = self.db.get_page_count()
        avgdl   = self.db.get_avg_word_count()
        postings = self.db.get_postings_for_doc(doc_id)

        rows = []
        for row in postings:
            term = row["term"]
            tf   = row["frequency"]
            df   = self.db.get_doc_freq(term)
            score = _bm25(tf, df, N, doc_length, avgdl)
            rows.append((score, term, doc_id))

        if rows:
            with self.db.get_conn() as conn:
                conn.executemany(
                    "UPDATE inverted_index SET bm25=? WHERE term=? AND doc_id=?",
                    rows
                )

    def refresh_all_bm25(self):
        """
        Recompute BM25 scores for every (term, doc) pair.
        Run after a bulk indexing session or after PageRank update.
        """
        logger.info("[Index] Refreshing all BM25 scores …")
        N     = self.db.get_page_count()
        avgdl = self.db.get_avg_word_count()

        if N == 0:
            return

        all_terms = self.db.get_all_terms()
        for term in all_terms:
            postings = self.db.get_postings(term)
            df = len(postings)
            rows = []
            for p in postings:
                score = _bm25(
                    p["frequency"], df, N, p["word_count"], avgdl
                )
                rows.append((score, term, p["doc_id"]))
            with self.db.get_conn() as conn:
                conn.executemany(
                    "UPDATE inverted_index SET bm25=? WHERE term=? AND doc_id=?",
                    rows
                )

        logger.info("[Index] BM25 refresh complete.")

    # ── Search ────────────────────────────────────────────────────────────────
    def search(self, query_terms: list, limit: int = 10,
               offset: int = 0) -> list:
        """
        Multi-term BM25 search with PageRank, Panda, Penguin re-ranking.
        Includes bigram bonus if consecutive query terms appear as a bigram.

        Returns list of result dicts ordered by combined score.
        """
        if not query_terms:
            return []

        scores: dict = {}   # doc_id → accum info

        # Also try bigrams from consecutive query terms
        bigram_terms = [
            f"{query_terms[i]} {query_terms[i + 1]}"
            for i in range(len(query_terms) - 1)
        ]
        all_query_terms = query_terms + bigram_terms

        for term in all_query_terms:
            is_bigram = " " in term
            postings  = self.db.get_postings(term)
            anchor_postings = {
                ap["dst_url"]: ap["frequency"]
                for ap in self.db.get_anchor_postings(term)
            }

            for p in postings:
                doc_id = p["doc_id"]
                if doc_id not in scores:
                    scores[doc_id] = {
                        "doc_id":        doc_id,
                        "url":           p["url"],
                        "title":         p["title"],
                        "description":   p["description"],
                        "content":       p.get("content", "") or "",
                        "pagerank":      p["pagerank_score"],
                        "panda":         p["panda_score"],
                        "penguin":       p["penguin_score"],
                        "text_score":    0.0,
                        "anchor_score":  0.0,
                        "term_coverage": 0,
                        "matched_terms": [],
                    }

                entry    = scores[doc_id]
                bm25     = p["bm25"]
                in_title = bool(p["in_title"])
                in_desc  = bool(p["in_description"])
                in_url   = bool(p.get("in_url", 0))

                multiplier = 1.0
                if in_title:
                    multiplier += TITLE_BOOST
                if in_desc:
                    multiplier += DESCRIPTION_BOOST
                if in_url:
                    multiplier += URL_BOOST
                if is_bigram:
                    multiplier += 2.0   # phrase match bonus

                entry["text_score"]   += bm25 * multiplier
                entry["anchor_score"] += anchor_postings.get(p["url"], 0) * ANCHOR_BOOST
                if not is_bigram:
                    entry["term_coverage"] += 1
                entry["matched_terms"].append(term)

        if not scores:
            return []

        n_terms = len(query_terms)

        results = []
        for info in scores.values():
            coverage_bonus = info["term_coverage"] / max(n_terms, 1)
            text_score     = info["text_score"] + info["anchor_score"]

            pr = info["pagerank"]
            pr_factor = math.log1p(pr * 1000) / 10.0 + 0.5   # 0.5–1.5

            combined = (
                text_score
                * max(0.1, coverage_bonus)
                * pr_factor
                * max(0.01, info["panda"])
                * max(0.01, info["penguin"])
            )

            results.append({
                **info,
                "score": round(combined, 6),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[offset: offset + limit]
