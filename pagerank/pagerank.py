"""
PageRank engine.

Implements the classic iterative PageRank (Brin & Page, 1998) with:

  • Proper dangling-node handling (no rank leak).
  • Teleportation factor d = 0.85 (configurable).
  • Convergence check via L1-norm of rank-delta vector.
  • Post-hoc adjustment by Panda + Penguin quality scores.

After PageRank converges, the "authority score" stored per page is:

    authority(P) = PageRank(P) × Panda(P) × Penguin(P)

This combined score drives the final search ranking alongside BM25.

Algorithm complexity: O(|V| + |E|) per iteration.

Checkpoint / resume behaviour
------------------------------
PageRankEngine.run() installs a SIGINT handler.  On CTRL-C:
  1. The current iteration finishes cleanly.
  2. Scores accumulated so far are written to the database.
  3. The next run warm-starts from those saved scores, so convergence
     resumes from near the interrupted point rather than from scratch.

Additionally, intermediate scores are saved to the DB every
PAGERANK_CHECKPOINT_EVERY iterations, so even a hard kill loses at
most that many iterations of work.
"""

import logging
import math
import signal
import threading
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import DAMPING_FACTOR, MAX_PAGERANK_ITERATIONS, PAGERANK_TOLERANCE
from pagerank.panda_scorer   import score_all_pages as panda_score_all
from pagerank.penguin_scorer import score_all_pages as penguin_score_all

logger = logging.getLogger("neurasearch.pagerank")

# Save intermediate scores every N iterations so CTRL-C loses minimal work.
PAGERANK_CHECKPOINT_EVERY = 20


def _build_graph(db):
    """
    Build in-memory graph structures needed for PageRank.

    Returns
    -------
    all_nodes     : set of all URLs (pages that exist in the DB)
    out_links     : dict {url: [dst_url, ...]}
    in_links      : dict {url: [src_url, ...]}
    dangling_nodes: set of URLs with no outgoing links
    """
    # All crawled pages
    pages = db.get_all_pages()
    all_nodes = {p["url"] for p in pages}

    # Link graph (only links between crawled pages)
    raw_graph = db.get_link_graph()   # {src: [dst, ...]}

    out_links: dict[str, list[str]] = defaultdict(list)
    in_links:  dict[str, list[str]] = defaultdict(list)

    for src, dsts in raw_graph.items():
        if src not in all_nodes:
            continue
        for dst in dsts:
            if dst in all_nodes and dst != src:   # no self-loops
                out_links[src].append(dst)
                in_links[dst].append(src)

    dangling_nodes = {n for n in all_nodes if len(out_links[n]) == 0}

    logger.info(
        f"[PageRank] Graph: {len(all_nodes)} nodes, "
        f"{sum(len(v) for v in out_links.values())} edges, "
        f"{len(dangling_nodes)} dangling nodes"
    )
    return all_nodes, out_links, in_links, dangling_nodes


def compute_pagerank(
    all_nodes,
    out_links,
    in_links,
    dangling_nodes,
    d: float = DAMPING_FACTOR,
    max_iter: int = MAX_PAGERANK_ITERATIONS,
    tol: float = PAGERANK_TOLERANCE,
    initial_scores: dict = None,
    checkpoint_fn=None,
    stop_event=None,
) -> dict:
    """
    Iterative PageRank with dangling-node correction.

    Parameters
    ----------
    all_nodes      : iterable of node strings
    out_links      : {src: [dst, ...]}
    in_links       : {dst: [src, ...]}
    dangling_nodes : set of nodes with no outgoing edges
    d              : damping factor (teleportation = 1 - d)
    max_iter       : maximum iterations
    tol            : L1 convergence threshold
    initial_scores : optional warm-start scores from a previous run
                     (converges in fewer iterations when provided)
    checkpoint_fn  : optional callable(scores_dict, iteration) — called
                     every PAGERANK_CHECKPOINT_EVERY iterations to persist
                     intermediate results so a crash loses minimal work.
    stop_event     : optional threading.Event — if set, the loop exits after
                     the current iteration and the best scores so far are
                     returned.  The caller should persist them.

    Returns
    -------
    dict {url: pagerank_score}  (scores sum to ≈ 1.0)
    """
    all_nodes = list(all_nodes)
    N = len(all_nodes)
    if N == 0:
        return {}

    # Warm start from previous scores (normalise to sum=1)
    if initial_scores:
        total = sum(initial_scores.get(n, 1.0 / N) for n in all_nodes)
        pr: dict = {
            n: initial_scores.get(n, 1.0 / N) / max(total, 1e-12)
            for n in all_nodes
        }
        logger.info(f"[PageRank] Warm-starting from {len(initial_scores)} existing scores.")
    else:
        pr: dict = {node: 1.0 / N for node in all_nodes}

    for iteration in range(1, max_iter + 1):
        # Check for external stop signal (e.g. CTRL-C)
        if stop_event and stop_event.is_set():
            logger.info(f"[PageRank] Stop requested — saving scores at iteration {iteration - 1}.")
            break

        new_pr: dict = {}
        dangling_sum = d * sum(pr[node] for node in dangling_nodes) / N

        for node in all_nodes:
            rank  = (1.0 - d) / N
            rank += dangling_sum
            for src in in_links.get(node, []):
                out_deg = len(out_links.get(src, []))
                if out_deg > 0:
                    rank += d * pr[src] / out_deg
            new_pr[node] = rank

        delta = sum(abs(new_pr[n] - pr[n]) for n in all_nodes)
        pr = new_pr

        if iteration % 10 == 0 or delta < tol:
            logger.info(f"[PageRank] Iteration {iteration:3d}  Δ={delta:.2e}")

        # Periodic checkpoint so a hard kill loses at most N iterations of work
        if checkpoint_fn and iteration % PAGERANK_CHECKPOINT_EVERY == 0:
            try:
                checkpoint_fn(pr, iteration)
            except Exception as exc:
                logger.warning(f"[PageRank] Checkpoint error: {exc}")

        if delta < tol:
            logger.info(f"[PageRank] Converged after {iteration} iterations.")
            break
    else:
        logger.warning(f"[PageRank] Did not converge within {max_iter} iterations.")

    return pr


def personalised_pagerank(
    all_nodes,
    out_links,
    in_links,
    dangling_nodes,
    seed_urls: list[str],
    d: float = 0.85,
    max_iter: int = 50,
) -> dict[str, float]:
    """
    Personalised PageRank: biases teleportation toward *seed_urls*.
    Useful for topic-sensitive ranking of a query cluster.
    """
    all_nodes = list(all_nodes)
    N = len(all_nodes)
    seed_set = {u for u in seed_urls if u in set(all_nodes)}
    if not seed_set:
        return compute_pagerank(all_nodes, out_links, in_links, dangling_nodes, d=d, max_iter=max_iter)

    seed_weight = 1.0 / len(seed_set)
    pr: dict[str, float] = {node: 1.0 / N for node in all_nodes}

    for _ in range(max_iter):
        new_pr: dict[str, float] = {}
        dangling_sum = d * sum(pr[n] for n in dangling_nodes) / N

        for node in all_nodes:
            teleport = seed_weight if node in seed_set else 0.0
            rank = (1.0 - d) * teleport + dangling_sum
            for src in in_links.get(node, []):
                out_deg = len(out_links.get(src, []))
                if out_deg > 0:
                    rank += d * pr[src] / out_deg
            new_pr[node] = rank
        pr = new_pr

    return pr


class PageRankEngine:
    """
    High-level orchestrator: runs PageRank → Panda → Penguin and
    writes results back to the database.

    Checkpoint / resume:
      - Scores are saved to the DB every PAGERANK_CHECKPOINT_EVERY iterations
        during the computation loop, so a crash loses at most that much work.
      - CTRL-C (SIGINT) is caught: the current iteration finishes, all
        accumulated scores are saved to the DB, and the process exits.
        The next run warm-starts from those saved scores.
    """

    def __init__(self, db):
        self.db        = db
        self._stop_evt = threading.Event()

    def run(self, run_panda: bool = True, run_penguin: bool = True):
        """
        Full pipeline:
          1. Panda scoring (content quality)
          2. Penguin scoring (link quality)
          3. PageRank computation with mid-run checkpointing
          4. Persist final scores to DB
        """
        logger.info("══ PageRank pipeline starting ══")

        # ── Step 1: Panda ────────────────────────────────────────────────────
        if run_panda:
            panda_score_all(self.db)

        # ── Step 2: Penguin ──────────────────────────────────────────────────
        if run_penguin:
            penguin_score_all(self.db)

        # ── Step 3: PageRank ─────────────────────────────────────────────────
        all_nodes, out_links, in_links, dangling = _build_graph(self.db)
        if not all_nodes:
            logger.warning("[PageRank] No pages in database.")
            return

        # Warm start from previous saved scores
        prev_pages  = self.db.get_all_pages()
        prev_scores = {p["url"]: p["pagerank_score"] for p in prev_pages
                       if p["pagerank_score"] and p["pagerank_score"] > 0}
        warm_start  = prev_scores if prev_scores else None

        # Install SIGINT handler
        self._stop_evt.clear()
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):
            logger.info("[PageRank] CTRL-C — will save scores after this iteration.")
            self._stop_evt.set()

        signal.signal(signal.SIGINT, _handle_sigint)

        # Checkpoint callback: save intermediate scores every N iterations
        def _checkpoint(scores: dict, iteration: int):
            self.db.bulk_update_pagerank(scores)
            logger.info(f"[PageRank] Checkpoint saved at iteration {iteration}.")
            try:
                from utils.json_exporter import export_pagerank_scores
                export_pagerank_scores(self.db)
            except Exception:
                pass

        pr = None
        try:
            pr = compute_pagerank(
                all_nodes, out_links, in_links, dangling,
                initial_scores=warm_start,
                checkpoint_fn=_checkpoint,
                stop_event=self._stop_evt,
            )
        finally:
            signal.signal(signal.SIGINT, prev_handler)

        if pr is None:
            logger.warning("[PageRank] Computation returned no results.")
            return None

        # ── Step 4: Persist final scores ─────────────────────────────────────
        self.db.bulk_update_pagerank(pr)
        logger.info(
            f"[PageRank] Scores written for {len(pr)} pages.  "
            f"Top-5: {sorted(pr.items(), key=lambda x: -x[1])[:5]}"
        )

        # ── Step 5: Export ───────────────────────────────────────────────────
        try:
            from utils.json_exporter import export_pagerank_scores
            export_pagerank_scores(self.db)
        except Exception as exc:
            logger.warning(f"[PageRank] JSON export error: {exc}")

        logger.info("══ PageRank pipeline complete ══")
        return pr
