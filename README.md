# NeuraSearch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-ff69b4?style=flat-square)

**NeuraSearch** is a fully self-built, open-source search engine backend written in Python.  It crawls the web, indexes pages through a multi-stage NLP pipeline, ranks results using a multi-signal PageRank system, and exposes everything through a clean REST API.

The project mimics the core architecture of how a real search engine works, from the priority-queue frontier that decides which URL to fetch next, through lemmatization-then-stemming text normalization, all the way to Panda/Penguin quality scoring on top of iterative PageRank.  Every component is readable, documented, and meant to be understood.  The data stored is also designed to be rich enough for AI/ML training use cases.

---

## Table of Contents

- [Why We Built This](#why-we-built-this)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Crawler](#crawler)
- [Text Processing Pipeline](#text-processing-pipeline)
- [Indexer and BM25 Scoring](#indexer-and-bm25-scoring)
- [PageRank Pipeline](#pagerank-pipeline)
- [Database Schema](#database-schema)
- [Data Exports (JSON)](#data-exports-json)
- [REST API](#rest-api)
- [Getting Started](#getting-started)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Checkpoint and Resume](#checkpoint-and-resume)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why We Built This

Most people treat search engines as a black box.  After reading through various papers on information retrieval and spending time understanding how Google's early architecture worked, it became clear that the only real way to understand it was to build one from scratch.

NeuraSearch is the result of that effort.  It implements every layer of the pipeline correctly: a polite resumable web crawler; a POS-aware lemmatization + stemming NLP pipeline; a BM25 inverted index with full term provenance; Named Entity Recognition; a PageRank algorithm with proper dangling-node handling; and Panda/Penguin content and link quality scorers.

Every stored data point is designed to be useful, not just for search, but also as a structured dataset for AI/ML training.

---

## Features

**Crawler**
- Multi-threaded crawling with a configurable thread pool (default 10 workers)
- Priority-based URL frontier using a max-heap scored by depth, TLD weight, and in-link count
- Full robots.txt compliance including Crawl-Delay directives
- Per-domain rate limiting with random jitter
- Automatic retry with exponential backoff on failed requests
- Content deduplication via MD5 hash
- Respects `noindex` and `nofollow` meta directives
- Extracts `h1`–`h6` headings, `meta keywords`, and canonical URL from each page
- NLP-processed anchor text (lemmatize + stem) for higher-quality anchor index
- Graceful shutdown on CTRL-C with automatic checkpoint save
- **Crash recovery**: interrupted URLs automatically resume on next run

**Text Processing**
- POS tagging (NLTK averaged_perceptron_tagger) before lemmatization
- **WordNet lemmatization** (POS-aware) before Porter stemming: `"running"→"run"→"run"`, `"geese"→"goose"→"goos"`
- Full provenance per token: `{ stem: { lemma, originals, positions, pos_tag } }`
- **Named Entity Recognition** via NLTK ne_chunk: extracts PERSON, ORGANIZATION, GPE, LOCATION, etc.
- **Bigram extraction**: common 2-gram phrases indexed as separate terms for phrase-match bonus
- Snippet generation: extracts the most relevant context window around query hits
- WordNet query expansion (optional, via `?expand=true`)

**Indexer**
- BM25 (Robertson-Walker) inverted index with per-(term, document) provenance
- Every index entry stores: stemmed term, lemma, original word forms, word positions, field flags (title / description / heading / URL / anchor), dominant POS tag
- Field-level boosts: title (4.0), headings (2.5), description (2.0), anchor (3.0), URL (1.5)
- Bigrams indexed alongside unigrams; phrase query gets a +2.0 multiplier at query time
- URL path tokenization: `/wiki/Machine_learning` → index terms `machine`, `learn`
- Named entities stored in a dedicated `named_entities` table per page
- **Incremental**: already-indexed pages are always skipped — safe to resume after any interruption
- **CTRL-C checkpointing**: current page finishes cleanly and a JSON snapshot is exported

**Ranking**
- Iterative PageRank with dangling-node handling and warm-start from previous run
- **Mid-run checkpointing**: scores saved every 20 iterations so CTRL-C loses minimal work
- Panda scorer: content quality from 5 signals (length, keyword density, readability, title quality, vocabulary diversity)
- Penguin scorer: link quality from 3 signals (anchor diversity, domain diversity, link count)
- Combined authority: `PageRank × Panda × Penguin` multiplied into BM25 at query time
- Phrase/bigram match bonus in final scoring

**API**
- FastAPI REST API with Swagger/ReDoc documentation
- Paginated search with per-query timing
- **Query expansion** (`?expand=true`) using WordNet synonyms
- **Snippet generation** in every search result
- Entity lookup endpoint: find pages by named entity
- Autocomplete using lemma forms (human-readable) rather than raw stems
- Background task management for all long-running operations
- Full export endpoint (`POST /export`) to write all JSON data snapshots

---

## System Architecture

```
                    +---------------------------+
                    |        Seed URLs          |
                    +------------+--------------+
                                 |
                                 v
+------------------------------------+------------------------------------+
|                              CRAWLER                                   |
|                                                                        |
|  +---------------------+         +------------------------------+     |
|  | Frontier (max-heap) | pop()   |   Thread Pool (10 workers)   |     |
|  | scored by:          +-------> |      _crawl_worker()         |     |
|  |  - depth            |         +---+----------+----------+----+     |
|  |  - TLD weight       |             |          |          |          |
|  |  - in-link count    |             v          v          v          |
|  +---------------------+        robots.txt  HTTP fetch  rate limit    |
|                                       |      + retry    per domain    |
|                                       v                               |
|                              +-------------------+                    |
|                              | HTML Parser       |                    |
|                              | - h1-h6 headings  |                    |
|                              | - meta_keywords   |                    |
|                              | - canonical URL   |                    |
|                              | - link extractor  |                    |
|                              | - NLP anchor text |                    |
|                              +--------+----------+                    |
|                                       |                               |
+---------------------------------------+-------------------------------+
                                        |
                                        v
                        +---------------+---------------+
                        |         SQLite Database       |
                        | pages / links / crawl_queue   |
                        | inverted_index / anchors      |
                        | named_entities / domain_stats |
                        +---+------------------+--------+
                            |                  |
               +------------+                  +-------------+
               |                                             |
               v                                             v
   +-----------+----------+                   +--------------+----------+
   |        INDEXER       |                   |    PAGERANK PIPELINE    |
   |                      |                   |                         |
   | NLP Pipeline:        |                   | 1. Panda (content)      |
   |  - POS tagging       |                   | 2. Penguin (links)      |
   |  - Lemmatization     |                   | 3. PageRank (graph)     |
   |  - Porter stemming   |                   | 4. authority =          |
   |  - Provenance        |                   |    PR × Panda × Penguin |
   |  - NER extraction    |                   |                         |
   |  - Bigram indexing   |                   | Mid-run checkpointing:  |
   |  - URL tokenization  |                   | saves every 20 iters    |
   |                      |                   |                         |
   | BM25 inverted index  |                   | CTRL-C: saves & exits   |
   | CTRL-C: saves & exits|                   |                         |
   +----------+-----------+                   +--------------+----------+
              |                                              |
              +--------------------+-------------------------+
                                   |
                                   v
                    +--------------+--------------+
                    |        FastAPI REST API      |
                    |                             |
                    | GET  /search?q=...&expand=  |
                    | GET  /suggest               |
                    | GET  /entity                |
                    | GET  /stats                 |
                    | POST /crawl                 |
                    | POST /index                 |
                    | POST /pagerank              |
                    | POST /export                |
                    +--------------+--------------+
                                   |
                                   v
                            Frontend Client
```

---

## Crawler

The crawler is multi-threaded, polite (robots.txt + rate limits), resumable (all state in SQLite), and extracts rich metadata from each page.

**What is extracted per page:**

| Field | Source |
|-------|--------|
| `title` | `<title>` tag |
| `description` | `<meta name="description">` or first `<p>` |
| `headings` | All `h1`–`h6` tags → JSON `{"h1":[...], "h2":[...]}` |
| `meta_keywords` | `<meta name="keywords">` |
| `canonical_url` | `<link rel="canonical">` |
| `content` | Full cleaned body text (boilerplate removed) |
| `word_count` | Word count of cleaned body |
| `content_hash` | MD5 for deduplication |

**Anchor text processing** uses the full NLP pipeline (lemmatize + stem) so anchor terms in the index match the same stems used for body content.

### Frontier (Priority Queue)

```
URL priority score = (1 / (1 + depth)) × TLD_boost × length_factor × clean_url_bonus

TLD boosts:  .edu → 1.4  |  .gov → 1.3  |  .org → 1.1  |  .com → 1.0
Heap: push O(log n) | pop O(log n) | dedup O(1) via MD5 seen-set
DB-backed: every URL written to crawl_queue so crashes are resumable
```

### Robots.txt and Politeness

Each domain: fetch `/robots.txt` once → cache in memory → check `NeuraSearchBot` permission → enforce `Crawl-Delay` with random jitter (0–0.3 s).

---

## Text Processing Pipeline

The NLP pipeline in `indexer/text_processor.py` is applied to both documents at index time and queries at search time, ensuring consistent normalization.

```
Raw text input
     |
     v
+---------------------------------------+
| Tokenize                              |
| regex: [a-zA-Z]+(?:['-][a-zA-Z]+)*   |
| lowercase, preserves hyphenation      |
+---------------------------------------+
     |
     v
+---------------------------------------+
| Filter                                |
| - remove stop words (NLTK + built-in)|
| - discard tokens len < 2 or > 50     |
| - discard non-alpha tokens           |
+---------------------------------------+
     |
     v
+---------------------------------------+
| POS Tagging (NLTK)                    |
| averaged_perceptron_tagger            |
| tags: NN, VB, JJ, RB ...             |
+---------------------------------------+
     |
     v
+---------------------------------------+
| WordNet Lemmatization (POS-aware)     |
| "running"  → "run"   (VB)            |
| "better"   → "good"  (JJ)            |
| "geese"    → "goose" (NN)            |
| "quickly"  → "quickly" (RB)          |
| cached: lru_cache(500,000)           |
+---------------------------------------+
     |
     v
+---------------------------------------+
| Porter Stemming                       |
| "running"  → "run"                   |
| "language" → "languag"               |
| applied to the lemma, not raw token  |
| cached: lru_cache(500,000)           |
+---------------------------------------+
     |
     v
Provenance dict per stemmed term:
{
  "languag": {
    "lemma":     "language",
    "originals": ["language", "languages"],
    "positions": [3, 17, 42],
    "pos":       "NN"
  },
  ...
}
```

**Named Entity Recognition** — runs separately over the first 6,000 characters of `title + description + headings + content`:

```
NLTK ne_chunk pipeline:
  word_tokenize → pos_tag → ne_chunk

Entity types returned:
  PERSON, ORGANIZATION, GPE (geo-political), LOCATION, FACILITY
```

**Bigram extraction** — sliding window over stemmed body tokens, stored in the same inverted index as terms like `"machin learn"`.  At query time, consecutive query stems are also tried as a bigram and given a +2.0 phrase-match multiplier.

---

## Indexer and BM25 Scoring

Each `(term, document)` pair stored in `inverted_index` now carries:

| Column | Description |
|--------|-------------|
| `term` | Porter-stemmed form (search key) |
| `lemma` | WordNet-lemmatized intermediate |
| `original_forms` | JSON list of distinct raw word forms seen |
| `positions` | JSON list of word positions in cleaned stream |
| `frequency` | Raw count + synthetic field-boost counts |
| `bm25` | Robertson-Walker BM25 score |
| `in_title` | Term found in `<title>` |
| `in_description` | Term found in meta description |
| `in_url` | Term found in URL path segments |
| `in_anchor` | Term found in inbound anchor texts |
| `pos_tag` | Dominant NLTK POS tag for this term in this doc |

**BM25 formula:**

```
IDF = log( (N - df + 0.5) / (df + 0.5) + 1 )

TF_norm = tf × (k1 + 1) / (tf + k1 × (1 - b + b × dl / avgdl))

BM25(t, D) = IDF × TF_norm      k1=1.5  b=0.75
```

**Query-time scoring:**

```
final_score(Q, D) =

  ( Σ_t [  BM25(t, D) × field_multiplier(t, D)
           + anchor_freq(t, D) × ANCHOR_BOOST     ]
    + bigram_bonus if consecutive query terms match a bigram
  )
  × coverage_bonus          (matched_terms / total_query_terms)
  × log-normalized PageRank factor
  × panda_score(D)
  × penguin_score(D)

Field multipliers:
  TITLE_BOOST       = 4.0
  DESCRIPTION_BOOST = 2.0
  HEADING_BOOST     = 2.5
  ANCHOR_BOOST      = 3.0
  URL_BOOST         = 1.5
  Bigram phrase match = +2.0
```

---

## PageRank Pipeline

```
Step 1: Panda Scorer (content quality)
  5 signals → panda_score ∈ (0, 1]
  - Content length (logistic curve, saturates ~800 words)
  - Keyword density (ideal 2–8%, penalizes stuffing)
  - Flesch-Kincaid readability (bell curve around 60)
  - Title quality (word overlap with body content)
  - Vocabulary diversity (type-token ratio)

Step 2: Penguin Scorer (link quality)
  3 signals → penguin_score ∈ (0, 1]
  - Anchor text diversity (unique/total, log scale)
  - Linking domain diversity (log scale, caps at 50+ domains)
  - In-link count (log scale, saturates at 200)

Step 3: PageRank (iterative, Brin & Page 1998)
  Initialize: PR(p) = 1/N
  Each iteration:
    dangling_sum = d × Σ(PR(dangling)) / N
    PR(p) = (1-d)/N + dangling_sum + d × Σ(PR(q)/out_deg(q))
  Convergence: L1-delta < 1e-8  or  200 iterations
  Warm-start from previous DB scores (fewer iterations needed)
  Checkpoint: saves to DB every 20 iterations

Step 4: authority(P) = PageRank(P) × Panda(P) × Penguin(P)
  Written to pages.final_score

Step 5: BM25 refresh so updated authority affects search immediately
```

---

## Database Schema

All data is stored in `data/neurasearch.db` (SQLite, WAL mode, per-thread connections).

```
pages
  id, url, title, description, content, content_hash
  word_count, language, crawled_at, last_modified
  status_code, depth, in_link_count, out_link_count
  pagerank_score, panda_score, penguin_score, final_score
  is_indexed
  headings        -- JSON {"h1":[...], "h2":[...], ...}
  entities_json   -- JSON [{"text":"...","label":"..."}]
  meta_keywords   -- raw meta keywords string
  canonical_url   -- canonical URL if different from url

links
  id, src_url, dst_url, anchor_text, rel

crawl_queue
  id, url, priority, depth, added_at
  status: pending | processing | done | failed

inverted_index
  id, term, doc_id
  frequency, positions        -- count + JSON positions
  tf_idf, bm25                -- relevance scores
  in_title, in_description    -- field flags (0/1)
  in_url, in_anchor           -- field flags (0/1)
  lemma                       -- lemmatized form
  original_forms              -- JSON list of raw forms
  pos_tag                     -- dominant POS tag

named_entities
  id, doc_id, entity, entity_type, frequency

anchor_index
  id, term, dst_url, frequency

domain_stats
  domain, page_count, avg_pagerank, last_crawled
  crawl_delay, robots_txt, is_blocked

Indexes:
  idx_inv_term, idx_inv_doc, idx_inv_lemma
  idx_links_src, idx_links_dst
  idx_pages_score, idx_queue_pri, idx_anchor_term
  idx_ne_entity, idx_ne_doc, idx_ne_type
```

---

## Data Exports (JSON)

Run `python main.py export` (or `POST /export`) to write all data to human-readable JSON files inside `data/`.

```
data/
├── crawled/
│   ├── pages.json          all crawled pages + headings + entities + scores
│   └── links.json          full link graph {src: [dst, ...]}
├── pagerank/
│   └── scores.json         per-page authority scores, sorted best-first
├── index/
│   ├── vocabulary.json     each term with FULL PROVENANCE per source page:
│   │                         term, lemma, doc_freq,
│   │                         sources: [{url, title, field_flags,
│   │                                    frequency, positions, bm25,
│   │                                    original_forms, pos_tag,
│   │                                    page_scores}]
│   └── indexed_pages.json  all indexed pages with headings
└── entities/
    ├── named_entities.json all extracted entities + source pages
    └── entity_types.json   entity type distribution (PERSON, ORG, GPE ...)
```

The `vocabulary.json` file is the richest data store — every word in the index comes with a full history: where it was found, in which field, at which positions, with what BM25 score, what its original capitalized form was, and what part of speech it was tagged as.  This makes it directly usable for AI/ML training tasks.

---

## REST API

The API runs on `http://localhost:8000` by default.  All long-running operations are dispatched to background threads.

```
GET  /
     Health check.

GET  /stats
     { pages_indexed, links_found, unique_terms, unique_bigrams,
       named_entities, domains }

GET  /tasks
     Status of background tasks: "running" | "done" | "error"

GET  /search?q=...&page=1&limit=10&expand=false
     - q       : search query (required)
     - page    : result page (default 1)
     - limit   : results per page, max 50 (default 10)
     - expand  : if true, expand query with WordNet synonyms
     Returns: {
       query, processed_terms, expanded_terms,
       total, page, limit, took_ms,
       results: [{
         url, title, description,
         snippet,          ← context window around query term hits
         score, pagerank, panda, penguin,
         matched_terms     ← which query stems matched this result
       }]
     }

GET  /suggest?q=...&limit=8
     Autocomplete using lemma forms (real words, not stems).

GET  /entity?name=...&limit=10
     Find pages by named entity name.
     Returns: { entity, page_count, pages: [{url, title, pagerank}] }

POST /crawl    { seeds, limit, max_depth, max_workers }
POST /index
POST /pagerank
POST /export   → write all JSON data files in background
```

Interactive docs: `http://localhost:8000/docs` (Swagger) | `http://localhost:8000/redoc`

---

## Getting Started

### Requirements

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/neura-spheres/NeuraSearch.git
cd NeuraSearch/backend

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download NLTK data (run once — includes POS tagger and NER models)
python main.py setup
```

### Quick Start

```bash
# Step 1: Crawl pages from a seed URL
python main.py crawl https://en.wikipedia.org/wiki/Python --limit 500 --depth 3

# Step 2: Index crawled pages (builds NLP pipeline, BM25, NER, bigrams)
python main.py index

# Step 3: Run PageRank + Panda + Penguin scoring
python main.py pagerank

# Step 4: Export all data to JSON
python main.py export

# Step 5: Start the API server
python main.py serve
```

Or in one command:

```bash
python main.py all --seeds https://en.wikipedia.org/wiki/Python --limit 500
```

Then search:

```bash
curl "http://localhost:8000/search?q=python+programming&expand=true"
```

---

## CLI Usage

```
python main.py <command> [options]

Commands:
  setup
    Download NLTK data (punkt, wordnet, POS tagger, NER models).
    Run this once before first use.

  serve [--port INT] [--reload]
    Start the FastAPI server (default port 8000).

  crawl [seeds...] [--limit INT] [--workers INT] [--depth INT]
    Crawl from one or more seed URLs.
    Automatically resumes any previously interrupted session.

  index
    Build the inverted index for all un-indexed pages.
    CTRL-C exits cleanly after the current page; next run resumes.

  pagerank
    Run Panda + Penguin + PageRank pipeline.
    CTRL-C saves scores accumulated so far; next run warm-starts.

  export
    Write all data to JSON files under data/.

  all [seeds...] [--limit] [--workers] [--depth] [--port] [--reload]
    Run crawl → index → pagerank → export → serve in sequence.
```

Examples:

```bash
# Crawl Wikipedia with 15 threads, depth 4
python main.py crawl https://en.wikipedia.org/wiki/Machine_learning \
               --limit 2000 --workers 15 --depth 4

# Resume the indexer (automatically skips already-indexed pages)
python main.py index

# Search with query expansion
curl "http://localhost:8000/search?q=neural+network&expand=true"

# Find pages mentioning a named entity
curl "http://localhost:8000/entity?name=Guido+van+Rossum"
```

---

## Checkpoint and Resume

Every process is designed to survive interruption and resume from where it stopped.

| Process | How it checkpoints |
|---------|-------------------|
| **Crawler** | Every URL is saved to `crawl_queue` with status `pending/processing/done`. On restart, any `processing` URL is reset to `pending`. JSON is saved every 500 pages. CTRL-C triggers a final save. |
| **Indexer** | Each page is marked `is_indexed=1` immediately after indexing. On restart, `get_pages_not_indexed()` skips all already-indexed pages. CTRL-C finishes the current page, exports a JSON checkpoint, and exits. |
| **PageRank** | Scores are written to the database every 20 iterations. CTRL-C saves scores accumulated so far. On next run, warm-start loads those saved scores and resumes near the interrupted point. |

There is **no manual intervention needed** to resume.  Simply re-run the same command and it picks up from where it left off.

---

## Language Filter

You can restrict crawling to specific languages by setting `CRAWL_LANGUAGES` in `config.py`.

```python
# English only
CRAWL_LANGUAGES = ["en"]

# English + Indonesian
CRAWL_LANGUAGES = ["en", "id"]

# No filter — crawl everything (default)
CRAWL_LANGUAGES = []
```

**How detection works (in priority order):**

1. `<html lang="...">` attribute — most reliable; present on almost all modern sites
2. `<meta http-equiv="Content-Language">` tag
3. [`langdetect`](https://pypi.org/project/langdetect/) library — install with `pip install langdetect` for content-based fallback detection

If the language cannot be detected and `CRAWL_LANGUAGES` is non-empty, the page content is **skipped** (not stored), but its outbound links are still followed — so if a French page links to an English page, the English page is still discovered and crawled.

**Supported language codes (ISO 639-1):**

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `af` | Afrikaans | `id` | Indonesian | `sk` | Slovak |
| `ar` | Arabic | `it` | Italian | `sl` | Slovenian |
| `bg` | Bulgarian | `ja` | Japanese | `sq` | Albanian |
| `bn` | Bengali | `ka` | Georgian | `sr` | Serbian |
| `ca` | Catalan | `ko` | Korean | `sv` | Swedish |
| `cs` | Czech | `lt` | Lithuanian | `sw` | Swahili |
| `cy` | Welsh | `lv` | Latvian | `ta` | Tamil |
| `da` | Danish | `mk` | Macedonian | `te` | Telugu |
| `de` | German | `ml` | Malayalam | `th` | Thai |
| `el` | Greek | `mr` | Marathi | `tl` | Filipino |
| `en` | English | `ms` | Malay | `tr` | Turkish |
| `es` | Spanish | `mt` | Maltese | `uk` | Ukrainian |
| `et` | Estonian | `nl` | Dutch | `ur` | Urdu |
| `fa` | Persian | `no` | Norwegian | `vi` | Vietnamese |
| `fi` | Finnish | `pl` | Polish | `zh` | Chinese |
| `fr` | French | `pt` | Portuguese | `zu` | Zulu |
| `gu` | Gujarati | `ro` | Romanian | | |
| `hi` | Hindi | `ru` | Russian | | |
| `hr` | Croatian | `hu` | Hungarian | | |

---

## Configuration

All tuneable values live in `config.py`.

```python
# ── Crawler ───────────────────────────────────────────────────────────────────
MAX_WORKERS      = 10          # concurrent crawler threads
CRAWL_LIMIT      = 50_000      # max pages per crawl session
REQUEST_TIMEOUT  = 10          # HTTP request timeout (seconds)
MIN_DELAY        = 0.5         # minimum per-domain politeness delay (seconds)
MAX_DELAY        = 3.0         # maximum per-domain delay (seconds)
MAX_RETRIES      = 3           # HTTP retry attempts before giving up
MAX_DEPTH        = 6           # max link depth from seed URLs
MAX_CONTENT_SIZE = 5_242_880   # max page content: 5 MB

# ── BM25 / Indexer ────────────────────────────────────────────────────────────
BM25_K1           = 1.5        # term frequency saturation
BM25_B            = 0.75       # document length normalization
TITLE_BOOST       = 4.0
DESCRIPTION_BOOST = 2.0
HEADING_BOOST     = 2.5        # h1-h6 headings
ANCHOR_BOOST      = 3.0
URL_BOOST         = 1.5

# ── NLP ───────────────────────────────────────────────────────────────────────
BIGRAM_MIN_FREQ    = 2         # minimum frequency for a bigram to be indexed
MAX_BIGRAMS_PER_DOC = 200      # cap bigrams per document
MAX_ENTITIES_PER_DOC = 50      # cap named entities per document
NER_TEXT_LIMIT     = 6000      # characters fed to NER (capped for speed)
SNIPPET_WINDOW     = 40        # words of context around a query hit

# ── PageRank ──────────────────────────────────────────────────────────────────
DAMPING_FACTOR          = 0.85
MAX_PAGERANK_ITERATIONS = 200
PAGERANK_TOLERANCE      = 1.0e-8

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST                = "0.0.0.0"
API_PORT                = 8000
DEFAULT_RESULTS_PER_PAGE = 10
MAX_RESULTS             = 100
```

---

## Project Structure

```
backend/
│
├── main.py                  CLI entry point
├── config.py                All tunable constants
├── setup_nltk.py            NLTK data downloader (run once)
│
├── crawler/
│   ├── crawler.py           Crawler class, thread pool, heading extraction
│   ├── frontier.py          Priority queue (max-heap) + DB persistence
│   ├── url_utils.py         URL normalization, scoring, link extraction
│   └── robots_handler.py    robots.txt fetching, caching, checking
│
├── indexer/
│   ├── indexer.py           Indexing orchestrator (NER, bigrams, checkpoint)
│   ├── inverted_index.py    BM25 builder + query engine (provenance, bigrams)
│   └── text_processor.py    NLP: POS tag → lemmatize → stem, NER, bigrams,
│                             snippet generation, readability
│
├── pagerank/
│   ├── pagerank.py          Iterative PageRank + checkpoint callback + SIGINT
│   ├── panda_scorer.py      Content quality (5 signals)
│   └── penguin_scorer.py    Link quality (3 signals)
│
├── database/
│   └── db.py                SQLite layer (schema, migration, all queries)
│
├── api/
│   └── app.py               FastAPI routes: search, suggest, entity, export
│
├── utils/
│   └── json_exporter.py     Rich JSON exports (vocabulary with full provenance,
│                             named entities, pages, pagerank, link graph)
│
└── data/                    Auto-generated at runtime (gitignored)
    ├── neurasearch.db
    ├── crawled/             pages.json, links.json
    ├── index/               vocabulary.json (with provenance), indexed_pages.json
    ├── pagerank/            scores.json
    └── entities/            named_entities.json, entity_types.json
```

---

## Roadmap

- [ ] Phrase query using stored token positions (beyond current bigram approach)
- [ ] Personalised PageRank as an API parameter for topic-sensitive ranking
- [ ] SimHash near-duplicate detection (currently using exact MD5 only)
- [ ] Language detection and multilingual tokenization
- [ ] Scheduled re-crawling of stale indexed pages
- [ ] Trie-based autocomplete (faster + smarter than current SQL LIKE)
- [ ] Incremental PageRank updates (avoid full recomputation on small additions)
- [ ] Distributed crawling across multiple machines
- [ ] A frontend UI (search bar, result cards, facets, entity highlighting)
- [ ] Docker image and docker-compose for easy deployment
- [ ] Prometheus metrics endpoint

---

## Known Limitations

- **Scale.** SQLite works well up to a few hundred thousand pages. Millions of documents would require Elasticsearch or a custom shard-based index.
- **NER accuracy.** NLTK's ne_chunk is a statistical chunker trained on newswire. Accuracy on general web text varies. SpaCy would be significantly more accurate.
- **PageRank recomputation.** The full graph is recomputed every run. Incremental updates would help for large crawls.
- **English only.** The NLP pipeline and stop word list are English. Non-English pages index poorly.
- **Memory.** The frontier and PageRank graph are held in memory. Very large crawls (10M+ URLs) would need on-disk alternatives.

---

## Contributing

Contributions of any kind are welcome — bug fixes, new features, documentation improvements, or ideas.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

1. Fork the repository and create your branch from `main`.
2. Write clear, readable code that matches the existing style.
3. Describe what you changed and why in your PR.
4. If fixing a bug, include reproduction steps.

To report a bug or suggest a feature, open an issue on GitHub.

---

## License

This project is licensed under the **MIT License**.  See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements

- **The Anatomy of a Large-Scale Hypertextual Web Search Engine** — Brin & Page (1998).  The original PageRank paper.
- **Introduction to Information Retrieval** — Manning, Raghavan, and Schütze.  BM25 derivation in chapter 11.
- **Google Panda and Penguin** — algorithm documentation and SEO research for the quality signal design.
- The Python ecosystem: FastAPI, BeautifulSoup, NLTK, SQLite, and the standard library.

---

*Built by students, for anyone who wants to understand how search really works.*
