import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "neurasearch.db"
INDEX_PATH = DATA_DIR / "index"
CRAWLED_PATH = DATA_DIR / "crawled"

# ─── Language filter ─────────────────────────────────────────────────────────
# List of BCP-47 language codes to crawl.
# Only pages whose detected language is in this list will be stored and indexed.
# Set to an empty list [] to crawl ALL languages (no filter).
#
# Detection strategy (in order):
#   1. <html lang="..."> attribute — most reliable for modern sites
#   2. <meta http-equiv="Content-Language"> tag
#   3. langdetect library (optional; install with: pip install langdetect)
#
# Supported language codes (ISO 639-1):
# ┌──────┬─────────────────────────┐  ┌──────┬─────────────────────────┐
# │ Code │ Language                │  │ Code │ Language                │
# ├──────┼─────────────────────────┤  ├──────┼─────────────────────────┤
# │  af  │ Afrikaans               │  │  lt  │ Lithuanian              │
# │  ar  │ Arabic                  │  │  lv  │ Latvian                 │
# │  bg  │ Bulgarian               │  │  mk  │ Macedonian              │
# │  bn  │ Bengali                 │  │  ml  │ Malayalam               │
# │  ca  │ Catalan                 │  │  mr  │ Marathi                 │
# │  cs  │ Czech                   │  │  ms  │ Malay                   │
# │  cy  │ Welsh                   │  │  mt  │ Maltese                 │
# │  da  │ Danish                  │  │  nl  │ Dutch                   │
# │  de  │ German                  │  │  no  │ Norwegian               │
# │  el  │ Greek                   │  │  pl  │ Polish                  │
# │  en  │ English                 │  │  pt  │ Portuguese              │
# │  es  │ Spanish                 │  │  ro  │ Romanian                │
# │  et  │ Estonian                │  │  ru  │ Russian                 │
# │  fa  │ Persian (Farsi)         │  │  sk  │ Slovak                  │
# │  fi  │ Finnish                 │  │  sl  │ Slovenian               │
# │  fr  │ French                  │  │  sq  │ Albanian                │
# │  ga  │ Irish                   │  │  sr  │ Serbian                 │
# │  gl  │ Galician                │  │  sv  │ Swedish                 │
# │  gu  │ Gujarati                │  │  sw  │ Swahili                 │
# │  hi  │ Hindi                   │  │  ta  │ Tamil                   │
# │  hr  │ Croatian                │  │  te  │ Telugu                  │
# │  hu  │ Hungarian               │  │  th  │ Thai                    │
# │  hy  │ Armenian                │  │  tl  │ Filipino (Tagalog)      │
# │  id  │ Indonesian              │  │  tr  │ Turkish                 │
# │  it  │ Italian                 │  │  uk  │ Ukrainian               │
# │  ja  │ Japanese                │  │  ur  │ Urdu                    │
# │  ka  │ Georgian                │  │  vi  │ Vietnamese              │
# │  ko  │ Korean                  │  │  zh  │ Chinese (any variant)   │
# │  lt  │ Lithuanian              │  │  zu  │ Zulu                    │
# └──────┴─────────────────────────┘  └──────┴─────────────────────────┘
#
# Examples:
#   CRAWL_LANGUAGES = ["en"]              # English only
#   CRAWL_LANGUAGES = ["en", "id"]        # English + Indonesian
#   CRAWL_LANGUAGES = ["en", "de", "fr"]  # English + German + French
#   CRAWL_LANGUAGES = []                  # no filter — crawl everything
#
CRAWL_LANGUAGES: list = ['en', 'id']   # set by GUI

# ─── Crawler ────────────────────────────────────────────────────────────────
MAX_WORKERS = 10          # Concurrent crawler threads
CRAWL_LIMIT = 50_000      # Max pages per crawl session
REQUEST_TIMEOUT = 10      # HTTP timeout (seconds)
MIN_DELAY = 0.5           # Minimum per-domain request delay
MAX_DELAY = 3.0           # Maximum per-domain request delay
MAX_RETRIES = 3           # HTTP retry attempts
MAX_DEPTH = 6             # Max link depth from seed URLs
MAX_CONTENT_SIZE = 5 * 1024 * 1024   # 5 MB per page

USER_AGENT = "NeuraSearchBot/1.0 (+https://neurasearch.io/bot)"

# Domains that are always skipped
BLOCKED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".mp4", ".mp3", ".avi", ".mov", ".zip", ".tar", ".gz",
    ".exe", ".dmg", ".apk", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".css", ".js", ".woff", ".woff2", ".ttf",
}

# ─── BM25 Indexer ───────────────────────────────────────────────────────────
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 50
BM25_K1 = 1.5             # Term frequency saturation
BM25_B = 0.75             # Document length normalization

# Field ranking boosts
TITLE_BOOST = 4.0
DESCRIPTION_BOOST = 2.0
HEADING_BOOST = 2.5       # h1–h6 headings
ANCHOR_BOOST = 3.0
URL_BOOST = 1.5

# ─── NLP / Indexing ─────────────────────────────────────────────────────────
BIGRAM_MIN_FREQ = 2       # Minimum times a bigram must appear in a doc to be indexed
MAX_BIGRAMS_PER_DOC = 200 # Cap bigrams stored per document
MAX_ENTITIES_PER_DOC = 50 # Cap named entities stored per document
NER_TEXT_LIMIT = 6000     # Characters fed to NER (capped for speed)
POS_TAG_LIMIT = 15000     # Characters POS-tagged per document
SNIPPET_WINDOW = 40       # Words of context around a hit for snippet generation

# ─── PageRank ───────────────────────────────────────────────────────────────
DAMPING_FACTOR = 0.85
MAX_PAGERANK_ITERATIONS = 200
PAGERANK_TOLERANCE = 1.0e-8

# ─── FastAPI ────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://neurasearch.io",
]

# ─── Search ─────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_PER_PAGE = 10
MAX_RESULTS = 100

# ─── Init directories ───────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH.mkdir(parents=True, exist_ok=True)
CRAWLED_PATH.mkdir(parents=True, exist_ok=True)
