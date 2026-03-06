"""
NLP text processing pipeline

Pipeline per document / query token:
  raw text
    -> tokenize  (smart regex: preserves hyphenation, apostrophes)
    -> lowercase
    -> filter    (stop-words, length, alpha)
    -> POS-tag   (NLTK averaged_perceptron_tagger — batch, fast)
    -> lemmatize (WordNet, POS-aware: "running"->"run", "geese"->"goose")
    -> stem      (Porter: "running"->"run", applied to lemma)
    -> provenance dict

Provenance dict returned by process_with_provenance():
    {
        stemmed_form: {
            "lemma":     str,        # intermediate lemmatized form
            "originals": list[str],  # distinct raw forms seen (order of first occurrence)
            "positions": list[int],  # positions in the cleaned token stream
            "pos":       str,        # dominant NLTK POS tag  (NN, VB, JJ, ...)
        }
    }

Additional helpers:
    extract_named_entities(text)   -> list[{"text":str, "label":str}]
    extract_bigrams(stemmed_list)  -> list[str]   ("w1 w2" format)
    extract_headings(soup)         -> dict         {"h1":[...], "h2":[...], ...}
    build_snippet(content, query_terms, window) -> str
"""

import re
import logging
from collections import Counter
from functools import lru_cache

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import MIN_WORD_LENGTH, MAX_WORD_LENGTH, BIGRAM_MIN_FREQ, NER_TEXT_LIMIT

logger = logging.getLogger("neurasearch.text")

# ── NLTK bootstrap ─────────────────────────────────────────────────────────────
_NLTK_OK    = False
_LEMMA_OK   = False
_NER_OK     = False
_stemmer    = None
_lemmatizer = None
_nltk_stop  = set()

try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords

    for _pkg in (
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker", "maxent_ne_chunker_tab", "words",
    ):
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

    _stemmer    = PorterStemmer()
    _lemmatizer = WordNetLemmatizer()
    _nltk_stop  = set(nltk_stopwords.words("english"))
    _NLTK_OK    = True
    _LEMMA_OK   = True

    # Verify NER chain works end-to-end
    try:
        from nltk import pos_tag, ne_chunk, word_tokenize, Tree
        _test = ne_chunk(pos_tag(word_tokenize("OpenAI released GPT-4.")), binary=False)
        _NER_OK = True
    except Exception as _e:
        logger.debug(f"NER unavailable: {_e}")

except Exception as exc:
    logger.warning(f"NLTK unavailable ({exc}). Using built-in fallbacks.")


# ── Combined stop-word set ─────────────────────────────────────────────────────
_STOP_WORDS: frozenset = frozenset({
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Be verbs
    "be", "is", "are", "was", "were", "been", "being",
    "am", "will", "would", "shall", "should",
    # Do / have
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    # Modals
    "can", "could", "may", "might", "must", "ought",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once",
    "about", "against", "along", "around", "near", "up", "down",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "not", "no", "only", "own", "same",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "whose",
    # Common low-value verbs
    "get", "got", "go", "goes", "went", "come", "came",
    "make", "made", "take", "took", "put", "see", "say", "said",
    "know", "think", "look", "want", "give", "use", "find", "tell",
    "ask", "seem", "feel", "try", "leave", "call",
    # Common adverbs / adjectives
    "very", "just", "more", "also", "how", "all", "each", "every",
    "any", "few", "other", "some", "such", "too", "well",
    "than", "then", "when", "where", "why", "there", "here",
    "now", "new", "old", "good", "great", "first", "last", "long",
    "little", "right", "big", "high", "different", "small",
    # Numbers / web noise
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "http", "https", "www", "com", "org", "net", "html",
    # Merge NLTK stop words
    *_nltk_stop,
})


# ── Regex tokenizer ────────────────────────────────────────────────────────────
# Preserves hyphenated words ("state-of-the-art") and apostrophes ("it's")
_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:['-][a-zA-Z]+)*")


def tokenize(text: str) -> list:
    """Split text into lowercase alpha tokens, preserving hyphenation/apostrophes."""
    return _TOKEN_RE.findall(text.lower())


# ── POS → WordNet POS mapping ──────────────────────────────────────────────────
def _nltk_pos_to_wordnet(tag: str):
    """Map NLTK POS tag string to a wordnet POS constant, or None."""
    if not _LEMMA_OK:
        return None
    try:
        from nltk.corpus import wordnet as wn
        if tag.startswith("J"):
            return wn.ADJ
        if tag.startswith("V"):
            return wn.VERB
        if tag.startswith("N"):
            return wn.NOUN
        if tag.startswith("R"):
            return wn.ADV
    except Exception:
        pass
    return None


@lru_cache(maxsize=500_000)
def _lemmatize(word: str, pos=None) -> str:
    """
    Lemmatize a single lowercase word using WordNet.
    pos should be a WordNet POS constant (wn.NOUN, wn.VERB, etc.) or None.
    Cached: identical (word, pos) pairs are looked up in O(1) after the first call.
    """
    if _LEMMA_OK and _lemmatizer:
        try:
            if pos is not None:
                return _lemmatizer.lemmatize(word, pos=pos)
            return _lemmatizer.lemmatize(word)
        except Exception:
            pass
    return word


@lru_cache(maxsize=500_000)
def _stem(word: str) -> str:
    """
    Porter-stem a single word.  Applied *after* lemmatization so the input
    is already a canonical form ("running" -> lemma "run" -> stem "run").
    """
    if _NLTK_OK and _stemmer:
        try:
            return _stemmer.stem(word)
        except Exception:
            pass
    # Naive suffix-stripping fallback
    for suffix in ("ing", "tion", "ed", "er", "ly", "ness", "ment"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _is_clean(token: str) -> bool:
    """Return True if a token should be kept for indexing."""
    return (
        token not in _STOP_WORDS
        and MIN_WORD_LENGTH <= len(token) <= MAX_WORD_LENGTH
        and token.isalpha()
    )


# ── Public NLP pipeline ────────────────────────────────────────────────────────

def process_text(text: str) -> list:
    """
    Full pipeline: tokenize → filter → lemmatize → stem.
    Returns a list of stemmed, cleaned tokens (same API as before).
    """
    tokens = tokenize(text)
    clean  = [t for t in tokens if _is_clean(t)]
    lemmas = [_lemmatize(t) for t in clean]
    return [_stem(l) for l in lemmas]


def process_with_positions(text: str) -> dict:
    """
    Full pipeline that tracks token positions in the cleaned stream.
    Returns {stemmed_term: [pos1, pos2, …]}
    (Thin wrapper around process_with_provenance for backward compatibility.)
    """
    return {term: data["positions"] for term, data in process_with_provenance(text).items()}


def process_with_provenance(text: str, pos_tag_limit: int = 15000) -> dict:
    """
    Full pipeline with complete provenance per stemmed term.

    Returns:
        {
            stemmed_form: {
                "lemma":     str,        # lemmatized intermediate
                "originals": list[str],  # distinct raw forms, insertion-ordered
                "positions": list[int],  # positions in cleaned token stream
                "pos":       str,        # dominant NLTK POS tag, e.g. "NN"
            }
        }

    Steps:
      1. Tokenize the full text.
      2. POS-tag the first pos_tag_limit characters worth of tokens.
      3. For each clean token: lemmatize (using POS), then stem.
      4. Accumulate provenance per stemmed form.
    """
    raw_tokens = tokenize(text)

    # POS tag in one pass for speed.  Cap so huge pages don't stall.
    cap = pos_tag_limit // 5   # rough chars-per-word = 5
    tag_tokens = raw_tokens[:cap]
    if _NLTK_OK and tag_tokens:
        try:
            from nltk import pos_tag as _pos_tag
            pos_pairs = _pos_tag(tag_tokens)
            # Pad with default NN tag for tokens beyond the cap
            if len(raw_tokens) > len(pos_pairs):
                pos_pairs = pos_pairs + [("", "NN")] * (len(raw_tokens) - len(pos_pairs))
        except Exception:
            pos_pairs = [(t, "NN") for t in raw_tokens]
    else:
        pos_pairs = [(t, "NN") for t in raw_tokens]

    result: dict = {}
    pos_counter = 0

    for raw_token, pair in zip(raw_tokens, pos_pairs):
        # pos_pairs is list of (word, tag) tuples from nltk.pos_tag
        actual_tag = pair[1] if isinstance(pair, tuple) else "NN"

        if not _is_clean(raw_token):
            continue

        wn_pos = _nltk_pos_to_wordnet(actual_tag)
        lemma  = _lemmatize(raw_token, wn_pos)
        stem   = _stem(lemma)

        if stem not in result:
            result[stem] = {
                "lemma":     lemma,
                "originals": [],
                "positions": [],
                "pos":       actual_tag,
            }
        entry = result[stem]
        entry["positions"].append(pos_counter)
        if raw_token not in entry["originals"]:
            entry["originals"].append(raw_token)

        pos_counter += 1

    return result


# ── Named Entity Recognition ───────────────────────────────────────────────────

def extract_named_entities(text: str) -> list:
    """
    Extract named entities from text using NLTK's ne_chunk pipeline.

    Returns list of dicts:
        [{"text": "Python Software Foundation", "label": "ORGANIZATION"}, ...]

    Labels include: PERSON, ORGANIZATION, GPE (geopolitical), LOCATION,
                    FACILITY, GSP, CARDINAL, etc.

    Falls back to [] when NER packages are unavailable.
    """
    if not _NER_OK:
        return []
    try:
        from nltk import pos_tag, ne_chunk, word_tokenize, Tree

        # Cap text for speed; NER is O(n) but chunker can be slow on huge texts
        sample = text[:NER_TEXT_LIMIT]
        tokens = word_tokenize(sample)
        tagged = pos_tag(tokens)
        tree   = ne_chunk(tagged, binary=False)

        entities = []
        seen: set = set()
        for subtree in tree:
            if isinstance(subtree, Tree):
                label       = subtree.label()
                entity_text = " ".join(leaf[0] for leaf in subtree.leaves())
                key         = (entity_text.lower(), label)
                if key not in seen:
                    seen.add(key)
                    entities.append({"text": entity_text, "label": label})

        return entities

    except Exception as exc:
        logger.debug(f"NER extraction failed: {exc}")
        return []


# ── Bigram extraction ──────────────────────────────────────────────────────────

def extract_bigrams(stemmed_tokens: list, min_freq: int = BIGRAM_MIN_FREQ) -> list:
    """
    Extract meaningful 2-gram phrases from a list of already-stemmed tokens.

    Returns list of "w1 w2" strings whose in-document frequency >= min_freq.
    For single-document indexing you may call with min_freq=1.
    """
    if len(stemmed_tokens) < 2:
        return []
    counts: Counter = Counter()
    for i in range(len(stemmed_tokens) - 1):
        counts[f"{stemmed_tokens[i]} {stemmed_tokens[i + 1]}"] += 1
    return [bg for bg, cnt in counts.items() if cnt >= min_freq]


# ── Heading extraction (from BeautifulSoup) ────────────────────────────────────

def extract_headings(soup) -> dict:
    """
    Extract h1–h6 heading texts from a BeautifulSoup object.

    Returns:
        {
            "h1": ["Main Title"],
            "h2": ["Section A", "Section B"],
            ...
        }
    """
    headings: dict = {}
    for level in range(1, 7):
        tag_name = f"h{level}"
        texts = [
            tag.get_text(strip=True)
            for tag in soup.find_all(tag_name)
            if tag.get_text(strip=True)
        ]
        if texts:
            headings[tag_name] = texts
    return headings


# ── Snippet generation ─────────────────────────────────────────────────────────

def build_snippet(content: str, query_terms: list, window: int = 40) -> str:
    """
    Extract the most relevant snippet from *content* for the given query_terms.

    Strategy:
      1. Find the token position of the first query-term hit.
      2. Extract a window of ±window words around that position.
      3. Return as a plain-text string.

    Falls back to the first *window*×2 words if no hit is found.
    """
    if not content:
        return ""

    words = content.split()
    if not words:
        return ""

    query_stems = set(query_terms)

    # Find best anchor position
    best_pos = None
    best_hits = 0
    for start in range(0, min(len(words), 3000), window):
        chunk = words[start: start + window]
        chunk_stems = {_stem(_lemmatize(w.lower())) for w in chunk if w.isalpha()}
        hits = len(query_stems & chunk_stems)
        if hits > best_hits:
            best_hits = hits
            best_pos  = start

    if best_pos is None:
        best_pos = 0

    start = max(0, best_pos - window // 2)
    end   = min(len(words), start + window * 2)
    snippet = " ".join(words[start:end])

    # Trim to sentence boundary if possible
    if start > 0 and not snippet[0].isupper():
        snippet = "…" + snippet
    if end < len(words):
        snippet += "…"

    return snippet


# ── Readability ────────────────────────────────────────────────────────────────

def _estimate_syllables(word: str) -> int:
    """Very rough syllable count heuristic."""
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def estimated_reading_level(text: str) -> float:
    """
    Flesch-Kincaid readability score (0–100, higher = easier to read).
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if not sentences:
        return 0.0
    words = tokenize(text)
    if not words:
        return 0.0
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables    = sum(_estimate_syllables(w) for w in words) / len(words)
    score = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
    return max(0.0, min(100.0, score))


def compute_keyword_density(text: str, keywords: list) -> float:
    """
    Fraction of cleaned tokens that are in *keywords* (lemmatized+stemmed).
    Used by the Panda scorer.
    """
    tokens = process_text(text)
    if not tokens:
        return 0.0
    kw_stems = {_stem(_lemmatize(k.lower())) for k in keywords}
    count = sum(1 for t in tokens if t in kw_stems)
    return count / len(tokens)
