# ============================================================
# GuardGPT - dataset_loader.py
# ============================================================
# PURPOSE:
#   Load the harm dataset (JSON file) into memory once,
#   build a TF-IDF search index over all records,
#   and answer "which record best matches this prompt?" queries.
#
# HOW TF-IDF WORKS (simple explanation):
#   - Rare words get HIGH weight  (e.g. "ransomware" → very significant)
#   - Common words get LOW weight (e.g. "the" → filtered out as stopword)
#   - When a prompt arrives, we find the dataset record that shares
#     the most high-weight words with it.
# ============================================================

import json, math, re, time, logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the dataset file (must be in the Engine/ root folder)
DATASET_PATH = Path(__file__).parent.parent / "harm_only_400k_dataset.json"

# ── Stopwords ─────────────────────────────────────────────────────────────────
# These common English words appear everywhere and carry NO safety signal.
# Removing them dramatically improves matching accuracy.
# Example: "fuck you" → after removing "you" → only "fuck" remains
#          → correctly matches hate_speech records instead of any record with "you"
_STOPWORDS = frozenset({
    # Articles, prepositions, conjunctions
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","by","from","into","about","up","out","than","then","there",
    # Common verbs
    "is","was","are","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might",
    "get","got","go","going","know","want","need","make","use",
    "tell","give","show","see","look","come","take","help","said","say",
    # Pronouns
    "i","you","he","she","we","they","it","me","him","her","us","them",
    "my","your","his","our","their","its","this","that","these","those",
    # Common filler words
    "what","which","who","how","when","where","why","if","as","so",
    "not","no","can","just","now","also","more","please","am","very",
    "some","any","all","one","two","new","good","other","like",
})


def _tokenize(text: str) -> list[str]:
    """
    Convert any text into a list of clean, meaningful tokens.

    Steps:
      1. Lowercase everything
      2. Remove punctuation (keep only letters, numbers, spaces)
      3. Split into words
      4. Remove short words (1 char) and stopwords

    Example:
      Input : "How do I hack into someone's account?"
      Output: ["hack", "account"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # remove punctuation
    tokens = [t for t in text.split() if len(t) > 1]  # remove 1-char words
    return [t for t in tokens if t not in _STOPWORDS]  # remove stopwords


# ── DatasetLoader class ───────────────────────────────────────────────────────
class DatasetLoader:
    """
    Loads the harm dataset once and answers record-matching queries.

    Usage:
        loader = DatasetLoader()
        loader.load()                    # loads dataset + builds index
        record = loader.query(prompt)    # finds best matching record
    """

    def __init__(self, path: str | Path = DATASET_PATH) -> None:
        self._path      = Path(path)
        self._records   = []              # all dataset records in memory
        self._index     = defaultdict(list)  # token → [record indices]
        self._idf       = {}              # token → IDF weight
        self._loaded    = False
        self._load_time = 0.0

    def load(self) -> None:
        """
        Load dataset from disk and build the TF-IDF index.
        Calling this multiple times is safe — it only loads once.
        """
        if self._loaded:
            return  # already loaded, skip

        if not self._path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._path}\n"
                "Place harm_only_400k_dataset.json in the Engine/ folder."
            )

        logger.info("Loading dataset from %s ...", self._path)
        t0 = time.monotonic()

        with open(self._path, "r", encoding="utf-8") as f:
            self._records = json.load(f)

        logger.info("Loaded %d records. Building TF-IDF index ...", len(self._records))
        self._build_index()

        self._load_time = time.monotonic() - t0
        self._loaded = True
        logger.info(
            "Dataset ready in %.2fs | %d unique tokens indexed",
            self._load_time, len(self._index)
        )

    def query(self, text: str) -> Optional[dict]:
        """
        Find the dataset record that best matches the given prompt.

        How it works:
          1. Tokenize the prompt (remove stopwords)
          2. For each token, find all records that contain it
          3. Score each record by summing IDF weights of shared tokens
          4. Return the highest-scoring record

        Returns None if no tokens match anything in the dataset.
        """
        if not self._loaded:
            self.load()

        tokens = _tokenize(text)
        if not tokens:
            return None  # nothing meaningful in the prompt

        # Score every record that shares at least one token with the prompt
        scores = defaultdict(float)
        for token in tokens:
            if token not in self._index:
                continue  # this word is not in the dataset at all
            for record_idx in self._index[token]:
                scores[record_idx] += self._idf.get(token, 0.0)

        if not scores:
            return None  # no matching records found

        # Return the record with the highest total score
        best_idx = max(scores, key=lambda k: scores[k])
        return self._records[best_idx]

    def _build_index(self) -> None:
        """
        Build the inverted index and compute IDF weights.

        For each record:
          - Tokenize its input_text
          - Map each token → record index
          - Count how many records contain each token (document frequency)

        Then compute IDF for each token:
          IDF = log(total_records / records_containing_token) + 1
          → rare tokens get high IDF, common tokens get low IDF
        """
        doc_freq = defaultdict(int)  # token → count of records containing it
        n = len(self._records)

        for idx, record in enumerate(self._records):
            # Use a SET so each token is counted once per record
            tokens = set(_tokenize(record.get("input_text", "")))
            for token in tokens:
                self._index[token].append(idx)
                doc_freq[token] += 1

        # Compute smoothed IDF for every token
        for token, df in doc_freq.items():
            self._idf[token] = math.log(n / df) + 1.0

    # ── Read-only properties ──────────────────────────────────────────────────
    @property
    def is_loaded(self)    -> bool:  return self._loaded
    @property
    def record_count(self) -> int:   return len(self._records)
    @property
    def load_time(self)    -> float: return self._load_time