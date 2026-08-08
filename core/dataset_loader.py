import json
import logging
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "harm_only_400k_dataset.json"
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_PATH = CACHE_DIR / "guardgpt_faiss.index"
RECORDS_PATH = CACHE_DIR / "guardgpt_records.json"
MODEL_NAME = "all-MiniLM-L6-v2"


class DatasetLoader:
    """Loads dataset and executes vector search via Sentence-BERT + FAISS."""

    def __init__(
        self,
        path: str | Path = DATASET_PATH,
        model_name: str = MODEL_NAME,
        max_records: Optional[int] = None,  # Full dataset indexing enabled
    ) -> None:
        self._path = Path(path)
        self._model_name = model_name
        self.max_records = max_records
        self._records: list[dict] = []
        self._texts: list[str] = []
        self._model: Optional[SentenceTransformer] = None
        self._index = None
        self._loaded = False
        self._load_time = 0.0

    def load(self) -> None:
        if self._loaded:
            return

        start_time = time.monotonic()

        if not self._path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._path}\n"
                "Place harm_only_400k_dataset.json in the project root folder."
            )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Loading Sentence-BERT model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

        if INDEX_PATH.exists() and RECORDS_PATH.exists():
            try:
                logger.info("Loading cached FAISS index...")
                self._index = faiss.read_index(str(INDEX_PATH))
                with open(RECORDS_PATH, "r", encoding="utf-8") as file:
                    self._records = json.load(file)

                self._texts = [str(r.get("input_text", "")) for r in self._records]

                if self._index.ntotal != len(self._records):
                    raise ValueError("Cached FAISS index and records size mismatch.")

                self._loaded = True
                self._load_time = time.monotonic() - start_time
                logger.info("Cached index loaded successfully (%d records).", len(self._records))
                return
            except Exception as error:
                logger.warning("Could not use cached index: %s. Rebuilding...", error)

        logger.info("Loading dataset from %s", self._path)
        with open(self._path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            self._records = data.get("records", [])
        elif isinstance(data, list):
            self._records = data
        else:
            raise ValueError("Invalid dataset format.")

        if not self._records:
            raise ValueError("Dataset contains no records.")

        if self.max_records is not None and len(self._records) > self.max_records:
            logger.info("Limiting dataset to first %d records.", self.max_records)
            self._records = self._records[: self.max_records]

        self._texts = [str(r.get("input_text", "")) for r in self._records]
        logger.info("Generating embeddings for %d records...", len(self._records))

        embeddings = self._model.encode(
            self._texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")
        dimension = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        try:
            faiss.write_index(self._index, str(INDEX_PATH))
            with open(RECORDS_PATH, "w", encoding="utf-8") as file:
                json.dump(self._records, file, ensure_ascii=False)
            logger.info("FAISS index saved to cache.")
        except OSError as error:
            logger.warning("Could not save cache: %s", error)

        self._load_time = time.monotonic() - start_time
        self._loaded = True

    def query(self, text: str, top_k: int = 1) -> Optional[dict]:
        if not self._loaded:
            self.load()
        if not text or not text.strip():
            return None

        query_embedding = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")

        top_k = max(1, min(top_k, len(self._records)))
        similarities, indices = self._index.search(query_embedding, top_k)

        if indices.size == 0 or indices[0][0] < 0:
            return None

        best_index = int(indices[0][0])
        best_score = float(similarities[0][0])

        record = dict(self._records[best_index])
        record["_similarity"] = round(best_score, 4)
        return record

    def query_top_k(self, text: str, top_k: int = 5) -> list[dict]:
        if not self._loaded:
            self.load()
        if not text or not text.strip():
            return []

        query_embedding = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        top_k = max(1, min(top_k, len(self._records)))
        similarities, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, index in zip(similarities[0], indices[0]):
            if index < 0:
                continue
            record = dict(self._records[int(index)])
            record["_similarity"] = round(float(score), 4)
            results.append(record)

        return results

    def clear_cache(self) -> None:
        for cache_file in (INDEX_PATH, RECORDS_PATH):
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except OSError as error:
                logger.warning("Could not delete %s: %s", cache_file, error)

        self._index = None
        self._records = []
        self._texts = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def embedding_dimension(self) -> int:
        return 0 if self._index is None else self._index.d