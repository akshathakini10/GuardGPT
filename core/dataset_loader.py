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
DATASET_PATH = PROJECT_ROOT / "data" / "guardgpt_augmented_clean.json"
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_PATH = PROJECT_ROOT / "data" / "guardgpt_faiss.index"
RECORDS_PATH = PROJECT_ROOT / "data" / "guardgpt_id_map.json"
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

        self._load_prebuilt()

    def _load_prebuilt(self) -> None:
        """Validate the three artifacts before publishing any loaded state."""
        start = time.monotonic()
        index_path = self._path.parent / "guardgpt_faiss.index"
        map_path = self._path.parent / "guardgpt_id_map.json"
        for path in (self._path, index_path, map_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required dataset artifact missing: {path}")
        if self.max_records is not None:
            raise ValueError("max_records is unsupported for a prebuilt index.")
        if self._model_name not in {MODEL_NAME, f"sentence-transformers/{MODEL_NAME}"}:
            raise ValueError(f"This index requires {MODEL_NAME}.")
        with self._path.open(encoding="utf-8-sig") as file:
            dataset = json.load(file)
        with map_path.open(encoding="utf-8-sig") as file:
            mapping = json.load(file)
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Dataset must be a nonempty JSON list.")
        if not isinstance(mapping, dict):
            raise ValueError("ID map must be a dictionary keyed by vector number.")
        index = faiss.read_index(str(index_path))
        if not isinstance(index, faiss.IndexFlatIP) or index.d != 384:
            raise ValueError("Expected a 384-dimensional IndexFlatIP index.")
        if index.ntotal != len(dataset) or set(mapping) != {str(i) for i in range(index.ntotal)}:
            raise ValueError("Dataset, index and ID-map counts/keys do not match.")
        by_id = {}
        for record in dataset:
            if not isinstance(record, dict):
                raise ValueError("Each dataset record must be an object.")
            rid = record.get("request_id")
            if not isinstance(rid, str) or not rid or rid in by_id:
                raise ValueError("request_id must be a unique nonempty string.")
            if not isinstance(record.get("input_text"), str) or not record["input_text"].strip():
                raise ValueError(f"Missing input_text: {rid}")
            if record.get("intent") not in {"safe", "unsafe"}:
                raise ValueError(f"Expected safe/unsafe intent: {rid}")
            if record.get("target_verdict") not in {"ALLOW", "BLOCK", "SANITIZE"}:
                raise ValueError(f"Invalid target_verdict: {rid}")
            if (record["intent"] == "safe") != (record["target_verdict"] == "ALLOW"):
                raise ValueError(f"Conflicting intent and target_verdict: {rid}")
            scores = record.get("category_scores")
            if not isinstance(scores, dict) or not {"harm", "toxicity", "jailbreak", "prompt_injection"} <= scores.keys():
                raise ValueError(f"Missing category scores: {rid}")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
                raise ValueError(f"Invalid category score: {rid}")
            if record["intent"] == "safe" and any(scores[k] >= 0.5 for k in ("harm", "toxicity", "jailbreak", "prompt_injection")):
                raise ValueError(f"Safe label conflicts with high harm scores: {rid}")
            by_id[rid] = record
        records = [mapping[str(i)] for i in range(index.ntotal)]
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each ID-map value must be a full record.")
            rid = record.get("request_id")
            if not isinstance(rid, str) or rid in seen or by_id.get(rid) != record:
                raise ValueError("ID map has duplicate, missing or changed dataset records.")
            seen.add(rid)
        for i in np.linspace(0, index.ntotal - 1, min(100, index.ntotal), dtype=int):
            if not np.isclose(np.linalg.norm(index.reconstruct(int(i))), 1.0, atol=1e-3):
                raise ValueError("Index vectors must be L2 normalized.")
        try:
            model = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(self._model_name)
        dimension_method = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        if dimension_method() != index.d:
            raise ValueError("Embedding model dimension does not match index.")
        self._model, self._index, self._records = model, index, records
        self._texts = [r["input_text"] for r in records]
        self._load_time = time.monotonic() - start
        self._loaded = True
        logger.info("Loaded validated prebuilt index: %d records", len(records))


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
        """Release memory; never delete the supplied dataset artifacts."""
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
