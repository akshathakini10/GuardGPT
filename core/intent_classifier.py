# ============================================================
# GuardGPT - intent_classifier.py
# ============================================================
# Trained six-class intent classifier ONLY.
#
# Current-turn flow:
#   Prompt -> tokenizer -> fine-tuned MiniLM encoder
#   -> attention-mask mean pooling -> trained linear head
#   -> six-class softmax probabilities (c_t)
#
# The temporal/history model is intentionally NOT implemented here.
# This module produces the current-turn evidence c_t that the
# temporal module will consume later.
# ============================================================

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Stores security classification results for a user prompt."""
    prompt: str
    intent: str
    intent_confidence: float
    risk_level: str
    category_scores: Dict[str, float] = field(default_factory=dict)
    dataset_match_confidence: float = 0.0
    matched_record_id: Optional[str] = None
    matched_record_intent: Optional[str] = None
    final_blocked: bool = False
    block_reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    history_triggered: bool = False
    history_block_reason: str = ""


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_LABELS = [
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
]


class IntentClassifier:
    """Inference wrapper for GuardGPT's trained six-class intent model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name

        # Kept only for compatibility with existing SafetyService code that may
        # assign classifier._model. It is NOT used for intent classification.
        self._model = None

        self._tokenizer = None
        self._trained_encoder = None
        self._trained_head = None
        self._trained_labels: list[str] = []
        self._max_length = 256
        self._trained_loaded = False
        self._load_error: Optional[str] = None

    @property
    def labels(self) -> list[str]:
        self._ensure_loaded()
        return list(self._trained_labels)

    def _checkpoint_path(self) -> Path:
        return Path(
            os.getenv(
                "GUARDGPT_INTENT_MODEL",
                str(Path(__file__).resolve().parents[1] / "intent_classifier" / "best_model.pt"),
            )
        )

    def _ensure_loaded(self) -> None:
        if self._trained_loaded:
            if self._load_error:
                raise RuntimeError(self._load_error)
            return

        self._trained_loaded = True
        checkpoint_path = self._checkpoint_path()

        if not checkpoint_path.is_file():
            self._load_error = f"Trained intent checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(self._load_error)

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            labels = checkpoint.get("labels")
            state = checkpoint.get("model_state_dict")
            checkpoint_model_name = checkpoint.get("model_name", self.model_name)
            max_length = checkpoint.get("max_length", 256)

            if not isinstance(labels, list) or not labels:
                raise ValueError("Checkpoint is missing a valid 'labels' list.")
            if not isinstance(state, dict):
                raise ValueError("Checkpoint is missing 'model_state_dict'.")

            labels = [str(label) for label in labels]
            if set(labels) != set(EXPECTED_LABELS):
                raise ValueError(
                    "Unexpected trained labels. "
                    f"Expected {EXPECTED_LABELS}, found {labels}."
                )

            model_dir = checkpoint_path.parent

            # Prefer the tokenizer saved with training.
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                )
            except Exception:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint_model_name,
                        local_files_only=True,
                    )
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)

            # Load a dedicated encoder. Do NOT reuse DatasetLoader's SentenceTransformer
            # object because the checkpoint contains fine-tuned encoder weights and
            # FAISS retrieval should keep its own embedding model unchanged.
            try:
                encoder = AutoModel.from_pretrained(
                    checkpoint_model_name,
                    local_files_only=True,
                )
            except Exception:
                encoder = AutoModel.from_pretrained(checkpoint_model_name)

            encoder_state = {
                key.removeprefix("encoder."): value
                for key, value in state.items()
                if key.startswith("encoder.")
            }
            if not encoder_state:
                raise ValueError("Checkpoint contains no encoder.* weights.")

            load_result = encoder.load_state_dict(encoder_state, strict=False)
            if load_result.unexpected_keys:
                logger.warning(
                    "Unexpected encoder checkpoint keys: %s",
                    load_result.unexpected_keys,
                )

            weight = state.get("classifier.weight")
            bias = state.get("classifier.bias")
            if weight is None or bias is None:
                raise ValueError("Checkpoint is missing classifier.weight/classifier.bias.")

            head = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
            head.load_state_dict({"weight": weight, "bias": bias})

            encoder.eval()
            head.eval()

            self.model_name = str(checkpoint_model_name)
            self._max_length = int(max_length)
            self._trained_labels = labels
            self._tokenizer = tokenizer
            self._trained_encoder = encoder
            self._trained_head = head

            logger.info(
                "Loaded trained GuardGPT intent classifier from %s with labels=%s",
                checkpoint_path,
                labels,
            )

        except Exception as error:
            self._load_error = f"Failed to load trained intent classifier: {error}"
            logger.exception(self._load_error)
            raise RuntimeError(self._load_error) from error

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        """Attention-mask mean pooling, matching the training architecture."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _probabilities(self, text: str) -> Dict[str, float]:
        self._ensure_loaded()

        import torch

        encoded = self._tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        )

        with torch.no_grad():
            outputs = self._trained_encoder(**encoded)
            pooled = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            logits = self._trained_head(pooled)
            probabilities = torch.softmax(logits, dim=-1)[0]

        return {
            label: float(probabilities[index].item())
            for index, label in enumerate(self._trained_labels)
        }

    def classify(self, text: str) -> dict:
        """Return the current-turn trained intent prediction."""
        if not text or not text.strip():
            return {"intent": "unknown", "confidence": 0.0}

        scores = self._probabilities(text)
        intent = max(scores, key=scores.get)
        return {
            "intent": intent,
            "confidence": round(scores[intent], 4),
        }

    def classify_with_scores(self, text: str) -> dict:
        """Return prediction plus all six softmax scores (the c_t vector)."""
        if not text or not text.strip():
            return {"intent": "unknown", "confidence": 0.0, "scores": {}}

        scores = self._probabilities(text)
        intent = max(scores, key=scores.get)
        return {
            "intent": intent,
            "confidence": round(scores[intent], 4),
            "scores": {label: round(score, 6) for label, score in scores.items()},
        }

    def current_turn_vector(self, text: str) -> dict:
        """Explicit alias for the temporal model's current evidence vector c_t."""
        return self.classify_with_scores(text)["scores"]

    def predict(self, text: str) -> str:
        return self.classify(text)["intent"]

    def predict_intent(self, text: str) -> str:
        return self.predict(text)

    def top_intents(self, text: str, top_k: int = 3) -> list[dict]:
        """Return top-k trained intents ordered by softmax probability."""
        if not text or not text.strip():
            return []

        scores = self._probabilities(text)
        top_k = max(1, min(int(top_k), len(scores)))
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {"intent": intent, "score": round(score, 6)}
            for intent, score in ordered
        ]

    def reset(self) -> None:
        self._tokenizer = None
        self._trained_encoder = None
        self._trained_head = None
        self._trained_labels = []
        self._trained_loaded = False
        self._load_error = None
