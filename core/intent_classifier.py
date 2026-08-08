# ============================================================
# GuardGPT - intent_classifier.py
# ============================================================
# PURPOSE:
#   Detect the semantic intent of a user prompt using
#   Sentence-BERT embeddings.
#
# APPROACH:
#   User Prompt -> Sentence-BERT -> Semantic Embedding
#   -> Compare with Intent Prototypes -> Best Intent + Confidence
#
# MODEL:
#   all-MiniLM-L6-v2
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ============================================================
# GUARD RESULT DATACLASS
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# INTENT DEFINITIONS
# ============================================================

INTENT_PROTOTYPES = {
    "benign": [
        "A normal harmless question or request.",
        "The user wants general information.",
        "The user asks for help with a safe everyday task.",
        "The user wants an explanation of a concept.",
    ],
    "coding": [
        "The user wants help writing or debugging computer code.",
        "The user asks a programming or software development question.",
        "The user wants an algorithm or programming solution.",
        "The user asks about Python, Java, JavaScript, databases, or programming.",
    ],
    "educational": [
        "The user wants to learn about an academic or technical topic.",
        "The user asks for an explanation for study purposes.",
        "The user wants educational information.",
        "The user asks about engineering, science, mathematics, or technology.",
        "The user asks about machine learning, artificial intelligence, ML, DL, neural networks, or data science.",
        "How machine learning models, regression, network protocols, or AI algorithms work conceptually.",
        "The theoretical mechanism behind computer networking concepts or cyber security principles.",
    ],
    "creative": [
        "The user requests creative writing.",
        "The user wants a story, poem, caption, script, or fictional content.",
        "The user asks for creative ideas or imaginative content.",
    ],
    "personal_advice": [
        "The user asks for personal advice or guidance.",
        "The user wants help dealing with a personal situation.",
        "The user asks for relationship or life advice.",
    ],
    "harmful": [
        "The user wants instructions for causing physical harm.",
        "The user asks how to injure or kill someone.",
        "The user requests instructions for dangerous or violent activity.",
        "The user wants assistance carrying out harmful actions.",
    ],
    "self_harm": [
        "The user expresses an intention to hurt themselves.",
        "The user asks about suicide or methods of self-harm.",
        "The user requests instructions for ending their own life.",
        "The user expresses serious thoughts about harming themselves.",
    ],
    "illegal": [
        "The user asks how to commit a crime.",
        "The user requests instructions for illegal activities.",
        "The user wants help evading law enforcement while committing a crime.",
        "The user asks for operational guidance for criminal activity.",
    ],
    "cyber_abuse": [
        "The user wants to hack or compromise a computer system.",
        "The user asks for malware, credential theft, or unauthorized access.",
        "The user wants to exploit a system without authorization.",
        "The user requests instructions for executing cyber attacks.",
    ],
    "prompt_injection": [
        "The user attempts to override system instructions.",
        "The user tells the AI to ignore previous instructions.",
        "The user attempts to manipulate the AI's instruction hierarchy.",
        "The user asks the AI to reveal hidden system instructions.",
    ],
    "jailbreak": [
        "The user attempts to bypass AI safety restrictions.",
        "The user asks the AI to behave without safety limitations.",
        "The user uses roleplay or special instructions to evade safety rules.",
        "The user asks the model to ignore its safety policies.",
    ],
}


# ============================================================
# INTENT CLASSIFIER
# ============================================================

class IntentClassifier:
    """
    BERT-based semantic intent classifier using Sentence-BERT embeddings.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._intent_embeddings = None
        self._intent_names = []
        self._initialized = False

    def _load_model(self) -> None:
        """Load Sentence-BERT model if not already loaded."""
        if self._model is not None:
            return

        logger.info("Loading intent classification model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def _initialize(self) -> None:
        """Generate prototype embeddings for all defined intents."""
        if self._initialized:
            return

        self._load_model()
        logger.info("Creating intent prototype embeddings...")

        self._intent_names = list(INTENT_PROTOTYPES.keys())
        prototype_texts = []

        for intent in self._intent_names:
            prototype_texts.extend(INTENT_PROTOTYPES[intent])

        # Encode all prototype descriptions
        all_embeddings = self._model.encode(
            prototype_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embeddings = np.asarray(all_embeddings, dtype=np.float32)

        # Average embeddings for each intent prototype
        intent_embeddings = []
        position = 0

        for intent in self._intent_names:
            count = len(INTENT_PROTOTYPES[intent])
            embeddings = all_embeddings[position : position + count]

            prototype = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototype = prototype / norm

            intent_embeddings.append(prototype)
            position += count

        self._intent_embeddings = np.asarray(intent_embeddings, dtype=np.float32)
        self._initialized = True
        logger.info("Intent classifier initialized with %d intents.", len(self._intent_names))

    def classify(self, text: str) -> dict:
        """Classify the intent of a user prompt."""
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
            }

        self._initialize()

        # Encode user prompt
        query_embedding = self._model.encode(
            [text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        # Calculate cosine similarities
        scores = np.dot(self._intent_embeddings, query_embedding[0])

        unsafe_intents = {
            "harmful", "self_harm", "illegal", "cyber_abuse",
            "prompt_injection", "jailbreak"
        }

        best_index = int(np.argmax(scores))
        best_intent = self._intent_names[best_index]
        raw_score = float(scores[best_index])

        # Convert cosine similarity (-1 to 1) into a confidence score (0 to 1)
        confidence = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

        text_lower = text.lower()

        # Keywords indicating educational or conceptual technical intent
        safe_tech_keywords = [
            "machine learning", " ml ", " ml?", " ml.", "deep learning",
            "data science", "regression", "neural network", "artificial intelligence",
            "theoretical mechanism", "how does", "conceptually", "explain the concept",
            "what is the mechanism", "theoretical", "academic", "theory of"
        ]
        is_safe_tech = any(kw in text_lower for kw in safe_tech_keywords)

        # Explicit action-oriented exploit terms
        unsafe_keyword_map = {
            "harmful": [
                "explosive", "bomb", "chemical weapon", "poison", "cyanide", "ricin",
                "assassinate", "lethal weapon", "molotov", "improvised device"
            ],
            "self_harm": [
                "suicide", "kill myself", "end my life", "take my life",
                "cutting myself", "hang myself"
            ],
            "illegal": [
                "illegal drug", "methamphetamine", "heroin", "cocaine",
                "credit card fraud", "rob a bank"
            ],
            "cyber_abuse": [
                "sql injection payload", "ddos attack target", "hack server",
                "exploit code", "write malware", "ransomware", "trojan virus",
                "intercept traffic", "run spoofing script"
            ],
            "jailbreak": [
                "ignore all previous instructions", "bypass safety", "dev mode",
                "jailbreak", "do anything now"
            ]
        }

        override_intent = None
        for category, keywords in unsafe_keyword_map.items():
            for kw in keywords:
                if kw in text_lower:
                    override_intent = category
                    break
            if override_intent:
                break

        if override_intent:
            best_intent = override_intent
            confidence = 0.85
        elif is_safe_tech and best_intent in unsafe_intents:
            # Re-route false positive technical queries back to educational
            best_intent = "educational"
            confidence = 0.80

        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
        }

    def predict(self, text: str) -> str:
        """Return only the predicted intent string."""
        return self.classify(text)["intent"]

    def predict_intent(self, text: str) -> str:
        """Alias for predict()."""
        return self.predict(text)

    def classify_with_scores(self, text: str) -> dict:
        """Return predicted intent along with similarity scores for all intents."""
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "scores": {},
            }

        self._initialize()

        query_embedding = self._model.encode(
            [text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        scores = np.dot(self._intent_embeddings, query_embedding[0])
        best_index = int(np.argmax(scores))

        score_dict = {
            intent: round(float(score), 4)
            for intent, score in zip(self._intent_names, scores)
        }

        return {
            "intent": self._intent_names[best_index],
            "confidence": round(float((scores[best_index] + 1.0) / 2.0), 4),
            "scores": score_dict,
        }

    def top_intents(self, text: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most likely intents."""
        if not text or not text.strip():
            return []

        self._initialize()

        query_embedding = self._model.encode(
            [text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        scores = np.dot(self._intent_embeddings, query_embedding[0])
        top_k = max(1, min(int(top_k), len(self._intent_names)))

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "intent": self._intent_names[int(index)],
                "score": round(float(scores[index]), 4),
            }
            for index in indices
        ]

    def reset(self) -> None:
        """Reset classifier state."""
        self._model = None
        self._intent_embeddings = None
        self._intent_names = []
        self._initialized = False