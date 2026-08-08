# ============================================================
# GuardGPT - guard_engine.py
# ============================================================
# PURPOSE:
#   Main controller for the GuardGPT security pipeline.
#
# PIPELINE:
#   User Prompt -> IntentClassifier -> DatasetLoader (FAISS)
#   -> ConversationGuard -> DecisionEngine -> ALLOW / BLOCK -> Llama
# ============================================================

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from core.conversation_guard import ConversationGuard
from core.dataset_loader import DatasetLoader
from core.decision_engine import DecisionEngine, DecisionOutput
from core.intent_classifier import GuardResult, IntentClassifier
from core.llama_backend import LlamaBackend

logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are GuardGPT, a helpful and responsible AI assistant. "
    "Answer the user's question clearly and accurately. "
    "Do not provide harmful or unsafe instructions."
)


# ============================================================
# TERMINAL COLOURS
# ============================================================

_COLOURS = {
    "safe": "\033[92m",
    "low": "\033[93m",
    "medium": "\033[33m",
    "high": "\033[91m",
    "critical": "\033[1;91m",
}

_RESET = "\033[0m"


# ============================================================
# ENGINE RESPONSE
# ============================================================

@dataclass
class EngineResponse:
    """Complete response returned by GuardEngine."""

    allowed: bool
    intent: str
    risk_level: str
    decision: DecisionOutput
    llama_response: Optional[str] = None
    blocked_message: Optional[str] = None
    error: Optional[str] = None


# ============================================================
# GUARD ENGINE
# ============================================================

class GuardEngine:
    """Main GuardGPT controller."""

    def __init__(
        self,
        session_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())

        self._loader = DatasetLoader()
        self._classifier = IntentClassifier()
        self._guard = ConversationGuard(self.session_id)
        self._decision = DecisionEngine()
        self._llama = LlamaBackend()
        self._ready = False

        logger.info("GuardEngine created | session=%s", self.session_id)

    def startup(self) -> None:
        """Prepare GuardGPT components and models."""
        if self._ready:
            return

        logger.info("Starting GuardGPT...")

        try:
            self._loader.load()

            # Share Sentence-BERT model instance to optimize startup and memory
            if hasattr(self._loader, "_model") and self._loader._model is not None:
                self._classifier._model = self._loader._model

            self._ready = True
            logger.info("GuardGPT ready | Dataset: %d records", self._loader.record_count)

        except Exception as error:
            logger.exception("GuardGPT startup failed.")
            raise RuntimeError(f"GuardGPT could not start: {error}") from error

    def process(
        self,
        prompt: str,
    ) -> EngineResponse:
        """Process one prompt through the complete GuardGPT pipeline."""
        if not self._ready:
            self.startup()

        prompt = prompt.strip() if prompt else ""

        if not prompt:
            return self._empty_response()

        # Step 1: Intent Classification
        try:
            classifier_output = self._classifier.classify(prompt)
        except Exception as error:
            logger.exception("Intent classification failed.")
            return self._error_response(prompt, f"Intent classification failed: {error}")

        result = self._build_guard_result(prompt, classifier_output)

        # Step 2: Semantic Dataset Matching
        try:
            dataset_record = self._loader.query(prompt)
        except Exception as error:
            logger.warning("FAISS semantic search failed: %s", error)
            dataset_record = None

        if dataset_record:
            result.dataset_match_confidence = float(dataset_record.get("_similarity", 0.0) or 0.0)
            result.matched_record_id = self._get_record_id(dataset_record)
            result.matched_record_intent = dataset_record.get("intent")
            result.category_scores = self._extract_category_scores(dataset_record)
        else:
            result.dataset_match_confidence = 0.0
            result.matched_record_id = None
            result.matched_record_intent = None
            result.category_scores = {}

        # Step 3: Combine Intent + Semantic Information
        result.risk_level = self._estimate_risk(
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            similarity=result.dataset_match_confidence,
        )

        result.final_blocked = self._should_preliminarily_block(result)

        if result.final_blocked and not result.block_reason:
            result.block_reason = f"High-risk intent detected: {result.intent}"

        # Step 4: Conversation Guard Multi-turn Tracking
        conversation = self._guard.evaluate_result(result)
        result.history_triggered = conversation.history_triggered
        result.history_block_reason = conversation.history_block_reason

        if conversation.history_triggered:
            result.final_blocked = True
            if conversation.history_block_reason:
                result.reason_codes.append("history_unsafe")

        # Step 5: Final Decision
        decision = self._decision.decide(
            result,
            turn_index=conversation.turn_index,
        )

        # Step 6: Blocked Request
        if not decision.allowed:
            return EngineResponse(
                allowed=False,
                intent=decision.intent,
                risk_level=decision.risk_level,
                decision=decision,
                blocked_message=decision.user_message,
            )

        # Step 7: Allowed Request -> Llama
        llama_text, error = self._call_llama(prompt)

        return EngineResponse(
            allowed=True,
            intent=decision.intent,
            risk_level=decision.risk_level,
            decision=decision,
            llama_response=llama_text,
            error=error,
        )

    @staticmethod
    def _build_guard_result(prompt: str, classifier_output) -> GuardResult:
        if isinstance(classifier_output, GuardResult):
            classifier_output.prompt = prompt
            return classifier_output

        if isinstance(classifier_output, dict):
            intent = str(classifier_output.get("intent", "unknown"))
            confidence = float(
                classifier_output.get(
                    "confidence", classifier_output.get("intent_confidence", 0.0)
                ) or 0.0
            )
            risk_level = str(classifier_output.get("risk_level", "safe"))

            return GuardResult(
                prompt=prompt,
                intent=intent,
                intent_confidence=confidence,
                risk_level=risk_level,
                category_scores=dict(classifier_output.get("category_scores", {}) or {}),
                dataset_match_confidence=0.0,
                matched_record_id=None,
                matched_record_intent=None,
                final_blocked=False,
                block_reason="",
                reason_codes=list(classifier_output.get("reason_codes", []) or []),
                history_triggered=False,
                history_block_reason="",
            )

        raise TypeError("IntentClassifier.classify() must return a GuardResult or dictionary.")

    @staticmethod
    def _should_preliminarily_block(result: GuardResult) -> bool:
        """Optimized preliminary block thresholds based on confidence and similarity."""
        intent = result.intent
        confidence = result.intent_confidence
        similarity = result.dataset_match_confidence

        if intent in {"self_harm", "self_harm_risk"}:
            return confidence >= 0.40

        if intent in {"harmful", "cyber_abuse", "illegal", "prompt_injection", "jailbreak"}:
            return confidence >= 0.45

        matched_record_is_safe = (
            result.matched_record_intent == "safe"
            or (result.category_scores.get("safe", 0.0) > 0.5)
        )

        is_educational = intent in {"educational", "coding", "benign"}
        threshold = 0.85 if is_educational else 0.70

        if similarity >= threshold and not matched_record_is_safe:
            return True

        return False

    @staticmethod
    def _estimate_risk(intent: str, intent_confidence: float, similarity: float) -> str:
        """Estimate risk level based on normalized metrics."""
        if intent in {"self_harm", "self_harm_risk"}:
            return "critical" if intent_confidence >= 0.50 else "high"

        if intent in {"harmful", "cyber_abuse", "illegal", "prompt_injection", "jailbreak"}:
            return "high" if intent_confidence >= 0.55 else "medium"

        is_educational = intent in {"educational", "coding", "benign"}
        high_threshold = 0.85 if is_educational else 0.70
        med_threshold = 0.80 if is_educational else 0.55

        if similarity >= high_threshold:
            return "high"
        if similarity >= med_threshold:
            return "medium"
        if similarity >= 0.40:
            return "low"

        return "safe"

    @staticmethod
    def _get_record_id(record: dict) -> Optional[str]:
        for key in ("id", "record_id", "prompt_id", "uuid"):
            value = record.get(key)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _extract_category_scores(record: dict) -> dict:
        for key in ("category_scores", "scores", "safety_scores"):
            value = record.get(key)
            if isinstance(value, dict):
                return dict(value)
        return {}

    def new_conversation(self) -> None:
        """Reset conversation history."""
        self._guard.reset()
        logger.info("Conversation reset | session=%s", self.session_id)

    def _call_llama(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        try:
            response = self._llama.generate(prompt, system_prompt=SYSTEM_PROMPT)
            return response, None
        except Exception as error:
            return None, str(error)

    def _empty_response(self) -> EngineResponse:
        decision = DecisionOutput(
            allowed=False,
            intent="empty",
            risk_level="safe",
            user_message="Please enter a message.",
            technical_reason="Empty input.",
            category_scores={},
            reason_codes=["empty_input"],
            dataset_match_confidence=0.0,
            matched_record_id=None,
            history_triggered=False,
            turn_index=self._guard.turn_count + 1,
        )
        return EngineResponse(
            allowed=False,
            intent="empty",
            risk_level="safe",
            decision=decision,
            blocked_message="Please enter a message.",
        )

    def _error_response(self, prompt: str, error: str) -> EngineResponse:
        decision = DecisionOutput(
            allowed=False,
            intent="unknown",
            risk_level="high",
            user_message="The prompt could not be safely classified and was blocked.",
            technical_reason=error,
            category_scores={},
            reason_codes=["classifier_error"],
            dataset_match_confidence=0.0,
            matched_record_id=None,
            history_triggered=False,
            turn_index=self._guard.turn_count + 1,
        )
        return EngineResponse(
            allowed=False,
            intent="unknown",
            risk_level="high",
            decision=decision,
            blocked_message=decision.user_message,
            error=error,
        )

    def print_status(self) -> None:
        """Print current GuardGPT system status."""
        try:
            llama_available = self._llama.is_available()
        except Exception:
            llama_available = False

        print(
            "\n"
            "================ GuardGPT Status ================\n"
            f"Session ID : {self.session_id}\n"
            f"Ready      : {self._ready}\n"
            f"Turns      : {self._guard.turn_count}\n"
            f"Flagged    : {self._guard.is_flagged}\n"
            f"Dataset    : {self._loader.record_count:,} records\n"
            f"Llama      : {llama_available}\n"
            "=================================================\n"
        )

    def run_interactive(self) -> None:
        """Start GuardGPT interactive terminal mode."""
        self.startup()
        _print_banner()

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
                print("\nGoodbye!")
                break

            if user_input.lower() in {"/new", "/reset"}:
                self.new_conversation()
                print("\nConversation history cleared.")
                continue

            if user_input.lower() == "/status":
                self.print_status()
                continue

            response = self.process(user_input)
            print_response(response)


# ============================================================
# RESPONSE DISPLAY & HELPER FUNCTIONS
# ============================================================

def print_response(response: EngineResponse) -> None:
    """Display an EngineResponse in the terminal."""
    decision = response.decision
    colour = _COLOURS.get(response.risk_level, "")

    print("\n" + "-" * 60)
    print(f"Intent     : {response.intent}")

    if hasattr(decision, "intent_confidence"):
        print(f"Intent Conf: {decision.intent_confidence:.1%}")

    print(f"Risk       : {colour}{response.risk_level.upper()}{_RESET}")
    print(f"Similarity : {decision.dataset_match_confidence:.1%}")

    if response.allowed:
        print("Decision   : \033[92mALLOWED\033[0m")
        if response.llama_response:
            print(f"\nGuardGPT: {response.llama_response}")
        elif response.error:
            print(f"\nLlama unavailable: {response.error}")
    else:
        print("Decision   : \033[91mBLOCKED\033[0m")
        print(f"\nGuardGPT: {response.blocked_message}")

    if decision.reason_codes:
        print("\nReason     : " + ", ".join(decision.reason_codes))

    print("-" * 60)


def _print_banner() -> None:
    """Display the GuardGPT terminal banner."""
    print(
        "\n"
        "==============================================\n"
        "              GuardGPT\n"
        "      Intelligent Prompt Analysis\n"
        "==============================================\n"
        "Commands:\n"
        "  /new     Reset conversation\n"
        "  /status  Show system status\n"
        "  exit     Quit\n"
        "=============================================="
    )