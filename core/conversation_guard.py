"""Per-session conversation state for GuardGPT temporal intent processing."""

from dataclasses import dataclass
from typing import Any

from core.temporal_intent import TemporalIntent


@dataclass
class ConversationResult:
    history_triggered: bool
    history_block_reason: str
    unsafe_ratio: float
    previous_block: bool
    session_flagged: bool
    session_flag_reason: str
    turn_index: int

    # Temporal intent outputs
    contextual_intent: str = "unknown"
    contextual_confidence: float = 0.0
    contextual_vector: dict | None = None
    history_vector: dict | None = None
    risk_vector: dict | None = None
    gate: float | None = None
    learned_temporal_model: bool = False


class ConversationGuard:
    """
    Maintains temporal intent state for exactly one conversation/session.

    The previous sticky unsafe-ratio mechanism has been removed. History is now
    represented by TemporalIntent state (H_t and I_t).

    Important:
    The installed TemporalIntent implements the learned temporal equations and historical-context
    update H_t and loads trained W matrices/biases
    exist from the temporal checkpoint. Its contextual
    I_t is the learned context-aware intent used downstream.
    """

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.temporal = TemporalIntent()
        self._turn_counter = 0

    def evaluate_result(self, result: Any) -> ConversationResult:
        current_vector = self._get_value(result, "current_intent_vector", {})
        risk_vector = self._get_value(result, "risk_vector", None)

        temporal_result = self.temporal.update(
            current_vector=current_vector,
            risk_vector=risk_vector,
        )

        self._turn_counter += 1

        return ConversationResult(
            history_triggered=False,
            history_block_reason="",
            unsafe_ratio=0.0,
            previous_block=False,
            session_flagged=False,
            session_flag_reason="",
            turn_index=self._turn_counter,
            contextual_intent=temporal_result.contextual_intent,
            contextual_confidence=temporal_result.contextual_confidence,
            contextual_vector=dict(temporal_result.contextual_vector),
            history_vector=dict(temporal_result.history_vector),
            risk_vector=dict(temporal_result.risk_vector),
            gate=temporal_result.gate,
            learned_temporal_model=temporal_result.learned_temporal_model,
        )

    def reset(self) -> None:
        self.temporal.reset()
        self._turn_counter = 0

    @staticmethod
    def _get_value(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def is_flagged(self) -> bool:
        # Retained only for compatibility with callers of the old guard API.
        return False

    @property
    def flag_reason(self) -> str:
        return ""

    @property
    def history(self) -> list:
        # Raw prompt history is no longer needed for temporal intent state.
        return []

    @property
    def unsafe_ratio(self) -> float:
        # Retained only for compatibility with callers of the old guard API.
        return 0.0
