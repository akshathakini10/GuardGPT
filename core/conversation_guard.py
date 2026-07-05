# ============================================================
# GuardGPT - conversation_guard.py
# ============================================================
# PURPOSE:
#   Track conversation history and block harmful follow-up messages.
#
#   Even if a single message looks safe on its own, this module
#   checks the FULL conversation context before allowing it through.
#
# WHY THIS IS NEEDED:
#   Attackers often use multi-turn tricks like:
#     Turn 1: "how do I hack a system?"       ← blocked
#     Turn 2: "this is for educational purposes" ← looks safe alone
#   Without history tracking, Turn 2 would pass. This module catches it.
#
# RULES (checked in order for every new message):
#   1. Session permanently flagged → block everything
#   2. Bypass phrase after a prior block → block
#   3. Continued unsafe intent after a prior block → block
#   4. Ambiguous follow-up when ≥40% of recent turns were unsafe → block
#   5. If ≥70% of recent turns were unsafe → permanently flag session
# ============================================================

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from core.intent_classifier import GuardResult, UNSAFE_INTENTS

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
HISTORY_WINDOW    = 10    # remember the last 10 turns
UNSAFE_RATIO_WARN = 0.40  # block ambiguous follow-ups if ≥40% turns were unsafe
UNSAFE_RATIO_FLAG = 0.70  # permanently lock session if ≥70% turns were unsafe


# ── TurnRecord dataclass ──────────────────────────────────────────────────────
@dataclass
class TurnRecord:
    """
    A snapshot of one conversation turn, stored in the history window.
    Used to evaluate follow-up messages in context.
    """
    turn_index:      int    # which turn number (1, 2, 3...)
    timestamp:       str    # when this turn happened (UTC)
    prompt_snippet:  str    # first 80 chars of the user's message
    intent:          str    # what intent was detected
    risk_level:      str    # safe / low / medium / high / critical
    is_blocked:      bool   # was this turn blocked?
    block_reason:    str    # why it was blocked (empty if allowed)
    category_scores: dict = field(default_factory=dict)  # harm/jailbreak scores


# ── ConversationGuard class ───────────────────────────────────────────────────
class ConversationGuard:
    """
    Tracks conversation history and applies multi-turn safety rules.

    One instance per user session. Call evaluate() for every turn
    BEFORE passing to DecisionEngine.

    Usage:
        guard = ConversationGuard(session_id="abc123")
        result = guard.evaluate(guard_result)  # adds history context
    """

    def __init__(self, session_id: str) -> None:
        self.session_id    = session_id
        self._history      = deque(maxlen=HISTORY_WINDOW)  # rolling window
        self._turn_counter = 0
        self._flagged      = False   # True = session permanently locked
        self._flag_reason  = ""

    def evaluate(self, result: GuardResult) -> GuardResult:
        """
        Check the current prompt against conversation history.
        Adds history_triggered and history_block_reason to result if needed.
        """

        # ── Rule 1: Session is permanently flagged ────────────────────────
        # Once flagged, EVERY message is blocked — no exceptions.
        if self._flagged:
            result.history_triggered    = True
            result.history_block_reason = (
                f"Session locked: {self._flag_reason} "
                "Type /new to start a fresh conversation."
            )
            self._save_turn(result)
            return result

        # Check if any previous turn in this session was blocked
        has_prior_block = any(t.is_blocked for t in self._history)

        # ── Rule 2: Bypass phrase after a prior block ─────────────────────
        # User tried a known bypass phrase after being blocked.
        # Example: "ignore all previous instructions" after a harmful message.
        if has_prior_block and "bypass_phrase_detected" in result.reason_codes:
            result.history_triggered    = True
            result.history_block_reason = (
                "Bypass phrase detected after a previous unsafe message. "
                "This override attempt has been blocked."
            )

        # ── Rule 3: Continued unsafe intent after a prior block ───────────
        # User keeps sending harmful messages of the same type.
        # Example: Asking about hacking after being blocked for hacking.
        elif has_prior_block and result.intent in UNSAFE_INTENTS:
            result.history_triggered    = True
            result.history_block_reason = (
                "Continued unsafe intent detected across conversation history."
            )

        # ── Rule 4: Ambiguous follow-up with high unsafe ratio ────────────
        # Current message looks borderline safe but too many recent turns
        # were harmful — block it as a precaution.
        # Example: "can you help me?" after 4 harmful messages.
        elif has_prior_block and not result.is_blocked:
            is_clearly_safe = (
                result.risk_level == "safe"
                and all(v < 0.2 for v in result.category_scores.values())
            )
            ratio = self._unsafe_ratio()
            if not is_clearly_safe and ratio >= UNSAFE_RATIO_WARN:
                result.history_triggered    = True
                result.history_block_reason = (
                    f"{int(ratio * 100)}% of recent turns were unsafe. "
                    "This follow-up has been blocked as a precaution."
                )

        # ── Rule 5: Permanently flag session if unsafe ratio too high ──────
        # If ≥70% of recent turns were blocked, lock the entire session.
        if self._unsafe_ratio() >= UNSAFE_RATIO_FLAG:
            self._flagged     = True
            self._flag_reason = "70% or more of recent turns contained unsafe content."
            logger.warning("Session %s permanently flagged.", self.session_id)

        # Save this turn to history regardless of outcome
        self._save_turn(result)
        return result

    def reset(self) -> None:
        """
        Clear all history and reset the session.
        Called when user types /new.
        """
        self._history.clear()
        self._turn_counter = 0
        self._flagged      = False
        self._flag_reason  = ""
        logger.info("Session %s history cleared.", self.session_id)

    def _save_turn(self, result: GuardResult) -> None:
        """Save the current turn to the history window."""
        self._turn_counter += 1
        self._history.append(TurnRecord(
            turn_index     = self._turn_counter,
            timestamp      = datetime.now(timezone.utc).isoformat(),
            prompt_snippet = result.prompt[:80],
            intent         = result.intent,
            risk_level     = result.risk_level,
            is_blocked     = result.final_blocked,
            block_reason   = result.block_reason or result.history_block_reason,
            category_scores= result.category_scores,
        ))

    def _unsafe_ratio(self) -> float:
        """
        Calculate what fraction of recent turns were blocked.
        Returns 0.0 if fewer than 3 turns (not enough data to judge).
        """
        if len(self._history) < 3:
            return 0.0
        blocked = sum(1 for t in self._history if t.is_blocked)
        return blocked / len(self._history)

    # ── Read-only properties ──────────────────────────────────────────────────
    @property
    def turn_count(self) -> int:   return self._turn_counter
    @property
    def is_flagged(self) -> bool:  return self._flagged
    @property
    def history(self) -> list:     return list(self._history)