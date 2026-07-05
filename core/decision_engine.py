# ============================================================
# GuardGPT - decision_engine.py
# ============================================================
# PURPOSE:
#   Takes the GuardResult from IntentClassifier + ConversationGuard
#   and makes the FINAL decision: ALLOW or BLOCK.
#
#   Also writes every decision to an audit log file so you can
#   review all decisions later (useful for debugging and viva demo).
#
# FLOW:
#   GuardResult → decide() → DecisionOutput + audit log entry
# ============================================================

import json, logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from core.intent_classifier import GuardResult

logger = logging.getLogger(__name__)

# Audit log location — every prompt decision is saved here
AUDIT_LOG_PATH = Path(__file__).parent.parent / "logs" / "guardgpt_audit.jsonl"

# ── User-facing messages ──────────────────────────────────────────────────────
# These are shown to the user when their message is blocked.
# One message per intent label (must match your dataset's intent labels).
_BLOCK_MESSAGES = {

    "self_harm_risk": (
        "Your message indicates you may be experiencing distress. "
        "Please reach out to a mental health professional or crisis helpline. "
        "You are not alone. 🆘"
    ),

    "prompt_injection": (
        "A prompt injection attempt was detected. "
        "This request cannot be processed."
    ),

    "jailbreak_attempt": (
        "A jailbreak attempt was detected. "
        "This request cannot be processed."
    ),

    "cyberattack_intent": (
        "Your message requests assistance with hacking, weapons, drugs, "
        "explosives, or other harmful activities. "
        "This request has been blocked."
    ),

    "hate_speech_harassment": (
        "Your message contains hate speech or harassment. "
        "This request has been blocked."
    ),
}

# Shown when no specific intent message exists
_DEFAULT_BLOCK_MESSAGE = (
    "Your message has been identified as potentially harmful "
    "and cannot be processed."
)

# Shown when conversation history caused the block (not the current message)
_HISTORY_BLOCK_MESSAGE = (
    "Based on this conversation's history, this message cannot be processed. "
    "Previous messages in this session contained unsafe content. "
    "Type /new to start a fresh conversation."
)


# ── DecisionOutput dataclass ──────────────────────────────────────────────────
@dataclass
class DecisionOutput:
    """
    The final verdict for one prompt. Returned by DecisionEngine.decide().

    Fields:
      allowed                 : True = prompt goes to Llama, False = blocked
      intent                  : detected intent label
      risk_level              : safe / low / medium / high / critical
      user_message            : message shown to user when blocked
      technical_reason        : internal reason for logging/debugging
      category_scores         : harm/jailbreak/toxicity scores from dataset
      reason_codes            : list of rules that fired (e.g. high_harm_score)
      dataset_match_confidence: how well the prompt matched a dataset record
      matched_record_id       : which dataset record was matched
      history_triggered       : True if history (not current message) caused block
      turn_index              : which turn number this is in the session
    """
    allowed: bool
    intent: str
    risk_level: str
    user_message: str
    technical_reason: str
    category_scores: dict
    reason_codes: list
    dataset_match_confidence: float
    matched_record_id: Optional[str]
    history_triggered: bool
    turn_index: int


# ── DecisionEngine class ──────────────────────────────────────────────────────
class DecisionEngine:
    """
    Makes the final ALLOW / BLOCK decision and logs every turn.

    Usage:
        engine = DecisionEngine()
        output = engine.decide(guard_result, turn_index=1)
    """

    def __init__(self) -> None:
        # Make sure the logs/ directory exists when the engine starts
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def decide(self, result: GuardResult, turn_index: int = 0) -> DecisionOutput:
        """
        Convert a GuardResult into a final DecisionOutput.

        Logic:
          - If result.final_blocked is True → BLOCK
          - If result.final_blocked is False → ALLOW

        Also selects the right user-facing message and writes audit log.
        """
        allowed = not result.final_blocked

        # Pick the right message to show the user
        if allowed:
            # No message needed for allowed prompts
            user_message = ""

        elif result.history_triggered:
            # History caused the block, not the current message
            user_message = _HISTORY_BLOCK_MESSAGE

        else:
            # Current message caused the block — use intent-specific message
            user_message = _BLOCK_MESSAGES.get(result.intent, _DEFAULT_BLOCK_MESSAGE)

        # Build a technical reason string for internal logging
        parts = [r for r in (result.block_reason, result.history_block_reason) if r]
        technical_reason = " | ".join(parts) if parts else "no issues detected"

        # Build the output object
        output = DecisionOutput(
            allowed                  = allowed,
            intent                   = result.intent,
            risk_level               = result.risk_level,
            user_message             = user_message,
            technical_reason         = technical_reason,
            category_scores          = result.category_scores,
            reason_codes             = result.reason_codes,
            dataset_match_confidence = result.dataset_match_confidence,
            matched_record_id        = result.matched_record_id,
            history_triggered        = result.history_triggered,
            turn_index               = turn_index,
        )

        # Write to audit log
        self._write_audit_log(result, output)
        return output

    def _write_audit_log(self, result: GuardResult, output: DecisionOutput) -> None:
        """
        Append one JSON line to the audit log for this turn.

        The audit log is append-only — it records every single decision
        made by the engine. Useful for reviewing system behaviour.

        Note: Only the first 120 chars of the prompt are stored (privacy).
        """
        entry = {
            "timestamp"               : datetime.now(timezone.utc).isoformat(),
            "turn_index"              : output.turn_index,
            "allowed"                 : output.allowed,
            "intent"                  : output.intent,
            "risk_level"              : output.risk_level,
            "history_triggered"       : output.history_triggered,
            "technical_reason"        : output.technical_reason,
            "reason_codes"            : output.reason_codes,
            "category_scores"         : output.category_scores,
            "dataset_match_confidence": output.dataset_match_confidence,
            "prompt_snippet"          : result.prompt[:120],  # truncated for privacy
        }
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning("Could not write audit log: %s", e)