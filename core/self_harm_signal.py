"""Conservative, current-turn self-harm evidence; not a replacement classifier."""
import re

# Explicit first-person expressions only; do not treat general educational discussion
# or a mere 'educational purpose' claim as a self-harm disclosure.
_DISCLOSURE = re.compile(
    r"\b(?:i\s*(?:am|['’]?m|want\s+to|plan\s+to|will|['’]?ll|might|could|feel\s+like|intend\s+to)\s+)"
    r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+(?:own\s+)?life|commit\s+suicide|hurt\s+myself|harm\s+myself)\b",
    re.I,
)
_METHOD = re.compile(
    r"\b(?:how\s+(?:can|could|do|would|to)\s+i\s+|tell\s+me\s+how\s+to\s+)"
    r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+(?:own\s+)?life|commit\s+suicide|hurt\s+myself|harm\s+myself)\b",
    re.I,
)


def self_harm_evidence(prompt: str, scores: dict) -> str | None:
    """Return disclosure/method for explicit current-turn evidence, otherwise None.

    Score gate supplements the rules; it never re-labels the MiniLM prediction.
    """
    if not isinstance(prompt, str):
        return None
    if _METHOD.search(prompt):
        return "method"
    if _DISCLOSURE.search(prompt):
        return "disclosure"
    return None
