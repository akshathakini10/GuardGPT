"""Shared risk-vector construction for GuardGPT temporal intent.

R_t is a six-dimensional evidence vector aligned with the trained intent
classifier labels. This module contains no learned parameters.

Use the same function during temporal dataset preparation and live inference
to avoid train/runtime feature mismatch.
"""

LABELS = (
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
)

# Map deterministic detector reason codes into the six-class taxonomy.
# Unknown codes are deliberately ignored rather than guessed.
PATTERN_TO_LABEL = {
    "prompt_injection": "prompt_injection",
    "instruction_override": "prompt_injection",
    "system_prompt_extraction": "prompt_injection",
    "jailbreak": "jailbreak",
    "safety_bypass": "jailbreak",
    "harmful_instructions": "harmful_instructions",
    "manipulation": "manipulation",
    "self_harm_risk": "self_harm_risk",
}


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def build_risk_vector(
    *,
    matched_intent=None,
    similarity=0.0,
    patterns=None,
    pattern_strength=1.0,
):
    """Return R_t as {six-class-label: evidence in [0,1]}.

    Evidence sources:
    1. FAISS: similarity contributes to the matched six-class label.
    2. Deterministic pattern hits: recognized reason codes contribute to the
       mapped label.

    Multiple signals for one label are combined with max(), not summed, so
    every feature remains in [0,1].

    This is an evidence vector, not a probability distribution; therefore it
    is intentionally NOT normalized to sum to 1.
    """
    vector = {label: 0.0 for label in LABELS}

    if matched_intent in vector:
        vector[matched_intent] = max(
            vector[matched_intent], _clip01(similarity)
        )

    strength = _clip01(pattern_strength)
    for pattern in patterns or ():
        label = PATTERN_TO_LABEL.get(str(pattern))
        if label in vector:
            vector[label] = max(vector[label], strength)

    return vector


def vector_as_list(vector):
    """Return R_t in the fixed LABELS order expected by a neural layer."""
    return [float(vector.get(label, 0.0)) for label in LABELS]
