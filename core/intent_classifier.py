# ============================================================
# GuardGPT - intent_classifier.py
# ============================================================
# PURPOSE:
#   Classify every user prompt using a TWO-LAYER defence system.
#
# LAYER 1 - Keyword Blocklist (runs first, instant):
#   Checks if any word in the prompt is in our blocklist.
#   If yes → block immediately, no need to check dataset.
#   Catches: "fuck you", "how to hack", "cocaine recipe", etc.
#
# LAYER 2 - TF-IDF Dataset Match (runs if Layer 1 passes):
#   Finds the closest matching record in the 599k harm dataset.
#   Checks category scores (harm, jailbreak, toxicity, etc.)
#   to decide whether to block.
#   Catches: complex harmful prompts, jailbreak attempts, etc.
#
# DATASET INTENTS (actual labels in your dataset):
#   safe, self_harm_risk, prompt_injection,
#   jailbreak_attempt, cyberattack_intent, hate_speech_harassment
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional
from core.dataset_loader import DatasetLoader, _tokenize

logger = logging.getLogger(__name__)

# ── Score thresholds ──────────────────────────────────────────────────────────
# A dataset record only causes a block if its score exceeds these values.
# These are tuned for your dataset's score distribution.
HARM_THRESHOLD             = 0.30   # blocks cyberattack, self harm records
JAILBREAK_THRESHOLD        = 0.25   # blocks jailbreak records
TOXICITY_THRESHOLD         = 0.30   # blocks hate speech records
PROMPT_INJECTION_THRESHOLD = 0.25   # blocks prompt injection records
MEDICAL_THRESHOLD          = 0.50   # blocks high medical risk records

# Minimum token overlap needed before we trust the dataset match.
# Set low (0.15) because dataset prompts are short (avg 72 chars).
# For self harm specifically, we use an even lower threshold (0.10)
# because missing a self harm message is far more dangerous than
# a false positive.
OVERLAP_CONFIDENCE_THRESHOLD      = 0.15
SELF_HARM_CONFIDENCE_THRESHOLD    = 0.10  # extra sensitive for self harm

# ── Intent labels (must match your dataset exactly) ───────────────────────────
UNSAFE_INTENTS = frozenset({
    "self_harm_risk",           # harm score dominant
    "prompt_injection",         # prompt_injection score dominant
    "jailbreak_attempt",        # jailbreak score dominant
    "cyberattack_intent",       # harm score dominant
    "hate_speech_harassment",   # toxicity score dominant
})

# ── Layer 1: Keyword blocklist ────────────────────────────────────────────────
# Organised by intent category for easy maintenance.
# Add new words here whenever you find gaps in detection.
_KEYWORDS = {

    "hate_speech_harassment": {
        # Profanity
        "fuck","fucking","fucker","fucks","bitch","bitches",
        "bastard","asshole","arsehole","dickhead","douchebag",
        "shit","bullshit","cunt","whore","slut","damn",
        # Insults
        "idiot","moron","retard","dumbass","stupid",
        # Slurs
        "nigger","nigga","faggot","kike","spic","chink",
        "hate", "hatred", "despise", "loathe",

    },

    "self_harm_risk": {
        # Direct self harm words
        "suicide","suicidal","selfharm","self-harm",
        #"overdose",
        "slit","hanging","noose",
        # Emotional distress signals combined with harm
        "kill myself","end my life","end my pain",
        "want to die","want to disappear","no reason to live",
        "better off dead","nobody would care","worthless",
        "hopeless","disappear forever",
    },

    "cyberattack_intent": {
        # Cybercrime
        "hack","hacking","hacks","hacker","ddos","malware","ransomware",
        "phishing","exploit","keylogger","botnet","trojan","spyware",
        # Weapons and explosives
        "explosive","explosives","detonator","detonation",
        "gunpowder","napalm","thermite","molotov","grenade",
        "dynamite","semtex","c4","ammunition","silencer",
        # Harmful substances
        "poison","poisoning","cyanide","arsenic","sarin",
        "anthrax","ricin","chlorine","synthesis","synthesize","synthesise",
        # Drugs
        "cocaine","meth","methamphetamine","heroin","crack",
        "fentanyl","ecstasy","mdma","lsd",
        # Violence
        "murder","rape","stab","strangle","massacre","genocide",
        "torture","trafficking","traffick","smuggle","smuggling",
        # Abuse
        "pedophile","pedophilia","grooming",
        # Story wrapping harmful nouns
        "make explosives", "build a bomb", "synthesize drugs",
        "create malware", "write a virus",
    },

    "prompt_injection": {
        # Classic jailbreak phrases
        "jailbreak","jailbreaking","unrestricted","no restrictions",
        "ignore previous","ignore all","bypass","dan",
        "do anything now","no rules","no limits",
        "pretend you have no","act as if you have no",
    },
}

# Build a flat lookup dictionary: keyword → intent
# This is built once at import time for fast O(1) lookups
_KEYWORD_TO_INTENT: dict[str, str] = {}
for _intent, _words in _KEYWORDS.items():
    for _word in _words:
        _KEYWORD_TO_INTENT[_word] = _intent


# ── GuardResult dataclass ─────────────────────────────────────────────────────
@dataclass
class GuardResult:
    """
    Holds the complete safety verdict for one user prompt.

    This object is passed through the pipeline:
      IntentClassifier → ConversationGuard → DecisionEngine

    Fields:
      prompt                  : original user message
      intent                  : detected intent label
      risk_level              : safe / low / medium / high / critical
      is_blocked              : True if THIS prompt triggered a block
      block_reason            : why it was blocked (human readable)
      category_scores         : harm/jailbreak/toxicity scores from dataset
      reason_codes            : machine-readable list of triggered rules
      dataset_match_confidence: how closely prompt matched dataset record (0-1)
      matched_record_id       : which dataset record was matched
      history_triggered       : True if conversation history caused the block
      history_block_reason    : why history caused the block
    """
    prompt: str
    intent: str = "safe"
    risk_level: str = "safe"
    is_blocked: bool = False
    block_reason: str = ""
    category_scores: dict = field(default_factory=dict)
    reason_codes: list    = field(default_factory=list)
    dataset_match_confidence: float = 0.0
    matched_record_id: Optional[str] = None
    history_triggered: bool = False
    history_block_reason: str = ""

    @property
    def final_blocked(self) -> bool:
        """True if blocked for ANY reason — direct block OR history block."""
        return self.is_blocked or self.history_triggered


# ── IntentClassifier class ────────────────────────────────────────────────────
class IntentClassifier:
    """
    Classifies user prompts using two layers of defence.

    Layer 1: Instant keyword check — no dataset needed.
    Layer 2: TF-IDF dataset match — for complex harmful prompts.
    """

    def __init__(self, loader: DatasetLoader) -> None:
        self._loader = loader

    def classify(self, prompt: str) -> GuardResult:
        """
        Main entry point. Runs both layers and returns a GuardResult.

        Layer 1 runs first. If it finds a blocked keyword, returns immediately.
        Layer 2 only runs if Layer 1 finds nothing suspicious.
        """
        if not self._loader.is_loaded:
            self._loader.load()

        result = GuardResult(prompt=prompt)

        # ── LAYER 1: Keyword check ────────────────────────────────────────
        # We check BOTH raw words (before stopword removal) and filtered
        # tokens (after stopword removal) to catch all variations.
        raw_words     = set(prompt.lower().split())
        filtered_words = set(_tokenize(prompt))
        all_words     = raw_words | filtered_words

        # Check multi-word phrases first (e.g. "ignore all", "end my life")
        prompt_lower = prompt.lower()
        for phrase, intent in _KEYWORD_TO_INTENT.items():
            if " " in phrase and phrase in prompt_lower:
                # Multi-word phrase matched
                result.is_blocked   = True
                result.intent       = intent
                result.risk_level   = "high"
                result.block_reason = f"Blocked phrase: '{phrase}'"
                result.reason_codes.append("blocked_keyword")
                logger.debug("Layer 1 phrase block: '%s' → %s", phrase, intent)
                return result

        # Check single words
        for word in all_words:
            if word in _KEYWORD_TO_INTENT:
                intent = _KEYWORD_TO_INTENT[word]
                result.is_blocked   = True
                result.intent       = intent
                result.risk_level   = "high"
                result.block_reason = f"Blocked keyword: '{word}'"
                result.reason_codes.append("blocked_keyword")
                logger.debug("Layer 1 word block: '%s' → %s", word, intent)
                return result

        # ── LAYER 2: Dataset match ────────────────────────────────────────
        record = self._loader.query(prompt)
        if record:
            result = self._apply_record(prompt, record, result)

        return result

    def _apply_record(self, prompt: str, record: dict, result: GuardResult) -> GuardResult:
        """
        Use a matched dataset record to decide if the prompt should be blocked.

        Steps:
          1. Extract intent and scores from the matched record
          2. Calculate token overlap (confidence) between prompt and record
          3. Apply blocking rules based on scores and intent
          4. Override everything if the record has a high 'safe' score
        """
        intent       = record.get("intent", "unknown")
        scores       = record.get("category_scores", {})
        reason_codes = record.get("reason_codes", [])

        # Store metadata about the match
        result.intent            = intent
        result.category_scores   = scores
        result.matched_record_id = record.get("request_id")

        # ── Calculate token overlap (confidence) ──────────────────────────
        # Overlap = how many of the prompt's tokens appear in the matched record
        # High overlap = good match, scores are reliable
        # Low overlap  = poor match, scores may not apply
        query_tokens  = set(_tokenize(prompt))
        record_tokens = set(_tokenize(record.get("input_text", "")))

        if query_tokens:
            overlap = len(query_tokens & record_tokens) / len(query_tokens)
        else:
            overlap = 0.0

        result.dataset_match_confidence = round(overlap, 3)

        # Use lower threshold for self harm — never miss a crisis signal
        if intent == "self_harm_risk":
            confident = overlap >= SELF_HARM_CONFIDENCE_THRESHOLD
        else:
            confident = overlap >= OVERLAP_CONFIDENCE_THRESHOLD

        # ── Extract all score dimensions ──────────────────────────────────
        harm       = scores.get("harm",             0.0)
        jailbreak  = scores.get("jailbreak",        0.0)
        toxicity   = scores.get("toxicity",         0.0)
        injection  = scores.get("prompt_injection", 0.0)
        medical    = scores.get("medical_help",     0.0)
        safe_score = scores.get("safe",             0.0)

        # ── Apply blocking rules ──────────────────────────────────────────
        # Each rule is independent — multiple rules can fire on one prompt.
        # We only apply rules if confidence is high enough.

        # Rule 1: Intent is in our unsafe list
        if intent in UNSAFE_INTENTS and confident:
            result.is_blocked   = True
            result.block_reason = f"Unsafe intent detected: {intent}"
            result.reason_codes.extend(reason_codes)

        # Rule 2: High harm score (covers cyberattack, self harm)
        if harm >= HARM_THRESHOLD and confident:
            result.is_blocked   = True
            result.block_reason = result.block_reason or "High harm score"
            result.reason_codes.append("high_harm_score")

        # Rule 3: High jailbreak score
        if jailbreak >= JAILBREAK_THRESHOLD and confident:
            result.is_blocked   = True
            result.block_reason = result.block_reason or "Jailbreak detected"
            result.reason_codes.append("jailbreak_attempt")

        # Rule 4: High toxicity score (covers hate speech)
        if toxicity >= TOXICITY_THRESHOLD and confident:
            result.is_blocked   = True
            result.block_reason = result.block_reason or "High toxicity"
            result.reason_codes.append("high_toxicity")

        # Rule 5: High prompt injection score
        if injection >= PROMPT_INJECTION_THRESHOLD and confident:
            result.is_blocked   = True
            result.block_reason = result.block_reason or "Prompt injection"
            result.reason_codes.append("prompt_injection_detected")

        # Rule 6: High medical risk score
        if medical >= MEDICAL_THRESHOLD and confident:
            result.is_blocked   = True
            result.block_reason = result.block_reason or "Medical risk detected"
            result.reason_codes.append("medical_risk_detected")

        # ── Safe score override ───────────────────────────────────────────
        # If the matched record is clearly safe (safe score ≥ 0.80),
        # the match was likely a false positive — clear all blocks.
        # Example: "tell me about udupi" matching a weak harmful record
        if safe_score >= 0.80:
            result.is_blocked   = False
            result.block_reason = ""
            result.reason_codes = []

        # ── Compute final risk level ──────────────────────────────────────
        result.risk_level = self._compute_risk(
            harm, jailbreak, toxicity, injection, intent, medical, safe_score
        )
        return result

    @staticmethod
    def _compute_risk(harm, jailbreak, toxicity, injection,
                      intent, medical=0.0, safe_score=0.0) -> str:
        """
        Convert scores into a risk label.

        Risk ladder (checked from highest to lowest):
          critical : any score ≥ 0.90  OR  unsafe intent
          high     : any score ≥ 0.85  OR  unsafe intent
          medium   : any score ≥ 0.60
          low      : any score ≥ 0.35
          safe     : everything below low  OR  safe_score ≥ 0.80
        """
        # Safe record overrides everything
        if safe_score >= 0.80:
            return "safe"

        top_score = max(harm, jailbreak, toxicity, injection, medical)

        if intent in UNSAFE_INTENTS or top_score >= 0.85:
            return "critical" if top_score >= 0.90 else "high"
        if top_score >= 0.60:
            return "medium"
        if top_score >= 0.35:
            return "low"
        return "safe"