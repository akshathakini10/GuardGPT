"""Input checks shared by the complete request pipeline."""
from dataclasses import asdict
from functools import lru_cache
from core.decision_engine import DecisionEngine
from core.risk_estimator import estimate_risk, should_preliminarily_block
from core.risk_vector import build_risk_vector


@lru_cache(maxsize=1)
def resources():
    # Lazy imports allow diagnostics, help and offline tests without model loading.
    from core.dataset_loader import DatasetLoader
    from core.intent_classifier import IntentClassifier
    loader = DatasetLoader()
    loader.load()  # Missing or invalid artifacts must stop analysis.
    classifier = IntentClassifier()
    classifier._model = loader._model
    return loader, classifier


class SafetyService:
    def __init__(self, resource_provider=resources):
        self.resource_provider = resource_provider
        self.engine = DecisionEngine(write_audit=False)

    def analyze(self, prompt):
        loader, classifier = self.resource_provider()
        classified = classifier.classify_with_scores(prompt)
        intent = str(classified.get("intent", "unknown"))
        confidence = float(classified.get("confidence", 0.0))
        current_intent_vector = dict(classified.get("scores", {}))  # c_t
        record = loader.query(prompt)
        if not record:
            raise ValueError("Semantic search returned no record")
        similarity = float(record["_similarity"])
        matched_intent = record["intent"]
        raw_scores = dict(record.get("category_scores", {}))
        threshold = 0.8 if intent == "safe" else 0.55
        scores = raw_scores if similarity >= threshold else {}

        from core.jailbreak_patterns import _pattern_hits
        patterns = _pattern_hits(prompt.lower())
        risk_vector = build_risk_vector(
            matched_intent=matched_intent,
            similarity=similarity,
            patterns=patterns,
        )
        risk = estimate_risk(intent, confidence, similarity, matched_intent)
        signal = dict(prompt=prompt, intent=intent, intent_confidence=confidence,
            risk_level=risk, dataset_match_confidence=similarity,
            matched_record_id=record.get("request_id"), matched_record_intent=matched_intent,
            category_scores=scores, reason_codes=[],
            current_intent_vector=current_intent_vector,
            risk_vector=risk_vector)
        preliminary = should_preliminarily_block(signal)
        if patterns:
            preliminary = True
            signal["reason_codes"].extend(patterns)
        signal["final_blocked"] = preliminary
        signal["block_reason"] = "Input safety check triggered." if preliminary else ""
        decision = asdict(self.engine.decide(signal))
        attacks = list(patterns)
        if confidence >= 0.65 and intent in {"jailbreak", "prompt_injection", "harmful_instructions", "manipulation"}:
            attacks.append(intent)
        # Self-harm is a support need, not an attack on the system.
        return dict(signal=signal, decision=decision,
                    detected_attacks=list(dict.fromkeys(attacks)),
                    matched_category_scores=raw_scores)

    def apply_history(self, analysis, guard):
        signal = dict(analysis["signal"])
        signal["final_blocked"] = analysis["decision"]["action"] == "BLOCK"

        # Exactly one temporal update for this input turn.
        history = guard.evaluate_result(signal)

        # Preserve the raw MiniLM result for diagnostics.
        signal["raw_intent"] = signal.get("intent", "unknown")
        signal["raw_intent_confidence"] = signal.get("intent_confidence", 0.0)

        # I_t is now the intent evidence used by downstream decision logic.
        signal["intent"] = history.contextual_intent
        signal["intent_confidence"] = history.contextual_confidence
        signal["contextual_intent_vector"] = dict(history.contextual_vector or {})
        signal["history_vector"] = dict(history.history_vector or {})
        signal["temporal_gate"] = history.gate
        signal["learned_temporal_model"] = history.learned_temporal_model

        # Recalculate risk from the contextual intent. FAISS evidence is unchanged.
        signal["risk_level"] = estimate_risk(
            signal["intent"],
            signal["intent_confidence"],
            float(signal.get("dataset_match_confidence", 0.0)),
            signal.get("matched_record_intent", ""),
        )

        # Re-run the decision on every session turn because contextual I_t can
        # differ from the standalone MiniLM winner even without an old-style
        # history_triggered flag.
        signal["final_blocked"] = should_preliminarily_block(signal)
        if signal.get("reason_codes"):
            signal["final_blocked"] = True
        signal["block_reason"] = (
            "Input safety check triggered." if signal["final_blocked"] else ""
        )

        analysis["signal"] = signal
        analysis["decision"] = asdict(
            self.engine.decide(signal, turn_index=history.turn_index)
        )
        return history.turn_index
