"""Enrich GuardGPT temporal candidates with the trained MiniLM c_t vector.

Reads:
  data/guardgpt_temporal_context_candidates.jsonl

Writes a NEW file:
  data/guardgpt_temporal_context_with_ct.jsonl

The source candidate file and trained classifier checkpoint are not modified.
"""

import argparse
import json
from pathlib import Path

from core.intent_classifier import IntentClassifier

LABELS = [
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/guardgpt_temporal_context_candidates.jsonl",
    )
    ap.add_argument(
        "--output",
        default="data/guardgpt_temporal_context_with_ct.jsonl",
    )
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    classifier = IntentClassifier()

    conversations = 0
    turns = 0

    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8-sig") as fin, \
         dst.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            conversations += 1

            for turn in row.get("turns", []):
                text = turn.get("input_text", "")
                result = classifier.classify_with_scores(text)

                scores = result.get("scores", {})
                missing = [label for label in LABELS if label not in scores]
                if missing:
                    raise ValueError(
                        f"Line {line_no}: classifier scores missing labels: {missing}"
                    )

                # Preserve the exact trained classifier evidence.
                turn["c_t"] = {
                    label: float(scores[label])
                    for label in LABELS
                }
                turn["current_classifier_intent"] = result["intent"]
                turn["current_classifier_confidence"] = float(
                    result["confidence"]
                )

                turns += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Conversations enriched:", conversations)
    print("Turns enriched:", turns)
    print("Output:", dst)
    print("c_t source: trained six-class MiniLM classifier")
    print("Original candidate file: unchanged")
    print("Classifier checkpoint: unchanged")

if __name__ == "__main__":
    main()
