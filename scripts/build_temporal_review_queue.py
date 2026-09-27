"""Create a compact review queue for GuardGPT temporal labels.

Prioritizes turns where the standalone MiniLM prediction differs from the
proposed contextual target. It does not approve or modify any labels.

Run:
    python -m scripts.build_temporal_review_queue
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/guardgpt_temporal_context_with_ct_rt.jsonl",
    )
    ap.add_argument(
        "--output",
        default="data/guardgpt_temporal_review_queue.csv",
    )
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    rows = [
        json.loads(line)
        for line in src.open("r", encoding="utf-8-sig")
        if line.strip()
    ]

    review = []
    for conv in rows:
        turns = conv["turns"]
        for idx, turn in enumerate(turns):
            predicted = turn["current_classifier_intent"]
            target = turn["contextual_intent_target"]

            # Context-sensitive disagreements are the highest-value review cases.
            if predicted != target:
                previous_text = turns[idx - 1]["input_text"] if idx > 0 else ""
                review.append({
                    "conversation_id": conv["conversation_id"],
                    "pattern": conv.get("pattern", ""),
                    "turn": turn["turn"],
                    "previous_text": previous_text,
                    "current_text": turn["input_text"],
                    "minilm_prediction": predicted,
                    "minilm_confidence": turn["current_classifier_confidence"],
                    "proposed_contextual_target": target,
                    "faiss_matched_intent": turn.get("faiss_matched_intent", ""),
                    "faiss_similarity": turn.get("faiss_similarity", ""),
                    "risk_level": turn.get("risk_level_at_enrichment", ""),
                    "review_decision": "",
                    "reviewed_contextual_target": "",
                    "review_notes": "",
                })

    # Highest-confidence disagreements first.
    review.sort(
        key=lambda x: float(x["minilm_confidence"]),
        reverse=True,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "conversation_id", "pattern", "turn",
        "previous_text", "current_text",
        "minilm_prediction", "minilm_confidence",
        "proposed_contextual_target",
        "faiss_matched_intent", "faiss_similarity",
        "risk_level",
        "review_decision",
        "reviewed_contextual_target",
        "review_notes",
    ]

    with dst.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)

    print("Source conversations:", len(rows))
    print("Priority disagreement cases:", len(review))
    print("Review queue:", dst)
    print("No labels were approved or modified.")
    print("Fill review_decision with APPROVE, CHANGE, or REJECT.")


if __name__ == "__main__":
    main()
