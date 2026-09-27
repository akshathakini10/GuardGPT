"""Create a GuardGPT temporal-dataset annotation template.

This script does not modify the original single-turn dataset and does not
invent temporal labels. Contextual labels must be human reviewed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = {
    "safe", "prompt_injection", "jailbreak",
    "harmful_instructions", "manipulation", "self_harm_risk",
}

def validate_source(path):
    counts = Counter()
    total = 0
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "input_text" not in obj or "intent" not in obj:
                raise ValueError(f"Line {line_no}: input_text/intent missing")
            if obj["intent"] not in LABELS:
                raise ValueError(f"Line {line_no}: unexpected intent {obj['intent']!r}")
            counts[obj["intent"]] += 1
            total += 1
    return total, counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/guardgpt_dataset.jsonl")
    parser.add_argument("--output", default="data/guardgpt_temporal_seed.jsonl")
    args = parser.parse_args()

    total, counts = validate_source(args.source)
    example = {
        "conversation_id": "conv_0001",
        "review_status": "needs_human_review",
        "turns": [{
            "turn": 1,
            "input_text": "REPLACE_WITH_REVIEWED_PROMPT",
            "current_intent": "safe",
            "contextual_intent": "safe",
            "relation_to_previous": "new_topic",
            "notes": "Human reviewer must verify contextual_intent."
        }]
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Validated source records: {total}")
    for label in sorted(LABELS):
        print(f"{label}: {counts[label]}")
    print(f"Created: {output}")
    print("No temporal labels were generated automatically.")

if __name__ == "__main__":
    main()
