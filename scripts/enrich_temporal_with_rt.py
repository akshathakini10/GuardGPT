"""Enrich GuardGPT temporal data with the live R_t risk vector.

Reads the file that already contains real MiniLM c_t vectors and writes a
NEW file containing R_t produced by the same SafetyService used at runtime.

Run from the project root as:
    python -m scripts.enrich_temporal_with_rt
"""

import argparse
import json
from pathlib import Path

from core.safety_service import SafetyService

LABELS = (
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/guardgpt_temporal_context_with_ct.jsonl",
    )
    ap.add_argument(
        "--output",
        default="data/guardgpt_temporal_context_with_ct_rt.jsonl",
    )
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    service = SafetyService()
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
                text = str(turn.get("input_text", ""))
                analysis = service.analyze(text)
                signal = analysis["signal"]

                risk_vector = signal.get("risk_vector")
                if not isinstance(risk_vector, dict):
                    raise ValueError(
                        f"Line {line_no}: live SafetyService returned no risk_vector"
                    )

                missing = [x for x in LABELS if x not in risk_vector]
                if missing:
                    raise ValueError(
                        f"Line {line_no}: R_t missing labels: {missing}"
                    )

                turn["R_t"] = {
                    label: float(risk_vector[label]) for label in LABELS
                }

                # Keep useful provenance/debug information without replacing
                # the already stored c_t or contextual target.
                turn["risk_level_at_enrichment"] = signal.get("risk_level")
                turn["faiss_matched_intent"] = signal.get(
                    "matched_record_intent"
                )
                turn["faiss_similarity"] = float(
                    signal.get("dataset_match_confidence", 0.0)
                )
                turn["pattern_reason_codes"] = list(
                    signal.get("reason_codes", [])
                )
                turns += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Conversations enriched:", conversations)
    print("Turns enriched:", turns)
    print("Output:", dst)
    print("R_t source: live SafetyService")
    print("Input file: unchanged")
    print("MiniLM checkpoint: unchanged")


if __name__ == "__main__":
    main()
