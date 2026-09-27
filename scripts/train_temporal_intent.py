"""Train GuardGPT's temporal intent layer only.

Uses PRECOMPUTED c_t and R_t vectors, so the existing MiniLM classifier,
FAISS index, and dataset are not retrained or modified.

Input:
    data/guardgpt_temporal_context_unique.jsonl

Output:
    models/temporal_intent/temporal_intent.pt
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from core.temporal_intent import LABELS, TemporalIntentLayer


SEED = 42
DATA = Path("data/guardgpt_temporal_context_unique.jsonl")
OUT = Path("models/temporal_intent/temporal_intent.pt")

LAMBDA = 0.70
EPOCHS = 200
LR = 0.01
PATIENCE = 25

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def vec(d):
    return torch.tensor([float(d[k]) for k in LABELS], dtype=torch.float32)


def load_data():
    rows = [
        json.loads(line)
        for line in DATA.open("r", encoding="utf-8-sig")
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("Temporal dataset is empty.")

    # Guard against accidental exact duplicate conversations.
    seen = set()
    for row in rows:
        key = tuple(
            turn["input_text"].strip().lower()
            for turn in row["turns"]
        )
        if key in seen:
            raise RuntimeError(
                "Duplicate conversation found. Use the deduplicated temporal dataset."
            )
        seen.add(key)

    return rows


def split_conversations(rows):
    """Split whole conversations, never individual turns."""
    rng = random.Random(SEED)
    rows = rows[:]
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    return train, val, test


def run_conversation(model, conversation, training):
    turns = conversation["turns"]

    # Same initialization convention as runtime:
    # H_1 = c_1 and I_1 = c_1.
    c_first = vec(turns[0]["c_t"])
    h_prev = c_first
    i_prev = c_first

    losses = []
    correct = 0
    counted = 0

    # Temporal recurrence starts at turn 2.
    for turn in turns[1:]:
        c_t = vec(turn["c_t"])
        r_t = vec(turn["R_t"])
        target_id = LABEL_TO_ID[turn["contextual_intent_target"]]

        candidate, gate, i_t = model(c_t, h_prev, r_t, i_prev)
        i_t = i_t / i_t.sum().clamp_min(1e-12)

        # Cross entropy on final contextual intent I_t.
        loss = F.nll_loss(
            torch.log(i_t.clamp_min(1e-8)).unsqueeze(0),
            torch.tensor([target_id]),
        )
        losses.append(loss)

        pred = int(torch.argmax(i_t).item())
        correct += int(pred == target_id)
        counted += 1

        # Exact history equation for the next turn.
        h_t = LAMBDA * h_prev + (1.0 - LAMBDA) * c_t

        # Keep sequence graph during training; detach for evaluation.
        h_prev = h_t if training else h_t.detach()
        i_prev = i_t if training else i_t.detach()

    if not losses:
        return None, 0, 0

    return torch.stack(losses).mean(), correct, counted


def evaluate(model, rows):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for conv in rows:
            loss, c, n = run_conversation(model, conv, training=False)
            if loss is None:
                continue
            total_loss += float(loss.item()) * n
            correct += c
            count += n

    return (
        total_loss / max(count, 1),
        correct / max(count, 1),
        count,
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = load_data()
    train_rows, val_rows, test_rows = split_conversations(rows)

    print("Conversations:", len(rows))
    print("Train:", len(train_rows))
    print("Validation:", len(val_rows))
    print("Test:", len(test_rows))
    print("MiniLM retrained: NO")
    print("FAISS retrained: NO")
    print()

    model = TemporalIntentLayer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        shuffled = train_rows[:]
        random.shuffle(shuffled)

        train_loss = 0.0
        train_correct = 0
        train_count = 0

        for conv in shuffled:
            optimizer.zero_grad()
            loss, correct, count = run_conversation(
                model, conv, training=True
            )
            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * count
            train_correct += correct
            train_count += count

        train_loss /= max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)

        val_loss, val_acc, _ = evaluate(model, val_rows)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_count = evaluate(model, test_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "labels": list(LABELS),
            "history_lambda": LAMBDA,
            "architecture": "GuardGPT temporal intent equations",
            "input_features": ["c_t", "H_t_minus_1", "R_t", "I_t_minus_1"],
            "train_conversations": len(train_rows),
            "validation_conversations": len(val_rows),
            "test_conversations": len(test_rows),
            "test_turns": test_count,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "seed": SEED,
        },
        OUT,
    )

    print()
    print("Best validation loss:", round(best_val, 4))
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("Checkpoint:", OUT)
    print("Existing MiniLM checkpoint unchanged.")


if __name__ == "__main__":
    main()
