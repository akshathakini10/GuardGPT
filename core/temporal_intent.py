"""Learned temporal intent layer for GuardGPT.

Implements:
    H_t = lambda * H_(t-1) + (1-lambda) * c_t

    I_hat_t = softmax(
        W_c c_t + W_h H_(t-1) + W_r R_t + W_i I_(t-1) + b
    )

    g_t = sigmoid(
        W_g^T [c_t ; H_(t-1) ; R_t ; I_(t-1)] + b_g
    )

    I_t = (1-g_t) I_(t-1) + g_t I_hat_t

The existing MiniLM intent classifier is NOT retrained by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


LABELS = (
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
)

N = len(LABELS)


def _vector(values: Dict[str, float] | Iterable[float]) -> torch.Tensor:
    if isinstance(values, dict):
        x = torch.tensor([float(values.get(k, 0.0)) for k in LABELS],
                         dtype=torch.float32)
    else:
        x = torch.tensor(list(values), dtype=torch.float32)

    if x.numel() != N:
        raise ValueError(f"Expected {N} values, got {x.numel()}.")

    return x


def _as_dict(x: torch.Tensor) -> Dict[str, float]:
    values = x.detach().cpu().tolist()
    return {label: round(float(v), 6) for label, v in zip(LABELS, values)}


class TemporalIntentLayer(nn.Module):
    """The trainable W/b parameters from the temporal equations."""

    def __init__(self):
        super().__init__()

        # Each term maps a 6-dimensional signal to six intent logits.
        self.W_c = nn.Linear(N, N, bias=False)
        self.W_h = nn.Linear(N, N, bias=False)
        self.W_r = nn.Linear(N, N, bias=False)
        self.W_i = nn.Linear(N, N, bias=False)
        self.b = nn.Parameter(torch.zeros(N))

        # [c_t ; H_(t-1) ; R_t ; I_(t-1)] = 24 dimensions -> scalar gate.
        self.W_g = nn.Linear(N * 4, 1, bias=True)

    def forward(
        self,
        c_t: torch.Tensor,
        h_prev: torch.Tensor,
        r_t: torch.Tensor,
        i_prev: torch.Tensor,
    ):
        logits = (
            self.W_c(c_t)
            + self.W_h(h_prev)
            + self.W_r(r_t)
            + self.W_i(i_prev)
            + self.b
        )

        candidate = torch.softmax(logits, dim=-1)

        gate_input = torch.cat((c_t, h_prev, r_t, i_prev), dim=-1)
        gate = torch.sigmoid(self.W_g(gate_input)).squeeze(-1)

        # Works for both a single vector [6] and batched vectors [B, 6].
        while gate.dim() < candidate.dim():
            gate = gate.unsqueeze(-1)

        contextual = (1.0 - gate) * i_prev + gate * candidate
        return candidate, gate, contextual


@dataclass
class TemporalIntentResult:
    current_vector: Dict[str, float]
    previous_history: Dict[str, float]
    history_vector: Dict[str, float]
    previous_intent: Dict[str, float]
    candidate_vector: Dict[str, float]
    contextual_vector: Dict[str, float]
    contextual_intent: str
    contextual_confidence: float
    risk_vector: Dict[str, float]
    gate: float
    learned_temporal_model: bool


class TemporalIntent:
    """Per-conversation temporal state plus the learned temporal layer."""

    def __init__(
        self,
        model_path: str | Path = "models/temporal_intent/temporal_intent.pt",
        history_lambda: float = 0.70,
        device: str = "cpu",
    ):
        if not 0.0 <= history_lambda <= 1.0:
            raise ValueError("history_lambda must be between 0 and 1.")

        self.history_lambda = float(history_lambda)
        self.device = torch.device(device)
        self.layer = TemporalIntentLayer().to(self.device)
        self.model_path = Path(model_path)

        self._loaded = False
        self._h: Optional[torch.Tensor] = None
        self._i: Optional[torch.Tensor] = None

        # Do NOT use randomly initialized temporal weights in GuardGPT.
        # Runtime becomes available only after a trained checkpoint is loaded.
        if self.model_path.exists():
            self.load(self.model_path)

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def reset(self) -> None:
        self._h = None
        self._i = None

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        checkpoint_labels = tuple(checkpoint.get("labels", LABELS))
        if checkpoint_labels != LABELS:
            raise ValueError(
                f"Temporal checkpoint labels {checkpoint_labels} do not match {LABELS}."
            )

        state = checkpoint.get("model_state_dict", checkpoint)
        self.layer.load_state_dict(state, strict=True)
        self.layer.eval()
        self._loaded = True
        self.model_path = path

    def update(
        self,
        current_vector: Dict[str, float] | Iterable[float],
        risk_vector: Dict[str, float] | Iterable[float],
    ) -> TemporalIntentResult:
        """Apply one conversation turn.

        First-turn convention:
        c_1 initializes H_1 and I_1. The learned recurrence is applied from
        turn 2 onward. This keeps both states valid six-class distributions
        without inventing an I_0 distribution.
        """
        c_t = _vector(current_vector).to(self.device)
        r_t = _vector(risk_vector).to(self.device)

        if self._h is None or self._i is None:
            self._h = c_t.detach().clone()
            self._i = c_t.detach().clone()

            winner = int(torch.argmax(self._i).item())
            return TemporalIntentResult(
                current_vector=_as_dict(c_t),
                previous_history={k: 0.0 for k in LABELS},
                history_vector=_as_dict(self._h),
                previous_intent={k: 0.0 for k in LABELS},
                candidate_vector=_as_dict(c_t),
                contextual_vector=_as_dict(self._i),
                contextual_intent=LABELS[winner],
                contextual_confidence=float(self._i[winner].item()),
                risk_vector=_as_dict(r_t),
                gate=1.0,
                learned_temporal_model=self._loaded,
            )

        h_prev = self._h.detach().clone()
        i_prev = self._i.detach().clone()

        # Exact historical-context recurrence supplied for GuardGPT.
        h_t = (
            self.history_lambda * h_prev
            + (1.0 - self.history_lambda) * c_t
        )

        if not self._loaded:
            raise RuntimeError(
                "Temporal weights are not trained/loaded yet. "
                "Refusing to use random W_c/W_h/W_r/W_i/W_g parameters."
            )

        with torch.no_grad():
            candidate, gate_tensor, i_t = self.layer(
                c_t, h_prev, r_t, i_prev
            )

        # Convex combination of probability distributions; small numerical
        # normalization only protects against floating-point drift.
        i_t = i_t / i_t.sum().clamp_min(1e-12)

        self._h = h_t.detach()
        self._i = i_t.detach()

        winner = int(torch.argmax(i_t).item())
        gate = float(gate_tensor.reshape(-1)[0].item())

        return TemporalIntentResult(
            current_vector=_as_dict(c_t),
            previous_history=_as_dict(h_prev),
            history_vector=_as_dict(h_t),
            previous_intent=_as_dict(i_prev),
            candidate_vector=_as_dict(candidate),
            contextual_vector=_as_dict(i_t),
            contextual_intent=LABELS[winner],
            contextual_confidence=float(i_t[winner].item()),
            risk_vector=_as_dict(r_t),
            gate=gate,
            learned_temporal_model=True,
        )
