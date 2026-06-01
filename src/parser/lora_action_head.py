"""LoRA action head — bridges an AI model's decisions to the action library.

Replaces the regex-based CommandParser path with a trainable LoRA layer::

    base_model(input) -> hidden_state -> LoRAActionHead -> [ActionCall, ...]

The base AI model stays frozen — only the low-rank ``A`` / ``B`` matrices
inside the head are trained on ``(hidden_state, target_action_sequence)``
pairs, so adapting to the action library doesn't require fine-tuning the
underlying model.

Bug fixes vs. the old CommandParser path:

  - Out-of-vocabulary actions are structurally impossible — argmax is taken
    over exactly ``[library_names..., STOP]``.
  - All six ActionCall parameters are decoded (speed, extension, repeat,
    distance_m, rotation_rad, duration_s). The old parser only captured
    distance in meters and silently dropped every other modifier.
  - A failed decode raises ``ActionDecodingError`` instead of silently
    returning ``[ActionCall("stand")]``.
  - Sequence length is explicit: the head emits STOP when the plan ends,
    and truncation at ``max_seq_len`` is reported via ``DecodeReport``.

The module ships with ``HashTextEncoder`` so the head is runnable
end-to-end without loading a transformer. For real deployments, plug in
any encoder that produces ``[batch, time, hidden_dim]`` tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadenza.actions.library import ActionCall
from cadenza.stack.vocabulary import ActionDescriptor, ActionVocabulary


# ─────────────────────────────────────────────────────────────────────────────
# Errors and small types
# ─────────────────────────────────────────────────────────────────────────────


class ActionDecodingError(RuntimeError):
    """Raised when the LoRA head cannot produce a valid action sequence.

    The previous CommandParser silently returned ``[stand]`` on any failure.
    This exception is the deliberate replacement: callers must handle a
    decode failure explicitly instead of getting a quiet wrong answer.
    """


# Slot layout for ActionCall parameters. Order is fixed and load-bearing
# (the head's parameter projection emits exactly these dims in this order).
_PARAM_NAMES: tuple[str, ...] = (
    "speed", "extension", "repeat",
    "distance_m", "rotation_rad", "duration_s",
)
_PARAM_DIM = len(_PARAM_NAMES)

STOP_TOKEN = "<stop>"


@dataclass
class DecodeReport:
    """Diagnostic envelope returned alongside decoded ActionCalls.

    Lets callers observe truncation and per-slot confidence without those
    facts being silently lost — the failure mode that motivated this
    replacement.
    """
    calls: list[ActionCall] = field(default_factory=list)
    truncated: bool = False
    stop_emitted: bool = False
    action_confidence: list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA primitive
# ─────────────────────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight + trainable low-rank adapter.

        y = x @ W + (alpha / r) * x @ A @ B

    ``W`` is initialised and frozen at construction. Only ``A`` and ``B``
    receive gradients, which is the whole point of LoRA: cheap, composable
    adaptation without touching the base model.

    ``B`` is initialised to zero so the layer is exactly equivalent to its
    frozen base at step 0. The adapter only diverges from the base once
    training has actually moved ``B``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"LoRALinear needs positive dims, "
                f"got in={in_features}, out={out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank

        W = torch.empty(in_features, out_features)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        self.W = nn.Parameter(W, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.W + self.bias
        adapter = (x @ self.A @ self.B) * self.scale
        return base + adapter

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.A, self.B]


# ─────────────────────────────────────────────────────────────────────────────
# Action head
# ─────────────────────────────────────────────────────────────────────────────


def _param_range(name: str, descriptor: ActionDescriptor) -> tuple[float, float, bool]:
    """Return (min, max, is_int) for one slot of ``descriptor``.

    Defaults to a zero range when the descriptor doesn't expose this slot —
    the head will then produce 0 for that param, which matches ActionCall's
    own field defaults (no surprise mutations).
    """
    for p in descriptor.params:
        if p.name == name:
            lo = float(p.min) if p.min is not None else 0.0
            hi = float(p.max) if p.max is not None else lo
            return lo, max(lo, hi), p.type == "int"
    return 0.0, 0.0, name == "repeat"


class LoRAActionHead(nn.Module):
    """Maps base-model hidden states to action library sequences via LoRA.

    Inputs:
      ``hidden``  — ``[B, T, H]`` (or ``[T, H]``) hidden states from the
                    base AI model. Each of the ``T`` slots is one decision
                    step; the head decides per-slot which library action to
                    emit and its parameters.

    Outputs:
      ``decode``  — for each batch row, a list of ``ActionCall`` objects.
                    Use ``decode_report`` to additionally see whether the
                    sequence stopped naturally vs. was truncated.

    Constrained decoding:
      - Action choice is argmax over ``[library_names..., STOP]``. There is
        no "out-of-vocab" code path — the only emittable indices map 1:1 to
        library actions.
      - Parameter values are sigmoid-normalised then linearly mapped into
        ``[ParamSchema.min, ParamSchema.max]`` for that ``(action, param)``
        pair. Integer params (``repeat``) are rounded and clamped to >= 1.
    """

    def __init__(
        self,
        vocabulary: ActionVocabulary,
        hidden_dim: int,
        max_seq_len: int = 16,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        names = list(vocabulary.names())
        if not names:
            raise ValueError(f"Vocabulary for {vocabulary.robot!r} is empty.")

        self.vocabulary = vocabulary
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self._names: list[str] = names
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(names)}
        self._vocab_size = len(names) + 1   # +1 for STOP
        self._stop_index = len(names)

        # Precompute per-action parameter range tables: shape [V, 6].
        mins, maxs, ints = [], [], []
        for name in names:
            desc = vocabulary.get(name)
            row_min, row_max, row_int = [], [], []
            for pname in _PARAM_NAMES:
                lo, hi, is_int = _param_range(pname, desc)
                row_min.append(lo)
                row_max.append(hi)
                row_int.append(1.0 if is_int else 0.0)
            mins.append(row_min)
            maxs.append(row_max)
            ints.append(row_int)
        self.register_buffer("_param_min", torch.tensor(mins, dtype=torch.float32))
        self.register_buffer("_param_max", torch.tensor(maxs, dtype=torch.float32))
        self.register_buffer("_param_is_int", torch.tensor(ints, dtype=torch.float32))

        self.action_proj = LoRALinear(
            hidden_dim, self._vocab_size, rank=lora_rank, alpha=lora_alpha,
        )
        self.param_proj = LoRALinear(
            hidden_dim, _PARAM_DIM, rank=lora_rank, alpha=lora_alpha,
        )

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def action_names(self) -> list[str]:
        return list(self._names)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Only the LoRA A/B matrices receive gradients."""
        return [
            *self.action_proj.trainable_parameters(),
            *self.param_proj.trainable_parameters(),
        ]

    # ── Forward / training ───────────────────────────────────────────────────

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute raw action logits and pre-clamp parameter outputs.

        Returns:
            action_logits: ``[B, T, V+1]`` (last index is STOP).
            param_raw:     ``[B, T, 6]`` sigmoid-normalised in ``[0, 1]``.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden must have shape [B, T, {self.hidden_dim}], "
                f"got {tuple(hidden.shape)}"
            )
        action_logits = self.action_proj(hidden)
        param_raw = torch.sigmoid(self.param_proj(hidden))
        return action_logits, param_raw

    def training_loss(
        self,
        hidden: torch.Tensor,
        target_action_idx: torch.Tensor,
        target_params: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy on action choice + MSE on parameter values.

        Caller supplies indices into ``action_names + [STOP]`` and the
        normalised target params (already mapped into ``[0, 1]`` over each
        action's range). Padding slots can be masked out via ``action_mask``.
        """
        action_logits, param_raw = self.forward(hidden)
        B, T, V = action_logits.shape
        ce = F.cross_entropy(
            action_logits.reshape(B * T, V),
            target_action_idx.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        mse = ((param_raw - target_params) ** 2).mean(dim=-1)
        loss = ce + mse
        if action_mask is not None:
            loss = loss * action_mask
            denom = action_mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def decode(self, hidden: torch.Tensor) -> list[list[ActionCall]]:
        """Decode hidden states into ActionCall sequences (per batch row)."""
        return [r.calls for r in self.decode_report(hidden)]

    @torch.inference_mode()
    def decode_report(self, hidden: torch.Tensor) -> list[DecodeReport]:
        """Decode with diagnostics (truncation, stop-emitted, confidences)."""
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3:
            raise ActionDecodingError(
                f"hidden must be [B, T, H] or [T, H]; got {tuple(hidden.shape)}"
            )
        if hidden.shape[1] == 0:
            raise ActionDecodingError(
                "hidden has zero time steps; nothing to decode."
            )
        original_T = hidden.shape[1]
        if original_T > self.max_seq_len:
            hidden = hidden[:, : self.max_seq_len]
        was_clipped = original_T > self.max_seq_len

        action_logits, param_raw = self.forward(hidden)
        probs = F.softmax(action_logits, dim=-1)
        chosen = action_logits.argmax(dim=-1)               # [B, T]
        confidence = probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # Map per-slot params through the chosen action's range table.
        safe_idx = chosen.clamp(max=len(self._names) - 1)
        mins = self._param_min[safe_idx]                    # [B, T, 6]
        maxs = self._param_max[safe_idx]
        is_int = self._param_is_int[safe_idx]
        params = mins + (maxs - mins) * param_raw
        params = torch.where(is_int > 0.5, params.round(), params)

        reports: list[DecodeReport] = []
        B, T = chosen.shape
        for b in range(B):
            calls: list[ActionCall] = []
            confs: list[float] = []
            stop_emitted = False
            for t in range(T):
                idx = int(chosen[b, t].item())
                confs.append(float(confidence[b, t].item()))
                if idx == self._stop_index:
                    stop_emitted = True
                    break
                if not (0 <= idx < len(self._names)):
                    # Structurally unreachable — argmax is over [0, V+1).
                    raise ActionDecodingError(
                        f"action index {idx} outside [0, {len(self._names)}); "
                        "this indicates a corrupted vocabulary table."
                    )
                p = params[b, t].tolist()
                calls.append(ActionCall(
                    action_name=self._names[idx],
                    speed=float(p[0]),
                    extension=float(p[1]),
                    repeat=max(1, int(p[2])),
                    distance_m=float(p[3]),
                    rotation_rad=float(p[4]),
                    duration_s=float(p[5]),
                ))
            reports.append(DecodeReport(
                calls=calls,
                truncated=was_clipped and not stop_emitted,
                stop_emitted=stop_emitted,
                action_confidence=confs,
            ))
        return reports

    # ── Training-target construction ─────────────────────────────────────────

    def encode_targets(
        self,
        sequences: Iterable[Iterable[ActionCall]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert ground-truth ActionCall sequences into training tensors.

        Returns:
            target_action_idx: ``[B, T]`` int64 indices into vocab + STOP.
            target_params:     ``[B, T, 6]`` float in ``[0, 1]`` (normalised
                               into each action's range; 0 for STOP slots).
            action_mask:       ``[B, T]`` float — 1 on real slots and the
                               STOP slot, 0 on right-padding.
        """
        seqs = [list(s) for s in sequences]
        T = max((len(s) for s in seqs), default=0) + 1   # +1 for STOP
        if T > self.max_seq_len:
            raise ValueError(
                f"target sequence too long ({T} > max_seq_len={self.max_seq_len})"
            )
        B = len(seqs)
        idx = torch.full((B, T), 0, dtype=torch.int64)
        params = torch.zeros((B, T, _PARAM_DIM), dtype=torch.float32)
        mask = torch.zeros((B, T), dtype=torch.float32)
        for b, seq in enumerate(seqs):
            for t, call in enumerate(seq):
                if call.action_name not in self._name_to_idx:
                    raise ActionDecodingError(
                        f"training target action {call.action_name!r} is not "
                        f"in vocabulary for {self.vocabulary.robot}."
                    )
                a = self._name_to_idx[call.action_name]
                idx[b, t] = a
                lo = self._param_min[a]
                hi = self._param_max[a]
                rng = (hi - lo).clamp(min=1e-9)
                raw = torch.tensor([
                    call.speed, call.extension, float(call.repeat),
                    call.distance_m, call.rotation_rad, call.duration_s,
                ], dtype=torch.float32)
                params[b, t] = ((raw - lo) / rng).clamp(0.0, 1.0)
                mask[b, t] = 1.0
            # STOP slot at the end of each real sequence.
            idx[b, len(seq)] = self._stop_index
            mask[b, len(seq)] = 1.0
        return idx, params, mask


# ─────────────────────────────────────────────────────────────────────────────
# Built-in text encoder (so the head is runnable without a transformer)
# ─────────────────────────────────────────────────────────────────────────────


class HashTextEncoder(nn.Module):
    """Tiny deterministic text encoder.

    Hashes character trigrams of each whitespace-separated token into a
    fixed-width bag-of-features, then runs that bag through a frozen random
    linear projection to ``hidden_dim``. One slot per token, capped at
    ``max_seq_len``.

    This exists so ``LoRAActionDecoder`` is runnable end-to-end without
    requiring a transformer. For real performance you should plug in a
    learned text encoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 16,
        feature_dim: int = 256,
        seed: int = 0xC4DE,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        g = torch.Generator().manual_seed(seed)
        proj = torch.empty(feature_dim, hidden_dim)
        nn.init.kaiming_uniform_(proj, a=math.sqrt(5), generator=g)
        self.proj = nn.Parameter(proj, requires_grad=False)

    def _featurise(self, token: str) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float32)
        padded = f"^{token.lower()}$"
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            h = hash(tri) % self.feature_dim
            vec[h] += 1.0
        norm = vec.norm()
        return vec / norm if float(norm) > 0 else vec

    def forward(self, text: str | list[str]) -> torch.Tensor:
        batch = [text] if isinstance(text, str) else list(text)
        slots = torch.zeros(
            (len(batch), self.max_seq_len, self.hidden_dim), dtype=torch.float32,
        )
        for b, sample in enumerate(batch):
            tokens = sample.split()[: self.max_seq_len]
            if not tokens:
                continue
            feats = torch.stack([self._featurise(t) for t in tokens])  # [N, F]
            slots[b, : len(tokens)] = feats @ self.proj
        return slots


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end decoder (encoder + LoRA head)
# ─────────────────────────────────────────────────────────────────────────────


class LoRAActionDecoder(nn.Module):
    """Composable wrapper: ``encoder -> LoRAActionHead -> ActionCalls``.

    The encoder is any ``nn.Module`` whose ``forward(text)`` returns hidden
    states shaped ``[B, T, hidden_dim]`` — or any callable matching that
    contract. By default a :class:`HashTextEncoder` is used so the decoder
    is runnable without external model weights; swap it for a real text
    encoder in production.

    Decoding always raises ``ActionDecodingError`` on:
      - empty / whitespace-only goals, and
      - the STOP token being emitted at slot 0 (the model "refused" to plan).
    Both are silent failures in the previous CommandParser system.
    """

    def __init__(
        self,
        vocabulary: ActionVocabulary,
        hidden_dim: int = 128,
        max_seq_len: int = 16,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else HashTextEncoder(
            hidden_dim=hidden_dim, max_seq_len=max_seq_len,
        )
        self.head = LoRAActionHead(
            vocabulary=vocabulary,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.head.trainable_parameters()

    @torch.inference_mode()
    def decode(self, goal: str) -> list[ActionCall]:
        if not goal or not goal.strip():
            raise ActionDecodingError("decode() called with empty goal text.")
        hidden = self.encoder(goal)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        report = self.head.decode_report(hidden)[0]
        if not report.calls:
            raise ActionDecodingError(
                f"LoRA head emitted STOP at slot 0 for goal {goal!r}; "
                "the model refused to plan. The previous parser would have "
                "returned [stand] here — surfacing this as an error instead."
            )
        return report.calls

    @torch.inference_mode()
    def decode_report(self, goal: str) -> DecodeReport:
        if not goal or not goal.strip():
            raise ActionDecodingError("decode_report() called with empty goal text.")
        hidden = self.encoder(goal)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        return self.head.decode_report(hidden)[0]
