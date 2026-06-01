"""SmolVLA-based world-model adapter tuned for the G1 humanoid.

G1's action vocabulary is narrower than Go1 (no side_step, climb_step,
turn primitives in the current library). The goal-text -> plan path now
runs through :class:`cadenza.parser.lora_action_head.LoRAActionDecoder`,
a trainable LoRA head whose output is structurally constrained to the
action library — replacing the regex CommandParser whose silent drops
caused divergence between intended and executed plans. SmolVLA inference
still runs each tick for perception/logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadenza import WorldModelAdapter, AdapterReply, ProposedAction
from cadenza.stack.vocabulary import ActionVocabulary


_LOG = logging.getLogger("ai_models.g1.vla")


@dataclass
class _EpisodeState:
    plan: list[ProposedAction] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    started: bool = False


class VLA(WorldModelAdapter):
    """SmolVLA-driven adapter for G1 (plan-queue mode)."""

    name = "g1_vla"
    description = "HuggingFace SmolVLA plan-queue adapter for G1."
    DEFAULT_MODEL_ID = "lerobot/smolvla_base"

    MAX_TICKS = 250

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        return None

    def _load_impl(self) -> None:
        if self.model is not None:
            return
        try:
            from lerobot.policies.smolvla.modeling_smolvla import (    # type: ignore
                SmolVLAPolicy,
            )
        except ImportError as e:
            raise RuntimeError(
                "ai_models.g1.VLA requires `lerobot[smolvla]`. Install with:\n"
                "    pip install 'lerobot[smolvla]'"
            ) from e
        model_id = str(self.checkpoint) if self.checkpoint else self.DEFAULT_MODEL_ID
        self.model = SmolVLAPolicy.from_pretrained(model_id)
        try:
            self.model.eval()
        except AttributeError:
            pass
        self._episode = _EpisodeState()

    def _ensure_episode(self) -> _EpisodeState:
        if not hasattr(self, "_episode"):
            self._episode = _EpisodeState()
        return self._episode

    def _build_plan(self, goal: str, vocabulary: ActionVocabulary) -> list[ProposedAction]:
        decoder = self._ensure_lora_decoder(vocabulary)
        report = decoder.decode_report(goal)
        if report.truncated:
            _LOG.warning(
                "g1.VLA: plan truncated at max_seq_len=%d; raise the head's "
                "max_seq_len if longer plans are needed.",
                decoder.head.max_seq_len,
            )
        plans: list[ProposedAction] = []
        for call, conf in zip(report.calls, report.action_confidence):
            params: dict[str, Any] = {}
            if call.distance_m > 0:
                params["distance_m"] = call.distance_m
            if call.duration_s > 0:
                params["duration_s"] = call.duration_s
            plans.append(ProposedAction(
                name=call.action_name, params=params,
                rationale=f"lora head (conf={conf:.2f})",
            ))
        return plans

    def _ensure_lora_decoder(self, vocabulary: ActionVocabulary):
        """Lazily build the LoRA decoder keyed to this vocabulary."""
        from cadenza.parser.lora_action_head import LoRAActionDecoder
        cache = getattr(self, "_lora_decoder", None)
        if cache is None or cache[0] is not vocabulary:
            decoder = LoRAActionDecoder(
                vocabulary=vocabulary,
                hidden_dim=128,
                max_seq_len=16,
                lora_rank=8,
            )
            self._lora_decoder = (vocabulary, decoder)
        return self._lora_decoder[1]

    def _infer(self, observation: dict, goal: str) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            import torch
            cam = observation.get("camera")
            frame: dict[str, Any] = {
                "task": goal,
                "observation.state": torch.tensor(
                    np.asarray(observation.get("qpos", []), dtype=np.float32),
                ),
            }
            if cam is not None:
                arr = np.asarray(cam, dtype=np.float32) / 255.0
                if arr.ndim == 3:
                    arr = arr.transpose(2, 0, 1)[None, ...]
                frame["observation.image"] = torch.tensor(arr)
            with torch.inference_mode():
                action = self.model.select_action(frame)
            return np.asarray(action.detach().cpu().numpy(), dtype=np.float32).flatten()
        except Exception as e:
            _LOG.debug("vla.select_action failed (%s); using perception-only loop", e)
            return None

    def propose_actions(
        self,
        observation: dict,
        goal: str,
        vocabulary: ActionVocabulary,
        history: list[ProposedAction] | None = None,
    ) -> AdapterReply:
        if not self.is_loaded:
            self.load()
        ep = self._ensure_episode()

        if not ep.started:
            ep.plan = self._build_plan(goal, vocabulary)
            ep.started = True
            _LOG.info("g1.VLA: episode start, plan=%d actions", len(ep.plan))

        action_vec = self._infer(observation, goal)
        norm_str = (
            f"{float(np.linalg.norm(action_vec)):.3f}"
            if action_vec is not None else "nan"
        )

        if not ep.plan:
            return AdapterReply(actions=[], done=True, note="g1.VLA: plan exhausted")

        if len(ep.completed) >= self.MAX_TICKS:
            return AdapterReply(
                actions=[], done=True,
                note=f"g1.VLA: tick budget {self.MAX_TICKS} reached",
            )

        action = ep.plan.pop(0)
        ep.completed.append(action.name)
        note = (
            f"g1.VLA tick {len(ep.completed)}: vla_norm={norm_str} -> "
            f"{action.name} [PLAN remain={len(ep.plan)}]"
        )
        return AdapterReply(actions=[action], done=False, note=note)
