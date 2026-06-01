"""SmolVLA-based world-model adapter tuned for the Go1 quadruped.

Subclasses cadenza.WorldModelAdapter. Instantiate and pass to
``go1.setup(model=VLA())``; the cadenza stack will drive its
``propose_actions`` each tick.

Closed-loop perceive-reason-act loop (when ``target=(x, y)`` is supplied
to ``go1.run``):

  1. Goal reached  -> emit ``sit`` and signal done.
  2. Stuck         -> hand off to depth + VLM vision navigator.
  3. Climbable step ahead -> ``climb_step``.
  4. Obstacle blocking the path -> side-step toward the clearer side.
  5. Misaligned with target heading -> ``turn_left`` / ``turn_right``.
  6. Path clear and aligned -> ``walk_forward`` (chunked).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadenza import WorldModelAdapter, AdapterReply, ProposedAction
from cadenza.actions.library import ActionCall  # noqa: F401  (re-export point)
from cadenza.stack.trajectory import TrajectoryMonitor
from cadenza.stack.vocabulary import ActionVocabulary


_LOG = logging.getLogger("ai_models.go1.vla")


@dataclass
class _EpisodeState:
    plan: list[ProposedAction] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    last_action_vec: np.ndarray | None = None
    settled_ticks: int = 0
    started: bool = False
    sequential_dodges: int = 0
    trajectory: TrajectoryMonitor = field(default_factory=TrajectoryMonitor)
    consecutive_recoveries: int = 0


class VLA(WorldModelAdapter):
    """SmolVLA-driven adapter for Go1."""

    name = "go1_vla"
    description = "HuggingFace SmolVLA closed-loop adapter for Go1."
    DEFAULT_MODEL_ID = "lerobot/smolvla_base"

    # ── Tunables ─────────────────────────────────────────────────────────────
    MAX_TICKS = 250
    SETTLE_THRESHOLD = 3
    LOW_MOTION_NORM = 0.05

    OBSTACLE_TRIGGER_M = 0.55
    OBSTACLE_CLEAR_M = 0.85
    MAX_SEQUENTIAL_DODGES = 6
    CLIMB_STEP_MIN_M = 0.05
    CLIMB_STEP_MAX_M = 0.20

    HEADING_TOL_DEG = 25.0
    MAX_TURN_RAD = 1.2

    WALK_CHUNK_MAX_M = 0.7
    WALK_CHUNK_FRAC = 0.4

    PROGRESS_WINDOW = 4
    PROGRESS_MIN_M = 0.10
    ARRIVAL_M = 0.45
    MAX_RECOVERIES = 12

    # ── Detection ────────────────────────────────────────────────────────────

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        """User instantiates directly via ``go1.setup(model=VLA())``."""
        return None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _load_impl(self) -> None:
        if self.model is not None:
            return
        try:
            from lerobot.policies.smolvla.modeling_smolvla import (    # type: ignore
                SmolVLAPolicy,
            )
        except ImportError as e:
            raise RuntimeError(
                "ai_models.go1.VLA requires `lerobot[smolvla]`. Install with:\n"
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

    # ── Vision navigator (lazy) ──────────────────────────────────────────────

    def _ensure_navigator(self):
        if getattr(self, "_navigator", None) is not None:
            return self._navigator
        try:
            from cadenza.stack.vision import VisionNavigator
        except ImportError:
            self._navigator = None
            return None
        self._navigator = VisionNavigator()
        return self._navigator

    def _attempt_vision_recovery(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        cam = observation.get("camera")
        target = observation.get("target_xy")
        pos = observation.get("pos")
        rpy = observation.get("rpy")
        if cam is None or target is None or pos is None or rpy is None:
            return None
        navigator = self._ensure_navigator()
        if navigator is None:
            return None
        try:
            decision = navigator.decide(
                rgb=np.asarray(cam),
                target_xy=(float(target[0]), float(target[1])),
                robot_xy=(float(pos[0]), float(pos[1])),
                robot_yaw=float(rpy[2]),
            )
        except Exception as e:
            _LOG.warning("vision navigator failed: %s", e)
            return None
        if decision.action not in vocabulary:
            return None
        params: dict[str, Any] = {}
        if decision.action == "walk_forward":
            params["distance_m"] = 0.6
        elif decision.action in {"side_step_left", "side_step_right"}:
            params["distance_m"] = 0.30
        elif decision.action in {"turn_left", "turn_right"}:
            params["rotation_rad"] = max(
                0.5,
                min(1.5, math.radians(abs(decision.target_bearing_deg))),
            )
        return ProposedAction(
            name=decision.action,
            params=params,
            rationale=decision.rationale,
        )

    # ── Inference (perception) ───────────────────────────────────────────────

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

    # ── Reasoning helpers ────────────────────────────────────────────────────

    def _build_plan(self, goal: str, vocabulary: ActionVocabulary) -> list[ProposedAction]:
        decoder = self._ensure_lora_decoder(vocabulary)
        report = decoder.decode_report(goal)
        if report.truncated:
            _LOG.warning(
                "go1.VLA: plan truncated at max_seq_len=%d; raise the head's "
                "max_seq_len if longer plans are needed.",
                decoder.head.max_seq_len,
            )
        plans: list[ProposedAction] = []
        for call, conf in zip(report.calls, report.action_confidence):
            params: dict[str, Any] = {"speed": call.speed, "repeat": call.repeat}
            if call.distance_m > 0:
                params["distance_m"] = call.distance_m
            if call.rotation_rad != 0:
                params["rotation_rad"] = call.rotation_rad
            if call.duration_s > 0:
                params["duration_s"] = call.duration_s
            plans.append(ProposedAction(
                name=call.action_name, params=params,
                rationale=f"lora head (conf={conf:.2f})",
            ))
        return plans

    def _ensure_lora_decoder(self, vocabulary: ActionVocabulary):
        """Lazily build the LoRA decoder keyed to this vocabulary.

        The decoder is rebuilt if the vocabulary identity changes — important
        because the action-name -> index table is baked into the head.
        """
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

    def _terrain_override(
        self,
        next_action: ProposedAction,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction:
        if next_action.name not in {"walk_forward", "trot_forward", "pace_forward"}:
            return next_action
        terrain = observation.get("terrain_ahead") or {}
        step = float(terrain.get("max_step_up") or 0.0)
        if (self.CLIMB_STEP_MIN_M < step < self.CLIMB_STEP_MAX_M
                and "climb_step" in vocabulary):
            return ProposedAction(
                name="climb_step",
                params={},
                rationale=f"perception: step_up={step:.2f}m ahead",
            )
        return next_action

    def _obstacle_detour(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        obs = observation.get("obstacles_ahead") or {}
        center = obs.get("center_m")
        left = obs.get("left_m")
        right = obs.get("right_m")
        if center is None and left is None and right is None:
            return None
        nearest = min(d for d in (center, left, right) if d is not None)
        if nearest >= self.OBSTACLE_TRIGGER_M:
            return None
        terrain = observation.get("terrain_ahead") or {}
        step = float(terrain.get("max_step_up") or 0.0)
        if self.CLIMB_STEP_MIN_M < step < self.CLIMB_STEP_MAX_M:
            return None
        max_range = float(obs.get("max_range_m") or 1.5)
        left_clear = max_range if left is None else float(left)
        right_clear = max_range if right is None else float(right)
        if left_clear > right_clear + 0.05 and "side_step_left" in vocabulary:
            name, chosen, available = "side_step_left", "left", left_clear
        elif "side_step_right" in vocabulary:
            name, chosen, available = "side_step_right", "right", right_clear
        else:
            return None
        return ProposedAction(
            name=name,
            params={"distance_m": 0.30},
            rationale=(
                f"obstacle reasoning: nearest={nearest:.2f}m "
                f"(side={obs.get('side')}); going {chosen} "
                f"(clearance={available:.2f}m)"
            ),
        )

    def _is_settled(self, action_vec: np.ndarray | None) -> bool:
        ep = self._ensure_episode()
        if action_vec is None:
            return False
        norm = float(np.linalg.norm(action_vec))
        if norm < self.LOW_MOTION_NORM:
            ep.settled_ticks += 1
        else:
            ep.settled_ticks = 0
        return ep.settled_ticks >= self.SETTLE_THRESHOLD

    @staticmethod
    def _target_bearing(
        target_xy: tuple[float, float],
        robot_xy: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float]:
        dx = target_xy[0] - robot_xy[0]
        dy = target_xy[1] - robot_xy[1]
        target_world_heading = math.atan2(dy, dx)
        # Cadenza convention: forward = -x in body frame ⇒ world = yaw + π.
        robot_world_heading = robot_yaw + math.pi
        bearing = target_world_heading - robot_world_heading
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(bearing), math.hypot(dx, dy)

    def _depth_modality_detour(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        d_left = observation.get("depth_left")
        d_center = observation.get("depth_center")
        d_right = observation.get("depth_right")
        if d_left is None or d_center is None or d_right is None:
            return None
        max_side = max(float(d_left), float(d_right))
        if d_center >= max_side * 0.75:
            return None
        if d_left > d_right and "side_step_left" in vocabulary:
            name, side, clear = "side_step_left", "left", float(d_left)
        elif "side_step_right" in vocabulary:
            name, side, clear = "side_step_right", "right", float(d_right)
        else:
            return None
        return ProposedAction(
            name=name,
            params={"distance_m": 0.30},
            rationale=(
                f"depth modality: centre={d_center:.2f} < sides "
                f"({d_left:.2f}/{d_right:.2f}); going {side}"
            ),
        )

    def _beacon_alignment(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
    ) -> ProposedAction | None:
        """If the RGB modality sees a beacon, nudge heading toward it."""
        if not observation.get("beacon_visible"):
            return None
        bearing_px = float(observation.get("beacon_bearing_px") or 0.0)
        if abs(bearing_px) < 0.20:
            return None  # already roughly centered
        # bearing_px > 0 means beacon is to the right of frame center.
        # Cadenza convention: turn_left is positive yaw, turn_right negative.
        rad = max(0.3, min(0.8, abs(bearing_px) * 1.0))
        name = "turn_right" if bearing_px > 0 else "turn_left"
        if name not in vocabulary:
            return None
        return ProposedAction(
            name=name,
            params={"rotation_rad": rad},
            rationale=f"beacon alignment: bearing_px={bearing_px:+.2f}",
        )

    # ── Loop step ────────────────────────────────────────────────────────────

    def _closed_loop_step(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
        ep: _EpisodeState,
    ) -> tuple[ProposedAction, str]:
        pos = observation["pos"]
        rpy = observation.get("rpy", [0.0, 0.0, 0.0])
        target = ep.trajectory.target_xy
        bearing_deg, distance_m = self._target_bearing(
            (float(target[0]), float(target[1])),
            (float(pos[0]), float(pos[1])),
            float(rpy[2]),
        )

        if ep.trajectory.is_stuck and ep.consecutive_recoveries < self.MAX_RECOVERIES:
            recovery = self._attempt_vision_recovery(observation, vocabulary)
            if recovery is not None:
                ep.consecutive_recoveries += 1
                ep.trajectory.reset_after_recovery()
                return recovery, f"VISION_RECOVERY[{ep.consecutive_recoveries}/{self.MAX_RECOVERIES}]"

        terrain = observation.get("terrain_ahead") or {}
        step = float(terrain.get("max_step_up") or 0.0)
        if (self.CLIMB_STEP_MIN_M < step < self.CLIMB_STEP_MAX_M
                and "climb_step" in vocabulary):
            return ProposedAction(
                name="climb_step",
                params={},
                rationale=f"climbable step ahead ({step:.2f}m)",
            ), "CLIMB"

        if ep.sequential_dodges <= self.MAX_SEQUENTIAL_DODGES:
            detour = self._obstacle_detour(observation, vocabulary)
            if detour is None:
                detour = self._depth_modality_detour(observation, vocabulary)
            if detour is not None:
                ep.sequential_dodges += 1
                return detour, f"DETOUR[{ep.sequential_dodges}/{self.MAX_SEQUENTIAL_DODGES}]"
        ep.sequential_dodges = 0

        if abs(bearing_deg) > self.HEADING_TOL_DEG:
            turn_rad = max(0.4, min(self.MAX_TURN_RAD, math.radians(abs(bearing_deg))))
            name = "turn_left" if bearing_deg > 0 else "turn_right"
            if name in vocabulary:
                return ProposedAction(
                    name=name,
                    params={"rotation_rad": turn_rad},
                    rationale=f"align: bearing={bearing_deg:+.0f}°",
                ), "TURN"

        beacon_turn = self._beacon_alignment(observation, vocabulary)
        if beacon_turn is not None:
            return beacon_turn, "BEACON_ALIGN"

        chunk = max(0.20, min(self.WALK_CHUNK_MAX_M, distance_m * self.WALK_CHUNK_FRAC))
        return ProposedAction(
            name="walk_forward",
            params={"distance_m": chunk},
            rationale=f"advance: dist={distance_m:.2f}m bearing={bearing_deg:+.0f}°",
        ), "ADVANCE"

    def _legacy_plan_step(
        self,
        observation: dict,
        vocabulary: ActionVocabulary,
        ep: _EpisodeState,
    ) -> tuple[ProposedAction | None, str, bool]:
        if not ep.plan:
            return None, "PLAN_EMPTY", True
        next_action = ep.plan.pop(0)
        next_action = self._terrain_override(next_action, observation, vocabulary)
        return next_action, f"PLAN[remain={len(ep.plan)}]", False

    # ── Per-tick API ─────────────────────────────────────────────────────────

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
            target = observation.get("target_xy")
            ep.trajectory = TrajectoryMonitor(
                target_xy=target,
                window=self.PROGRESS_WINDOW,
                min_progress_m=self.PROGRESS_MIN_M,
                arrival_distance_m=self.ARRIVAL_M,
            )
            ep.plan = [] if target is not None else self._build_plan(goal, vocabulary)
            ep.started = True
            mode = "closed-loop" if target is not None else f"plan({len(ep.plan)})"
            _LOG.info("go1.VLA: episode start, mode=%s, target=%s", mode, target)

        pos = observation.get("pos")
        if pos is not None and ep.trajectory.target_xy is not None:
            ep.trajectory.update((float(pos[0]), float(pos[1])))

        action_vec = self._infer(observation, goal)
        ep.last_action_vec = action_vec
        norm_str = (
            f"{float(np.linalg.norm(action_vec)):.3f}"
            if action_vec is not None else "nan"
        )

        n_ticks = len(ep.completed)

        if ep.trajectory.at_target:
            return AdapterReply(
                actions=[ProposedAction(name="sit", params={},
                                        rationale="reached target")],
                done=True,
                note=f"go1.VLA: arrived ({ep.trajectory.progress_summary()})",
            )
        if n_ticks >= self.MAX_TICKS:
            return AdapterReply(
                actions=[], done=True,
                note=f"go1.VLA: tick budget {self.MAX_TICKS} reached "
                     f"({ep.trajectory.progress_summary()})",
            )

        if ep.trajectory.target_xy is not None:
            action, branch = self._closed_loop_step(observation, vocabulary, ep)
        else:
            action, branch, exhausted = self._legacy_plan_step(observation, vocabulary, ep)
            if action is None:
                done = True
                if exhausted and self._is_settled(action_vec):
                    note = "go1.VLA: plan exhausted and robot settled"
                else:
                    note = "go1.VLA: plan exhausted"
                return AdapterReply(actions=[], done=done, note=note)

        ep.completed.append(action.name)
        progress = ep.trajectory.progress_summary() if ep.trajectory.target_xy else "no target"
        note = (
            f"go1.VLA tick {n_ticks + 1}: "
            f"vla_norm={norm_str} -> {action.name} "
            f"[{branch}; {action.rationale}; {progress}]"
        )
        return AdapterReply(actions=[action], done=False, note=note)
