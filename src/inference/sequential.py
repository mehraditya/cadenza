"""Sequential — the original cadenza VLA flow as a named orchestrator.

For each action the robot executes, a guardian (default: ``VLAGuardian``)
monitors the forward camera. When an obstacle is detected mid-action:

  1. The current step is cut short at whatever fraction was completed.
  2. The guardian emits an avoidance sequence (turn / side-step / wait).
  3. The avoidance sequence runs with the guardian off (no recursive interrupts).
  4. The original action resumes with the remaining distance, until completed
     or interrupted again — bounded by ``retries`` so the robot can't thrash.

Constructor knobs:

    Sequential(
        show_camera=True,                # show guardian's live camera window
        model_id=None,                   # override the guardian's VLM checkpoint
        min_resume_distance_m=0.1,       # below this we abandon, don't resume
        retries=5,                       # max VLA interrupts per step (None=∞)
        guardian=None,                   # custom detector — class, factory, or instance
        logging="runs/episode.jsonl",    # write JSON-Lines event log for later training
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from cadenza.inference.base import InferenceOrchestrator


class Sequential(InferenceOrchestrator):
    """Single-threaded think-then-move VLA orchestration."""

    name = "Sequential"

    def __init__(
        self,
        *,
        show_camera: bool = True,
        model_id: str | None = None,
        min_resume_distance_m: float = 0.1,
        retries: int | None = 5,
        guardian: Any | None = None,
        logging: str | Path | None = None,
    ):
        """
        Args:
            show_camera: show the guardian's live camera window.
            model_id: override the guardian's VLM checkpoint (default
                guardian's own default — usually SmolVLM-Instruct).
            min_resume_distance_m: if an interrupted action has less than
                this much distance left, drop it instead of resuming.
            retries: hard cap on how many VLA interrupts a single step may
                trigger before we abandon it and move on. ``None`` = ∞.
            guardian: custom obstacle detector. Accepts a class, a factory
                callable, or an already-built instance. If ``None``, uses
                ``cadenza.vla.VLAGuardian``. The object must expose
                ``load()`` and ``get_avoidance_steps(result)``, and the
                results it returns must duck-type the VLAGuardian
                ``ObstacleResult`` shape (``detected``, ``position``,
                ``size``, ``_steps_completed``, ``_steps_total``).
            logging: path to a JSON-Lines log file. Every lifecycle event
                (step_start, detect, avoid_start/step/complete, resume,
                skip, retries_exceeded, step_complete, session_start/end)
                is appended as one JSON object per line — easy to load for
                offline analysis or training data.
        """
        self.show_camera = show_camera
        self.model_id = model_id
        self.min_resume_distance_m = float(min_resume_distance_m)
        self.retries = retries
        self._guardian_factory = guardian
        self._log_path = Path(logging) if logging is not None else None

        # Populated in setup() / teardown().
        self._guardian: Any = None
        self._robot_name: str | None = None
        self._log_file = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self, robot_name: str, sim: Any, lib: Any) -> None:
        self._robot_name = robot_name
        self._guardian = self._build_guardian(robot_name)
        self._guardian.load()

        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", buffering=1)
            self._emit("session_start", robot=robot_name)

    def teardown(self) -> None:
        if self._log_file is not None:
            try:
                self._emit("session_end")
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self._guardian = None
        self._robot_name = None

    # ── Guardian construction ────────────────────────────────────────────────

    def _build_guardian(self, robot_name: str) -> Any:
        """Resolve the ``guardian=`` argument into a live guardian instance."""
        gf = self._guardian_factory
        if gf is None:
            from cadenza.vla import VLAGuardian
            kwargs: dict[str, Any] = {"show_camera": self.show_camera}
            if self.model_id is not None:
                kwargs["model_id"] = self.model_id
            return VLAGuardian(robot_name, **kwargs)
        # Class → instantiate with the robot name. Check this BEFORE the
        # duck-type test, since classes also expose unbound `load` etc.
        if isinstance(gf, type):
            return gf(robot_name)
        # Already built? Use it directly.
        if hasattr(gf, "get_avoidance_steps") and hasattr(gf, "load"):
            return gf
        # Factory callable → call with the robot name.
        if isinstance(gf, Callable):  # type: ignore[arg-type]
            try:
                return gf(robot_name=robot_name)
            except TypeError:
                return gf(robot_name)
        raise TypeError(
            "`guardian=` must be a class, factory callable, or pre-built "
            f"instance with .load()/.get_avoidance_steps() — got {type(gf).__name__!r}"
        )

    # ── JSONL logging ────────────────────────────────────────────────────────

    def _emit(self, event: str, **payload: Any) -> None:
        if self._log_file is None:
            return
        record = {"ts": time.time(), "event": event, **payload}
        try:
            self._log_file.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Logging must never break execution.
            pass

    @staticmethod
    def _step_payload(step: Any) -> dict[str, Any]:
        """Compact JSON-friendly snapshot of a Step."""
        return {
            "name": getattr(step, "name", None),
            "speed": getattr(step, "speed", None),
            "extension": getattr(step, "extension", None),
            "distance_m": getattr(step, "distance_m", None),
            "rotation_rad": getattr(step, "rotation_rad", None),
            "repeat": getattr(step, "repeat", None),
        }

    # ── Per-step ─────────────────────────────────────────────────────────────

    def run_step(
        self,
        step: Any,
        sim: Any,
        lib: Any,
        viewer: Any,
        robot: Any,
    ) -> None:
        """Run one step, retrying with reduced distance on every interruption,
        bounded by ``self.retries``."""
        current = step
        retry_count = 0
        self._emit("step_start", step=self._step_payload(step))

        while True:
            if not viewer.is_running():
                return

            result = robot._execute_single(
                current, sim, lib, viewer, vla_guardian=self._guardian,
            )

            if result is None or not getattr(result, "detected", False):
                self._emit("step_complete", step=self._step_payload(current))
                return

            # Interruption: figure out how much of the action was completed.
            completed_frac = 0.0
            if hasattr(result, "_steps_completed") and hasattr(result, "_steps_total"):
                completed_frac = result._steps_completed / max(result._steps_total, 1)
            original = float(getattr(current, "distance_m", 0.0) or 0.0)
            distance_done = original * completed_frac
            distance_left = original - distance_done

            self._emit(
                "detect",
                attempt=retry_count + 1,
                position=getattr(result, "position", None),
                size=getattr(result, "size", None),
                distance_done=distance_done,
                distance_left=distance_left,
                step=self._step_payload(current),
            )

            print(f"\n  [Sequential] VLA INTERRUPT: obstacle {result.position} "
                  f"({result.size})")
            print(f"               Completed {distance_done:.1f}m of {original:.1f}m")
            print(f"               Remaining: {distance_left:.1f}m  "
                  f"(retry {retry_count + 1}"
                  + (f"/{self.retries}" if self.retries is not None else "") + ")")

            # Retry cap: stop thrashing after `retries` interrupts.
            if self.retries is not None and retry_count >= self.retries:
                self._emit(
                    "retries_exceeded",
                    attempts=retry_count + 1,
                    distance_left=distance_left,
                    step=self._step_payload(current),
                )
                print(f"\n  [Sequential] retries exhausted ({self.retries}); "
                      f"abandoning {current.name}\n")
                return

            # Run guardian-provided avoidance with VLA off (no recursive checks).
            avoidance = self._guardian.get_avoidance_steps(result) or []
            self._emit(
                "avoid_start",
                attempt=retry_count + 1,
                steps=[self._step_payload(s) for s in avoidance],
            )
            if avoidance:
                print(f"               Avoidance: {[s.name for s in avoidance]}\n")
                for av_step in avoidance:
                    if not viewer.is_running():
                        return
                    self._emit("avoid_step", step=self._step_payload(av_step))
                    print(f"    >> {av_step.name}", end="")
                    if getattr(av_step, "distance_m", 0) > 0:
                        print(f"  {av_step.distance_m:.1f}m", end="")
                    print()
                    robot._execute_single(av_step, sim, lib, viewer)
                    viewer.cam.lookat[:] = sim.data.qpos[0:3]
                    viewer.cam.lookat[2] = max(float(sim.data.qpos[2]) * 0.8, 0.15)
            self._emit("avoid_complete", attempt=retry_count + 1)

            retry_count += 1

            # Resume the original step with whatever distance is left.
            if distance_left > self.min_resume_distance_m:
                current = replace(current, distance_m=distance_left)
                self._emit(
                    "resume",
                    attempt=retry_count,
                    distance_left=distance_left,
                    step=self._step_payload(current),
                )
                print(f"\n  [Sequential] RESUME: {current.name} "
                      f"{distance_left:.1f}m remaining\n")
                continue

            self._emit(
                "skip",
                attempt=retry_count,
                distance_left=distance_left,
                reason="below min_resume_distance_m",
                step=self._step_payload(current),
            )
            print(f"\n  [Sequential] action was nearly complete; moving on\n")
            return
