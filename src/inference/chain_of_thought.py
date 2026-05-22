"""ChainOfThought — concurrent inference + motor execution.

The VLA model holds every decision. While the robot is *moving* through
action N, a background thread is already running the next inference pass
on the latest multi-modal observation to produce action N+1. By the time
N's motors finish, N+1 is queued and fires immediately — no dead time
waiting for the model to think.

Pipeline (1-deep, single background worker)::

    bootstrap:     [ infer A0 ]
    tick 0  main:  ════════════════ exec A0 ════════════════
            bg:                        [ infer A1 ]
    tick 1  main:  ════════════════ exec A1 ════════════════
            bg:                        [ infer A2 ]
    ...

The model receives the observation captured *at the start* of each
execution slot, so its decision for N+1 reflects the world the moment
N begins — close to "ground truth at the boundary" for short actions.

Construction::

    from cadenza.inference import ChainOfThought
    from ai_models.go1 import VLA, Depth, RGB

    go1 = cadenza.go1(
        xml_path="scene.xml",
        inference=ChainOfThought(
            model=VLA(),                # any WorldModelAdapter
            sense=[Depth(), RGB()],     # multi-modal inputs merged in obs
            goal="reach the far wall",
            target=(-4.5, 0.0),
            max_steps=80,
            logging="run.jsonl",
        ),
    )
    go1.run([go1.walk_forward()])       # any trigger step — orchestrator takes over
"""

from __future__ import annotations

import concurrent.futures as _futures
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from cadenza.inference.base import InferenceOrchestrator


class ChainOfThought(InferenceOrchestrator):
    """Concurrent inference + motor execution. Model decides every action."""

    name = "ChainOfThought"

    def __init__(
        self,
        *,
        model: Any = None,
        sense: list | None = None,
        goal: str = "navigate the field",
        target: tuple[float, float] | None = None,
        max_steps: int = 80,
        arrival_m: float = 0.45,
        chunk_distance_m: float = 0.40,
        chunk_rotation_rad: float = 0.35,
        logging: str | Path | None = None,
    ):
        """
        Args:
            model: ``WorldModelAdapter`` that returns ONE action per tick.
                Default: ``ai_models.go1.VLA`` (closed-loop SmolVLA-based).
            sense: list of ``Modality`` instances. Their outputs are merged
                into the observation dict the model sees.
            goal: natural-language goal passed to the model each tick.
            target: optional (x, y). Used both for arrival termination and
                handed to the model via ``observation["target_xy"]``.
            max_steps: hard cap on the number of actions executed.
            arrival_m: arrival ring around ``target`` that ends the run.
            chunk_distance_m: cap on distance per ``walk_forward``-style
                action. Smaller = shorter motor bursts, faster transitions,
                tighter coupling to incoming model decisions. The motors
                run continuously across chunks because the loop never
                waits for inference — if the next decision isn't ready it
                just repeats the current action.
            chunk_rotation_rad: same idea for ``turn_*`` actions.
            logging: optional JSONL log path for offline analysis.
        """
        self.model = model
        self.sense = list(sense or [])
        self.goal = goal
        self.target = (
            (float(target[0]), float(target[1])) if target is not None else None
        )
        self.max_steps = int(max_steps)
        self.arrival_m = float(arrival_m)
        self.chunk_distance_m = float(chunk_distance_m)
        self.chunk_rotation_rad = float(chunk_rotation_rad)

        self._log_path = Path(logging) if logging is not None else None
        self._log_file = None
        self._pool: _futures.ThreadPoolExecutor | None = None
        self._vocab: Any = None
        self._renderer: Any = None
        self._robot_name: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self, robot_name: str, sim: Any, lib: Any) -> None:
        self._robot_name = robot_name
        self._lib = lib

        if self.model is None:
            try:
                from ai_models.go1 import VLA
                self.model = VLA()
            except Exception as e:
                raise RuntimeError(
                    "ChainOfThought needs a `model=` (WorldModelAdapter). "
                    "Default `ai_models.go1.VLA` could not be imported."
                ) from e

        from cadenza.stack.vocabulary import build_vocabulary
        self._vocab = build_vocabulary(robot_name, library=lib)

        # Camera renderer + a separate depth renderer (cheap once built).
        # The depth renderer is the input to SpatialMemory's visual-only map.
        try:
            import mujoco
            self._renderer = mujoco.Renderer(sim.model, 224, 224)
            self._depth_renderer = mujoco.Renderer(sim.model, 224, 224)
            try:
                self._depth_renderer.enable_depth_rendering()
            except AttributeError:
                # Older MuJoCo builds — fall back to no depth.
                self._depth_renderer = None
        except Exception:
            self._renderer = None
            self._depth_renderer = None

        # ── EAGER MODEL LOADING ─────────────────────────────────────────────
        # Every model is fully resident before the robot moves. No lazy
        # initialisation during the action loop.
        print(f"  [ChainOfThought] warming up models …")

        # 1. The world-model adapter itself (SmolVLA weights, etc.).
        if not getattr(self.model, "is_loaded", False):
            self.model.load()

        # 2. Modalities — each one's setup() pulls its own weights.
        for m in self.sense:
            m.setup()

        # 3. Sub-models the adapter would otherwise lazy-load mid-run.
        #    `ai_models.go1.VLA` keeps a VisionNavigator (DepthAnything +
        #    SmolVLM) behind `_ensure_navigator()` that only initialises
        #    when stuck-recovery first fires — we force it now.
        ensure_nav = getattr(self.model, "_ensure_navigator", None)
        if callable(ensure_nav):
            try:
                navigator = ensure_nav()
                if navigator is not None and hasattr(navigator, "load"):
                    navigator.load()
            except Exception as e:
                print(f"  [ChainOfThought] navigator preload skipped: {e}")

        # 4. One bootstrap inference + a render pass. Triggers any other
        #    one-shot initialisation in PyTorch / MuJoCo / model graphs so
        #    the first tick of the live loop pays no warm-up cost either.
        try:
            self._infer(self._observe(sim))
        except Exception as e:
            print(f"  [ChainOfThought] warm-up inference skipped: {e}")
        print(f"  [ChainOfThought] models ready — starting motion")
        self._stream_emit("models_ready",
                          model=type(self.model).__name__,
                          modalities=[type(m).__name__ for m in self.sense],
                          target=self.target)

        # ── Concurrency + persistent gait state ────────────────────────────
        self._pool = _futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cot-infer",
        )
        # Single gait engine spans the whole run; replaced ONLY when the
        # model picks a different gait kind. Same-gait chunks just pump it
        # for more steps → no per-chunk blend ramp → no jitter.
        self._gait_engine: Any = None
        self._current_gait_name: str | None = None

        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", buffering=1)
            self._emit("session_start", robot=robot_name, goal=self.goal,
                       target=self.target)

    def teardown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        for m in self.sense:
            try:
                m.teardown()
            except Exception:
                pass
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        if getattr(self, "_depth_renderer", None) is not None:
            try:
                self._depth_renderer.close()
            except Exception:
                pass
            self._depth_renderer = None
        self._gait_engine = None
        self._current_gait_name = None
        if self._log_file is not None:
            try:
                self._emit("session_end")
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    # ── Per-step (drives the whole episode) ──────────────────────────────────

    def run_step(self, step, sim, lib, viewer, robot) -> None:
        """`step` is a trigger; ChainOfThought drives the rest itself.

        True non-blocking pipeline: motors execute *short, chunked* actions
        in a tight loop and inference fires-and-forgets in the background.
        Every tick we *poll* the future — if a fresher decision is ready,
        we swap; if not, we just repeat the current action. The robot
        never stops to wait for the model.
        """
        assert self._pool is not None and self._vocab is not None
        from cadenza.stack.adapters.base import AdapterReply

        # Bootstrap action #0 synchronously (only blocking inference call).
        reply = self._infer(self._observe(sim))
        if not reply.actions:
            self._emit("done", reason="bootstrap_returned_no_action")
            self._stream_emit("done", reason="bootstrap_returned_no_action")
            return
        current = self._chunk_action(reply.actions[0])
        self._stream_emit("bootstrap", action=current.name,
                          params=current.params)

        in_flight = None         # next-action inference future, or None
        final_pending = False    # set when model emits done=True
        repeat_count = 0         # consecutive repeats of the same action
        done_reason = None

        for tick in range(self.max_steps):
            if not viewer.is_running():
                done_reason = "viewer_closed"
                break

            # Always have an inference in flight (unless one is mid-collection).
            if in_flight is None and not final_pending:
                in_flight = self._pool.submit(self._infer, self._observe(sim))

            # Execute the current chunked action — motors moving, viewer
            # syncing. This is the wall-clock time inference overlaps with.
            t0 = time.time()
            self._emit("execute_start", tick=tick, action=current.name,
                       params=current.params, rationale=current.rationale,
                       repeat=repeat_count)
            self._stream_emit("execute_start", tick=tick,
                              action=current.name,
                              dist=current.params.get("distance_m"),
                              rot=current.params.get("rotation_rad"),
                              repeat=repeat_count)
            self._drive_action(current, sim, lib, viewer, robot)
            exec_elapsed = round(time.time() - t0, 3)
            self._emit("execute_done", tick=tick, action=current.name,
                       elapsed_s=exec_elapsed)
            self._stream_emit("execute_done", tick=tick, elapsed_s=exec_elapsed)

            # Arrival predicate — cheap, runs in main.
            if self.target is not None:
                pos = sim.data.qpos[0:3]
                dist = float(np.hypot(pos[0] - self.target[0], pos[1] - self.target[1]))
                if dist <= self.arrival_m:
                    if in_flight is not None:
                        in_flight.cancel()
                    self._emit("target_reached", tick=tick, distance_m=dist)
                    self._stream_emit("target_reached", tick=tick,
                                      distance_m=dist)
                    done_reason = "target_reached"
                    break

            if final_pending:
                done_reason = "model_signaled_done"
                break

            # Non-blocking pickup: only swap if the in-flight inference is
            # done. Otherwise keep `current` and loop — motors continue.
            if in_flight is not None and in_flight.done():
                try:
                    reply = in_flight.result()
                except Exception as e:
                    self._emit("infer_error", tick=tick, error=str(e))
                    reply = AdapterReply(actions=[], done=False, note=str(e))
                in_flight = None
                if reply.actions:
                    new_action = self._chunk_action(reply.actions[0])
                    self._emit("infer_picked", tick=tick,
                               action=new_action.name,
                               params=new_action.params,
                               wait_s=0.0)
                    self._stream_emit("infer_picked", tick=tick,
                                      action=new_action.name,
                                      rationale=new_action.rationale or "")
                    # Same action? keep repeat counter; different? reset.
                    if (new_action.name == current.name
                            and new_action.params == current.params):
                        repeat_count += 1
                    else:
                        repeat_count = 0
                    current = new_action
                if reply.done:
                    final_pending = True
            else:
                # Inference still computing — keep current, motors keep moving.
                repeat_count += 1
                self._emit("infer_pending", tick=tick,
                           repeating=current.name,
                           repeat=repeat_count)
                self._stream_emit("infer_pending", tick=tick,
                                  repeating=current.name,
                                  repeat=repeat_count)

        if done_reason is None:
            done_reason = "max_steps_reached"
        if in_flight is not None:
            in_flight.cancel()
        self._emit("done", reason=done_reason)

    # ── Action chunking — keeps each motor burst short so transitions are
    # imperceptible and we re-read the model's intent often.
    def _chunk_action(self, action):
        from cadenza.stack.adapters.base import ProposedAction
        params = dict(action.params or {})
        gait_names = {"walk_forward", "walk_backward", "trot_forward",
                      "pace_forward", "crawl_forward", "bound_forward",
                      "side_step_left", "side_step_right"}
        turn_names = {"turn_left", "turn_right",
                      "precision_turn_left", "precision_turn_right"}
        if action.name in gait_names:
            existing = float(params.get("distance_m", 0.0) or 0.0)
            params["distance_m"] = (
                min(self.chunk_distance_m, existing) if existing > 0
                else self.chunk_distance_m
            )
        elif action.name in turn_names:
            existing = float(params.get("rotation_rad", 0.0) or 0.0)
            params["rotation_rad"] = (
                min(self.chunk_rotation_rad, existing) if existing > 0
                else self.chunk_rotation_rad
            )
        return ProposedAction(
            name=action.name, params=params,
            rationale=action.rationale,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _observe(self, sim: Any) -> dict[str, Any]:
        """Snapshot sim state + run every modality. Returns a dict the model
        adapter can read directly."""
        from cadenza.stack.gym_adapter import Observation

        state = sim.get_state()

        camera = None
        depth = None
        if self._renderer is not None:
            try:
                self._renderer.update_scene(sim.data, camera="forward")
                camera = self._renderer.render()
            except Exception:
                try:
                    self._renderer.update_scene(sim.data)
                    camera = self._renderer.render()
                except Exception:
                    camera = None
        if getattr(self, "_depth_renderer", None) is not None:
            try:
                self._depth_renderer.update_scene(sim.data, camera="forward")
                depth = self._depth_renderer.render()
            except Exception:
                try:
                    self._depth_renderer.update_scene(sim.data)
                    depth = self._depth_renderer.render()
                except Exception:
                    depth = None

        obs = Observation(
            pos=np.asarray(state["pos"], dtype=np.float32),
            rpy=np.array([state["roll"], state["pitch"], state["yaw"]],
                         dtype=np.float32),
            body_height=float(state["body_height"]),
            qpos=sim.data.qpos.copy(),
            qvel=sim.data.qvel.copy(),
            foot_contacts=tuple(bool(c) for c in state.get("foot_contacts", ())),
            terrain_ahead=state.get("terrain_ahead", {}),
            obstacles_ahead=state.get("obstacles_ahead", {}),
            camera=camera,
            depth=depth,
        )
        d = obs.to_dict()
        if self.target is not None:
            d["target_xy"] = self.target
        # When streaming is on, hand the channel to the model so it can
        # narrate ("I see a box on the left, going right") via .say().
        if self._stream is not None:
            d["stream"] = self._stream

        for m in self.sense:
            try:
                res = m.compute(obs)
                d.update(res.keys)
                # Modalities surface a non-empty ``summary`` only on
                # meaningful change (e.g. SpatialMemory just acquired a
                # landmark). Forward those to the stream — quiet ticks
                # produce no output.
                if res.summary:
                    self._stream_emit("modality_update", message=res.summary)
            except Exception:
                continue
        return d

    def _infer(self, observation: dict[str, Any]):
        """Run the model on a fully-formed observation dict.

        Called from the bg thread for every tick after bootstrap.
        """
        return self.model.propose_actions(
            observation=observation,
            goal=self.goal,
            vocabulary=self._vocab,
            history=None,
        )

    def _drive_action(self, action, sim, lib, viewer, robot) -> None:
        """Run one chunk's worth of motion.

        Gait actions (walk/turn/side_step) are pumped through a *persistent*
        ``GaitEngine`` so consecutive same-gait chunks join seamlessly —
        no per-chunk re-blend, no jitter. Phase actions (sit, jump,
        climb_step) keep going through ``robot._execute_single`` since
        they're one-shot motions, not steady-state gaits.
        """
        spec = lib.get(action.name)
        if spec.is_gait:
            self._pump_gait_chunk(action, spec, sim, viewer)
        else:
            from cadenza.go1 import Step
            params = action.params or {}
            step = Step(
                name=action.name,
                speed=float(params.get("speed", 1.0)),
                extension=float(params.get("extension", 1.0)),
                repeat=int(params.get("repeat", 1)),
                distance_m=float(params.get("distance_m", 0.0)),
                rotation_rad=float(params.get("rotation_rad", 0.0)),
            )
            robot._execute_single(step, sim, lib, viewer)
            # Phase action ends in a different pose; drop persistent gait so
            # the next gait chunk re-initialises cleanly.
            self._gait_engine = None
            self._current_gait_name = None

        viewer.cam.lookat[:] = sim.data.qpos[0:3]
        viewer.cam.lookat[2] = max(float(sim.data.qpos[2]) * 0.8, 0.15)

    # ── Persistent gait pump ─────────────────────────────────────────────────

    def _pump_gait_chunk(self, action, spec, sim, viewer) -> None:
        """Run one chunk of a gait by stepping a persistent ``GaitEngine``.

        Skips ``_run_gait``'s blend ramp on continuations — that ramp is what
        causes the visible jitter when actions are chunked.
        """
        import time
        from cadenza.locomotion.gait_engine import GaitEngine
        from cadenza.sim import _rpy

        _HZ = 50.0
        gait = spec.gait

        # Rebuild only when the gait kind genuinely changes (walk → trot, etc.).
        if self._gait_engine is None or self._current_gait_name != gait.gait_name:
            self._gait_engine = GaitEngine(
                sim.spec, gait_name=gait.gait_name, body_height=gait.body_height,
            )
            self._current_gait_name = gait.gait_name

        params = action.params or {}
        speed_scale = float(params.get("speed", 1.0))
        speed = float(spec.speed_ms) * speed_scale if spec.speed_ms > 0 else 0.5

        distance = float(params.get("distance_m", 0.0) or 0.0)
        rotation = float(params.get("rotation_rad", 0.0) or 0.0)
        if distance > 0:
            duration = distance / max(speed, 0.05)
        elif rotation > 0:
            yaw_speed = max(abs(gait.cmd_yaw) * speed_scale, 0.2)
            duration = rotation / yaw_speed
        else:
            duration = self.chunk_distance_m / max(speed, 0.05)

        n_steps = max(1, int(duration * _HZ))
        cmd = np.array([
            gait.cmd_vx * speed_scale,
            gait.cmd_vy * speed_scale,
            gait.cmd_yaw * speed_scale,
        ], dtype=np.float32)

        dt = 1.0 / _HZ
        t0 = time.monotonic()
        for s in range(n_steps):
            if not viewer.is_running():
                return
            rpy = _rpy(sim.data.qpos[3:7])
            q_gait = self._gait_engine.step(dt, cmd, rpy)
            sim.data.ctrl[:] = q_gait
            for _ in range(sim._phys):
                sim._step()
            viewer.sync()
            wait = t0 + (s + 1) / _HZ - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    # ── JSONL logging ────────────────────────────────────────────────────────

    def _emit(self, event: str, **payload: Any) -> None:
        if self._log_file is None:
            return
        try:
            record = {"ts": time.time(), "event": event, **payload}
            self._log_file.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass
