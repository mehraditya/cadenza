"""cadenza.g1 — Developer API for the Unitree G1 humanoid.

Same structure as Go1: define actions, run them.
The robot moves ONLY through motor commands. No teleporting.

Usage::

    import cadenza

    g1 = cadenza.g1()
    g1.run([
        g1.stand(),
        g1.crouch(),
        g1.walk_forward(distance_m=1.0),
        g1.stand(),
        g1.jump(),
        g1.stand(),
    ])
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import mujoco, mujoco.viewer

from cadenza.go1 import Step  # reuse the same Step descriptor


class G1:
    """Unitree G1 humanoid controller.

    Example::

        import cadenza
        g1 = cadenza.g1()
        g1.run([
            g1.stand(),
            g1.crouch(),
            g1.walk_forward(distance_m=1.0),
            g1.stand(),
        ])
    """

    def __init__(self, cam_distance: float = 4.0, cam_elevation: float = -15,
                 cam_azimuth: float = 120, xml_path: str | None = None):
        self._cam_distance = cam_distance
        self._cam_elevation = cam_elevation
        self._cam_azimuth = cam_azimuth
        self._xml_path = xml_path
        self._model = None
        self._sense: list = []

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self, *, model=None, sense=None):
        """Attach a world-model adapter and perception modalities for the next run.

        Example::

            from ai_models.g1 import VLA, Depth, RGB
            g1 = cadenza.g1()
            g1.setup(model=VLA(), sense=[Depth(), RGB()])
            g1.run(goal="walk to the chair and sit", target=(2.0, 0.0))
        """
        self._model = model
        self._sense = list(sense or [])
        return self

    # ── Action methods (return descriptors, no execution) ────────────────

    def stand(self, duration=2.0):
        return Step("stand", speed=duration)

    def crouch(self, duration=2.0):
        return Step("crouch", speed=duration)

    def walk_forward(self, distance_m=1.0, **kw):
        return Step("walk_forward", distance_m=distance_m)

    def jump(self, **kw):
        return Step("jump")

    def hold(self, duration=1.0):
        return Step("hold", speed=duration)

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self, sequence: list | None = None, *,
            goal: str | None = None,
            scene: str | None = None,
            target: tuple[float, float] | None = None,
            on: str | None = None,
            model=None, sense=None,
            max_iterations: int = 250,
            headless: bool = False,
            verbose: bool = True):
        """Execute a sequence of actions, or drive a goal with a world model.

        Two shapes:

        1. Scripted::

            g1.run([g1.stand(), g1.walk_forward(distance_m=1.0), g1.crouch()])

        2. World-model-driven (after ``setup``)::

            g1.setup(model=VLA(), sense=[Depth(), RGB()])
            g1.run(goal="walk to the chair", target=(2.0, 0.0))
        """
        if goal is not None:
            return self._run_goal(
                goal=goal, scene=scene, target=target, on=on,
                model=model if model is not None else self._model,
                sense=sense if sense is not None else self._sense,
                max_iterations=max_iterations, headless=headless, verbose=verbose,
            )
        if sequence is None:
            raise TypeError(
                "G1.run() requires either a list of Steps or goal='...'"
            )
        return self._run_sequence(sequence)

    def _run_goal(self, *, goal, scene, target, on, model, sense,
                  max_iterations, headless, verbose):
        if on is not None:
            raise NotImplementedError(
                f"on={on!r} not yet wired for goal mode; default sim only."
            )
        from cadenza.stack import run as stack_run
        return stack_run(
            robot="g1",
            goal=goal,
            target=target,
            world_model=model,
            modalities=sense or [],
            xml_path=scene if scene else self._xml_path,
            max_iterations=max_iterations,
            headless=headless,
            verbose=verbose,
        )

    def _run_sequence(self, sequence: list):
        """Execute actions in MuJoCo. Continuous physics, no teleporting."""
        from cadenza.g1_gait import (
            setup_model, _exec_stand, _exec_crouch,
            _exec_walk, _exec_jump, _hold,
        )

        model, data = setup_model()
        steps = [s if isinstance(s, Step) else Step(s) for s in sequence]

        print(f"\n  Cadenza G1  |  {len(steps)} steps\n")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = self._cam_distance
            viewer.cam.elevation = self._cam_elevation
            viewer.cam.azimuth = self._cam_azimuth

            for i, step in enumerate(steps):
                if not viewer.is_running():
                    break

                print(f"  [{i+1}/{len(steps)}] {step.name}", end="")
                if step.distance_m > 0:
                    print(f"  {step.distance_m}m", end="")
                print()

                if step.name == "stand":
                    _exec_stand(model, data, step.speed, viewer)
                elif step.name == "crouch":
                    _exec_crouch(model, data, step.speed, viewer)
                elif step.name == "walk_forward":
                    _exec_walk(model, data, step.distance_m or 1.0, viewer)
                elif step.name == "jump":
                    _exec_jump(model, data, viewer)
                elif step.name == "hold":
                    _hold(model, data, step.speed, viewer)

            print("\n  Done. Close viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.02)

    # ── Deploy ───────────────────────────────────────────────────────────

    def deploy(self, sequence: list, **kw):
        """Deploy actions to a physical G1 over DDS."""
        from cadenza.deploy.g1_driver import G1Driver
        steps = [s if isinstance(s, Step) else Step(s) for s in sequence]
        driver = G1Driver(**kw)
        driver.connect()
        try:
            driver.deploy(steps)
        finally:
            driver.disconnect()

    def deploy_ssh(self, script: str, host: str = "192.168.123.164",
                   user: str = "unitree", key: str | None = None, **kw):
        """Deploy and run a script on the physical G1 over SSH."""
        from cadenza.deploy.ssh import SSHDeploy
        conn = SSHDeploy(host=host, user=user, key=key, **kw)
        conn.deploy_and_run(script)
