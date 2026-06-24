"""cadenza.arm — Developer API for the Cadenza 6-axis articulated arm.

Same shape as :mod:`cadenza.go1` / :mod:`cadenza.g1`: action methods return
lightweight descriptors, and ``run([...])`` executes them in MuJoCo. The arm is
fixed-base and Cartesian, so motion is driven by closed-form damped-least-squares
inverse kinematics (top-down approach) rather than gaits, and "grasping" is a
weld constraint the controller activates once the gripper is on the object.

Usage::

    import cadenza

    arm = cadenza.arm()
    arm.run([
        arm.home(),
        arm.pick((0.5, 0.0, 0.43)),     # grab the cube on the table
        arm.place((0.4, 0.22, 0.43)),   # set it down to the side
        arm.home(),
    ])

Run headless (no window) for tests/CI::

    arm.run([arm.pick((0.5, 0, 0.43)), arm.place((0.4, 0.22, 0.43))],
            headless=True)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import mujoco

from cadenza.actions.arm_library import ArmAction

_DEFAULT_XML = Path(__file__).resolve().parent.parent / "models" / "arm" / "scene.xml"

# Gripper finger commands (metres of slide travel per finger).
_GRIP_OPEN = 0.04
_GRIP_CLOSED = 0.022

# Pick/place geometry.
_HOVER = 0.11          # height to approach above a target before descending
_LIFT = 0.18           # height to lift to after grasping

# Hand/table safety. The fingertips hang ~11mm below the pinch site (the IK
# target) when the gripper points straight down. We never command the pinch
# lower than the work surface plus this clearance, so the hand always stops
# right above the table instead of being driven into (or through) it.
_FINGERTIP_BELOW_PINCH = 0.011
_SURFACE_MARGIN = 0.003
_TABLE_CLEARANCE = _FINGERTIP_BELOW_PINCH + _SURFACE_MARGIN

# Desired top-down gripper orientation (palm +z pointing world -z): Rx(pi).
_Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])

# Speed control. Motions ramp their command from the current pose to the target
# over a number of physics steps; ``speed`` scales that step count inversely
# (faster speed → fewer steps → higher velocity). It is clamped so a runaway
# value can't snap the servo or stall the arm. ``_CONVERGE`` is a short fixed
# hold at the target after the ramp so the pose is reached regardless of speed.
_SPEED_MIN, _SPEED_MAX = 0.1, 8.0
_CONVERGE = 120

# Sub-move phases of the compound actions, in execution order. ``pick``/``place``
# accept a ``speed`` dict keyed by these names so each phase runs at its own pace.
_PICK_PHASES = ("approach", "descend", "grasp", "lift")
_PLACE_PHASES = ("carry", "lower", "release", "retract")


def _speed_for(speed, phase: str) -> float:
    """Resolve the multiplier for one phase from a scalar or per-phase dict."""
    if isinstance(speed, dict):
        return float(speed.get(phase, 1.0))
    return float(speed)


class _Runtime:
    """The live MuJoCo session: IK, motion, and grasp for one run."""

    def __init__(self, xml_path: str, hz: float = 50.0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._phys = max(1, int(round(1.0 / (self.model.opt.timestep * hz))))

        self._scratch = mujoco.MjData(self.model)  # for IK / FK probes
        m = self.model
        self._site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self._cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
        self._lo = m.jnt_range[:6, 0].copy()
        self._hi = m.jnt_range[:6, 1].copy()

        # Start at the "home" keyframe if present, else zero pose.
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)
        self._grip = _GRIP_OPEN

        # Work-surface no-go floor: the top of the table and its (x, y) footprint,
        # read from the model so the clamp tracks the geometry, not magic numbers.
        tt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        if tt >= 0:
            self._surface_z = float(self.data.geom_xpos[tt][2] + m.geom_size[tt][2])
            center = self.data.geom_xpos[tt][:2].copy()
            half = m.geom_size[tt][:2].copy()
            self._table_lo = center - half
            self._table_hi = center + half
        else:
            self._surface_z = 0.0
            self._table_lo = self._table_hi = None

    # ── Inverse kinematics (position + top-down orientation) ──────────────────

    def _ik(self, target: np.ndarray, iters: int = 500, wrot: float = 0.4) -> np.ndarray:
        """Damped-least-squares IK for the 6 arm joints to reach ``target``.

        Solves on a scratch copy so the live sim is untouched; returns the joint
        vector that puts the pinch site at ``target`` with the gripper pointing
        straight down.
        """
        m = self.model
        ds = mujoco.MjData(m)
        ds.qpos[:] = self.data.qpos
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_forward(m, ds)
            perr = target - ds.site_xpos[self._site]
            qc = np.zeros(4)
            mujoco.mju_mat2Quat(qc, ds.site_xmat[self._site])
            qci = np.zeros(4)
            mujoco.mju_negQuat(qci, qc)
            qerr = np.zeros(4)
            mujoco.mju_mulQuat(qerr, _Q_DOWN, qci)
            if qerr[0] < 0:
                qerr = -qerr
            rerr = 2.0 * qerr[1:4]
            if np.linalg.norm(perr) < 5e-4 and np.linalg.norm(rerr) < 1e-2:
                break
            mujoco.mj_jacSite(m, ds, jacp, jacr, self._site)
            J = np.vstack([jacp[:, :6], wrot * jacr[:, :6]])
            err = np.concatenate([perr, wrot * rerr])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.08**2 * np.eye(6), err)
            ds.qpos[:6] = np.clip(ds.qpos[:6] + np.clip(dq, -0.1, 0.1),
                                  self._lo, self._hi)
        return ds.qpos[:6].copy()

    # ── Motion primitives ─────────────────────────────────────────────────────

    def move_joints(self, qarm: np.ndarray, settle: int = 500, render=None,
                    speed: float = 1.0) -> float:
        """Ramp the arm joints to ``qarm`` at ``speed``; returns final joint error.

        The command is interpolated from the current pose to ``qarm`` over
        ``settle / speed`` steps, giving the motion a finite, speed-scaled
        velocity, then held at the target for ``_CONVERGE`` steps so the servo
        settles whatever the speed.
        """
        speed = float(np.clip(speed, _SPEED_MIN, _SPEED_MAX))
        q0 = self.data.qpos[:6].copy()
        travel = max(1, int(round(settle / speed)))
        self.data.ctrl[6] = self._grip
        self.data.ctrl[7] = self._grip
        for k in range(1, travel + 1):
            a = k / travel
            self.data.ctrl[:6] = (1.0 - a) * q0 + a * qarm
            mujoco.mj_step(self.model, self.data)
            if render is not None:
                render()
        self.data.ctrl[:6] = qarm
        for _ in range(_CONVERGE):
            mujoco.mj_step(self.model, self.data)
            if render is not None:
                render()
        return float(np.linalg.norm(qarm - self.data.qpos[:6]))

    def _over_table(self, t) -> bool:
        return (
            self._table_lo is not None
            and self._table_lo[0] <= t[0] <= self._table_hi[0]
            and self._table_lo[1] <= t[1] <= self._table_hi[1]
        )

    def _surface_at(self, t) -> float:
        """Height of the no-go floor under (x, y): tabletop over the table, else
        the ground."""
        return self._surface_z if self._over_table(t) else 0.0

    def _fingertip_z(self, qarm) -> float:
        """World z of the lowest hand point for a candidate arm pose (top-down,
        so that's the fingertips: ``_FINGERTIP_BELOW_PINCH`` under the site)."""
        s = self._scratch
        s.qpos[:] = self.data.qpos
        s.qpos[:6] = qarm
        mujoco.mj_forward(self.model, s)
        return float(s.site_xpos[self._site][2]) - _FINGERTIP_BELOW_PINCH

    def _solve_above_surface(self, target) -> np.ndarray:
        """IK that provably keeps the hand above the work surface.

        First clamp the target to the no-go floor, then solve. Because the IK
        can undershoot on awkward low reaches (returning a pose whose fingertips
        still dip below the table), check the *solution's* fingertip height and
        nudge the target up until the hand clears — so the commanded pose never
        buries the gripper in the table, regardless of IK accuracy.
        """
        t = np.asarray(target, dtype=float).copy()
        min_tip = self._surface_at(t) + _SURFACE_MARGIN
        if t[2] < min_tip + _FINGERTIP_BELOW_PINCH:
            t[2] = min_tip + _FINGERTIP_BELOW_PINCH
        q = self._ik(t)
        for _ in range(8):
            deficit = min_tip - self._fingertip_z(q)
            if deficit <= 1e-4:
                break
            t[2] += deficit + 0.002
            q = self._ik(t)
        return q

    def move_to(self, target, settle: int = 500, render=None,
                speed: float = 1.0) -> float:
        return self.move_joints(self._solve_above_surface(target), settle,
                                render, speed)

    def set_grip(self, opening: float, settle: int = 150, render=None,
                 speed: float = 1.0) -> None:
        speed = float(np.clip(speed, _SPEED_MIN, _SPEED_MAX))
        g0 = self._grip
        self._grip = float(opening)
        travel = max(1, int(round(settle / speed)))
        for k in range(1, travel + 1):
            a = k / travel
            g = (1.0 - a) * g0 + a * self._grip
            self.data.ctrl[6] = g
            self.data.ctrl[7] = g
            mujoco.mj_step(self.model, self.data)
            if render is not None:
                render()

    def grasp(self) -> None:
        """Weld the cube to the palm at their current relative pose."""
        if self._cube < 0 or self._eq < 0:
            return
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        p1, q1 = d.xpos[self._palm].copy(), d.xquat[self._palm].copy()
        p2, q2 = d.xpos[self._cube].copy(), d.xquat[self._cube].copy()
        q1i = np.zeros(4)
        mujoco.mju_negQuat(q1i, q1)
        prel = np.zeros(3)
        mujoco.mju_rotVecQuat(prel, p2 - p1, q1i)
        qrel = np.zeros(4)
        mujoco.mju_mulQuat(qrel, q1i, q2)
        m.eq_data[self._eq, :3] = 0.0
        m.eq_data[self._eq, 3:6] = prel
        m.eq_data[self._eq, 6:10] = qrel
        m.eq_data[self._eq, 10] = 1.0
        d.eq_active[self._eq] = 1

    def release(self) -> None:
        if self._eq >= 0:
            self.data.eq_active[self._eq] = 0

    def hold(self, steps: int = 200, render=None) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if render is not None:
                render()


class Arm:
    """Cadenza 6-axis articulated arm controller.

    Example::

        import cadenza
        arm = cadenza.arm()
        arm.run([
            arm.home(),
            arm.pick((0.5, 0.0, 0.43)),
            arm.place((0.4, 0.22, 0.43)),
        ])
    """

    def __init__(self, xml_path: str | None = None, *,
                 cam_distance: float = 1.7, cam_azimuth: float = 135,
                 cam_elevation: float = -22):
        self._xml_path = str(xml_path or _DEFAULT_XML)
        if not Path(self._xml_path).exists():
            raise FileNotFoundError(f"Arm model not found: {self._xml_path}")
        self._cam = (cam_distance, cam_azimuth, cam_elevation)

    # ── Action methods (return descriptors, no execution) ─────────────────────

    def home(self, *, speed: float = 1.0) -> ArmAction:
        return ArmAction("home", speed=speed)

    def move_to(self, x, y=None, z=None, *, speed: float = 1.0) -> ArmAction:
        x, y, z = self._xyz(x, y, z)
        return ArmAction("move_to", x, y, z, speed=speed)

    def open_gripper(self, *, speed: float = 1.0) -> ArmAction:
        return ArmAction("open_gripper", speed=speed)

    def close_gripper(self, *, speed: float = 1.0) -> ArmAction:
        return ArmAction("close_gripper", speed=speed)

    def pick(self, x, y=None, z=None, *,
             speed: float | dict[str, float] = 1.0) -> ArmAction:
        """Pick up the object at a location (x, y, z) in the arm's base frame.

        ``speed`` scales the motion. Pass a float to set every phase, or a dict
        to set them individually — phases: ``approach``, ``descend``, ``grasp``,
        ``lift`` (omitted phases default to ``1.0``)::

            arm.pick((0.5, 0, 0.43), speed={"descend": 0.5, "lift": 2.0})
        """
        x, y, z = self._xyz(x, y, z)
        self._check_phase_speed(speed, _PICK_PHASES, "pick")
        return ArmAction("pick", x, y, z, speed=speed)

    def place(self, x, y=None, z=None, *,
              speed: float | dict[str, float] = 1.0) -> ArmAction:
        """Place the held object down at a location (x, y, z).

        ``speed`` scales the motion. Pass a float to set every phase, or a dict
        to set them individually — phases: ``carry``, ``lower``, ``release``,
        ``retract`` (omitted phases default to ``1.0``)::

            arm.place((0.4, 0.22, 0.43), speed={"lower": 0.5})
        """
        x, y, z = self._xyz(x, y, z)
        self._check_phase_speed(speed, _PLACE_PHASES, "place")
        return ArmAction("place", x, y, z, speed=speed)

    @staticmethod
    def _check_phase_speed(speed, phases, action: str) -> None:
        if isinstance(speed, dict):
            unknown = set(speed) - set(phases)
            if unknown:
                raise ValueError(
                    f"Unknown {action} phase(s) {sorted(unknown)}; "
                    f"valid phases: {list(phases)}")

    @staticmethod
    def _xyz(x, y, z):
        """Accept either ``move_to(0.5, 0, 0.43)`` or ``move_to((0.5, 0, 0.43))``."""
        if y is None and z is None:
            x, y, z = x
        return float(x), float(y), float(z)

    def actions(self) -> list[str]:
        """Names of the arm's available primitives."""
        from cadenza.actions import get_library
        return get_library("arm").list_actions()

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self, sequence: list, *, headless: bool = False,
            realtime: bool = True, verbose: bool = True,
            camera: bool = True) -> "Arm":
        """Execute a sequence of arm actions in MuJoCo (rendered by default).

        When ``camera`` is True (and not headless), a live window shows the
        eye-in-hand ``grip_cam`` — the arm's onboard view down the grasp axis —
        updating in real time as the arm moves. Pass ``camera=False`` to hide it.
        """
        steps = [s if isinstance(s, ArmAction) else ArmAction(s) for s in sequence]
        rt = _Runtime(self._xml_path)

        if headless:
            if verbose:
                print(f"\n  Cadenza Arm (headless)  |  {len(steps)} steps\n")
            self._execute(rt, steps, render=None, verbose=verbose)
            return self

        import mujoco.viewer
        from cadenza.sensor_view import make_view
        if verbose:
            print(f"\n  Cadenza Arm  |  {len(steps)} steps\n")
        view = make_view("arm", enabled=camera)
        with mujoco.viewer.launch_passive(rt.model, rt.data) as viewer:
            dist, azi, elev = self._cam
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = dist, azi, elev
            viewer.cam.lookat[:] = (0.45, 0.0, 0.45)

            def render():
                if viewer.is_running():
                    viewer.sync()
                    view.maybe_update(rt.model, rt.data)
                    if realtime:
                        time.sleep(rt.model.opt.timestep)

            try:
                self._execute(rt, steps, render=render, verbose=verbose)
                if verbose:
                    print("\n  Done. Close viewer to exit.")
                while viewer.is_running():
                    mujoco.mj_step(rt.model, rt.data)
                    viewer.sync()
                    view.maybe_update(rt.model, rt.data)
                    time.sleep(0.02)
            finally:
                view.close()
        return self

    def _execute(self, rt: _Runtime, steps: list, render, verbose: bool) -> None:
        for i, step in enumerate(steps):
            if verbose:
                tgt = f"  ({step.x:.2f}, {step.y:.2f}, {step.z:.2f})" if step.is_cartesian else ""
                if step.speed == 1.0:
                    spd = ""
                elif isinstance(step.speed, dict):
                    spd = f"  @{step.speed}"
                else:
                    spd = f"  @{step.speed:g}x"
                print(f"  [{i+1}/{len(steps)}] {step.action_name}{tgt}{spd}")
            self._run_one(rt, step, render)

    def _run_one(self, rt: _Runtime, step: ArmAction, render) -> None:
        name = step.action_name
        spd = step.speed
        if name == "home":
            if rt.model.nkey > 0:
                rt.move_joints(rt.model.key_qpos[0][:6], render=render, speed=spd)
            rt.set_grip(_GRIP_OPEN, render=render, speed=spd)
        elif name == "open_gripper":
            rt.set_grip(_GRIP_OPEN, render=render, speed=spd)
        elif name == "close_gripper":
            rt.set_grip(_GRIP_CLOSED, render=render, speed=spd)
        elif name == "move_to":
            rt.move_to(step.target, render=render, speed=spd)
        elif name == "pick":
            self._pick(rt, np.asarray(step.target, float), render, spd)
        elif name == "place":
            self._place(rt, np.asarray(step.target, float), render, spd)
        else:
            raise ValueError(f"Unknown arm action {name!r}")

    def _pick(self, rt: _Runtime, loc: np.ndarray, render, speed=1.0) -> None:
        sp = lambda phase: _speed_for(speed, phase)
        rt.set_grip(_GRIP_OPEN, settle=80, render=render, speed=sp("approach"))
        rt.move_to(loc + [0, 0, _HOVER], render=render, speed=sp("approach"))  # approach above
        rt.move_to(loc, render=render, speed=sp("descend"))                    # descend onto object
        rt.grasp()                                                             # weld on
        rt.set_grip(_GRIP_CLOSED, render=render, speed=sp("grasp"))            # close fingers
        rt.move_to(loc + [0, 0, _LIFT], render=render, speed=sp("lift"))       # lift

    def _place(self, rt: _Runtime, loc: np.ndarray, render, speed=1.0) -> None:
        sp = lambda phase: _speed_for(speed, phase)
        rt.move_to(loc + [0, 0, _LIFT], render=render, speed=sp("carry"))      # carry above
        rt.move_to(loc, render=render, speed=sp("lower"))                      # lower to surface
        rt.release()                                                           # weld off
        rt.set_grip(_GRIP_OPEN, render=render, speed=sp("release"))            # open fingers
        rt.move_to(loc + [0, 0, _HOVER], render=render, speed=sp("retract"))   # retract
