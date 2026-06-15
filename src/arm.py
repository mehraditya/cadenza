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

# Desired top-down gripper orientation (palm +z pointing world -z): Rx(pi).
_Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


class _Runtime:
    """The live MuJoCo session: IK, motion, and grasp for one run."""

    def __init__(self, xml_path: str, hz: float = 50.0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._phys = max(1, int(round(1.0 / (self.model.opt.timestep * hz))))

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

    def move_joints(self, qarm: np.ndarray, settle: int = 500, render=None) -> float:
        """Command the arm joints and let the PD servo settle; returns cart error."""
        self.data.ctrl[:6] = qarm
        self.data.ctrl[6] = self._grip
        self.data.ctrl[7] = self._grip
        for _ in range(settle):
            mujoco.mj_step(self.model, self.data)
            if render is not None:
                render()
        return float(np.linalg.norm(qarm - self.data.qpos[:6]))

    def move_to(self, target, settle: int = 500, render=None) -> float:
        return self.move_joints(self._ik(np.asarray(target, float)), settle, render)

    def set_grip(self, opening: float, settle: int = 150, render=None) -> None:
        self._grip = float(opening)
        self.data.ctrl[6] = self._grip
        self.data.ctrl[7] = self._grip
        for _ in range(settle):
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

    def home(self) -> ArmAction:
        return ArmAction("home")

    def move_to(self, x, y=None, z=None) -> ArmAction:
        x, y, z = self._xyz(x, y, z)
        return ArmAction("move_to", x, y, z)

    def open_gripper(self) -> ArmAction:
        return ArmAction("open_gripper")

    def close_gripper(self) -> ArmAction:
        return ArmAction("close_gripper")

    def pick(self, x, y=None, z=None) -> ArmAction:
        """Pick up the object at a location (x, y, z) in the arm's base frame."""
        x, y, z = self._xyz(x, y, z)
        return ArmAction("pick", x, y, z)

    def place(self, x, y=None, z=None) -> ArmAction:
        """Place the held object down at a location (x, y, z)."""
        x, y, z = self._xyz(x, y, z)
        return ArmAction("place", x, y, z)

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
            realtime: bool = True, verbose: bool = True) -> "Arm":
        """Execute a sequence of arm actions in MuJoCo (rendered by default)."""
        steps = [s if isinstance(s, ArmAction) else ArmAction(s) for s in sequence]
        rt = _Runtime(self._xml_path)

        if headless:
            if verbose:
                print(f"\n  Cadenza Arm (headless)  |  {len(steps)} steps\n")
            self._execute(rt, steps, render=None, verbose=verbose)
            return self

        import mujoco.viewer
        if verbose:
            print(f"\n  Cadenza Arm  |  {len(steps)} steps\n")
        with mujoco.viewer.launch_passive(rt.model, rt.data) as viewer:
            dist, azi, elev = self._cam
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = dist, azi, elev
            viewer.cam.lookat[:] = (0.45, 0.0, 0.45)

            def render():
                if viewer.is_running():
                    viewer.sync()
                    if realtime:
                        time.sleep(rt.model.opt.timestep)

            self._execute(rt, steps, render=render, verbose=verbose)
            if verbose:
                print("\n  Done. Close viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(rt.model, rt.data)
                viewer.sync()
                time.sleep(0.02)
        return self

    def _execute(self, rt: _Runtime, steps: list, render, verbose: bool) -> None:
        for i, step in enumerate(steps):
            if verbose:
                tgt = f"  ({step.x:.2f}, {step.y:.2f}, {step.z:.2f})" if step.is_cartesian else ""
                print(f"  [{i+1}/{len(steps)}] {step.action_name}{tgt}")
            self._run_one(rt, step, render)

    def _run_one(self, rt: _Runtime, step: ArmAction, render) -> None:
        name = step.action_name
        if name == "home":
            if rt.model.nkey > 0:
                rt.move_joints(rt.model.key_qpos[0][:6], render=render)
            rt.set_grip(_GRIP_OPEN, render=render)
        elif name == "open_gripper":
            rt.set_grip(_GRIP_OPEN, render=render)
        elif name == "close_gripper":
            rt.set_grip(_GRIP_CLOSED, render=render)
        elif name == "move_to":
            rt.move_to(step.target, render=render)
        elif name == "pick":
            self._pick(rt, np.asarray(step.target, float), render)
        elif name == "place":
            self._place(rt, np.asarray(step.target, float), render)
        else:
            raise ValueError(f"Unknown arm action {name!r}")

    def _pick(self, rt: _Runtime, loc: np.ndarray, render) -> None:
        rt.set_grip(_GRIP_OPEN, settle=80, render=render)
        rt.move_to(loc + [0, 0, _HOVER], render=render)   # approach above
        rt.move_to(loc, render=render)                    # descend onto object
        rt.grasp()                                         # weld on
        rt.set_grip(_GRIP_CLOSED, render=render)           # close fingers
        rt.move_to(loc + [0, 0, _LIFT], render=render)     # lift

    def _place(self, rt: _Runtime, loc: np.ndarray, render) -> None:
        rt.move_to(loc + [0, 0, _LIFT], render=render)     # carry above
        rt.move_to(loc, render=render)                     # lower to surface
        rt.release()                                       # weld off
        rt.set_grip(_GRIP_OPEN, render=render)             # open fingers
        rt.move_to(loc + [0, 0, _HOVER], render=render)    # retract
