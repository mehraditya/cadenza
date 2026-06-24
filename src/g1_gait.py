"""cadenza.g1_gait — G1 humanoid action engine.

All transitions are smooth motor commands through position actuators.
No direct qpos/qvel modification. The robot moves ONLY because
motors push it through real physics.
"""

import time as _time
import math
import numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "models" / "g1"
_DT = 0.002
_RENDER_HZ = 60


def _quintic(t):
    t = max(0.0, min(1.0, t))
    return 10*t**3 - 15*t**4 + 6*t**5


def _eval_spline(x_arr, c_arr, t):
    idx = max(0, min(np.searchsorted(x_arr, t, side='right') - 1, len(x_arr) - 2))
    dt = t - x_arr[idx]
    return float(c_arr[0, idx]*dt**3 + c_arr[1, idx]*dt**2 + c_arr[2, idx]*dt + c_arr[3, idx])


_SPLINE_CACHE = None

def _get_splines():
    global _SPLINE_CACHE
    if _SPLINE_CACHE is None:
        sd = np.load(str(_DATA_DIR / "walk_splines.npz"))
        nv = int(sd['nv'])
        sx = [sd[f'x_{i}'] for i in range(nv)]
        sc = [sd[f'c_{i}'] for i in range(nv)]
        walk_pose = np.array([_eval_spline(sx[6+j], sc[6+j], 0.0) for j in range(29)])
        _SPLINE_CACHE = {
            'sd': sd, 'nv': nv, 'sx': sx, 'sc': sc,
            'init_qpos': sd['init_qpos'],
            'solref': sd['solref'], 'solimp': sd['solimp'],
            'walk_pose': walk_pose,
        }
    return _SPLINE_CACHE


def _get_yaw(data):
    """Read pelvis yaw from quaternion."""
    w, x, y, z = data.qpos[3:7]
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _compensated_ctrl(target, data):
    """Add hip-yaw correction to ctrl so the robot stands straight
    despite any pelvis yaw drift from walking."""
    ctrl = target.copy()
    yaw = _get_yaw(data)
    ctrl[2] -= yaw   # L hip yaw: counter-rotate
    ctrl[8] -= yaw   # R hip yaw: counter-rotate
    return ctrl


def _blend(model, data, target, duration, viewer=None, compensate_yaw=False):
    """Smoothly drive ctrl from current to target. Pure motor commands."""
    import mujoco
    start = data.ctrl[:29].copy()
    n = max(1, int(duration / _DT))
    steps_per_sync = max(1, int(1.0 / (_RENDER_HZ * _DT)))
    for i in range(n):
        a = _quintic(i / n)
        raw = (1 - a) * start + a * target
        data.ctrl[:29] = _compensated_ctrl(raw, data) if compensate_yaw else raw
        mujoco.mj_step(model, data)
        if viewer and i % steps_per_sync == 0:
            viewer.sync()
            _time.sleep(steps_per_sync * _DT)


def _hold(model, data, duration, viewer=None, compensate_yaw=False):
    """Hold current target steady."""
    import mujoco
    target = data.ctrl[:29].copy()
    steps_per_sync = max(1, int(1.0 / (_RENDER_HZ * _DT)))
    for i in range(int(duration / _DT)):
        data.ctrl[:29] = _compensated_ctrl(target, data) if compensate_yaw else target
        mujoco.mj_step(model, data)
        if viewer and i % steps_per_sync == 0:
            viewer.sync()
            _time.sleep(steps_per_sync * _DT)


# ── Action executors ────────────────────────────────────────────────────────

def _exec_stand(model, data, duration, viewer):
    """Smoothly drive to standing."""
    _blend(model, data, np.zeros(29), duration, viewer)
    _hold(model, data, 1.0, viewer)


def _exec_crouch(model, data, duration, viewer):
    """Smoothly bend to walk-ready position."""
    sp = _get_splines()
    _blend(model, data, sp['walk_pose'], duration, viewer)
    _hold(model, data, 0.5, viewer)


_WALK_TRAJ_CACHE = None


def _get_walk_traj():
    """Load (and cache) the trajectory-optimized walk: per-step qpos + ctrl."""
    global _WALK_TRAJ_CACHE
    if _WALK_TRAJ_CACHE is None:
        _WALK_TRAJ_CACHE = {
            "qpos": np.load(str(_DATA_DIR / "walk_qpos.npy")),
            "ctrl": np.load(str(_DATA_DIR / "walk_ctrl.npy")),
        }
    return _WALK_TRAJ_CACHE


def _exec_walk(model, data, distance_m, viewer):
    """Walk forward by replaying the trajectory-optimized gait.

    ``walk_ctrl`` is an optimized open-loop motor trajectory. Open-loop
    playback only reproduces the gait from the exact state it was optimized
    from, so we seed the robot at ``walk_qpos[0]`` (the walk-ready crouch)
    before stepping. Each stride covers ~0.4 m; playback is capped at the
    stable open-loop horizon (~0.85 m). Beyond that, integration drift
    accumulates — sustained walking would need a closed-loop balance
    controller (tracked as follow-up).

    NOTE: seeding ``qpos`` is a deliberate one-time initialization for the
    replay (the rest of the library never teleports).
    """
    import mujoco
    traj = _get_walk_traj()
    qpos, ctrl = traj["qpos"], traj["ctrl"]

    # Seed the walk-ready start state — required for open-loop replay.
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Map requested distance → playback length. The first ~800 steps are an
    # in-place windup; thereafter each 833-step stride advances ~0.4 m.
    WINDUP, STRIDE_STEPS, STRIDE_M, STABLE_MAX = 800, 833, 0.40, 2200
    want = WINDUP + int(max(0.0, distance_m) / STRIDE_M * STRIDE_STEPS)
    n = min(STABLE_MAX, len(ctrl), max(1, want))

    steps_per_sync = max(1, int(1.0 / (_RENDER_HZ * _DT)))
    for i in range(n):
        if viewer and not viewer.is_running():
            return
        data.ctrl[:] = ctrl[i]
        mujoco.mj_step(model, data)
        if viewer and i % steps_per_sync == 0:
            # Camera follows the robot so it stays in frame while walking.
            viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1],
                                    max(float(data.qpos[2]) * 0.8, 0.4)]
            viewer.sync()
            _time.sleep(steps_per_sync * _DT)


def _exec_jump(model, data, viewer):
    """Jump: slow squat, fast push, smooth landing. All through motors."""
    stand = np.zeros(29)

    squat = stand.copy()
    squat[0] = -0.30;  squat[3] = 0.60;  squat[4] = -0.30
    squat[6] = -0.30;  squat[9] = 0.60;  squat[10] = -0.30

    # Correct yaw in squat targets
    yaw = _get_yaw(data)
    if abs(yaw) > 0.02:
        squat[2] = -yaw; squat[8] = -yaw
        stand[2] = -yaw; stand[8] = -yaw

    # 1. Slow squat
    _blend(model, data, squat, 1.5, viewer)
    _hold(model, data, 0.5, viewer)

    # 2. Push up to standing
    _blend(model, data, stand, 0.3, viewer)

    # 3. Settle
    _hold(model, data, 1.0, viewer)


def setup_model():
    """Load the G1 scene and return (model, data) at keyframe."""
    import mujoco
    sp = _get_splines()
    model = mujoco.MjModel.from_xml_path(str(_DATA_DIR / "scene.xml"))
    data = mujoco.MjData(model)

    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == 0:
            model.geom_solref[gi] = sp['solref']
            model.geom_solimp[gi] = sp['solimp']

    mujoco.mj_resetDataKeyframe(model, data, 0)
    return model, data
