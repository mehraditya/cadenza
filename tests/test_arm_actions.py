"""Cadenza 6-axis arm — pick-and-place demo + tests.

This IS the demo. Run it and the arm homes, picks the red cube off the table,
and places it 22cm to the side, then returns home — rendered in a live viewer.

Usage:
    mjpython tests/test_arm_actions.py            # live viewer demo
    python   tests/test_arm_actions.py --test     # headless unit tests (CI)
"""

import sys
from pathlib import Path

import numpy as np

ARM_SCENE = str(Path(__file__).resolve().parent.parent / "models" / "arm" / "scene.xml")


def run_demo():
    """Arm picks the cube and relocates it — live viewer."""
    import cadenza

    print("\n" + "=" * 60)
    print("  CADENZA 6-AXIS ARM — PICK & PLACE")
    print("  home -> pick (0.50, 0.00) -> place (0.40, 0.22) -> home")
    print("  Motion: damped-least-squares IK   Grasp: weld constraint")
    print("=" * 60)

    arm = cadenza.arm()
    arm.run([
        arm.home(),
        arm.pick((0.50, 0.00, 0.43)),
        arm.place((0.40, 0.22, 0.43)),
        arm.home(),
    ])


# ── Unit tests (run with --test, fully headless) ─────────────────────────────

def test_library():
    """The arm exposes a Cartesian action library via the standard API."""
    import cadenza

    lib = cadenza.get_library("arm")
    names = lib.list_actions()
    assert names == ["home", "move_to", "open_gripper",
                     "close_gripper", "pick", "place"], names
    assert "pick" in lib
    assert lib.get("pick").needs_target is True
    assert lib.get("home").needs_target is False
    # Controller surfaces the same list.
    assert cadenza.arm().actions() == names
    print(f"  PASS: action library ({len(names)} primitives)")


def test_model_renders():
    """The model loads and renders a populated frame (proves it's visible)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(ARM_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 224, 224)
    renderer.update_scene(data)
    frame = renderer.render()
    assert frame.shape == (224, 224, 3)
    # A meaningful fraction of pixels are non-black: the arm/table/cube show up.
    lit = int((frame.sum(axis=2) > 10).sum())
    assert lit > 0.5 * frame.shape[0] * frame.shape[1], f"mostly black: {lit}px"
    print(f"  PASS: model renders ({lit}/{224*224} px lit)")


def test_ik_reaches():
    """IK drives the pinch site onto a Cartesian target, gripper pointing down.

    Targets sit above the 0.40m table surface — the hand now collides with the
    table, so a reachable target must not bury the gripper inside it.
    """
    import mujoco
    import cadenza
    from cadenza.arm import _Runtime

    arm = cadenza.arm()
    for target in [(0.50, 0.0, 0.55), (0.45, 0.16, 0.58), (0.40, -0.16, 0.60)]:
        rt = _Runtime(arm._xml_path)
        rt.move_to(target, render=None)
        mujoco.mj_forward(rt.model, rt.data)
        reached = rt.data.site_xpos[rt._site]
        err = float(np.linalg.norm(np.asarray(target) - reached))
        assert err < 0.03, f"IK miss at {target}: {err*1000:.0f}mm"
    print("  PASS: IK reaches targets (<30mm)")


def test_hand_never_goes_below_table():
    """The hand is held right above the table and never dips below it.

    Reproduces the reported bug across the whole tabletop: for every (x, y) over
    the table, command the empty gripper well below the surface and confirm the
    controller keeps the lowest hand point (the fingertips) at or above the
    table — controlled, not relying on the hand bouncing off after the fact.
    """
    import mujoco
    import cadenza
    from cadenza.arm import _Runtime
    from cadenza.arm import _FINGERTIP_BELOW_PINCH

    rt = _Runtime(cadenza.arm()._xml_path)
    rt.set_grip(0.04, settle=60, render=None)          # open, clear of the cube
    surface = rt._surface_z

    worst_pen = 0.0
    for x in (0.32, 0.45, 0.58, 0.68):
        for y in (-0.28, 0.0, 0.28):
            rt.move_to((x, y, 0.05), render=None)       # ask for 5cm: under table
            mujoco.mj_forward(rt.model, rt.data)
            tip_z = float(rt.data.site_xpos[rt._site][2]) - _FINGERTIP_BELOW_PINCH
            worst_pen = max(worst_pen, surface - tip_z)
            # Never below the surface, and not flung high either — right above it.
            assert tip_z >= surface - 1e-3, (
                f"hand went through table at ({x},{y}): tip z={tip_z:.3f} "
                f"< surface {surface:.3f}")
            assert tip_z <= surface + 0.05, (
                f"hand parked too high at ({x},{y}): tip z={tip_z:.3f}")
    print(f"  PASS: hand stays right above table "
          f"(worst dip {worst_pen*1000:.1f}mm below surface across grid)")


def test_pick_and_place():
    """End-to-end: the cube is grasped and relocated to the place target."""
    import cadenza
    from cadenza.arm import _Runtime

    arm = cadenza.arm()
    rt = _Runtime(arm._xml_path)
    start = rt.data.xpos[rt._cube].copy()

    place_xy = np.array([0.40, 0.22])
    arm._pick(rt, np.array([0.50, 0.0, 0.43]), render=None)
    arm._place(rt, np.array([place_xy[0], place_xy[1], 0.43]), render=None)
    rt.hold(steps=400)  # let it settle so we know it didn't roll off

    end = rt.data.xpos[rt._cube].copy()
    moved = float(np.linalg.norm(end[:2] - start[:2]))
    on_target = float(np.linalg.norm(end[:2] - place_xy))
    assert moved > 0.15, f"cube barely moved ({moved*1000:.0f}mm)"
    assert on_target < 0.05, f"cube off target by {on_target*1000:.0f}mm"
    assert end[2] > 0.40, f"cube fell off the table (z={end[2]:.3f})"
    print(f"  PASS: pick & place (moved {moved*100:.0f}cm, "
          f"within {on_target*1000:.0f}mm of target)")


def run_tests():
    print("\n  Cadenza Arm — Unit Tests\n")
    test_library()
    test_model_renders()
    test_ik_reaches()
    test_hand_never_goes_below_table()
    test_pick_and_place()
    print("\n  All tests passed!\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        run_demo()
