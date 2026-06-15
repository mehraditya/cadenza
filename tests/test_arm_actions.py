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
    """IK drives the pinch site onto a Cartesian target, gripper pointing down."""
    import cadenza
    from cadenza.arm import _Runtime

    arm = cadenza.arm()
    rt = _Runtime(arm._xml_path)
    for target in [(0.5, 0.0, 0.50), (0.42, 0.18, 0.46), (0.40, -0.20, 0.55)]:
        rt.move_to(target, render=None)
        import mujoco
        mujoco.mj_forward(rt.model, rt.data)
        reached = rt.data.site_xpos[rt._site]
        err = float(np.linalg.norm(np.asarray(target) - reached))
        assert err < 0.03, f"IK miss at {target}: {err*1000:.0f}mm"
    print("  PASS: IK reaches targets (<30mm)")


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
    test_pick_and_place()
    print("\n  All tests passed!\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        run_demo()
