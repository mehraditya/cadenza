"""Cadenza 6-axis arm — pick-and-place use case.

Scenario: a fixed-base manipulator tending a workcell. A part (the red cube)
arrives at the in-feed location on the table; the arm picks it up and stages it
at a drop-off location, then parks at home ready for the next part.

This is the manipulator analogue of the legged `deploy_go1.py` / `deploy_g1.py`
demos: same `cadenza.<robot>()` -> build a list of actions -> `.run(...)` shape,
but the primitives are Cartesian (poses + grasps) instead of gaits.

    python examples/cadenza_arm/pick_and_place.py            # live viewer
    python examples/cadenza_arm/pick_and_place.py --headless # no window (CI)
"""

import sys

import cadenza

# Workcell coordinates in the arm's base frame (metres). z=0.43 is the top of
# the cube as it rests on the 0.38m-tall table.
INFEED = (0.50, 0.00, 0.43)   # where the part arrives
DROPOFF = (0.40, 0.22, 0.43)  # where the part is staged

arm = cadenza.arm()

# ── Option 1: high-level pick/place ──────────────────────────────────────────
# `pick` approaches from above, descends, grasps, and lifts; `place` carries,
# lowers, releases, and retracts. One line each.
program = [
    arm.home(),
    arm.pick(INFEED),
    arm.place(DROPOFF),
    arm.home(),
]

# ── Option 2: same motion, spelled out with the low-level primitives ─────────
# Uncomment to drive the gripper and poses yourself instead of pick/place.
#
#   program = [
#       arm.home(),
#       arm.open_gripper(),
#       arm.move_to(INFEED[0], INFEED[1], INFEED[2] + 0.11),  # hover
#       arm.move_to(*INFEED),                                  # descend
#       arm.close_gripper(),                                   # grasp
#       arm.move_to(INFEED[0], INFEED[1], INFEED[2] + 0.18),   # lift
#       arm.move_to(DROPOFF[0], DROPOFF[1], DROPOFF[2] + 0.18),
#       arm.move_to(*DROPOFF),                                 # lower
#       arm.open_gripper(),                                    # release
#       arm.home(),
#   ]

if __name__ == "__main__":
    headless = "--headless" in sys.argv
    print(f"Staging part: in-feed {INFEED[:2]} -> drop-off {DROPOFF[:2]}")
    arm.run(program, headless=headless)
