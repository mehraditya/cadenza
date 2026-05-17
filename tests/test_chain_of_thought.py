"""End-to-end test of the ChainOfThought orchestrator on a random box field.

    mjpython tests/test_chain_of_thought.py
"""

import random
from pathlib import Path

import cadenza
from cadenza.inference import ChainOfThought
from cadenza.robots.go1 import MODEL_XML

HERE = Path(__file__).resolve().parent

# Build the gym: a moderate-density box field in the corridor between
# the start (0, 0) and target (-5, 0). ~14 boxes with enough spacing
# that the robot can weave through cleanly.
random.seed(7)
scene = cadenza.Scene()
placed: list[tuple[float, float, float]] = []   # (x, y, half_extent) for spacing
for _ in range(14):
    for _attempt in range(20):                   # rejection-sample for spacing
        x = random.uniform(-4.0, -1.0)           # forward = -x
        y = random.uniform(-0.8, 0.8)
        s = random.uniform(0.08, 0.16)
        if abs(x) < 0.7 and abs(y) < 0.35:       # keep the start clear
            continue
        # Reject if this box would crowd an existing one — a 0.30m gap
        # between edges leaves room for the Go1's ~0.20m body to pass.
        if any((x - px) ** 2 + (y - py) ** 2 < (s + ps + 0.30) ** 2
               for px, py, ps in placed):
            continue
        placed.append((x, y, s))
        h = random.uniform(0.12, 0.22)
        scene.add_box(
            position=(x, y, h),
            size=(s, s, h),
            rgba=(0.72, 0.30, 0.22, 1.0),
        )
        break

# Compile the scene into a real MuJoCo XML. ``out_path=`` saves it
xml_path = scene.compile(MODEL_XML, "chain_of_thought_field.xml")

# Init Go1 with ChainOfThought (defaults to ai_models.go1.VLA for the model).
log = "chain_of_thought_run.jsonl"

go1 = cadenza.go1(
    xml_path=str(xml_path),
    inference=ChainOfThought(
        goal="reach the far wall, dodge boxes",
        target=(-5.0, 0.0),
        max_steps=40,
        logging=log,
    ),
)

go1.run([go1.walk_forward()])
