"""Cadenza Go1 — world-model driven demo.

Attach a VLA model and perception modalities, then hand the robot a
natural-language goal. Cadenza closes the loop: sense → think → act.

    mjpython example.py
"""

import cadenza
from ai_models.go1 import VLA, Depth, RGB

go1 = cadenza.go1()

go1.setup(
    model=VLA(),
    sense=[Depth(), RGB()],
)

go1.run(
    goal="reach the green beacon at the top of the stairs and sit",
    scene="stairs",
    target=(-5.5, 0.0),
)
