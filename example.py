"""Cadenza G1 — VLA + multi-modal sensing demo.

   mjpython example.py
"""

import cadenza
from ai_models.g1 import VLA, Depth, RGB

g1 = cadenza.g1()

g1.setup(
    model=VLA(),
    sense=[Depth(), RGB()],
)

g1.run(
    goal="reach the green beacon at the top of the stairs and sit",
    scene="stairs",
    target=(-5.5, 0.0),
)
