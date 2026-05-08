"""Concrete models for the Unitree Go1.

Usage::

    import cadenza
    from ai_models.go1 import VLA, Depth, RGB

    go1 = cadenza.go1()
    go1.setup(model=VLA(), sense=[Depth(), RGB()])
    go1.run(goal="reach the green beacon and sit",
            scene="stairs", target=(-5.5, 0.0))
"""

from ai_models.go1.vla import VLA
from ai_models.go1.depth import Depth
from ai_models.go1.rgb import RGB

__all__ = ["VLA", "Depth", "RGB"]
