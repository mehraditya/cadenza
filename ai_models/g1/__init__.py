"""Concrete models for the Unitree G1 humanoid.

Usage::

    import cadenza
    from ai_models.g1 import VLA, Depth, RGB

    g1 = cadenza.g1()
    g1.setup(model=VLA(), sense=[Depth(), RGB()])
    g1.run(goal="walk to the chair and sit", target=(2.0, 0.0))
"""

from ai_models.g1.vla import VLA
from ai_models.g1.depth import Depth
from ai_models.g1.rgb import RGB

__all__ = ["VLA", "Depth", "RGB"]
