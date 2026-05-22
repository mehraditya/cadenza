"""cadenza.spatial — on-board 3D spatial memory for VLA orchestration.

A lightweight 2.5D occupancy + landmark map that accumulates multi-modal
observations (robot pose, terrain probes, obstacle raycasts, body height)
tick by tick. From this growing map the system picks the right *subgoal*
when the geometry of the world demands it — e.g. "approach the stair base
first" when the final target sits on top of a staircase, even when that
detour moves the robot further from the target in XY.

Plugged in as a Modality::

    from cadenza.spatial import SpatialMemory
    go1 = cadenza.go1(
        inference=ChainOfThought(
            model=VLA(),
            sense=[SpatialMemory(target=(-5.5, 0.0))],
            target=(-5.5, 0.0),
        ),
    )

The modality writes ``target_xy`` in the observation to the current
subgoal; the original target is preserved as ``final_target_xy``. The
``ai_models.go1.VLA`` adapter steers toward whatever ``target_xy`` says,
so swapping in the stair base just works.
"""

from cadenza.spatial.map import SpatialMap, Landmark
from cadenza.spatial.memory import SpatialMemory

__all__ = ["SpatialMap", "Landmark", "SpatialMemory"]
