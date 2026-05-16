"""cadenza.inference — VLA orchestration layer.

An orchestrator decides *how* the VLA model interacts with action execution:
when to run it, how to handle interruptions, when to inject recovery actions.
The robot controller delegates all of that to whichever orchestrator was
passed at construction::

    from cadenza.inference import Sequential

    go1 = cadenza.go1(inference=Sequential())
    go1.run([go1.walk_forward(distance_m=5.0)])

Strategies ship in this module. Future ones (parallel reasoning, planner-
worker splits, multi-camera coordination, etc.) can be added by subclassing
``InferenceOrchestrator`` and dropping the class beside ``Sequential``.
"""

from cadenza.inference.base import InferenceOrchestrator
from cadenza.inference.sequential import Sequential

__all__ = [
    "InferenceOrchestrator",
    "Sequential",
]
