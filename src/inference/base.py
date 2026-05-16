"""InferenceOrchestrator — base class for VLA orchestration strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cadenza.go1 import Step


class InferenceOrchestrator(ABC):
    """Strategy for how the VLA layer interacts with action execution.

    The robot controller (``Go1`` / ``G1``) calls into the orchestrator at
    three points::

        orchestrator.setup(robot_name, sim, lib)           # once, before the loop
        for step in sequence:
            orchestrator.run_step(step, sim, lib, viewer, robot)
        orchestrator.teardown()                             # once, after the loop

    Default ``setup`` and ``teardown`` are no-ops so simple strategies only
    need to implement ``run_step``.
    """

    # Human-readable name shown in run logs.
    name: str = "base"

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self, robot_name: str, sim: Any, lib: Any) -> None:
        """One-time init before the action loop. Load models here."""
        return

    def teardown(self) -> None:
        """One-time cleanup after the action loop."""
        return

    # ── Per-step ─────────────────────────────────────────────────────────────

    @abstractmethod
    def run_step(
        self,
        step: "Step",
        sim: Any,
        lib: Any,
        viewer: Any,
        robot: Any,
    ) -> None:
        """Execute one ``Step`` end-to-end, with whatever VLA logic the
        strategy enforces (monitoring, interruption, recovery, retry, ...).

        Implementations should drive the robot via ``robot._execute_single``
        (and ``robot._execute_concurrent`` when applicable) so that motor
        control stays in one place.
        """
        raise NotImplementedError
