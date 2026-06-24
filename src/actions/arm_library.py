"""Action library for the Cadenza 6-axis articulated arm.

The quadruped/humanoid :class:`~cadenza.actions.library.ActionLibrary` is built
around legged primitives (gaits, phases, per-leg joint targets). A manipulator
is a different animal: its primitives are Cartesian — *go to this pose*, *grasp
the thing at this location* — so the arm gets its own small, self-contained
library that mirrors the same shape (``list_actions`` / ``get`` / ``describe``)
and plugs into ``cadenza.get_library("arm")``.

The primitives:

==================  ===========================================================
``home``            Return to the neutral folded pose, gripper open.
``move_to``         Move the gripper to a Cartesian point (x, y, z), top-down.
``open_gripper``    Open the fingers.
``close_gripper``   Close the fingers.
``pick``            Approach from above, descend, grasp, and lift a target.
``place``           Carry to a point, lower, release, and retract.
==================  ===========================================================

Each :class:`ArmAction` is a lightweight descriptor (like
:class:`~cadenza.actions.library.ActionCall`): the :class:`~cadenza.arm.Arm`
controller turns it into IK-driven motion in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass


# Actions that carry a Cartesian (x, y, z) target.
_CARTESIAN = {"move_to", "pick", "place"}


@dataclass
class ArmAction:
    """A single arm command with an optional Cartesian target.

    ``x``/``y``/``z`` are meaningful for ``move_to``/``pick``/``place`` (metres,
    in the arm's base frame); ignored by ``home``/``open_gripper``/
    ``close_gripper``.

    ``speed`` is a velocity multiplier on the motion: ``1.0`` is the nominal
    pace, ``2.0`` moves twice as fast, ``0.5`` half as fast. It is clamped to a
    safe range by the controller.

    For the compound actions ``pick`` and ``place``, ``speed`` may instead be a
    ``dict`` mapping phase name → multiplier, so each sub-move runs at its own
    speed (e.g. ``{"descend": 0.5, "lift": 2.0}``); phases left out default to
    ``1.0``. The simple actions take a scalar ``speed`` only.
    """

    action_name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed: float | dict[str, float] = 1.0

    @property
    def target(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_cartesian(self) -> bool:
        return self.action_name in _CARTESIAN

    def __repr__(self) -> str:
        if self.speed == 1.0:
            spd = ""
        elif isinstance(self.speed, dict):
            spd = f", speed={self.speed}"
        else:
            spd = f", speed={self.speed:g}"
        if self.is_cartesian:
            return (f"ArmAction({self.action_name}, "
                    f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f}){spd})")
        return f"ArmAction({self.action_name}{spd})"


@dataclass
class ArmActionSpec:
    """Metadata for one arm primitive (parallels ``ActionSpec`` in spirit)."""

    name: str
    description: str
    needs_target: bool = False


_ARM_SPECS: dict[str, ArmActionSpec] = {
    "home": ArmActionSpec(
        "home", "Return to the neutral folded pose with the gripper open."),
    "move_to": ArmActionSpec(
        "move_to", "Move the gripper to (x, y, z), approaching top-down.",
        needs_target=True),
    "open_gripper": ArmActionSpec(
        "open_gripper", "Open the parallel fingers."),
    "close_gripper": ArmActionSpec(
        "close_gripper", "Close the parallel fingers."),
    "pick": ArmActionSpec(
        "pick", "Approach from above (x, y, z), descend, grasp, and lift.",
        needs_target=True),
    "place": ArmActionSpec(
        "place", "Carry to (x, y, z), lower, release, and retract.",
        needs_target=True),
}


class ArmActionLibrary:
    """The 6-axis arm's action library.

    Duck-types the read API of :class:`~cadenza.actions.library.ActionLibrary`
    (``list_actions`` / ``get`` / ``describe`` / membership / iteration) so it
    can be served by ``cadenza.get_library("arm")``.
    """

    robot = "arm"

    def __init__(self) -> None:
        self._actions = dict(_ARM_SPECS)

    def get(self, name: str) -> ArmActionSpec:
        if name not in self._actions:
            raise KeyError(
                f"Action '{name}' not found for arm. "
                f"Available: {list(self._actions)}"
            )
        return self._actions[name]

    def list_actions(self) -> list[str]:
        return list(self._actions)

    def describe(self) -> str:
        lines = ["Action Library — ARM (6-axis)", ""]
        for name, spec in self._actions.items():
            tag = "cartesian" if spec.needs_target else "discrete"
            lines.append(f"  {name:<14s} [{tag:>9s}]  {spec.description}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions.values())
