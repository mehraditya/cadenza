"""MissionCoordinator — split one human goal into per-robot subgoals.

The :mod:`cadenza.mcp` bus gives connected robots a way to *talk*; this module
gives a script a way to *delegate*. A developer hands the coordinator a single
plain-language mission and the coordinator breaks it into smaller, individual
goals and hands one (or more) to each connected robot — over the very same MCP
the robots use to talk to each other, so the whole hand-off is narrated into the
one shared terminal a person is watching.

How a mission is split (all rule-based, dependency-light, deterministic):

1. **Clause splitting** — the mission is cut on the same connectors the
   :class:`~cadenza.parser.translator.CommandParser` uses (``then``/``and``)
   plus ``,`` ``;`` ``&``/``plus`` — each clause becomes one candidate subgoal.
2. **Explicit targeting** — a clause that *names* a connected robot
   (``"go1 scout the left corridor"``, ``"g1: hold the doorway"``) is assigned
   to it; ``everyone``/``all``/``both``/``team`` make the clause a broadcast.
3. **Capability routing** — an unnamed clause that clearly needs a humanoid
   (``grab``, ``wave``, ``reach``…) or a quadruped (``crawl``, ``climb``…) goes
   to a robot of that kind when exactly the right kind is available.
4. **Round-robin** — anything still unassigned is dealt out evenly across the
   connected robots in a stable order, so a generic ``"search the room"`` mission
   spreads the work instead of piling it on one robot.

Quick start::

    import cadenza

    go1, g1 = cadenza.go1(), cadenza.g1()
    with cadenza.connect(go1, g1) as term:
        plan = term.coordinate(
            "go1 scout the left corridor and g1 hold the doorway "
            "then everyone regroup at the pad"
        )
        for sg in plan.subgoals:
            print(sg.robot, "->", sg.goal)
        # each subgoal is already in the target robot's inbox AND narrated
        # into the one shared terminal; read them back with robot.comm.messages()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cadenza.actions.library import ActionCall
    from cadenza.mcp.link import CoordinationTerminal


# Connectors that separate one subgoal from the next. Mirrors CommandParser's
# then/and split, widened with the punctuation a person naturally writes.
_SPLIT = re.compile(r"\s*(?:\bthen\b|\band\b|\bplus\b|;|,|&)\s*", re.IGNORECASE)

# Leading filler dropped after an explicit target ("go1 should scout" -> "scout").
_FILLER = re.compile(
    r"^(?:should|shall|will|would|must|please|needs?\s+to|has\s+to|to|go|:|-)\s+",
    re.IGNORECASE,
)

# A clause addressed to these means "every other connected robot".
_BROADCAST = {"everyone", "everybody", "all", "both", "team", "all robots"}

# Capability hints: words that strongly imply one robot kind.
_HUMANOID_HINTS = (
    "grab", "grasp", "grip", "pick up", "pick-up", "carry", "reach", "lift",
    "wave", "hand", "arm", "button", "handle", "throw", "hold the door",
)
_QUADRUPED_HINTS = (
    "crawl", "trot", "bound", "pace", "climb", "sniff", "scurry",
    "under the", "low to the ground", "all fours",
)


@dataclass
class SubGoal:
    """One smaller goal carved out of the mission and aimed at a robot.

    ``robot`` is the recipient's bus name, or ``"*"`` when ``broadcast`` is set
    (the clause was addressed to everyone). ``kind`` is the robot's parser kind
    (``"go1"``/``"g1"``) when known, used for best-effort action parsing.
    """

    robot: str
    goal: str
    broadcast: bool = False
    kind: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot": "all" if self.broadcast else self.robot,
            "goal": self.goal,
            "broadcast": self.broadcast,
        }


@dataclass
class MissionPlan:
    """The result of splitting a mission: the original goal + its subgoals."""

    goal: str
    subgoals: list[SubGoal] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)  # robot name -> parser kind

    def __iter__(self):
        return iter(self.subgoals)

    def __len__(self) -> int:
        return len(self.subgoals)

    def for_robot(self, name: str) -> list[SubGoal]:
        """Subgoals a given robot must act on — its own plus any broadcasts."""
        return [sg for sg in self.subgoals if sg.broadcast or sg.robot == name]

    def assignments(self) -> dict[str, list[str]]:
        """Map every connected robot to the goal texts it received (in order)."""
        out: dict[str, list[str]] = {name: [] for name in self.kinds}
        for sg in self.subgoals:
            targets = list(out) if sg.broadcast else [sg.robot]
            for t in targets:
                out.setdefault(t, []).append(sg.goal)
        return out

    def actions(self, robot: str) -> list["ActionCall"]:
        """Best-effort parse of a robot's subgoals into ActionCalls.

        Uses :class:`~cadenza.parser.translator.CommandParser`; clauses that
        don't name known actions simply yield nothing (the subgoal text is still
        the source of truth a developer can act on however they like).
        """
        from cadenza.parser.translator import CommandParser

        kind = self.kinds.get(robot, "go1")
        parser = CommandParser(kind)
        calls: list[ActionCall] = []
        for sg in self.for_robot(robot):
            try:
                calls.extend(parser.parse(sg.goal))
            except Exception:
                pass
        return calls

    def as_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "subgoals": [sg.as_dict() for sg in self.subgoals]}


class MissionCoordinator:
    """Splits a mission into subgoals and delegates them over the MCP.

    Joins the terminal as its own participant (default name ``"coordinator"``)
    so every delegated subgoal travels through a real MCP tool call and is
    narrated into the shared terminal, exactly like robot↔robot messages.
    """

    def __init__(self, term: "CoordinationTerminal", *, name: str = "coordinator",
                 route_by_capability: bool = True):
        self.term = term
        self.name = name
        self.route_by_capability = route_by_capability
        self._link = None  # lazily connected RobotLink

    # ── Planning (pure: no messages sent) ─────────────────────────────────────

    def plan(self, goal: str) -> MissionPlan:
        """Split ``goal`` into a :class:`MissionPlan` without dispatching it."""
        targets = [r for r in self.term.robots() if r != self.name]
        if not targets:
            raise ValueError(
                "no robots to coordinate; connect at least one robot first"
            )
        kinds = {name: self._kind(name) for name in targets}
        plan = MissionPlan(goal=goal, kinds=kinds)

        rr = 0  # round-robin cursor over `targets`
        for clause in self._clauses(goal):
            who, rest = self._strip_target(clause, targets)
            if who == "*":
                plan.subgoals.append(SubGoal("*", rest, broadcast=True))
            elif who is not None:
                plan.subgoals.append(SubGoal(who, rest, kind=kinds[who]))
            else:
                routed = self._route(clause, targets, kinds)
                if routed is not None:
                    plan.subgoals.append(SubGoal(routed, clause, kind=kinds[routed]))
                else:
                    pick = targets[rr % len(targets)]
                    rr += 1
                    plan.subgoals.append(SubGoal(pick, clause, kind=kinds[pick]))
        return plan

    # ── Dispatch (sends each subgoal through the MCP) ─────────────────────────

    def dispatch(self, plan: MissionPlan) -> MissionPlan:
        """Send every subgoal to its target robot over the coordination MCP."""
        link = self._ensure_link()
        for sg in plan.subgoals:
            if sg.broadcast:
                link.broadcast(sg.goal)
            else:
                link.tell(sg.robot, sg.goal)
        return plan

    def run(self, goal: str) -> MissionPlan:
        """Split ``goal`` and dispatch it — the one call a script normally makes."""
        return self.dispatch(self.plan(goal))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_link(self):
        if self._link is None:
            self._link = self.term.connect(name=self.name, robot_type="coordinator")
        return self._link

    def _kind(self, name: str) -> str:
        """Map a connected robot to a CommandParser kind ('go1'/'g1')."""
        rtype = (self.term.robot_type(name) or "").lower()
        if rtype == "humanoid" or name.lower() == "g1":
            return "g1"
        return "go1"

    @staticmethod
    def _clauses(goal: str) -> list[str]:
        return [c.strip() for c in _SPLIT.split(goal) if c.strip()]

    @staticmethod
    def _strip_target(clause: str, targets: list[str]) -> tuple[str | None, str]:
        """Pull an explicit recipient off the front of a clause.

        Returns ``("*", rest)`` for a broadcast, ``(robot_name, rest)`` for a
        named robot, or ``(None, clause)`` when the clause names no target.
        """
        lower = {t.lower(): t for t in targets}
        words = clause.split()
        if not words:
            return None, clause

        # Two-word broadcast phrase ("all robots ...").
        if len(words) >= 2 and f"{words[0]} {words[1]}".lower() in _BROADCAST:
            return "*", _FILLER.sub("", " ".join(words[2:]).strip())

        head = words[0].lower().strip(":,-")
        rest = " ".join(words[1:]).strip()
        if head in _BROADCAST:
            return "*", _FILLER.sub("", rest)
        if head in lower:
            return lower[head], _FILLER.sub("", rest)
        return None, clause

    def _route(self, clause: str, targets: list[str],
               kinds: dict[str, str]) -> str | None:
        """Route an unnamed clause to a robot by capability, or None."""
        if not self.route_by_capability:
            return None
        text = clause.lower()
        wants = None
        if any(h in text for h in _HUMANOID_HINTS):
            wants = "g1"
        elif any(h in text for h in _QUADRUPED_HINTS):
            wants = "go1"
        if wants is None:
            return None
        matches = [t for t in targets if kinds[t] == wants]
        return matches[0] if len(matches) == 1 else None
