"""RobotLink + CoordinationTerminal — connect two robots into one terminal.

``CoordinationTerminal`` hosts the bus and the FastMCP server. Each robot you
``connect()`` gets a :class:`RobotLink`: a real FastMCP client (in-memory
transport) that the robot uses to call the coordination tools. So every robot
genuinely talks *through the MCP* to reach the other — there is no private back
door around it.

The MCP protocol is async, but robots in a script are driven synchronously, so
``RobotLink`` exposes a plain blocking API backed by a single dedicated event
loop running on a background thread.

Typical use::

    import cadenza

    go1 = cadenza.go1()
    g1  = cadenza.g1()

    with cadenza.connect(go1, g1) as term:        # both connected to one MCP
        go1.comm.tell("g1", "I'll scout left, you hold the door")
        g1.comm.broadcast("copy that, holding position")
        for m in go1.comm.messages():             # go1 reads its inbox
            print(m["from"], "said", m["text"])
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from cadenza.mcp.bus import CoordinationBus
from cadenza.mcp.server import build_server


# ── Background event loop ─────────────────────────────────────────────────────

class _Loop:
    """A dedicated asyncio loop on a daemon thread, giving links a sync API."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="cadenza-mcp-loop", daemon=True
        )
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        if not self._loop.is_closed():
            self._loop.close()


# ── Per-robot MCP client ──────────────────────────────────────────────────────

class RobotLink:
    """One robot's handle on the coordination MCP.

    Created by :meth:`CoordinationTerminal.connect`; reachable as ``robot.comm``.
    """

    def __init__(self, name: str, client: Any, loop: _Loop, *, robot_type: str = ""):
        self.name = name
        self.robot_type = robot_type
        self._client = client
        self._loop = loop
        self._open = False

    # internal: open the MCP session and register this robot
    def _connect(self) -> None:
        if self._open:
            return
        self._loop.run(self._client.__aenter__())
        self._open = True
        self._call("register", robot=self.name, robot_type=self.robot_type)

    def _call(self, tool: str, **args: Any) -> Any:
        async def _go() -> Any:
            result = await self._client.call_tool(tool, args)
            return result.data

        return self._loop.run(_go())

    # ── Public API (what a robot calls to coordinate) ─────────────────────────

    def tell(self, recipient: str, message: str) -> dict[str, Any]:
        """Send a directed message to another robot by name."""
        return self._call("send_message", sender=self.name, recipient=recipient,
                           message=message)

    def broadcast(self, message: str) -> dict[str, Any]:
        """Send a message to every other connected robot."""
        return self._call("broadcast", sender=self.name, message=message)

    def messages(self) -> list[dict[str, Any]]:
        """Read and clear this robot's inbox (list of message dicts)."""
        return self._call("check_messages", robot=self.name)["messages"]

    def peers(self) -> list[str]:
        """Names of all robots connected to the terminal (including self)."""
        return self._call("list_robots")["robots"]

    def close(self) -> None:
        if self._open:
            self._loop.run(self._client.__aexit__(None, None, None))
            self._open = False


# ── The terminal that ties two robots together ────────────────────────────────

def _infer_name(robot: Any) -> str:
    """Best-effort stable name for a robot object."""
    if isinstance(robot, str):
        return robot
    for attr in ("name", "robot", "_robot"):
        val = getattr(robot, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(robot).__name__.lower()


def _infer_type(robot: Any) -> str:
    cls = type(robot).__name__.lower()
    if "go1" in cls or "go2" in cls:
        return "quadruped"
    if "g1" in cls:
        return "humanoid"
    return ""


class CoordinationTerminal:
    """Hosts the coordination MCP and connects robots to it.

    One terminal, one shared bus, one FastMCP server; each robot gets its own
    MCP client. Use as a context manager to clean up sessions and the loop.
    """

    def __init__(self, *, narrate: bool = True, out: Any = None,
                 server_name: str = "cadenza-coordination"):
        self.bus = CoordinationBus(narrate=narrate, out=out)
        self.server = build_server(self.bus, name=server_name)
        self._loop = _Loop()
        self._links: dict[str, RobotLink] = {}

    def connect(self, robot: Any = None, *, name: str | None = None,
                robot_type: str | None = None) -> RobotLink:
        """Connect a robot (object or bare name) to the terminal.

        If ``robot`` is a cadenza robot object, its ``.comm`` attribute is set to
        the returned link so it can coordinate via ``robot.comm.tell(...)``.
        """
        from fastmcp import Client

        rname = name or _infer_name(robot)
        rtype = robot_type if robot_type is not None else _infer_type(robot)
        if rname in self._links:
            raise ValueError(
                f"a robot named {rname!r} is already connected; "
                f"pass name=... to connect a second one"
            )

        link = RobotLink(rname, Client(self.server), self._loop, robot_type=rtype)
        link._connect()
        self._links[rname] = link
        if robot is not None and not isinstance(robot, str):
            try:
                robot.comm = link
            except Exception:
                pass  # robot model may forbid new attrs; link is still usable
        return link

    def link(self, name: str) -> RobotLink:
        """Return the link for an already-connected robot."""
        return self._links[name]

    def robots(self) -> list[str]:
        return self.bus.robots()

    def robot_type(self, name: str) -> str:
        """Platform tag a robot registered with (e.g. 'quadruped'); '' if unset."""
        link = self._links.get(name)
        return link.robot_type if link is not None else ""

    def coordinate(self, goal: str, *, route_by_capability: bool = True) -> Any:
        """Split one human goal into per-robot subgoals and dispatch them.

        Hands the mission to a :class:`~cadenza.mcp.coordinator.MissionCoordinator`
        (created once and reused), which breaks ``goal`` into smaller individual
        goals and delegates each to a connected robot over this same MCP — so the
        hand-off is narrated into the shared terminal. Returns the
        :class:`~cadenza.mcp.coordinator.MissionPlan`.

        Example::

            with cadenza.connect(go1, g1) as term:
                plan = term.coordinate("go1 scout left and g1 hold the door")
                go1.comm.messages()   # -> "scout left"
        """
        from cadenza.mcp.coordinator import MissionCoordinator

        if getattr(self, "_coordinator", None) is None:
            self._coordinator = MissionCoordinator(
                self, route_by_capability=route_by_capability
            )
        return self._coordinator.run(goal)

    def history(self) -> list[dict[str, Any]]:
        """Full ordered transcript of the coordination conversation."""
        return [m.as_dict() for m in self.bus.history()]

    def close(self) -> None:
        for link in self._links.values():
            try:
                link.close()
            except Exception:
                pass
        self._links.clear()
        self._loop.close()

    def __enter__(self) -> "CoordinationTerminal":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def connect(*robots: Any, narrate: bool = True, out: Any = None) -> CoordinationTerminal:
    """Connect one or more robots into a single coordination terminal.

    Returns a :class:`CoordinationTerminal`. Each robot object passed in gets a
    ``.comm`` link it can use to message the others through the MCP::

        go1, g1 = cadenza.go1(), cadenza.g1()
        term = cadenza.connect(go1, g1)
        go1.comm.tell("g1", "ready when you are")
    """
    term = CoordinationTerminal(narrate=narrate, out=out)
    for robot in robots:
        term.connect(robot)
    return term
