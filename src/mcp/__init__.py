"""cadenza.mcp — Connect two robots into one terminal and let them coordinate.

Robots already have a channel to talk to the *user*: the
:class:`cadenza.inference.stream.Stream` narration line (``stream.say(...)``).
This package reuses that exact channel to let robots talk to *each other*,
routed through a FastMCP server that each robot accesses as an MCP client.

Quick start::

    import cadenza

    go1 = cadenza.go1()
    g1  = cadenza.g1()

    with cadenza.connect(go1, g1) as term:      # both join one MCP terminal
        go1.comm.tell("g1", "I'll take the left corridor")
        g1.comm.broadcast("copy, covering the right")
        print(g1.comm.messages())               # g1 reads what go1 said

FastMCP is an optional extra: ``pip install cadenza-lab[mcp]``.
"""

from __future__ import annotations

from cadenza.mcp.bus import CoordinationBus, Message
from cadenza.mcp.coordinator import MissionCoordinator, MissionPlan, SubGoal
from cadenza.mcp.link import CoordinationTerminal, RobotLink, connect
from cadenza.mcp.server import build_server

__all__ = [
    "CoordinationBus",
    "Message",
    "CoordinationTerminal",
    "RobotLink",
    "connect",
    "build_server",
    "MissionCoordinator",
    "MissionPlan",
    "SubGoal",
]
