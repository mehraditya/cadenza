"""FastMCP server exposing the CoordinationBus as tools.

Each connected robot accesses this server (in-process via an in-memory
transport, or over stdio for an external agent) and calls these tools to
coordinate with the other robot. The server is a thin, stateless wrapper — all
state lives in the :class:`~cadenza.mcp.bus.CoordinationBus` it closes over.

Build a server with :func:`build_server`; run one standalone with
``python -m cadenza.mcp`` (see :mod:`cadenza.mcp.__main__`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from cadenza.mcp.bus import CoordinationBus


def build_server(bus: "CoordinationBus", *, name: str = "cadenza-coordination") -> "FastMCP":
    """Return a FastMCP server whose tools route through ``bus``.

    Raises a clear ImportError if FastMCP isn't installed (it's an optional
    extra: ``pip install cadenza-lab[mcp]``).
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "Robot coordination needs FastMCP. Install it with "
            "`pip install cadenza-lab[mcp]` (or `pip install fastmcp`)."
        ) from exc

    mcp = FastMCP(name)

    @mcp.tool
    def register(robot: str, robot_type: str = "") -> dict[str, Any]:
        """Join the coordination terminal so other robots can reach you.

        Args:
            robot: Your unique name on the bus (e.g. "go1", "g1").
            robot_type: Optional platform tag (e.g. "quadruped", "humanoid").

        Returns the full roster of connected robots.
        """
        peers = bus.register(robot, robot_type or None)
        return {"robot": robot, "robots": peers}

    @mcp.tool
    def send_message(sender: str, recipient: str, message: str) -> dict[str, Any]:
        """Send a directed message to one other robot.

        The message is spoken into the shared terminal and dropped into the
        recipient's inbox for them to read with ``check_messages``.
        """
        msg = bus.send(sender, recipient, message)
        return {"delivered": True, **msg.as_dict()}

    @mcp.tool
    def broadcast(sender: str, message: str) -> dict[str, Any]:
        """Send a message to every other connected robot at once."""
        msg = bus.broadcast(sender, message)
        return {"delivered": True, **msg.as_dict()}

    @mcp.tool
    def check_messages(robot: str) -> dict[str, Any]:
        """Read and clear your inbox — the messages other robots sent you."""
        msgs = bus.receive(robot)
        return {
            "robot": robot,
            "count": len(msgs),
            "messages": [m.as_dict() for m in msgs],
        }

    @mcp.tool
    def list_robots() -> dict[str, Any]:
        """List every robot currently connected to the coordination terminal."""
        return {"robots": bus.robots()}

    return mcp
