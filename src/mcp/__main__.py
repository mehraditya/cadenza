"""Run the coordination MCP as a standalone stdio server.

    python -m cadenza.mcp

An external MCP client (e.g. an LLM agent standing in for a robot) can connect
over stdio and call the same tools — register, send_message, broadcast,
check_messages, list_robots — to coordinate with other connected robots.
"""

from __future__ import annotations

from cadenza.mcp.bus import CoordinationBus
from cadenza.mcp.server import build_server


def main() -> None:
    bus = CoordinationBus()
    server = build_server(bus)
    server.run()  # stdio transport by default


if __name__ == "__main__":
    main()
