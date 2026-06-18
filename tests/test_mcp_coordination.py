"""Two robots coordinating through the FastMCP coordination terminal.

Verifies that two robots connected into one terminal can message each other
*through the MCP* (real FastMCP tools, in-memory transport — no network), and
that every message is narrated into the single shared terminal using the same
``Stream`` channel a robot uses to talk to the user.

Run with pytest::

    pytest tests/test_mcp_coordination.py

Or standalone::

    python tests/test_mcp_coordination.py
"""

import io

import pytest

import cadenza
from cadenza.mcp import CoordinationBus, CoordinationTerminal, build_server


# ── Bus-level (the pure hub, reusing the Stream narration channel) ────────────

def test_directed_message_lands_in_recipient_inbox():
    bus = CoordinationBus(narrate=False)
    bus.register("go1", "quadruped")
    bus.register("g1", "humanoid")

    bus.send("go1", "g1", "move to the doorway")

    g1_inbox = bus.receive("g1")
    assert [m.text for m in g1_inbox] == ["move to the doorway"]
    assert bus.receive("g1") == []          # inbox drains
    assert bus.receive("go1") == []          # sender doesn't get its own message


def test_broadcast_reaches_everyone_but_sender():
    bus = CoordinationBus(narrate=False)
    for name in ("go1", "g1", "go2"):
        bus.register(name)

    bus.broadcast("go1", "regroup at the pad")

    assert [m.text for m in bus.receive("g1")] == ["regroup at the pad"]
    assert [m.text for m in bus.receive("go2")] == ["regroup at the pad"]
    assert bus.receive("go1") == []          # sender excluded from its broadcast


def test_messages_narrate_into_one_shared_terminal():
    """Robot↔robot messages reuse the user-facing Stream, in one terminal."""
    buf = io.StringIO()
    bus = CoordinationBus(narrate=True, out=buf)
    bus.register("go1")
    bus.register("g1")

    bus.send("go1", "g1", "I'll scout left")
    bus.broadcast("g1", "holding position")

    out = buf.getvalue()
    assert "[go1]" in out and "I'll scout left" in out      # sender's channel
    assert "[g1]" in out and "holding position" in out


def test_unknown_recipient_and_unregistered_sender_raise():
    bus = CoordinationBus(narrate=False)
    bus.register("go1")
    with pytest.raises(ValueError):
        bus.send("go1", "ghost", "hello")     # unknown recipient
    with pytest.raises(ValueError):
        bus.send("ghost", "go1", "hello")     # unregistered sender


# ── MCP-level (the tools are real FastMCP tools each robot calls) ─────────────

def test_server_exposes_the_coordination_tools():
    import asyncio

    from fastmcp import Client

    bus = CoordinationBus(narrate=False)
    server = build_server(bus)

    async def _go():
        async with Client(server) as client:
            tools = {t.name for t in await client.list_tools()}
            assert {"register", "send_message", "broadcast",
                    "check_messages", "list_robots"} <= tools

            await client.call_tool("register", {"robot": "go1"})
            await client.call_tool("register", {"robot": "g1"})
            await client.call_tool(
                "send_message",
                {"sender": "go1", "recipient": "g1", "message": "ping"},
            )
            res = await client.call_tool("check_messages", {"robot": "g1"})
            assert res.data["count"] == 1
            assert res.data["messages"][0]["text"] == "ping"

    asyncio.run(_go())


# ── Top-level API (two real robot objects coordinating) ───────────────────────

def test_two_robots_coordinate_via_cadenza_connect():
    buf = io.StringIO()
    go1 = cadenza.go1()
    g1 = cadenza.g1()

    with cadenza.connect(go1, g1, out=buf) as term:
        # Both robots are connected to the one terminal.
        assert term.robots() == ["g1", "go1"]
        # connect() attaches a .comm link onto each robot object.
        assert go1.comm is term.link("go1")
        assert g1.comm is term.link("g1")

        # Each robot reaches the other strictly through its MCP link.
        go1.comm.tell("g1", "I'll take the left corridor")
        g1.comm.broadcast("copy, covering the right")

        g1_msgs = g1.comm.messages()
        go1_msgs = go1.comm.messages()

    assert [(m["from"], m["text"]) for m in g1_msgs] == \
        [("go1", "I'll take the left corridor")]
    assert [(m["from"], m["text"]) for m in go1_msgs] == \
        [("g1", "copy, covering the right")]

    # The whole exchange showed up in the single shared terminal.
    out = buf.getvalue()
    assert "I'll take the left corridor" in out
    assert "copy, covering the right" in out


def test_local_link_prefers_explicit_name_for_duplicate_robots():
    """Two robots of the same kind can coexist with explicit names."""
    term = CoordinationTerminal(narrate=False)
    try:
        a = term.connect(name="scout")
        b = term.connect(name="anchor")
        a.tell("anchor", "in position")
        assert [m["text"] for m in b.messages()] == ["in position"]
        with pytest.raises(ValueError):
            term.connect(name="scout")        # duplicate name refused
    finally:
        term.close()


# ── Coordinator (split one human goal into per-robot subgoals) ────────────────

def test_coordinator_splits_explicit_targets_and_broadcast():
    go1 = cadenza.go1()
    g1 = cadenza.g1()
    buf = io.StringIO()

    with cadenza.connect(go1, g1, out=buf) as term:
        plan = term.coordinate(
            "go1 scout the left corridor and g1 hold the doorway "
            "then everyone regroup at the pad"
        )
        # Each subgoal was delegated over the MCP into the right inbox(es).
        # (read inboxes inside the terminal — the MCP loop closes on exit)
        go1_texts = [m["text"] for m in go1.comm.messages()]
        g1_texts = [m["text"] for m in g1.comm.messages()]

    # Three subgoals: two explicitly targeted, one broadcast.
    assert [(sg.robot, sg.goal, sg.broadcast) for sg in plan.subgoals] == [
        ("go1", "scout the left corridor", False),
        ("g1", "hold the doorway", False),
        ("*", "regroup at the pad", True),
    ]
    assert go1_texts == ["scout the left corridor", "regroup at the pad"]
    assert g1_texts == ["hold the doorway", "regroup at the pad"]
    # ...and narrated into the one shared terminal by the coordinator.
    out = buf.getvalue()
    assert "[coordinator]" in out
    assert "scout the left corridor" in out and "regroup at the pad" in out


def test_coordinator_routes_unnamed_clauses_by_capability():
    from cadenza.mcp import MissionCoordinator

    go1 = cadenza.go1()
    g1 = cadenza.g1()
    with cadenza.connect(go1, g1, narrate=False) as term:
        plan = MissionCoordinator(term).plan(
            "crawl under the table and grab the red cube"
        )

    assert [(sg.robot, sg.goal) for sg in plan.subgoals] == [
        ("go1", "crawl under the table"),   # quadruped hint -> go1
        ("g1", "grab the red cube"),         # humanoid hint -> g1
    ]


def test_coordinator_round_robins_generic_clauses():
    from cadenza.mcp import MissionCoordinator

    go1 = cadenza.go1()
    g1 = cadenza.g1()
    with cadenza.connect(go1, g1, narrate=False) as term:
        plan = MissionCoordinator(term).plan(
            "search the room, mark the exits, sweep the hallway"
        )

    # Three generic clauses dealt out evenly across two robots, stable order.
    robots = [sg.robot for sg in plan.subgoals]
    assert sorted(robots) == ["g1", "go1", "go1"] or sorted(robots) == ["g1", "g1", "go1"]
    assert len(plan.subgoals) == 3
    # Every connected robot gets work via assignments().
    assigned = plan.assignments()
    assert assigned["go1"] and assigned["g1"]


def test_coordinator_subgoals_parse_into_actions():
    from cadenza.mcp import MissionCoordinator

    go1 = cadenza.go1()
    g1 = cadenza.g1()
    with cadenza.connect(go1, g1, narrate=False) as term:
        plan = MissionCoordinator(term).plan("go1 walk_forward")

    assert [a.action_name for a in plan.actions("go1")] == ["walk_forward"]


def test_coordinator_needs_at_least_one_robot():
    from cadenza.mcp import CoordinationTerminal, MissionCoordinator

    term = CoordinationTerminal(narrate=False)
    try:
        with pytest.raises(ValueError):
            MissionCoordinator(term).plan("do something")
    finally:
        term.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
