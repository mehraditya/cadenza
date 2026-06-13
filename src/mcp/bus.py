"""CoordinationBus — the shared message hub two robots talk through.

A robot already has one way to communicate: ``cadenza.inference.stream.Stream``,
the narration channel it uses to speak to the *user* in the terminal
(``stream.say("...")``). This module reuses that exact channel for robot↔robot
coordination.

When two robots are connected into one terminal, every message one robot sends
another is:

  1. **Spoken into the shared terminal** through the sender's own ``Stream`` —
     the same human-readable channel the robot uses to talk to the user — so a
     person watching a single terminal sees the whole conversation live.
  2. **Queued into the recipient's inbox**, so the other robot can read it and
     act on it.

The bus itself is pure, in-process, and dependency-light (only stdlib + the
existing ``Stream``). The FastMCP layer in :mod:`cadenza.mcp.server` exposes it
as tools each robot can call; see :mod:`cadenza.mcp.link` for the per-robot
client wrapper.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, TextIO

from cadenza.inference.stream import Stream

# Recipient sentinels that mean "everyone but the sender".
_BROADCAST = {"", "*", "all", "everyone"}


@dataclass
class Message:
    """One message routed across the bus."""

    seq: int
    sender: str
    recipient: str          # "" / "*" / "all" → broadcast
    text: str
    ts: float

    @property
    def is_broadcast(self) -> bool:
        return self.recipient.lower() in _BROADCAST

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "from": self.sender,
            "to": "all" if self.is_broadcast else self.recipient,
            "text": self.text,
            "ts": self.ts,
            "broadcast": self.is_broadcast,
        }


class CoordinationBus:
    """In-process hub that lets connected robots coordinate in one terminal.

    Thread-safe: the FastMCP server runs sync tools in a worker-thread pool, so
    several robots may touch the bus concurrently.
    """

    def __init__(self, *, narrate: bool = True, out: TextIO | None = None):
        self._lock = threading.RLock()
        self._narrate = bool(narrate)
        self._out = out
        self._robots: dict[str, dict[str, Any]] = {}
        self._inbox: dict[str, list[Message]] = {}
        self._cond: dict[str, threading.Condition] = {}
        self._streams: dict[str, Stream] = {}
        self._log: list[Message] = []
        self._seq = 0

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, robot_type: str | None = None) -> list[str]:
        """Add a robot to the bus (idempotent). Returns all connected names."""
        if not name:
            raise ValueError("a robot must register with a non-empty name")
        with self._lock:
            self._robots[name] = {"name": name, "type": robot_type}
            self._inbox.setdefault(name, [])
            self._cond.setdefault(name, threading.Condition(self._lock))
            # Each robot gets its own narration channel — the very same kind of
            # Stream it uses to talk to the user — prefixed with its name.
            self._streams.setdefault(
                name, Stream(prefix=name, enabled=self._narrate, out=self._out)
            )
        return self.robots()

    def robots(self) -> list[str]:
        with self._lock:
            return sorted(self._robots)

    def is_registered(self, name: str) -> bool:
        with self._lock:
            return name in self._robots

    # ── Sending ──────────────────────────────────────────────────────────────

    def send(self, sender: str, recipient: str, text: str) -> Message:
        """Deliver a directed message from ``sender`` to ``recipient``."""
        with self._lock:
            self._require(sender)
            if recipient.lower() not in _BROADCAST and recipient not in self._robots:
                raise ValueError(
                    f"unknown recipient {recipient!r}; "
                    f"connected robots: {self.robots()}"
                )
            msg = self._mk(sender, recipient, text)
            if msg.is_broadcast:
                self._fan_out(msg, exclude=sender)
            else:
                self._deliver(msg, recipient)
            self._narrate_msg(msg)
            return msg

    def broadcast(self, sender: str, text: str) -> Message:
        """Send a message to every other connected robot."""
        return self.send(sender, "all", text)

    # ── Receiving ────────────────────────────────────────────────────────────

    def receive(
        self, name: str, *, block: bool = False, timeout: float | None = None
    ) -> list[Message]:
        """Drain (and return) ``name``'s inbox.

        With ``block=True`` and an empty inbox, wait up to ``timeout`` seconds
        for at least one message before returning.
        """
        with self._lock:
            self._require(name)
            if block and not self._inbox[name]:
                self._cond[name].wait(timeout)
            msgs = self._inbox[name]
            self._inbox[name] = []
            return msgs

    def peek(self, name: str) -> list[Message]:
        """Return ``name``'s pending messages without draining them."""
        with self._lock:
            self._require(name)
            return list(self._inbox[name])

    def history(self) -> list[Message]:
        """Full ordered transcript of everything sent on the bus."""
        with self._lock:
            return list(self._log)

    # ── Internals ────────────────────────────────────────────────────────────

    def _mk(self, sender: str, recipient: str, text: str) -> Message:
        self._seq += 1
        msg = Message(self._seq, sender, recipient, text, time.time())
        self._log.append(msg)
        return msg

    def _deliver(self, msg: Message, recipient: str) -> None:
        self._inbox[recipient].append(msg)
        self._cond[recipient].notify_all()

    def _fan_out(self, msg: Message, *, exclude: str) -> None:
        for name in self._robots:
            if name != exclude:
                self._deliver(msg, name)

    def _narrate_msg(self, msg: Message) -> None:
        """Speak the message into the shared terminal via the sender's Stream."""
        stream = self._streams.get(msg.sender)
        if stream is None:
            return
        if msg.is_broadcast:
            stream.say(f"📣 (all) {msg.text}")
        else:
            stream.say(f"→ {msg.recipient}: {msg.text}")

    def _require(self, name: str) -> None:
        if name not in self._robots:
            raise ValueError(
                f"robot {name!r} is not connected; connected robots: {self.robots()}"
            )
