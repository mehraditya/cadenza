"""Stream — live narration channel for VLA orchestrators.

Goals:
  * One natural-language line per interesting event — not a kv dump.
  * Coalesce repeats: when the same line would print again, increment a
    counter on the existing line (``(×N)``) rather than spamming.
  * Drop redundant events (``execute_done``, ``infer_picked`` of the same
    action, raw setup chatter) — they don't help a human reader.
  * Free-form model narration via ``.say(...)`` always prints fresh and
    flushes any pending repeat counter.

Turned on by ``go1.run(..., streaming=True)``. When off, every method is
a no-op so adapters can call ``.say()`` unconditionally.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO


class Stream:
    """Natural-language event sink with in-place repeat coalescing."""

    def __init__(
        self,
        *,
        prefix: str = "VLA",
        enabled: bool = True,
        out: TextIO | None = None,
    ):
        self.prefix = prefix
        self.enabled = bool(enabled)
        self._out: TextIO = out if out is not None else sys.stdout
        self._last: str | None = None
        self._repeat: int = 0
        # ANSI carriage-up-and-clear only when we're attached to a real tty.
        self._tty = bool(getattr(self._out, "isatty", lambda: False)())

    # ── Channel control ─────────────────────────────────────────────────────

    def enable(self) -> "Stream":
        self.enabled = True
        return self

    def disable(self) -> "Stream":
        self.enabled = False
        return self

    # ── Public API ──────────────────────────────────────────────────────────

    def emit(self, event: str, **details: Any) -> None:
        """Translate a structured event into one natural-language line.

        Many events map to ``None`` and are dropped. Identical consecutive
        lines coalesce into a single line that grows a ``(×N)`` counter.
        """
        if not self.enabled:
            return
        msg = self._render(event, details)
        if msg is None:
            return
        if msg == self._last:
            self._repeat += 1
            self._overwrite_last()
        else:
            self._end_current_line()
            self._last = msg
            self._repeat = 1
            self._print(self._format(msg, 1), newline=True)

    def say(self, message: str) -> None:
        """Free-form narration from the model — always fresh, never coalesced."""
        if not self.enabled:
            return
        self._end_current_line()
        self._last = None
        self._repeat = 0
        self._print(f"  [{self.prefix}] » {message}", newline=True)

    # ── Internals ───────────────────────────────────────────────────────────

    def _render(self, event: str, d: dict[str, Any]) -> str | None:
        """Map (event, details) → natural sentence, or None to suppress."""
        # Setup / startup chatter
        if event == "setup_start":
            return f"Setting up{self._tail(d, 'retries', ' with up to {} retries')}."
        if event == "setup_done":
            g = d.get("guardian")
            return f"Guardian ready ({g})." if g else "Guardian ready."
        if event == "models_ready":
            tgt = d.get("target")
            mods = d.get("modalities") or []
            mod_str = f" using {', '.join(mods)}" if mods else ""
            tgt_str = f", heading to {self._fmt_xy(tgt)}" if tgt else ""
            return f"Models loaded{mod_str}{tgt_str}."

        # Per-tick chatter
        if event == "bootstrap":
            return f"First action: {self._action_phrase(d)}."
        if event == "execute_start":
            phrase = self._action_phrase(d)
            return f"→ {phrase}" if phrase else None
        if event == "infer_pending":
            return "thinking… (motors keep moving)"

        # Sequential lifecycle
        if event == "step_start":
            return f"Starting {self._action_phrase(d)}."
        if event == "detect":
            pos = d.get("position", "ahead")
            size = d.get("size", "obstacle")
            rem = d.get("remaining_m")
            tail = f"; {rem:.1f}m left to go" if isinstance(rem, (int, float)) else ""
            return f"DETECTED OBSTACLES — Navigating around {pos} ({size}){tail}..."
        if event == "avoid_start":
            # The detect banner already announced the avoidance; the
            # individual steps render via avoid_step right below.
            return None
        if event == "avoid_step":
            step = d.get("step") or d
            phrase = self._action_phrase({"action": step.get("name"),
                                          "dist": step.get("distance_m"),
                                          "rot":  step.get("rotation_rad")})
            return f"   ↳ {phrase}" if phrase else None
        if event == "resume":
            return f"Path clear — resuming {d.get('action')} ({d.get('remaining_m', 0):.1f}m left)."
        if event == "retries_exceeded":
            return f"Retries exhausted — abandoning {d.get('abandoning')}."
        if event == "step_complete":
            return f"Step complete: {d.get('action')}."

        # Terminal
        if event == "target_reached":
            d_m = d.get("distance_m")
            return f"Target reached ({d_m:.2f}m to pad)." \
                if isinstance(d_m, (int, float)) else "Target reached."
        if event == "done":
            return f"Done — {d.get('reason', 'finished')}."

        # Suppress everything else (execute_done, infer_picked, avoid_step,
        # avoid_complete, infer_wait, …).
        return None

    # ── Phrase helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _action_phrase(d: dict[str, Any]) -> str:
        name = d.get("action") or d.get("step", {}).get("name") if isinstance(d.get("step"), dict) else d.get("action")
        if not name:
            return ""
        dist = d.get("dist") or d.get("distance_m")
        rot = d.get("rot") or d.get("rotation_rad")
        if isinstance(d.get("step"), dict):
            dist = dist or d["step"].get("distance_m")
            rot = rot or d["step"].get("rotation_rad")
        bits = [name.replace("_", " ")]
        if isinstance(dist, (int, float)) and dist:
            bits.append(f"{dist:.2f}m")
        if isinstance(rot, (int, float)) and rot:
            bits.append(f"{rot:.2f}rad")
        return " ".join(bits)

    @staticmethod
    def _fmt_xy(v: Any) -> str:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return f"({float(v[0]):.1f}, {float(v[1]):.1f})"
        return str(v)

    @staticmethod
    def _tail(d: dict[str, Any], key: str, template: str) -> str:
        v = d.get(key)
        return template.format(v) if v not in (None, "", 0) else ""

    # ── Line rendering ──────────────────────────────────────────────────────

    def _format(self, msg: str, count: int) -> str:
        body = f"  [{self.prefix}] {msg}"
        if count > 1:
            body += f"  (×{count})"
        return body

    def _print(self, text: str, *, newline: bool) -> None:
        self._out.write(text + ("\n" if newline else ""))
        try:
            self._out.flush()
        except Exception:
            pass

    def _overwrite_last(self) -> None:
        """Rewrite the previous line with an updated repeat counter."""
        if self._last is None:
            return
        line = self._format(self._last, self._repeat)
        if self._tty:
            # Move cursor up one line, clear it, print the new version.
            self._out.write("\x1b[F\x1b[2K")
            self._print(line, newline=True)
        else:
            # Not a tty — re-printing on a new line is the only option;
            # but we suppress that and only show the running total when the
            # line changes.
            pass

    def _end_current_line(self) -> None:
        """When transitioning to a new message and the prior line was a
        coalesced repeat on a non-tty stream, flush the final count once."""
        if (not self._tty) and self._last is not None and self._repeat > 1:
            self._print(self._format(self._last, self._repeat), newline=True)
