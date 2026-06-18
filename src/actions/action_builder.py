"""Read-only built-in action library — offline payload (de)serialization + lookup.

This module is the *credential-free, offline* read path for the curated built-in
action pack that ships with the SDK. The pack itself is an auto-generated module
(``cadenza._builtin_actions``) committed by a maintainer who exports it from a
private backend; this repo never imports that backend, reads a Supabase key, or
touches the network. When the generated module is absent, the library degrades
to empty (an empty dict) — never an ImportError.

The built-in pack is public, read-only data, not a secret. Anything the SDK can
decode and run, a determined user can extract. Do not describe built-ins as
hidden or encrypted (see CLAUDE.md §0).

Payload contract (matches ``cadenza._builtin_actions.BUILTIN_ACTIONS``)::

    {
        "name": "<action_name>",
        "robot": "go1",                # or "g1"
        "type": "group",               # or "custom"
        # group:
        "steps": [{"action": "walk_forward", "distance_m": 0.4}, ...],
        # custom (instead of steps):
        # "joint_names": [...],
        # "frames": [{"t": 0.0, "joints": [...]}, ...],
        # "computed": {"total_duration_s": ..., ...},
    }

A *group* action is an ordered list of action calls, so its steps reuse the
existing :class:`~cadenza.actions.library.ActionCall` model rather than a parallel
schema. A *custom* action is a keyframe joint animation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cadenza.actions.library import ActionCall

BUILTIN_FORMAT = "cadenza-builtin-actions/1"


class ReadOnlyBuiltinError(RuntimeError):
    """Raised when something tries to edit or remove a read-only built-in."""


# ActionCall fields that may appear as parameters on a group payload step.
_STEP_PARAM_FIELDS = (
    "speed", "extension", "repeat", "distance_m", "rotation_rad",
    "duration_s", "speed_override", "height_override",
)


@dataclass
class GroupAction:
    """A named, ordered sequence of action calls (payload ``type: "group"``)."""
    name: str
    robot: str
    steps: list[ActionCall] = field(default_factory=list)
    type: str = "group"


@dataclass
class CustomAction:
    """A keyframe joint animation (payload ``type: "custom"``)."""
    name: str
    robot: str
    joint_names: list[str] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    computed: dict = field(default_factory=dict)
    type: str = "custom"


# --------------------------------------------------------------------------- #
# Payload <-> object  (one code path, used by file loading and built-ins)
# --------------------------------------------------------------------------- #

def _step_from_dict(step: dict) -> ActionCall:
    name = step.get("action") or step.get("action_name")
    if not name:
        raise ValueError(f"group step is missing an 'action' name: {step!r}")
    kwargs: dict = {}
    for f in _STEP_PARAM_FIELDS:
        if f in step:
            kwargs[f] = int(step[f]) if f == "repeat" else float(step[f])
    return ActionCall(action_name=name, **kwargs)


def _step_to_dict(call: ActionCall) -> dict:
    out: dict = {"action": call.action_name}
    default = ActionCall(action_name=call.action_name)
    for f in _STEP_PARAM_FIELDS:
        value = getattr(call, f)
        if value != getattr(default, f):
            out[f] = value
    return out


def action_from_payload(payload: dict):
    """Build a GroupAction/CustomAction from a build_action_payload() dict.

    Returns ``None`` for an empty/falsy payload so callers can treat "absent"
    and "empty" identically.
    """
    if not payload:
        return None
    name = payload["name"]
    robot = payload["robot"]
    atype = payload.get("type", "group")
    if atype == "custom":
        return CustomAction(
            name=name,
            robot=robot,
            joint_names=list(payload.get("joint_names", [])),
            frames=[dict(f) for f in payload.get("frames", [])],
            computed=dict(payload.get("computed", {})),
        )
    steps = [_step_from_dict(s) for s in payload.get("steps", [])]
    return GroupAction(name=name, robot=robot, steps=steps)


def build_action_payload(action) -> dict:
    """Serialize a GroupAction/CustomAction back to the canonical payload dict.

    Inverse of :func:`action_from_payload` — round-trips an action to the same
    dict shape stored in the built-in pack.
    """
    if isinstance(action, CustomAction):
        return {
            "name": action.name,
            "robot": action.robot,
            "type": "custom",
            "joint_names": list(action.joint_names),
            "frames": [dict(f) for f in action.frames],
            "computed": dict(action.computed),
        }
    if isinstance(action, GroupAction):
        return {
            "name": action.name,
            "robot": action.robot,
            "type": "group",
            "steps": [_step_to_dict(s) for s in action.steps],
        }
    raise TypeError(f"cannot serialize action of type {type(action).__name__}")


def load_action(path):
    """Load a single action from a JSON file using the shared parse path."""
    return action_from_payload(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Read-only built-in library  (no credentials, no network)
# --------------------------------------------------------------------------- #

def _builtin_table() -> dict:
    """The bundled library, or ``{}`` if the generated module isn't present."""
    try:
        from cadenza._builtin_actions import BUILTIN_ACTIONS
        return BUILTIN_ACTIONS
    except Exception:
        return {}


def list_builtin_actions(robot: str) -> list[str]:
    """Names of the read-only built-in actions for a robot (e.g. 'go1','g1')."""
    return sorted(_builtin_table().get(robot, {}))


def load_builtin_action(robot: str, name: str):
    """Return a GroupAction/CustomAction from the bundled library, or None."""
    payload = _builtin_table().get(robot, {}).get(name)
    return action_from_payload(payload) if payload else None


def is_builtin_action(robot: str, name: str) -> bool:
    return name in _builtin_table().get(robot, {})


# --------------------------------------------------------------------------- #
# Resolution order + read-only guard
# --------------------------------------------------------------------------- #

def resolve_action(project_dir, robot, name):
    """Resolve an action by name: a local project action wins, otherwise the
    read-only built-in library.

    Returns ``(action, source)`` where ``source`` is ``'local'`` or
    ``'builtin'``, or ``(None, None)`` if the name is unknown everywhere.
    """
    if project_dir is not None:
        local = Path(project_dir) / "actions" / f"{name}.json"
        if local.is_file():
            return load_action(local), "local"
    builtin = load_builtin_action(robot, name)
    if builtin is not None:
        return builtin, "builtin"
    return None, None


def ensure_action_editable(robot: str, name: str) -> None:
    """Raise :class:`ReadOnlyBuiltinError` if ``name`` is a built-in.

    Used by create/edit/remove paths to refuse mutation of the bundled pack.
    """
    if is_builtin_action(robot, name):
        raise ReadOnlyBuiltinError(
            f"{name} is a read-only built-in; it can't be edited or removed."
        )
