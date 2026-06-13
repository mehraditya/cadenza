"""Tests for the read-only, offline built-in action library.

These run with NO network and NO Supabase env vars, and touch only the action
model + the bundled dict — no simulator or heavy extras required.
"""

import sys
import types

import pytest

from cadenza.actions import action_builder as ab
from cadenza.actions.action_builder import GroupAction, CustomAction


# A small, self-contained built-in pack used to stand in for the maintainer's
# generated cadenza._builtin_actions module.
_FAKE_PACK = {
    "go1": {
        "trot": {
            "name": "trot",
            "robot": "go1",
            "type": "group",
            "steps": [
                {"action": "walk_forward", "distance_m": 0.4},
                {"action": "turn_left", "rotation_rad": 1.57},
                {"action": "sit", "repeat": 2},
            ],
        },
    },
    "g1": {
        "wave": {
            "name": "wave",
            "robot": "g1",
            "type": "custom",
            "joint_names": ["shoulder", "elbow"],
            "frames": [
                {"t": 0.0, "joints": [0.0, 0.0]},
                {"t": 0.5, "joints": [1.2, 0.6]},
            ],
            "computed": {"total_duration_s": 0.5, "max_joint_speed_rad_s": 2.4},
        },
    },
}


@pytest.fixture
def with_pack(monkeypatch):
    """Inject a fake cadenza._builtin_actions module for the duration of a test."""
    mod = types.ModuleType("cadenza._builtin_actions")
    mod.BUILTIN_FORMAT = "cadenza-builtin-actions/1"
    mod.BUILTIN_ACTIONS = _FAKE_PACK
    monkeypatch.setitem(sys.modules, "cadenza._builtin_actions", mod)
    yield


@pytest.fixture
def without_pack(monkeypatch):
    """Ensure the generated module is absent (the default state of this repo)."""
    monkeypatch.setitem(sys.modules, "cadenza._builtin_actions", None)
    yield


# --------------------------------------------------------------------------- #
# 1. list_builtin_actions: non-empty when present, [] (no crash) when absent
# --------------------------------------------------------------------------- #

def test_list_builtin_actions_present(with_pack):
    assert ab.list_builtin_actions("go1") == ["trot"]
    assert ab.list_builtin_actions("g1") == ["wave"]


def test_list_builtin_actions_absent(without_pack):
    assert ab.list_builtin_actions("go1") == []
    assert ab.list_builtin_actions("g1") == []


def test_unknown_robot_is_empty(with_pack):
    assert ab.list_builtin_actions("go2") == []


# --------------------------------------------------------------------------- #
# 2. load_builtin_action returns an object that round-trips to the same dict
# --------------------------------------------------------------------------- #

def test_group_action_round_trips(with_pack):
    action = ab.load_builtin_action("go1", "trot")
    assert isinstance(action, GroupAction)
    assert [s.action_name for s in action.steps] == [
        "walk_forward", "turn_left", "sit",
    ]
    assert ab.build_action_payload(action) == _FAKE_PACK["go1"]["trot"]


def test_custom_action_round_trips(with_pack):
    action = ab.load_builtin_action("g1", "wave")
    assert isinstance(action, CustomAction)
    assert action.joint_names == ["shoulder", "elbow"]
    assert ab.build_action_payload(action) == _FAKE_PACK["g1"]["wave"]


def test_load_missing_builtin_is_none(with_pack):
    assert ab.load_builtin_action("go1", "does_not_exist") is None


# --------------------------------------------------------------------------- #
# 3. The built-in read path needs no Supabase env and no network
# --------------------------------------------------------------------------- #

def test_read_path_offline_no_supabase(with_pack, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    # The read path must work and must not pull in any supabase module.
    assert ab.is_builtin_action("go1", "trot") is True
    assert ab.load_builtin_action("go1", "trot") is not None

    assert not any("supabase" in name for name in sys.modules), \
        "the built-in read path must never import supabase"


# --------------------------------------------------------------------------- #
# 4. resolve_action prefers a local action over a built-in of the same name
# --------------------------------------------------------------------------- #

def test_resolve_prefers_local(with_pack, tmp_path):
    actions = tmp_path / "actions"
    actions.mkdir()
    (actions / "trot.json").write_text(
        '{"name": "trot", "robot": "go1", "type": "group", '
        '"steps": [{"action": "crawl_forward", "distance_m": 1.0}]}'
    )

    action, source = ab.resolve_action(str(tmp_path), "go1", "trot")
    assert source == "local"
    assert action.steps[0].action_name == "crawl_forward"


def test_resolve_falls_back_to_builtin(with_pack, tmp_path):
    action, source = ab.resolve_action(str(tmp_path), "go1", "trot")
    assert source == "builtin"
    assert action.steps[0].action_name == "walk_forward"


def test_resolve_unknown_is_none(with_pack):
    assert ab.resolve_action(None, "go1", "nope") == (None, None)


# --------------------------------------------------------------------------- #
# 5. Removing/editing a built-in raises the clean read-only error
# --------------------------------------------------------------------------- #

def test_ensure_editable_refuses_builtin(with_pack):
    with pytest.raises(ab.ReadOnlyBuiltinError) as exc:
        ab.ensure_action_editable("go1", "trot")
    assert "read-only built-in" in str(exc.value)


def test_ensure_editable_allows_non_builtin(with_pack):
    # No exception for a name that isn't bundled.
    ab.ensure_action_editable("go1", "my_local_action")


# --------------------------------------------------------------------------- #
# Decode/execute path: a mission can call a built-in by name (offline)
# --------------------------------------------------------------------------- #

def test_parser_expands_builtin_group_by_name(with_pack):
    from cadenza.parser import CommandParser

    calls = CommandParser("go1").parse("trot")
    assert [c.action_name for c in calls] == ["walk_forward", "turn_left", "sit"]


def test_parser_unaffected_when_pack_absent(without_pack):
    from cadenza.parser import CommandParser

    # "stand" is a real go1 library primitive — normal parse still works.
    calls = CommandParser("go1").parse("stand")
    assert [c.action_name for c in calls] == ["stand"]
