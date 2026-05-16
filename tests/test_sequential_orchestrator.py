"""End-to-end smoke test of the Sequential VLA orchestrator on a real Go1.

Real components, no mocks:
  - cadenza.go1()                  — real Go1 controller, real MuJoCo physics
  - cadenza.inference.Sequential   — the orchestrator under test
  - cadenza.vla.VLAGuardian        — default guardian (SmolVLM-256M-Instruct)
  - models/go1/obstacle_scene.xml  — bundled scene with box / barrier / cone

Demonstrates the three new Sequential params end-to-end:
  - logging=PATH      writes a JSONL event log usable for offline training
  - retries=N         caps re-attempts per step so the robot can't thrash
  - guardian=None     uses the real VLAGuardian (override to plug your own)

Run (macOS needs mjpython for the MuJoCo viewer)::

    mjpython tests/test_sequential_orchestrator.py
"""

import json
from pathlib import Path

import cadenza
from cadenza.inference import Sequential

LOG = Path("sequential_run.jsonl")

# Real Go1 wired to the real Sequential orchestrator. show_camera=False
# keeps the guardian's preview window from popping up so the only window
# is the MuJoCo viewer itself.
go1 = cadenza.go1(
    xml_path="models/go1/obstacle_scene.xml",
    inference=Sequential(
        retries=3,
        logging="sequential_run.jsonl",
        show_camera=False,
    ),
)

go1.run([go1.walk_forward(distance_m=8.0)])

# After the viewer is closed, dump the log.
print()
print("─" * 64)
print(f"  JSONL log → {LOG}")
print("─" * 64)
events = [json.loads(l) for l in LOG.read_text().splitlines()]
for e in events:
    extras = {k: v for k, v in e.items() if k not in ("ts", "event")}
    print(f"  {e['event']:18}  {extras if extras else ''}")
print(f"\n  {len(events)} events captured.")
