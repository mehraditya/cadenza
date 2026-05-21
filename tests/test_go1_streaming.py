"""End-to-end test of ``go1.run(streaming=True)`` on a real obstacle course.

Real components:
  - cadenza.go1()                  — real Go1 controller + MuJoCo physics
  - cadenza.inference.ChainOfThought — concurrent inference + execution
  - ai_models.go1.VLA              — real WorldModelAdapter (closed-loop VLA)
  - models/go1/obstacle_scene.xml  — bundled scene (box, barrier, cone, goal pad)

Demonstrates streaming:
  1. Lifecycle events from the orchestrator land in the terminal live.
  2. The model itself narrates ``before`` the run starts and ``when`` it
     reaches the goal, via ``observation["stream"].say(...)``.

    mjpython tests/test_go1_streaming.py
"""

import cadenza
from cadenza.inference import ChainOfThought
from ai_models.go1 import VLA


class NarratingVLA(VLA):
    """Thin wrapper around the real VLA that speaks at start and arrival."""

    def propose_actions(self, observation, goal, vocabulary, history):
        reply = super().propose_actions(observation, goal, vocabulary, history)
        stream = observation.get("stream")
        if stream is not None:
            if not history:
                stream.say(
                    f'Plan: "{goal}"  →  target {observation.get("target_xy")}'
                )
            if reply.done:
                pos = observation.get("pos")
                stream.say(
                    f"Goal reached at pos="
                    f"{tuple(round(float(v), 2) for v in pos)}"
                )
        return reply


go1 = cadenza.go1(
    xml_path="models/go1/obstacle_scene.xml",
    inference=ChainOfThought(
        model=NarratingVLA(),
        goal="reach the green pad past the box, barrier, and cone",
        target=(-9.0, 0.0),    # goal_pad in obstacle_scene.xml
        max_steps=80,
    ),
)

go1.run([go1.walk_forward()], streaming=True)
