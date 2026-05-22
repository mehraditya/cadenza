"""End-to-end test of ``cadenza.spatial.SpatialMemory`` on a complex gym.

Real components only:
  - cadenza.Scene                  — programmatic gym builder
  - cadenza.go1()                  — real Go1 + MuJoCo physics
  - cadenza.inference.ChainOfThought — concurrent VLA loop
  - ai_models.go1.VLA              — real WorldModelAdapter
  - cadenza.spatial.SpatialMemory  — visual-only 3D memory under test

OUTPUTS
-------
The single user-facing output is a STL file that **updates as the robot
walks**. Every few ticks the modality rewrites it atomically with the
current state of the map (height grid, landmarks, trajectory, current
pose), so a 3D viewer pointed at the file always shows the latest world.

    tests/spatial_memory.stl    ← live; opens in macOS Preview, MeshLab, …
    tests/spatial_memory.png    ← top-down snapshot saved at end of run

The hidden ``.spatial_memory.cadenza-scene.xml`` next to those is the
compiled MuJoCo scene file the simulator needs — it's an intermediate
artifact, not the spatial memory output.

    mjpython tests/test_spatial_memory.py
"""

from pathlib import Path

import cadenza
from cadenza.inference import ChainOfThought
from cadenza.spatial import SpatialMemory
from cadenza.robots.go1 import MODEL_XML
from ai_models.go1 import VLA

HERE = Path(__file__).resolve().parent
TARGET = (-6.5, 0.0)

PNG_PATH = HERE / "spatial_memory.png"
SCENE_XML = HERE / ".spatial_memory.cadenza-scene.xml"   # dotfile → MuJoCo intermediate

# ── Build a complex gym ────────────────────────────────────────────────
scene = cadenza.Scene()

# Ground obstacles offset off-axis from the corridor.
scene.add_box(position=(-0.9,  0.35, 0.12), size=(0.15, 0.15, 0.12),
              rgba=(0.75, 0.40, 0.25, 1.0))
scene.add_box(position=(-1.6, -0.30, 0.10), size=(0.18, 0.18, 0.10),
              rgba=(0.75, 0.40, 0.25, 1.0))

# Four-step staircase climbing to z=0.32.
for i, x in enumerate([-2.5, -2.9, -3.3, -3.7]):
    half_h = 0.04 * (i + 1)
    scene.add_box(position=(x, 0.0, half_h),
                  size=(0.20, 0.90, half_h),
                  rgba=(0.55, 0.55, 0.50, 1.0))

# Elevated platform at z=0.32 extending across the corridor.
scene.add_box(position=(-5.5, 0.0, 0.16),
              size=(1.80, 0.90, 0.16),
              rgba=(0.45, 0.45, 0.40, 1.0))

# Crate on top deck that blocks the straight path.
scene.add_box(position=(-5.5, 0.35, 0.50), size=(0.18, 0.18, 0.16),
              rgba=(0.70, 0.30, 0.25, 1.0))

# Tiny green pad at the goal for visual marking.
scene.add_box(position=(-6.5, 0.0, 0.33), size=(0.20, 0.20, 0.005),
              rgba=(0.10, 0.85, 0.20, 1.0))

xml_path = scene.compile(MODEL_XML, SCENE_XML)

# ── Run with SpatialMemory plugged in ──────────────────────────────────
# SpatialMemory writes a live 3D STL automatically. By default it goes to
# ./cadenza_memory.stl (cwd) — opening it in any 3D viewer shows the world
# as the robot maps it.
memory = SpatialMemory(target=TARGET)

go1 = cadenza.go1(
    xml_path=str(xml_path),
    inference=ChainOfThought(
        model=VLA(),
        sense=[memory],
        goal="reach the green pad on the upper deck",
        target=TARGET,
        max_steps=120,
    ),
)

print(f"  tip: open the STL it announces below in any 3D viewer and ")
print(f"       watch it fill in as the robot walks.")
print()

go1.run([go1.walk_forward()], streaming=True)

# ── Post-run: final sync + PNG snapshot + summary ──────────────────────
if memory.stl_path is not None:
    memory.map.to_stl(save_path=memory.stl_path, target=TARGET)   # final sync
memory.map.render(save_path=PNG_PATH, target=TARGET,
                  title="Spatial memory — final state")

s = memory.map.summary()
print()
print("─" * 64)
print("  Spatial memory — final state")
print("─" * 64)
print(f"  cells seen:             {s['n_cells_seen']}")
print(f"  cells flagged occupied: {s['n_cells_occupied']}")
print(f"  landmarks discovered:   {s['n_landmarks']}  ({', '.join(s['landmark_kinds'])})")
print(f"  max elevation logged:   {s['max_height_m']:.2f} m")
print(f"  trajectory samples:     {len(memory.map.trajectory)}")
print()
print(f"  3D map (STL):  {memory.stl_path}")
print(f"  2D snapshot:   {PNG_PATH}")
if memory.stl_path is not None:
    print(f"  open the STL:  open {memory.stl_path}     # macOS Preview")
