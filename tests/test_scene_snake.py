"""Slope + snake demo. Edit constants below and re-run.

    mjpython tests/test_scene_snake.py
"""
import cadenza

ANGLE_DEG = 0.375
SLOPE_NEAR_X = -4.5

slope = cadenza.Slope.from_ground(
    near_x=SLOPE_NEAR_X, hx=1.0, hy=1.2, angle_deg=ANGLE_DEG,
)

scene = (
    cadenza.Scene()
    .add(slope)
    .add_box(position=(-3.0, -1.2, 0.025), size=(2.0, 0.6, 0.025),
             rgba=(0.20, 0.55, 0.70, 1.0))   # long, low, wide-top platform
    .snake(start_x=-1.0, step_x=-0.64, count=6, snake_y=0.40)
    .snake_on_slope(slope, count=8, snake_y=0.40)
)

cadenza.view(robot="go1", scene=scene)
