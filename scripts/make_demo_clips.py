"""Render short GIF clips of each robot for the README, using cadenza-lab.

We monkeypatch ``mujoco.viewer.launch_passive`` with a frame-capturing fake
viewer, so each controller's real ``run([...])`` path executes exactly as it
would on screen — but renders offscreen into frames we save as a GIF. No display
needed.

    python scripts/make_demo_clips.py [arm|go1|g1|all]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "assets"
OUT.mkdir(exist_ok=True)


class _Cam:
    """Stand-in for viewer.cam — the controllers set these and slice lookat."""
    def __init__(self):
        self.distance = 3.0
        self.elevation = -20.0
        self.azimuth = 90.0
        self.lookat = np.zeros(3)


class FrameViewer:
    """Quacks like a passive viewer but renders to a list of frames."""
    def __init__(self, model, data, frames, *, every, max_frames, w=420, h=300):
        self.model, self.data = model, data
        self.cam = _Cam()
        self._frames = frames
        self._every, self._max = every, max_frames
        self._n = 0
        self._r = mujoco.Renderer(model, h, w)
        self._mjcam = mujoco.MjvCamera()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._r.close()
        return False

    def is_running(self):
        # Stays "running" until we've captured enough frames; then the
        # controller's action loop and trailing idle loop both exit cleanly.
        return len(self._frames) < self._max

    def sync(self):
        self._n += 1
        if self._n % self._every:
            return
        mc, c = self._mjcam, self.cam
        mc.distance, mc.elevation, mc.azimuth = c.distance, c.elevation, c.azimuth
        mc.lookat[:] = c.lookat
        self._r.update_scene(self.data, camera=mc)
        self._frames.append(self._r.render().copy())


def _capture(build_and_run, *, every, max_frames, fps, name):
    frames: list = []
    orig = mujoco.viewer.launch_passive
    mujoco.viewer.launch_passive = lambda m, d, **k: FrameViewer(
        m, d, frames, every=every, max_frames=max_frames)
    try:
        build_and_run()
    finally:
        mujoco.viewer.launch_passive = orig
    if not frames:
        raise RuntimeError(f"{name}: no frames captured")
    path = OUT / f"{name}.gif"
    # Quantize to a shared adaptive palette (steadier colors, much smaller file).
    imgs = [Image.fromarray(f) for f in frames]
    pal = imgs[0].quantize(colors=96, method=Image.MEDIANCUT)
    imgs = [im.quantize(colors=96, palette=pal, dither=Image.NONE) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    print(f"  {name}: {len(frames)} frames -> {path}  "
          f"({path.stat().st_size/1024:.0f} KB)")


def clip_arm():
    import cadenza_lab as cadenza
    arm = cadenza.arm()
    _capture(lambda: arm.run([
        arm.home(),
        arm.pick((0.50, 0.00, 0.43)),
        arm.place((0.40, 0.22, 0.43)),
        arm.home(),
    ], realtime=False, verbose=False),
        every=70, max_frames=80, fps=18, name="arm")


def clip_go1():
    import cadenza_lab as cadenza
    go1 = cadenza.go1()
    _capture(lambda: go1.run([
        go1.stand(),
        go1.walk_forward(speed=1.5, distance_m=2.0),
        [go1.turn_left(), go1.walk_forward()],
        go1.jump(speed=1.8, extension=1.2),
        go1.sit(),
    ], verbose=False),
        every=24, max_frames=90, fps=20, name="go1")


def clip_g1():
    import cadenza_lab as cadenza
    g1 = cadenza.g1()
    _capture(lambda: g1.run([
        g1.stand(),
        g1.walk_forward(distance_m=1.5),
        g1.crouch(),
        g1.stand(),
    ], verbose=False),
        every=24, max_frames=90, fps=20, name="g1")


CLIPS = {"arm": clip_arm, "go1": clip_go1, "g1": clip_g1}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = CLIPS if which == "all" else {which: CLIPS[which]}
    for nm, fn in targets.items():
        try:
            fn()
        except Exception as e:
            print(f"  {nm}: FAILED — {type(e).__name__}: {e}")
