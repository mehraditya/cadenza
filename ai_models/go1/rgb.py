"""RGB camera analyzer for the Go1 forward camera.

Numpy-only HSV beacon detector. Looks for vivid-color regions (default:
green) in the forward camera and reports bearing/size to the world model
under ``beacon_visible`` / ``beacon_bearing_px`` / ``beacon_size_frac``.

No external weights — keeps the demo runnable without extra downloads.
Swap with a learned vision model (CLIP, segformer, ...) by subclassing
``cadenza.Modality`` the same way.
"""

from __future__ import annotations

import numpy as np

from cadenza import Modality, ModalityResult


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB float in [0, 1] -> HSV; H in [0, 360), S/V in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 1e-6
    rmax = (cmax == r) & mask
    gmax = (cmax == g) & mask
    bmax = (cmax == b) & mask
    h[rmax] = (60.0 * ((g[rmax] - b[rmax]) / delta[rmax])) % 360.0
    h[gmax] = (60.0 * ((b[gmax] - r[gmax]) / delta[gmax]) + 120.0) % 360.0
    h[bmax] = (60.0 * ((r[bmax] - g[bmax]) / delta[bmax]) + 240.0) % 360.0
    s = np.where(cmax > 1e-6, delta / cmax, 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


class RGB(Modality):
    """HSV-based vivid-color beacon detector for the Go1 forward camera."""

    name = "go1_rgb"
    description = "Numpy HSV beacon detector for Go1 forward camera."

    def __init__(
        self,
        hue: tuple[float, float] = (80.0, 160.0),  # green
        s_min: float = 0.40,
        v_min: float = 0.30,
    ):
        self.hue = hue
        self.s_min = s_min
        self.v_min = v_min

    def compute(self, observation) -> ModalityResult:
        cam = observation.camera
        if cam is None:
            return ModalityResult(keys={}, summary="rgb: no camera")

        rgb = np.asarray(cam, dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        hsv = _rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask = (
            (h >= self.hue[0]) & (h <= self.hue[1])
            & (s >= self.s_min) & (v >= self.v_min)
        )
        if not mask.any():
            return ModalityResult(
                keys={"beacon_visible": False},
                summary="rgb: no beacon",
            )

        ys, xs = np.where(mask)
        cx = float(xs.mean()) / mask.shape[1]
        bearing = (cx - 0.5) * 2.0   # -1 = far left, +1 = far right
        size_frac = float(mask.sum()) / mask.size
        return ModalityResult(
            keys={
                "beacon_visible": True,
                "beacon_bearing_px": bearing,
                "beacon_size_frac": size_frac,
            },
            summary=f"rgb: beacon bearing={bearing:+.2f} size={size_frac:.3f}",
        )
