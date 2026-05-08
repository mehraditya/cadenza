"""Depth-Anything-V2-Small modality for the Go1 forward camera.

Wraps depth-anything/Depth-Anything-V2-Small-hf as a cadenza Modality.
Loaded lazily on the first tick.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from cadenza import Modality, ModalityResult


_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class Depth(Modality):
    """Monocular depth from depth-anything/Depth-Anything-V2-Small-hf."""

    name = "go1_depth"
    description = "Depth-Anything-V2-Small monocular depth for Go1."
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device: str | None = None):
        self.device = device or _DEVICE
        self._processor = None
        self._model = None

    def setup(self) -> None:
        if self._model is not None:
            return
        print(f"  loading {self.MODEL_ID} on {self.device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID)
        self._model.to(self.device).eval()

    def compute(self, observation) -> ModalityResult:
        if observation.camera is None:
            return ModalityResult(keys={}, summary="depth: no camera")
        if self._model is None:
            self.setup()

        depth_map = self._predict(observation.camera)
        h, w = depth_map.shape[:2]
        ymin, ymax = int(h * 0.45), int(h * 0.85)
        third = w // 3
        d_left = float(depth_map[ymin:ymax, :third].mean())
        d_center = float(depth_map[ymin:ymax, third:2 * third].mean())
        d_right = float(depth_map[ymin:ymax, 2 * third:].mean())

        return ModalityResult(
            keys={
                "depth_map": depth_map,
                "depth_left": d_left,
                "depth_center": d_center,
                "depth_right": d_right,
                "depth_min": float(depth_map.min()),
                "depth_max": float(depth_map.max()),
            },
            summary=f"depth: L={d_left:.2f} C={d_center:.2f} R={d_right:.2f}",
        )

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255 if rgb.max() <= 1.0 else rgb).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.predicted_depth.squeeze(0).cpu().numpy().astype(np.float32)
