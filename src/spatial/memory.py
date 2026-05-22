"""SpatialMemory — Modality wrapping ``SpatialMap`` for the stack.

Built on top of the existing modality plug-in interface so it composes with
``DepthAnythingV2Small`` and anything else the client adds:

    sense=[
        SpatialMemory(target=(-5.5, 0.0)),
        DepthAnythingV2Small(),
    ]
"""

from __future__ import annotations

from cadenza.spatial.map import SpatialMap
from cadenza.stack.modalities.base import (
    Modality,
    ModalityResult,
    register_modality,
)


@register_modality
class SpatialMemory(Modality):
    """3D spatial memory built up from multi-modal observations."""

    name = "spatial_memory"
    description = "On-board 2.5D map; picks subgoals that respect 3D geometry."

    #: Default name used when ``stl_path`` isn't overridden. The STL is
    #: written to this filename in the *current working directory* — i.e.
    #: wherever you launched the script from. Set ``stl_path=None`` to
    #: turn the live 3D output off entirely.
    DEFAULT_STL_NAME = "cadenza_memory.stl"

    def __init__(
        self,
        target: tuple[float, float],
        *,
        cell_m: float = 0.20,
        extent_m: float = 12.0,
        elevation_threshold: float = 0.15,
        stl_path: "str | None" = "AUTO",
        stl_every_ticks: int = 5,
    ):
        """
        Args:
            target: the user-supplied **final** target ``(x, y)``. The
                modality may rewrite ``observation['target_xy']`` to a
                subgoal (e.g. the stair base) when the world's geometry
                demands it — but the original target is always preserved
                as ``observation['final_target_xy']``.
            cell_m: grid resolution.
            extent_m: world extent.
            elevation_threshold: heights above this count as "elevated".
            stl_path: where to write the live 3D map.

                * ``"AUTO"`` (default) — writes ``cadenza_memory.stl``
                  in the current working directory. The STL is rewritten
                  atomically every ``stl_every_ticks`` ticks so a 3D
                  viewer (Preview, MeshLab, F3D, Blender) can be left
                  open and watch it fill in.
                * any ``str`` / ``Path`` — write there instead.
                * ``None`` — disable live STL output entirely.

            stl_every_ticks: how often to rewrite the live STL. Default
                5 ⇒ roughly every second on a 1 Hz tick. Lower = more
                up-to-date, higher = cheaper.
        """
        from pathlib import Path as _Path
        self.final_target = (float(target[0]), float(target[1]))
        self.elevation_threshold = float(elevation_threshold)
        self.map = SpatialMap(cell_m=cell_m, extent_m=extent_m)
        # Stream-rate-limit state — only emit a summary when something
        # meaningful changed, not every tick.
        self._prev_n_landmarks = 0
        self._prev_subgoal_reason: str | None = None
        # Live STL output. "AUTO" sentinel → ./cadenza_memory.stl. None disables.
        if stl_path == "AUTO":
            self.stl_path: "_Path | None" = _Path(self.DEFAULT_STL_NAME).resolve()
        elif stl_path is None:
            self.stl_path = None
        else:
            self.stl_path = _Path(stl_path).expanduser().resolve()
        self.stl_every_ticks = max(1, int(stl_every_ticks))
        self._tick_counter = 0
        # Wipe any stale file from a previous run so the user can tell
        # whether the new run actually produced output.
        if self.stl_path is not None and self.stl_path.exists():
            try:
                self.stl_path.unlink()
            except Exception:
                pass

    def setup(self) -> None:
        if self.stl_path is not None:
            print(f"  [SpatialMemory] live 3D map → {self.stl_path}")
            print(f"                  (rewritten every {self.stl_every_ticks} ticks)")
        return

    def teardown(self) -> None:
        return

    def compute(self, observation) -> ModalityResult:
        pos = observation.pos
        rpy = observation.rpy
        body_h = float(observation.body_height)

        # Always record where we are. Trajectory grows tick-by-tick.
        self.map.record_pose(float(pos[0]), float(pos[1]), body_h)

        # Build the map SOLELY from the forward camera's depth + the robot's
        # current pose. If no depth is available (no camera in scene), fall
        # back to pose-only so trajectory still draws.
        depth = getattr(observation, "depth", None)
        if depth is not None:
            self.map.observe_from_depth(
                depth,
                robot_pos=(float(pos[0]), float(pos[1])),
                robot_yaw=float(rpy[2]),
                robot_z=body_h,
            )

        robot_xy = (float(pos[0]), float(pos[1]))
        subgoal, reason = self.map.next_subgoal(
            target_xy=self.final_target,
            robot_xy=robot_xy,
            robot_z=body_h,
            elevation_threshold=self.elevation_threshold,
        )

        # Only surface a streaming summary on *change*. Two trigger cases:
        #   1. The map just acquired a new landmark this tick.
        #   2. The subgoal kind switched (e.g. direct → approach_stairs).
        # Quiet ticks (the common case) return an empty summary so the
        # stream stays clean.
        summary = ""
        n_landmarks = len(self.map.landmarks)
        if n_landmarks > self._prev_n_landmarks:
            new = self.map.landmarks[self._prev_n_landmarks:]
            kinds = ", ".join(
                f"{lm.kind} at ({lm.xy[0]:.1f}, {lm.xy[1]:.1f})" for lm in new
            )
            summary = f"map updated: {kinds}"
        if reason != self._prev_subgoal_reason and self._prev_subgoal_reason is not None:
            switch = (
                f"subgoal switched: {self._prev_subgoal_reason} → {reason} "
                f"({subgoal[0]:.1f}, {subgoal[1]:.1f})"
            )
            summary = f"{summary}  |  {switch}" if summary else switch

        self._prev_n_landmarks = n_landmarks
        self._prev_subgoal_reason = reason

        # Live STL — rewrite atomically so any 3D viewer reading the file
        # never sees a half-written mesh.
        self._tick_counter += 1
        if self.stl_path is not None and self._tick_counter % self.stl_every_ticks == 0:
            try:
                import os
                tmp = self.stl_path.with_suffix(self.stl_path.suffix + ".tmp")
                self.map.to_stl(save_path=tmp, target=self.final_target)
                os.replace(tmp, self.stl_path)
            except Exception:
                # Live writing must never break the run.
                pass

        return ModalityResult(
            keys={
                "target_xy": subgoal,                 # override what runtime set
                "final_target_xy": self.final_target,
                "subgoal_reason": reason,
                "spatial_map_summary": self.map.summary(),
                "landmarks": [(lm.kind, lm.xy, lm.z) for lm in self.map.landmarks],
            },
            summary=summary,
        )
