"""SpatialMap — sparse 2.5D occupancy + landmark map.

Sized in metres (default 12 m × 12 m at 0.20 m / cell ⇒ 60 × 60 cells).
Each cell tracks: ``height`` (max ground elevation seen), ``seen`` flag,
``occupied`` flag (a vertical raycast hit). A growing list of ``Landmark``
records discrete features (``"stairs"``, ``"elevated"``, ...) used by the
``next_subgoal`` reasoning step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmark:
    """One named feature in the world."""
    kind: str                              # "stairs" | "elevated" | "obstacle"
    xy: tuple[float, float]
    z: float = 0.0
    confidence: float = 1.0


_GO1_STAND_BODY_Z = 0.27                   # nominal trunk height when standing


class SpatialMap:
    """Lightweight 2.5D world model fed by the orchestrator tick-by-tick."""

    def __init__(self, cell_m: float = 0.20, extent_m: float = 12.0,
                 point_voxel_m: float = 0.05):
        self.cell_m = float(cell_m)
        self.extent_m = float(extent_m)
        n = int(self.extent_m / self.cell_m)
        self.n = n
        # Grid used ONLY for navigation reasoning (next_subgoal etc.).
        self.height = np.zeros((n, n), dtype=np.float32)
        self.seen = np.zeros((n, n), dtype=bool)
        self.occupied = np.zeros((n, n), dtype=bool)
        self.landmarks: list[Landmark] = []
        # Robot trajectory: list of (x, y, z) tuples accumulated tick-by-tick.
        # The last entry is the current pose.
        self.trajectory: list[tuple[float, float, float]] = []
        # 3D point cloud — the actual "what the camera saw" structure.
        # Stored as a set of voxel indices for deduplication; the resolution
        # determines how dense the LiDAR-style cloud looks.
        self.point_voxel_m = float(point_voxel_m)
        self.points: set[tuple[int, int, int]] = set()

    # ── Indexing ─────────────────────────────────────────────────────────

    def _ix(self, v: float) -> int:
        return max(0, min(self.n - 1, int(round(v / self.cell_m + self.n / 2))))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return self._ix(x), self._ix(y)

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return ((ix - self.n / 2) * self.cell_m,
                (iy - self.n / 2) * self.cell_m)

    # ── Update from one observation ─────────────────────────────────────

    def observe(
        self,
        robot_pos: tuple[float, float],
        robot_yaw: float,
        body_height: float,
        obstacles_ahead: dict | None = None,
        terrain_ahead: dict | None = None,
    ) -> None:
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        ix, iy = self.world_to_cell(rx, ry)
        self.seen[ix, iy] = True

        # Robot's own elevation: trunk-z minus the nominal stand height.
        ground_z = float(body_height) - _GO1_STAND_BODY_Z
        if ground_z > self.height[ix, iy]:
            self.height[ix, iy] = ground_z
        if ground_z > 0.10:                   # we're standing up high → elevated
            self._merge_landmark("elevated", (rx, ry), ground_z)

        # Forward raycasts → occupied cells. Mirrors Sim._probe_obstacles_ahead.
        if obstacles_ahead:
            fwd = np.array([math.cos(robot_yaw + math.pi),
                            math.sin(robot_yaw + math.pi)])
            right = np.array([math.cos(robot_yaw + math.pi - math.pi / 2),
                              math.sin(robot_yaw + math.pi - math.pi / 2)])
            origin = np.array([rx, ry]) + fwd * 0.30
            for label, lat in (("left", -0.15), ("center", 0.0), ("right", 0.15)):
                d = obstacles_ahead.get(f"{label}_m")
                if d is None:
                    continue
                hit = origin + right * lat + fwd * float(d)
                hi, hj = self.world_to_cell(float(hit[0]), float(hit[1]))
                self.occupied[hi, hj] = True
                self.seen[hi, hj] = True

        # Climbable step ahead → record stair landmark + lift cell height.
        if terrain_ahead:
            step = float(terrain_ahead.get("max_step_up") or 0.0)
            if 0.05 < step < 0.20:
                fwd_x = math.cos(robot_yaw + math.pi)
                fwd_y = math.sin(robot_yaw + math.pi)
                sx = rx + fwd_x * 0.35
                sy = ry + fwd_y * 0.35
                self._merge_landmark("stairs", (sx, sy), step)
                si, sj = self.world_to_cell(sx, sy)
                if step > self.height[si, sj]:
                    self.height[si, sj] = step

    def _merge_landmark(self, kind: str, xy: tuple[float, float], z: float,
                        merge_radius: float = 0.30) -> Landmark:
        """Append a landmark unless one of the same kind is already nearby."""
        r2 = merge_radius * merge_radius
        for existing in self.landmarks:
            if existing.kind == kind:
                dx = existing.xy[0] - xy[0]
                dy = existing.xy[1] - xy[1]
                if dx * dx + dy * dy < r2:
                    return existing
        lm = Landmark(kind=kind, xy=(float(xy[0]), float(xy[1])), z=float(z))
        self.landmarks.append(lm)
        return lm

    # ── Visual-only update (camera + pose only — no raycasts) ───────────

    def record_pose(self, x: float, y: float, z: float) -> None:
        """Append the robot's current world-frame pose to the trajectory."""
        self.trajectory.append((float(x), float(y), float(z)))

    def observe_from_depth(
        self,
        depth: np.ndarray,
        *,
        robot_pos: tuple[float, float],
        robot_yaw: float,
        robot_z: float,
        fov_y_deg: float = 70.0,
        camera_forward_offset_m: float = 0.22,
        camera_up_offset_m: float = 0.06,
        max_range_m: float = 6.0,
        stride: int = 4,
    ) -> None:
        """Build the map from one forward-camera depth frame + the robot's pose.

        For each subsampled pixel: compute the world-frame ray from the
        camera, scale by the depth value to get the world point that
        pixel sees, then update the corresponding map cell (mark
        ``seen``, raise ``height``, flag ``occupied`` for tall hits, and
        register a stair landmark in the climbable height band).

        ``depth`` is expected to be the camera's metric depth image
        (MuJoCo's depth buffer gives this directly). Pixels with depth
        beyond ``max_range_m`` are ignored — the robot can't reliably
        place far points.

        Cadenza convention: the robot's front faces -x in the body frame,
        so the world forward direction is ``yaw + π``.
        """
        if depth is None or depth.size == 0:
            return

        H, W = depth.shape[:2]
        s = max(1, int(stride))

        # World-frame camera basis at this tick.
        yaw_w = float(robot_yaw) + math.pi               # world forward heading
        fwd = np.array([math.cos(yaw_w), math.sin(yaw_w), 0.0])
        right = np.array([math.cos(yaw_w - math.pi / 2),
                          math.sin(yaw_w - math.pi / 2), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        cam_xy = np.array([float(robot_pos[0]), float(robot_pos[1]), 0.0]) \
                 + fwd * float(camera_forward_offset_m)
        cam_xy[2] = float(robot_z) + float(camera_up_offset_m)

        # Frustum half-extents at unit distance.
        fov_y = math.radians(fov_y_deg)
        aspect = W / max(H, 1)
        half_y = math.tan(fov_y / 2.0)
        half_x = half_y * aspect

        # Subsample pixel grid for speed (8-stride on 224×224 ⇒ ~28×28 = 784 rays).
        for v in range(0, H, s):
            for u in range(0, W, s):
                d = float(depth[v, u])
                if not math.isfinite(d) or d <= 0.05 or d > max_range_m:
                    continue
                # Normalised image coords in [-1, 1]; flip Y so screen-up = world-up.
                ndx = (u / W) * 2.0 - 1.0
                ndy = 1.0 - (v / H) * 2.0
                ray = fwd + right * (ndx * half_x) + up * (ndy * half_y)
                norm = float(np.linalg.norm(ray))
                if norm < 1e-6:
                    continue
                ray /= norm
                world = cam_xy + ray * d

                wx, wy, wz = float(world[0]), float(world[1]), float(world[2])

                # Point cloud — voxel-hashed dedupe so consecutive ticks
                # don't blow up the size, but every uniquely-seen point
                # persists for the life of the run.
                vx = int(round(wx / self.point_voxel_m))
                vy = int(round(wy / self.point_voxel_m))
                vz = int(round(wz / self.point_voxel_m))
                self.points.add((vx, vy, vz))

                ix, iy = self.world_to_cell(wx, wy)
                self.seen[ix, iy] = True

                # Update cell elevation (max-keep).
                if wz > self.height[ix, iy]:
                    self.height[ix, iy] = wz

                # Anything taller than ~25 cm above the floor at that XY is a
                # solid obstacle for navigation purposes.
                if wz > 0.25:
                    self.occupied[ix, iy] = True

                # Climbable-step band ⇒ stair landmark.
                if 0.05 < wz < 0.20:
                    self._merge_landmark("stairs", (wx, wy), wz)

                # Tall obstacle ⇒ obstacle landmark (sparsified by merge radius).
                elif wz > 0.30:
                    self._merge_landmark("obstacle", (wx, wy), wz)

                # Elevated ground ⇒ elevated landmark.
                elif 0.20 <= wz <= 0.30:
                    self._merge_landmark("elevated", (wx, wy), wz)

    # ── Reasoning ───────────────────────────────────────────────────────

    def target_appears_elevated(
        self,
        target_xy: tuple[float, float],
        elevation_threshold: float = 0.10,
        radius: float = 0.6,
    ) -> bool:
        """True if any known landmark says the target is up high."""
        r2 = radius * radius
        for lm in self.landmarks:
            if lm.kind not in ("elevated", "stairs"):
                continue
            dx = lm.xy[0] - target_xy[0]
            dy = lm.xy[1] - target_xy[1]
            if dx * dx + dy * dy < r2 and lm.z > elevation_threshold:
                return True
        return False

    def stair_on_path(
        self,
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
        max_perp: float = 1.0,
    ) -> "Landmark | None":
        """If a known stair landmark sits on the segment from robot to
        target (within ``max_perp`` lateral distance), return it.

        The whole point of spatial memory: a stair *between* us and the
        target tells us climbing it is on the way, even if its XY is
        slightly off the straight line."""
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
        dx, dy = tx - rx, ty - ry
        L2 = dx * dx + dy * dy
        if L2 < 1e-6:
            return None
        best: Landmark | None = None
        best_t = 1.0
        max_perp2 = max_perp * max_perp
        for lm in self.landmarks:
            if lm.kind != "stairs":
                continue
            sx, sy = lm.xy
            t = ((sx - rx) * dx + (sy - ry) * dy) / L2
            if t < 0.0 or t > 1.0:
                continue
            # Perpendicular distance from segment.
            px = rx + t * dx
            py = ry + t * dy
            d2 = (sx - px) ** 2 + (sy - py) ** 2
            if d2 < max_perp2 and t <= best_t:
                best = lm
                best_t = t
        return best

    def next_subgoal(
        self,
        target_xy: tuple[float, float],
        robot_xy: tuple[float, float],
        robot_z: float,
        elevation_threshold: float = 0.15,
    ) -> tuple[tuple[float, float], str]:
        """Pick the right XY to steer toward right now.

        Returns ``(subgoal_xy, reason)``. Reasons:
          * ``"direct"``               — just go at the target.
          * ``"approach_stairs"``      — a known stair landmark lies
                                          between us and the target, or the
                                          target itself sits in an elevated
                                          region. Subgoal becomes the stair
                                          entry.
          * ``"exploring_for_stairs"`` — target is known to be elevated but
                                          no stair entry has been observed
                                          yet; head toward the target so
                                          the terrain probe maps the stairs.
        """
        on_elevated = robot_z > _GO1_STAND_BODY_Z + elevation_threshold

        # Priority 1: a stair we've already seen sits on our path → route
        # through it. Works even before we know the target is "high".
        if not on_elevated:
            on_path = self.stair_on_path(robot_xy, target_xy)
            if on_path is not None:
                return on_path.xy, "approach_stairs"

        # Priority 2: the map flags the target as elevated → head to the
        # nearest stair if any, else explore.
        target_high = self.target_appears_elevated(target_xy, elevation_threshold)
        if target_high and not on_elevated:
            stairs = [lm for lm in self.landmarks if lm.kind == "stairs"]
            if stairs:
                nearest = min(
                    stairs,
                    key=lambda lm: (lm.xy[0] - robot_xy[0]) ** 2
                                   + (lm.xy[1] - robot_xy[1]) ** 2,
                )
                return nearest.xy, "approach_stairs"
            return target_xy, "exploring_for_stairs"

        return target_xy, "direct"

    def summary(self) -> dict:
        return {
            "n_cells_seen": int(self.seen.sum()),
            "n_cells_occupied": int(self.occupied.sum()),
            "n_landmarks": len(self.landmarks),
            "landmark_kinds": sorted({lm.kind for lm in self.landmarks}),
            "max_height_m": float(self.height.max()),
        }

    # ── Visualization ───────────────────────────────────────────────────

    def render(
        self,
        *,
        save_path: "str | None" = None,
        show: bool = False,
        title: str = "SpatialMap — what the robot mapped",
        target: "tuple[float, float] | None" = None,
        robot_start: tuple[float, float] = (0.0, 0.0),
        crop_to_seen: bool = True,
    ):
        """Render a top-down view of the map. Requires ``matplotlib``.

        Layers (bottom → top):
          1. Height heatmap (viridis). Unseen cells are transparent.
          2. Occupied cells (black squares).
          3. Landmarks (red triangles for stairs, orange dots for
             elevated regions).
          4. Robot start (white star) and target (lime X).

        Args:
            save_path: write the figure to this PNG path; parent dir is
                created if missing. Pass ``None`` to skip saving.
            show: also pop a window. False by default since most test
                runs go through ``mjpython`` and we don't want a
                blocking GUI call.
            target: optional final target marker.
            robot_start: where the robot began.
            crop_to_seen: auto-crop axes to the region the robot actually
                visited (true) vs. show the full map extent (false).

        Returns the matplotlib Figure so callers can further customize.
        """
        try:
            import matplotlib
            if not show:
                matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise RuntimeError(
                "SpatialMap.render() needs matplotlib. "
                "Install with: pip install matplotlib"
            ) from e

        extent_half = self.extent_m / 2
        extent = (-extent_half, extent_half, -extent_half, extent_half)

        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Heightmap — show only where seen.
        height = self.height.astype(float).copy()
        height[~self.seen] = np.nan
        vmax = max(0.4, float(np.nan_to_num(self.height.max())))
        img = ax.imshow(
            height.T, origin="lower", extent=extent,
            cmap="viridis", vmin=0.0, vmax=vmax, alpha=0.85,
        )
        cbar = fig.colorbar(img, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("elevation (m)")

        # 2. Occupied cells.
        occ_ix, occ_iy = np.where(self.occupied)
        if occ_ix.size:
            occ_x = (occ_ix - self.n / 2) * self.cell_m
            occ_y = (occ_iy - self.n / 2) * self.cell_m
            ax.scatter(occ_x, occ_y, s=18, c="black", marker="s",
                       linewidths=0, label=f"occupied ({occ_ix.size})")

        # 3. Landmarks.
        by_kind: dict[str, list[tuple[float, float]]] = {}
        for lm in self.landmarks:
            by_kind.setdefault(lm.kind, []).append(lm.xy)
        kind_style = {
            "stairs":   ("red",     "^", 110),
            "elevated": ("orange",  "o",  70),
            "obstacle": ("dimgray", "x",  70),
        }
        for kind, pts in by_kind.items():
            color, marker, size = kind_style.get(kind, ("magenta", "P", 70))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=size, c=color, marker=marker,
                       edgecolors="black", linewidths=1.0,
                       label=f"{kind} ({len(pts)})")

        # 4. Trajectory + current pose. The robot's path through the map is
        #    the primary "ego" overlay — devs need to see where it's been.
        if self.trajectory:
            tx = [p[0] for p in self.trajectory]
            ty = [p[1] for p in self.trajectory]
            ax.plot(tx, ty, color="white", linewidth=2.2, alpha=0.95,
                    label=f"trajectory ({len(self.trajectory)} samples)")
            # Current pose: highlighted dot at the end of the path.
            ax.plot(tx[-1], ty[-1], marker="o", markersize=10,
                    color="white", markeredgecolor="red",
                    markeredgewidth=2, linestyle="None", label="current pose")

        # 5. Start + target.
        ax.plot(*robot_start, marker="*", markersize=18, color="white",
                markeredgecolor="black", linestyle="None", label="start")
        if target is not None:
            ax.plot(*target, marker="X", markersize=16, color="lime",
                    markeredgecolor="black", linestyle="None", label="target")

        # Crop to where data lives.
        if crop_to_seen and self.seen.any():
            seen_ix, seen_iy = np.where(self.seen)
            x_min = (seen_ix.min() - self.n / 2) * self.cell_m
            x_max = (seen_ix.max() - self.n / 2) * self.cell_m
            y_min = (seen_iy.min() - self.n / 2) * self.cell_m
            y_max = (seen_iy.max() - self.n / 2) * self.cell_m
            pad = 0.6
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_xlabel("x (m)  →  -x is forward")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
        fig.tight_layout()

        if save_path is not None:
            from pathlib import Path
            p = Path(save_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── 3D mesh export ──────────────────────────────────────────────────

    def to_stl(
        self,
        save_path: "str",
        *,
        target: "tuple[float, float] | None" = None,
        robot_start: tuple[float, float] = (0.0, 0.0),
        point_size_m: float = 0.018,
        binary: bool = True,
    ) -> "Path":
        """Export the spatial memory as a 3D point-cloud STL.

        Every voxel the depth camera ever projected into becomes a tiny
        upward-pointing tetrahedron — like a LiDAR point cloud. Together
        these markers reconstruct the world as the robot's visual sensor
        saw it. Crucially, the geometry is **not** a quantized grid of
        cuboids — it's the actual 3D structure perceived from each pose
        along the trajectory, with sparser coverage at distance and
        denser coverage near the camera path, exactly like a real depth
        scan.

        Also includes:
          * the robot's trajectory (small pellets) connecting start →
            current pose, with a tall pillar at the current pose itself.
          * tiny markers at robot start and (if given) target.

        Args:
            save_path: file to write.
            point_size_m: side length of each per-point tetrahedron.
                Smaller = finer-looking cloud, bigger = more visible
                from far out.
            binary: write binary STL (default; 4× smaller, faster).
                Set ``False`` for ASCII STL if you want to inspect it.

        Open in macOS Preview, MeshLab, Blender, F3D, etc.
        """
        from pathlib import Path

        triangles: list[tuple[tuple[float, float, float],
                              tuple[float, float, float],
                              tuple[float, float, float]]] = []

        # 1. Point cloud — every unique voxel the depth camera touched.
        s = float(point_size_m) / 2.0
        for vx, vy, vz in self.points:
            wx = vx * self.point_voxel_m
            wy = vy * self.point_voxel_m
            wz = vz * self.point_voxel_m
            triangles.extend(self._point_tetra(wx, wy, wz, s))

        # 2. Trajectory pellets — sub-sampled to ≤80 markers.
        traj = self.trajectory
        if traj:
            step = max(1, len(traj) // 80)
            for x, y, z in traj[::step]:
                base = max(z - 0.27, 0.0)
                triangles.extend(self._box_tris(
                    (x - 0.02, y - 0.02, base),
                    (x + 0.02, y + 0.02, base + 0.05),
                ))
            # Current pose marker: a tall skinny pillar + capping pyramid.
            cx, cy, cz = traj[-1]
            base = max(cz - 0.27, 0.0)
            triangles.extend(self._box_tris(
                (cx - 0.035, cy - 0.035, base),
                (cx + 0.035, cy + 0.035, base + 0.28),
            ))
            triangles.extend(self._pyramid_tris(
                base_xy=(cx, cy), base_z=base + 0.28,
                height=0.10, half_size=0.05,
            ))

        # 3. Start + target.
        triangles.extend(self._box_tris(
            (robot_start[0] - 0.04, robot_start[1] - 0.04, 0.0),
            (robot_start[0] + 0.04, robot_start[1] + 0.04, 0.08),
        ))
        if target is not None:
            triangles.extend(self._box_tris(
                (target[0] - 0.05, target[1] - 0.05, 0.0),
                (target[0] + 0.05, target[1] + 0.05, 0.14),
            ))

        out = Path(save_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            self._write_binary_stl(out, triangles)
        else:
            self._write_ascii_stl(out, triangles)
        return out

    @staticmethod
    def _point_tetra(x: float, y: float, z: float, s: float):
        """Tiny upward-pointing tetrahedron centred (in XY) at (x, y, z).

        4 triangles total — minimum needed for a solid that's visible
        from any 3D camera angle.
        """
        a = (x - s, y - s, z)
        b = (x + s, y - s, z)
        c = (x,     y + s, z)
        tip = (x, y, z + s * 1.6)
        return [(a, b, c), (a, tip, b), (b, tip, c), (c, tip, a)]

    @staticmethod
    def _write_ascii_stl(path: "Path", triangles) -> None:
        # Build the whole string at once — 4× faster than per-line write
        # when there are tens of thousands of triangles.
        parts = ["solid cadenza_spatial_map\n"]
        for v1, v2, v3 in triangles:
            parts.append("  facet normal 0 0 0\n    outer loop\n")
            for v in (v1, v2, v3):
                parts.append(f"      vertex {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            parts.append("    endloop\n  endfacet\n")
        parts.append("endsolid cadenza_spatial_map\n")
        path.write_text("".join(parts))

    @staticmethod
    def _write_binary_stl(path: "Path", triangles) -> None:
        import struct
        n = len(triangles)
        # Pack everything as one bytes blob, then dump in one write.
        # 80-byte header + 4-byte count + per-tri (12 floats + 2 bytes)
        buf = bytearray(80 + 4 + n * 50)
        struct.pack_into("<80sI", buf, 0, b"cadenza_spatial_map", n)
        off = 84
        for v1, v2, v3 in triangles:
            struct.pack_into(
                "<3f3f3f3fH", buf, off,
                0.0, 0.0, 0.0,                 # normal (unused by most viewers)
                v1[0], v1[1], v1[2],
                v2[0], v2[1], v2[2],
                v3[0], v3[1], v3[2],
                0,                              # attribute byte count
            )
            off += 50
        path.write_bytes(bytes(buf))

    # ── STL geometry helpers ────────────────────────────────────────────

    @staticmethod
    def _box_tris(p0: tuple[float, float, float],
                  p1: tuple[float, float, float]):
        """12 triangles describing an axis-aligned box from corner p0 to p1."""
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        c = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        # Outward-facing winding for each face.
        faces = [
            (0, 2, 1), (0, 3, 2),       # bottom (-z)
            (4, 5, 6), (4, 6, 7),       # top (+z)
            (0, 1, 5), (0, 5, 4),       # front (-y)
            (1, 2, 6), (1, 6, 5),       # right (+x)
            (2, 3, 7), (2, 7, 6),       # back (+y)
            (3, 0, 4), (3, 4, 7),       # left (-x)
        ]
        return [(c[a], c[b], c[d]) for a, b, d in faces]

    @staticmethod
    def _pyramid_tris(*, base_xy: tuple[float, float], base_z: float,
                     height: float, half_size: float):
        """4 side triangles + a square base for a marker pyramid."""
        x, y = base_xy
        tip = (x, y, base_z + height)
        b = [
            (x - half_size, y - half_size, base_z),
            (x + half_size, y - half_size, base_z),
            (x + half_size, y + half_size, base_z),
            (x - half_size, y + half_size, base_z),
        ]
        tris = []
        for i in range(4):
            tris.append((b[i], b[(i + 1) % 4], tip))
        tris.append((b[0], b[2], b[1]))
        tris.append((b[0], b[3], b[2]))
        return tris
