"""Scene builder — programmatic obstacle configuration for the gym.

Devs construct a Scene, add boxes / spheres / slopes (each with position,
size, and a `fixed` flag), then pass it into `Sim` or `GymAdapter`. At sim
build time, the scene is compiled by injecting the requested geoms into
the robot's worldbody.

Usage::

    import cadenza
    scene = (
        cadenza.Scene()
        .add_box(position=(2.0, 0, 0.1), size=(0.2, 0.2, 0.1))
        .add_sphere(position=(3.0, 0, 0.5), radius=0.1, fixed=False)
        .add_slope(position=(5.0, 0, 0.0), size=(1.0, 0.5, 0.05), angle_deg=15)
    )

    gym = cadenza.GymAdapter(robot="go1", scene=scene)
    gym.reset()

`fixed=True` (the default) welds the object to the world. `fixed=False`
wraps it in a body with a free joint so it falls under gravity and can be
shoved by the robot.

Object types are limited by design: rectangular prism (box), sphere, and
tilted-box slope.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


Vec3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]


def _v3(v: Sequence[float]) -> Vec3:
    if len(v) != 3:
        raise ValueError(f"expected 3 floats, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _rgba(v: Sequence[float] | None, default: Rgba) -> Rgba:
    if v is None:
        return default
    if len(v) != 4:
        raise ValueError(f"rgba must be 4 floats, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangular prism. `size` is half-extents (MuJoCo convention)."""
    position: Vec3
    size: Vec3
    fixed: bool = True
    rgba: Rgba = (0.78, 0.32, 0.22, 1.0)


@dataclass(frozen=True)
class Sphere:
    position: Vec3
    radius: float
    fixed: bool = True
    rgba: Rgba = (0.22, 0.55, 0.82, 1.0)


@dataclass(frozen=True)
class Slope:
    """Tilted rectangular prism. `angle_deg` is the tilt around `axis`."""
    position: Vec3
    size: Vec3
    angle_deg: float
    axis: Vec3 = (0.0, 1.0, 0.0)
    fixed: bool = True
    rgba: Rgba = (0.55, 0.55, 0.58, 1.0)

    @classmethod
    def from_ground(cls, *, near_x: float, near_y: float = 0.0,
                    hx: float, hy: float, hz: float = 0.05,
                    angle_deg: float, axis: Sequence[float] = (0.0, 1.0, 0.0),
                    fixed: bool = True, rgba: Sequence[float] | None = None) -> "Slope":
        """Build a slope whose +x top edge meets the ground at (near_x, near_y, 0).

        Lets callers say "I want a 10° ramp starting at x=-4.5" without
        having to compute the slope's center-z themselves.
        """
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        cx = near_x - hx * c - hz * s
        cz = hx * s - hz * c
        return cls(
            position=(cx, float(near_y), cz),
            size=(float(hx), float(hy), float(hz)),
            angle_deg=float(angle_deg),
            axis=_v3(axis),
            fixed=bool(fixed),
            rgba=_rgba(rgba, cls.__dataclass_fields__["rgba"].default),
        )

    def surface_point(self, local_x: float, local_y: float) -> Vec3:
        """World-space point on the top face for slope-local coords (lx, ly).

        Currently supports the default y-axis tilt only.
        """
        if tuple(self.axis) != (0.0, 1.0, 0.0):
            raise NotImplementedError("surface_point requires axis=(0,1,0)")
        theta = math.radians(self.angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        cx, cy, cz = self.position
        _, _, hz = self.size
        return (
            cx + local_x * c + hz * s,
            cy + local_y,
            cz + hz * c - local_x * s,
        )


class Scene:
    """Mutable container of scene objects, compiled into MuJoCo XML on demand."""

    def __init__(self) -> None:
        self.objects: list[Box | Sphere | Slope] = []

    # ── Builder API ─────────────────────────────────────────────────────────

    def add(self, obj: "Box | Sphere | Slope") -> "Scene":
        """Append a pre-built object (e.g. one returned by `Slope.from_ground`)."""
        if not isinstance(obj, (Box, Sphere, Slope)):
            raise TypeError(f"expected Box/Sphere/Slope, got {type(obj).__name__}")
        self.objects.append(obj)
        return self

    def add_box(self, position: Sequence[float], size: Sequence[float],
                *, fixed: bool = True, rgba: Sequence[float] | None = None) -> "Scene":
        self.objects.append(Box(
            position=_v3(position), size=_v3(size), fixed=bool(fixed),
            rgba=_rgba(rgba, Box.__dataclass_fields__["rgba"].default),
        ))
        return self

    def add_sphere(self, position: Sequence[float], radius: float,
                   *, fixed: bool = True, rgba: Sequence[float] | None = None) -> "Scene":
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.objects.append(Sphere(
            position=_v3(position), radius=float(radius), fixed=bool(fixed),
            rgba=_rgba(rgba, Sphere.__dataclass_fields__["rgba"].default),
        ))
        return self

    def add_slope(self, position: Sequence[float], size: Sequence[float],
                  angle_deg: float, *, axis: Sequence[float] = (0.0, 1.0, 0.0),
                  fixed: bool = True, rgba: Sequence[float] | None = None) -> "Scene":
        self.objects.append(Slope(
            position=_v3(position), size=_v3(size), angle_deg=float(angle_deg),
            axis=_v3(axis), fixed=bool(fixed),
            rgba=_rgba(rgba, Slope.__dataclass_fields__["rgba"].default),
        ))
        return self

    def clear(self) -> "Scene":
        self.objects.clear()
        return self

    # ── Higher-level placement helpers ──────────────────────────────────────

    def snake(self, *, start_x: float, step_x: float, count: int,
              snake_y: float = 0.4, z: float | None = None,
              box_size: Sequence[float] = (0.07, 0.07, 0.07),
              rgba: Sequence[float] = (0.85, 0.30, 0.20, 1.0)) -> "Scene":
        """Lay a row of fixed boxes alternating ±snake_y at x = start + i*step."""
        size = _v3(box_size)
        z0 = float(z) if z is not None else size[2]
        for i in range(count):
            x = start_x + i * step_x
            y = snake_y if i % 2 == 0 else -snake_y
            self.add_box(position=(x, y, z0), size=size, fixed=True, rgba=rgba)
        return self

    def snake_on_slope(self, slope: "Slope", *, count: int,
                       snake_y: float = 0.4, margin: float = 0.08,
                       box_size: Sequence[float] = (0.06, 0.06, 0.06),
                       rgba: Sequence[float] = (0.85, 0.30, 0.20, 1.0)) -> "Scene":
        """Place `count` boxes along the slope's top face, climbing near→far.

        Uses `slope.surface_point` so the boxes sit flush on the tilted face
        regardless of `slope.angle_deg`.
        """
        if not isinstance(slope, Slope):
            raise TypeError(f"expected Slope, got {type(slope).__name__}")
        if count < 1:
            return self
        size = _v3(box_size)
        hx = slope.size[0]
        theta = math.radians(slope.angle_deg)
        lift = size[2] / math.cos(theta) + 0.005
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.0
            lx = (hx - margin) - t * (2 * (hx - margin))
            ly = snake_y if i % 2 == 0 else -snake_y
            sx, sy, sz = slope.surface_point(lx, ly)
            self.add_box(position=(sx, sy, sz + lift), size=size,
                         fixed=True, rgba=rgba)
        return self

    def __len__(self) -> int:
        return len(self.objects)

    # ── Compilation ─────────────────────────────────────────────────────────

    def compile(self, base_xml: Path,
                out_path: "Path | str | None" = None) -> Path:
        """Inject objects into ``base_xml`` and return path to the compiled XML.

        Args:
            base_xml: source MuJoCo scene file to extend.
            out_path: optional destination for the compiled XML.

                * ``None`` (default): write next to ``base_xml`` —
                  preserves backward compatibility with the bundled scenes;
                  mesh / texture references stay relative.
                * ``"some/dir/"`` (directory): a fresh
                  ``_cadenza_scene_<id>.xml`` is dropped inside; the dir is
                  created if missing.
                * ``"some/path/scene.xml"`` (file): written to that exact
                  path; ``.parent`` is created if missing.

                When ``out_path`` is given, the compiled XML's
                ``<compiler meshdir=… texturedir=…>`` are rewritten to the
                absolute base-scene directory (and any ``<include file=…>``
                resolved to absolute) so MuJoCo still finds every asset no
                matter where the file is saved.

        Returns:
            Absolute path to the compiled XML.
        """
        base_xml = Path(base_xml).resolve()
        if not self.objects:
            return base_xml

        tree = ET.parse(base_xml)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")

        for i, obj in enumerate(self.objects):
            self._inject(worldbody, obj, i)

        if out_path is None:
            # Legacy behaviour: write next to the base XML so relative
            # asset paths still resolve.
            out = base_xml.parent / f"_cadenza_scene_{id(self):x}.xml"
        else:
            out = Path(out_path).expanduser()
            if out.suffix.lower() != ".xml":
                # Treat as a directory.
                out = out / f"_cadenza_scene_{id(self):x}.xml"
            out = out.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            self._absolutize_asset_paths(root, base_xml.parent)

        tree.write(out, encoding="utf-8", xml_declaration=False)
        return out

    # ── Asset-path rewriting (for custom out_path) ──────────────────────────

    @staticmethod
    def _absolutize_asset_paths(root: ET.Element, base_dir: Path) -> None:
        """Rewrite asset references so the XML still loads from any location.

        MuJoCo resolves ``<mesh file=…>`` / ``<texture file=…>`` against the
        ``<compiler>`` element's ``meshdir`` / ``texturedir`` (each
        defaulting to the XML's own directory). Moving the XML breaks
        those lookups; we fix both compiler dirs to absolute paths derived
        from the base scene's directory, and rewrite any ``<include>``
        directives to absolute paths too.
        """
        base_dir = base_dir.resolve()

        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)

        mesh_rel = compiler.attrib.get("meshdir")
        compiler.attrib["meshdir"] = str(
            (base_dir / mesh_rel).resolve() if mesh_rel else base_dir
        )
        tex_rel = compiler.attrib.get("texturedir")
        compiler.attrib["texturedir"] = str(
            (base_dir / tex_rel).resolve() if tex_rel else base_dir
        )
        asset_rel = compiler.attrib.get("assetdir")
        if asset_rel:
            compiler.attrib["assetdir"] = str((base_dir / asset_rel).resolve())

        for include in root.iter("include"):
            f = include.attrib.get("file")
            if f and not Path(f).is_absolute():
                include.attrib["file"] = str((base_dir / f).resolve())

    # ── Internal injection ──────────────────────────────────────────────────

    @staticmethod
    def _fmt(*nums: float) -> str:
        return " ".join(f"{n:g}" for n in nums)

    def _inject(self, worldbody: ET.Element, obj, idx: int) -> None:
        name = f"cadenza_obs_{idx}"
        if isinstance(obj, Box):
            self._emit(worldbody, name, "box",
                       pos=obj.position, size=obj.size, rgba=obj.rgba,
                       fixed=obj.fixed)
        elif isinstance(obj, Sphere):
            self._emit(worldbody, name, "sphere",
                       pos=obj.position, size=(obj.radius,), rgba=obj.rgba,
                       fixed=obj.fixed)
        elif isinstance(obj, Slope):
            # The bundled scene compilers run in radian mode, so convert
            # before writing axisangle — keeps the Python-side `angle_deg`
            # field truly in degrees.
            axisangle = self._fmt(*obj.axis, math.radians(obj.angle_deg))
            self._emit(worldbody, name, "box",
                       pos=obj.position, size=obj.size, rgba=obj.rgba,
                       fixed=obj.fixed, axisangle=axisangle)
        else:
            raise TypeError(f"unsupported scene object: {type(obj).__name__}")

    def _emit(self, worldbody: ET.Element, name: str, gtype: str,
              *, pos: Sequence[float], size: Sequence[float], rgba: Rgba,
              fixed: bool, axisangle: str | None = None) -> None:
        geom_attrs = {
            "name": name,
            "type": gtype,
            "size": self._fmt(*size),
            "rgba": self._fmt(*rgba),
        }
        if fixed:
            geom_attrs["pos"] = self._fmt(*pos)
            if axisangle is not None:
                geom_attrs["axisangle"] = axisangle
            ET.SubElement(worldbody, "geom", geom_attrs)
        else:
            body_attrs = {"name": f"{name}_body", "pos": self._fmt(*pos)}
            if axisangle is not None:
                body_attrs["axisangle"] = axisangle
            body = ET.SubElement(worldbody, "body", body_attrs)
            ET.SubElement(body, "freejoint", {"name": f"{name}_joint"})
            ET.SubElement(body, "geom", geom_attrs)
