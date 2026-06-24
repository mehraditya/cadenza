"""cadenza.sensor_view — live onboard-camera feed.

Renders a robot's onboard visual sensor (a named ``<camera>`` in the MuJoCo
model) every frame and serves it as a live MJPEG video at ``http://127.0.0.1:<port>``
*while the sim runs* — so you can watch what the robot sees in real time, in a
browser window, not a screenshot or video saved after the fact.

Each robot has one onboard sensor wired into its model:

  * Go1 — ``head_cam``  : in the head, facing forward (-x).
  * G1  — ``head_cam``  : in the head, facing forward (-x).
  * Arm — ``grip_cam``  : eye-in-hand on the palm, facing the grasp axis (+z).

Why a browser feed and not a native window? On macOS the MuJoCo viewer must run
under ``mjpython`` and owns the main-thread Cocoa event loop. A second native
GUI window (OpenCV/Tk/Qt) cannot share that loop, so it throws. A local HTTP
stream lives in a *separate* process (your browser), so it coexists with the
viewer cleanly and works the same on macOS, Linux, and Windows. If the stream
server cannot start, it transparently falls back to writing the latest frame to
``/tmp/cadenza_<robot>_<camera>.png`` so a sim run never crashes over the feed.

Typical use (the controllers do this for you)::

    view = make_view("go1")
    ...                          # inside the stepping loop, per frame:
    view.maybe_update(model, data)
    ...
    view.close()

``maybe_update`` is throttled to ``fps`` by wall-clock, so it is cheap to call
from every ``viewer.sync()`` site.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
import time
import webbrowser

import numpy as np

# Default onboard sensor for each robot.
_DEFAULT_CAMERA = {"go1": "head_cam", "go2": "head_cam", "g1": "head_cam",
                   "arm": "grip_cam"}

# Local port range to search for the stream server.
_PORT_BASE = 8900
_PORT_TRIES = 40


class _StreamHandler(http.server.BaseHTTPRequestHandler):
    """Serves a landing page at ``/`` and an MJPEG feed at ``/stream``."""

    def log_message(self, *args):          # silence default request logging
        pass

    def do_GET(self):
        view: "SensorView" = self.server.view          # type: ignore[attr-defined]
        if self.path.startswith("/stream"):
            self._serve_stream(view)
        else:
            self._serve_page(view)

    def _serve_page(self, view: "SensorView"):
        html = (
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{view.robot} · {view.camera}</title>"
            f"<style>body{{margin:0;background:#111;color:#7f8;"
            f"font-family:monospace;text-align:center}}"
            f"img{{max-width:100vw;image-rendering:pixelated}}"
            f"h1{{font-size:14px;padding:8px;margin:0}}</style></head>"
            f"<body><h1>cadenza · {view.robot} · {view.camera} "
            f"(onboard sensor, live)</h1>"
            f"<img src='/stream'></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        try:
            self.wfile.write(html)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_stream(self, view: "SensorView"):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while not view._closed:
                jpg = view._latest_jpeg()
                if jpg is None:
                    time.sleep(0.03)
                    continue
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(view._min_dt)
        except (BrokenPipeError, ConnectionResetError):
            return


class _StreamServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class SensorView:
    """A live browser feed of one onboard camera of a running MuJoCo sim."""

    def __init__(self, robot: str, camera: str = "head_cam", *,
                 width: int = 480, height: int = 360, fps: float = 20.0,
                 enabled: bool = True, open_browser: bool = True):
        self.robot = robot
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.enabled = bool(enabled)
        self._open_browser = bool(open_browser)
        self._min_dt = 1.0 / max(1e-3, float(fps))

        self._renderer = None
        self._last = 0.0
        self._warned = False
        self._closed = False

        # Stream server state.
        self._server: _StreamServer | None = None
        self._url: str | None = None
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._mode: str | None = None       # None | "stream" | "png"

    # ── rendering ─────────────────────────────────────────────────────────────

    def _ensure_renderer(self, model) -> bool:
        if self._renderer is not None:
            return True
        import mujoco
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if cid < 0:
            self._disable(f"model has no camera {self.camera!r}")
            return False
        try:
            self._renderer = mujoco.Renderer(model, self.height, self.width)
        except Exception as exc:
            self._disable(f"could not create renderer ({exc})")
            return False
        return True

    def maybe_update(self, model, data, *, force: bool = False) -> None:
        """Render the sensor and publish the frame — throttled to ``fps``."""
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last) < self._min_dt:
            return
        self._last = now

        if not self._ensure_renderer(model):
            return
        try:
            self._renderer.update_scene(data, camera=self.camera)
            frame = self._renderer.render()             # RGB uint8 (H, W, 3)
        except Exception as exc:
            self._disable(f"render failed ({exc})")
            return
        self._publish(frame)

    # ── publishing ────────────────────────────────────────────────────────────

    def _publish(self, frame: np.ndarray) -> None:
        if self._mode is None:
            self._start_stream()                        # first frame: open feed
        if self._mode == "stream":
            jpg = self._encode(frame)
            if jpg is not None:
                with self._lock:
                    self._jpeg = jpg
                return
        self._write_png(frame)                          # fallback

    def _encode(self, frame: np.ndarray) -> bytes | None:
        """JPEG-encode an RGB frame (label drawn on top). Uses OpenCV's codec —
        no GUI, so it is safe under mjpython."""
        try:
            import cv2
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            cv2.putText(bgr, f"{self.robot}:{self.camera}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 90), 1,
                        cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes() if ok else None
        except Exception:
            try:
                import io
                from PIL import Image
                bio = io.BytesIO()
                Image.fromarray(frame).save(bio, format="JPEG", quality=80)
                return bio.getvalue()
            except Exception:
                return None

    def _latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _write_png(self, frame: np.ndarray) -> None:
        try:
            from PIL import Image
            Image.fromarray(frame).save(
                f"/tmp/cadenza_{self.robot}_{self.camera}.png")
        except Exception:
            pass

    # ── stream server ─────────────────────────────────────────────────────────

    def _start_stream(self) -> None:
        for port in range(_PORT_BASE, _PORT_BASE + _PORT_TRIES):
            try:
                srv = _StreamServer(("127.0.0.1", port), _StreamHandler)
            except OSError:
                continue                                # port busy, try next
            srv.view = self                             # type: ignore[attr-defined]
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            self._server = srv
            self._url = f"http://127.0.0.1:{port}"
            self._mode = "stream"
            print(f"  [sensor-view] {self.robot}:{self.camera} live → "
                  f"{self._url}")
            if self._open_browser:
                try:
                    webbrowser.open(self._url)
                except Exception:
                    pass
            return
        # Could not bind any port → fall back to PNG frames.
        if not self._warned:
            print(f"  [sensor-view] no free port for live feed; writing "
                  f"frames to /tmp/cadenza_{self.robot}_{self.camera}.png")
            self._warned = True
        self._mode = "png"

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _disable(self, msg: str) -> None:
        if not self._warned:
            print(f"  [sensor-view] {self.robot}:{self.camera} disabled — {msg}")
            self._warned = True
        self.enabled = False

    def close(self) -> None:
        self._closed = True
        try:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
        except Exception:
            pass
        self._server = None
        try:
            if self._renderer is not None:
                self._renderer.close()
        except Exception:
            pass
        self._renderer = None


def make_view(robot: str, *, enabled: bool = True,
              camera: str | None = None, **kw) -> SensorView:
    """Build the onboard ``SensorView`` for ``robot`` (picks its default camera)."""
    return SensorView(robot, camera or _DEFAULT_CAMERA.get(robot, "head_cam"),
                      enabled=enabled, **kw)
