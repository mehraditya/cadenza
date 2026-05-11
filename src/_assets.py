"""Lazy model-asset fetcher.

Robot meshes/XMLs are too heavy (~113MB for G1) to ship in the PyPI wheel.
Instead we fetch them on first use from a rolling GitHub Release and cache
them under the user's home dir.

Layout on disk after first use::

    ~/.cache/cadenza/models/
        go1/scene.xml
        g1/scene.xml
        g1/<meshes>.STL
        ...

Override the cache dir with ``CADENZA_CACHE_DIR``. Force a re-download with
``CADENZA_REFETCH_MODELS=1``.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_REPO = "aparekh02/cadenza"
_RELEASE_TAG = "models-latest"
_RELEASE_URL = f"https://github.com/{_REPO}/releases/download/{_RELEASE_TAG}"


def _cache_root() -> Path:
    override = os.environ.get("CADENZA_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "cadenza" / "models"


def ensure_robot_assets(robot: str) -> Path:
    """Return the local directory holding <robot>'s MuJoCo assets.

    Downloads and extracts <robot>.tar.gz from the rolling release on first
    call. Subsequent calls hit the cache.
    """
    root = _cache_root()
    target = root / robot
    refetch = os.environ.get("CADENZA_REFETCH_MODELS") == "1"

    if target.exists() and not refetch:
        return target

    root.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_URL}/{robot}.tar.gz"

    if refetch and target.exists():
        shutil.rmtree(target)

    print(
        f"cadenza: fetching {robot} model assets from {url} ...",
        file=sys.stderr,
        flush=True,
    )
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(root)
    except Exception as e:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(
            f"Failed to download {robot} model assets from {url}. "
            f"Set CADENZA_CACHE_DIR to a prepopulated dir, or pass xml_path= "
            f"to Sim()/GymAdapter(). Cause: {e}"
        ) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    if not target.exists():
        raise RuntimeError(
            f"Tarball extracted but expected {target} not found. "
            f"The {robot}.tar.gz must contain a top-level '{robot}/' directory."
        )
    return target
