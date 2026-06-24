"""Internal helpers for opening Zarr groups across local paths and cloud URIs.

Centralises the URI-detection pattern used by both ``writers/bf2raw.py`` and
``audit.py`` so cloud paths (``gs://``, ``s3://``) are handled the same way as
local paths.
"""

from pathlib import Path

import fsspec
import zarr
import zarr.storage


def _is_remote_uri(s: str) -> bool:
    return "://" in s and not s.startswith("file://")


def open_root_group(store_path: str | Path, mode: str = "a") -> zarr.Group:
    """Open the root group of a Zarr store from either a local path or a
    remote fsspec URI (``gs://``, ``s3://``, ...).
    """
    s = str(store_path)
    if _is_remote_uri(s):
        store = zarr.storage.FsspecStore.from_url(s, read_only=mode == "r")
        return zarr.open_group(store, mode=mode, zarr_format=3)
    return zarr.open_group(s, mode=mode, zarr_format=3)


def size_on_disk(path: str | Path) -> int:
    """Return the total bytes on disk for ``path``.

    Handles single files, directory trees (recursive), and remote fsspec URIs
    (``gs://``, ``s3://``, ``memory://``, ...). Returns 0 for missing paths.
    """
    s = str(path)
    if _is_remote_uri(s):
        fs, fpath = fsspec.core.url_to_fs(s)
        if not fs.exists(fpath):
            return 0
        return int(fs.du(fpath, total=True))
    p = Path(s)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for child in p.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def format_bytes(n: int) -> str:
    """Render a byte count using powers of 1024 (e.g. ``2.3 MB``).

    Below 1 KB renders as an integer count of bytes; above that, one decimal
    place. The unit suffixes match what ``ls -lh`` and Finder show.
    """
    if n < 1024:
        return f"{int(n)} B"
    units = ("KB", "MB", "GB", "TB", "PB", "EB")
    size = float(n) / 1024.0
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    # Unreachable, but keeps mypy happy.
    return f"{size:.1f} {units[-1]}"
