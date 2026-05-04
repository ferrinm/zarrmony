"""Internal helpers for opening Zarr groups across local paths and cloud URIs.

Centralises the URI-detection pattern used by both ``writers/bf2raw.py`` and
``audit.py`` so cloud paths (``gs://``, ``s3://``) are handled the same way as
local paths.
"""

from pathlib import Path

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
