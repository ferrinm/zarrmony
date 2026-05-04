"""Audit-trail metadata for converted Zarrs.

Every conversion writes ``attrs["zarrmony"]`` at the root of the output store,
recording: zarrmony version, reader plugin used (and its version), the input
file's path / size / mtime / optional SHA256, the conversion config the user
passed, started/finished timestamps, per-scene records returned by
``write_scene``, and any extractor-failure warnings.

Stored as a top-level ``attrs.zarrmony`` (not under ``attrs.ome``) to keep the
spec-defined namespace clean.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from zarrmony import __version__
from zarrmony._storage import open_root_group


def _file_forensics(path: str | Path, *, checksum: bool = False) -> dict[str, Any]:
    p = Path(path)
    info: dict[str, Any] = {
        "path": str(p.resolve()),
        "exists": p.exists(),
    }
    if p.exists():
        st = p.stat()
        info["size_bytes"] = st.st_size
        info["mtime_iso"] = datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()
        if checksum and p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(2**20), b""):
                    h.update(chunk)
            info["sha256"] = h.hexdigest()
    return info


def _try_pkg_version(pkg: str) -> str | None:
    try:
        return pkg_version(pkg)
    except PackageNotFoundError:
        return None


def build_audit_record(
    *,
    input_path: str | Path,
    reader_plugin: str | None,
    config: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    per_scene: list[dict[str, Any]] | None = None,
    metadata_warnings: list[dict[str, Any]] | None = None,
    checksum: bool = False,
) -> dict[str, Any]:
    """Assemble the audit-record dict that gets written to ``attrs.zarrmony``."""
    return {
        "version": __version__,
        "reader_plugin": reader_plugin,
        "reader_plugin_version": _try_pkg_version(reader_plugin) if reader_plugin else None,
        "input": _file_forensics(input_path, checksum=checksum),
        "config": config,
        "conversion_started_at": started_at.isoformat(),
        "conversion_finished_at": finished_at.isoformat(),
        "per_scene": per_scene or [],
        "metadata_warnings": metadata_warnings or [],
    }


def write_audit_record(store_path: str | Path, audit: dict[str, Any]) -> None:
    """Set ``root.attrs["zarrmony"]`` to the audit dict.

    Lives outside ``attrs.ome`` to avoid shadowing spec-defined OME content.
    """
    root = open_root_group(store_path, mode="a")
    root.attrs["zarrmony"] = audit
