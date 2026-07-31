"""Audit-trail metadata for converted Zarrs.

Every conversion writes ``attrs["zarrmony"]`` at the root of the output store,
recording: zarrmony version, the winning reader plugin (name, distribution,
source, version, match score), the input file's path / size / mtime / optional
SHA256, the ``output`` block declaring the writer's OME-NGFF version (ADR-0008,
one stable audit path so BigQuery ingest never hardcodes ``"0.5"``), the
conversion config the user passed, started/finished timestamps, per-scene
records returned by ``write_scene`` (which for LIF conversions may carry an
``objective`` sub-dict with ``nominal_magnification`` / ``numerical_aperture``
/ ``immersion`` / ``model`` / ``working_distance_um``), and any
extractor-failure warnings.

Stored as a top-level ``attrs.zarrmony`` (not under ``attrs.ome``) to keep the
spec-defined namespace clean. ``audit_schema_version`` is bumped whenever this
record's shape changes.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from zarrmony import __version__
from zarrmony._constants import NGFF_VERSION
from zarrmony._storage import format_bytes, open_root_group, size_on_disk
from zarrmony.readers.plugin import ReaderPlugin

# 8: adds top-level ``output: {ome_ngff_version}`` block sourced from the
#    writer's ``NGFF_VERSION`` constant so downstream consumers (Aperture
#    BigQuery ingest) have one stable audit path for the NGFF version instead
#    of hardcoding ``"0.5"`` or reading the OME-NGFF ``attrs.ome.version``.
#    Purely additive: consumers pinned to 7 can widen their pin. (#70, ADR-0008)
# 7: adds optional ``per_scene[i].objective`` (nominal_magnification /
#    numerical_aperture / immersion / model / working_distance_um) from the
#    LIF objective-lens extractor. Missing fields are omitted; scenes with no
#    objective info omit the ``objective`` key entirely. Purely additive:
#    consumers pinned to 6 can widen their pin. (#52)
AUDIT_SCHEMA_VERSION = 8


def _file_forensics(path: str | Path, *, checksum: bool = False) -> dict[str, Any]:
    p = Path(path)
    info: dict[str, Any] = {
        "path": str(p.resolve()),
        "exists": p.exists(),
    }
    if p.exists():
        st = p.stat()
        info["size_bytes"] = size_on_disk(p)
        info["size_human"] = format_bytes(info["size_bytes"])
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


def _reader_plugin_record(
    plugin: ReaderPlugin | None,
    match_score: int | None,
    distribution: str | None,
) -> dict[str, Any] | None:
    if plugin is None:
        return None
    # Caller-supplied distribution wins (lets the catch-all default plugin
    # surface the actual bioio sub-package, e.g. ``bioio-ome-tiff``).
    actual_distribution = (
        distribution if distribution is not None else plugin.distribution
    )
    return {
        "name": plugin.name,
        "version": (
            _try_pkg_version(actual_distribution) if actual_distribution else None
        ),
        "source": plugin.source,
        "distribution": actual_distribution,
        "match_score": match_score,
    }


def build_audit_record(
    *,
    input_path: str | Path,
    reader_plugin: ReaderPlugin | None,
    match_score: int | None = None,
    distribution: str | None = None,
    config: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    layout: str | None = None,
    per_scene: list[dict[str, Any]] | None = None,
    fields: list[dict[str, Any]] | None = None,
    plate: dict[str, Any] | None = None,
    metadata_warnings: list[dict[str, Any]] | None = None,
    checksum: bool = False,
) -> dict[str, Any]:
    """Assemble the audit-record dict written to ``attrs.zarrmony``.

    ``distribution`` overrides ``reader_plugin.distribution`` for the audit
    record only; pass it when the plugin's static ``distribution`` field is
    ``None`` (e.g. the catch-all default plugin) and the caller has dynamically
    resolved the actual underlying bioio sub-package.

    Per ADR-0004, plate-layout audits use ``fields`` + ``plate`` (and omit
    ``per_scene``); flat-layout audits keep using ``per_scene``. The top-level
    ``layout`` key is the discriminator consumers switch on.
    """
    record: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "version": __version__,
        "layout": layout,
        "reader_plugin": _reader_plugin_record(
            reader_plugin, match_score, distribution
        ),
        "input": _file_forensics(input_path, checksum=checksum),
        "output": {"ome_ngff_version": NGFF_VERSION},
        "config": config,
        "conversion_started_at": started_at.isoformat(),
        "conversion_finished_at": finished_at.isoformat(),
        "metadata_warnings": metadata_warnings or [],
    }
    if fields is not None or plate is not None:
        record["fields"] = fields or []
        record["plate"] = plate or {}
    else:
        record["per_scene"] = per_scene or []
    return record


def write_audit_record(store_path: str | Path, audit: dict[str, Any]) -> None:
    """Set ``root.attrs["zarrmony"]`` to the audit dict.

    Lives outside ``attrs.ome`` to avoid shadowing spec-defined OME content.
    """
    root = open_root_group(store_path, mode="a")
    root.attrs["zarrmony"] = audit
