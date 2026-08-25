#!/usr/bin/env python3
"""Check a written OME-Zarr store against the ADR-0010 geometry policy.

An operations script, not part of the ``zarrmony`` package or its public API —
it lives here rather than in ``src/`` so it is not shipped in the wheel. It
exists because the acceptance test for the ADR-0010 series (issue #90) is a
multi-hour conversion on a cluster, and "did the geometry come out right?" has
to be answerable from the store afterwards rather than by re-reading the run's
console output.

Three questions, none of which the conversion itself answers:

1. **Does the store match what the policy would plan for it?** The predicted
   geometry is recomputed here from the store's *own* recorded policy
   (``attrs.zarrmony.config.geometry``) plus level 0's shape, spacing and
   dtype, then compared against the levels actually on disk. Nothing is
   hardcoded, so this is a real check on any store, not a lookup table for one
   dataset.

2. **Will the intended consumer resolve a coarse level?** zarrmony's
   ``coarse_level_index`` applies two of Lucida's ``SourceCoarseConfig``
   bounds (decoded bytes per (t,c), lateral long axis). Lucida applies two
   more that ADR-0010 does not model — a 16 MiB cap on one chunk's decoded
   bytes and a 4096 cap on chunks per (t,c) — and it selects the *deepest*
   fitting level where zarrmony records the *shallowest*. Both are checked
   here, because a store that satisfies zarrmony's rule and fails Lucida's
   would look fine in the audit and still fall back to a server-generated
   coarse tier in the viewer.

3. **How big did it get?** Bytes on disk and object count, which is what the
   ADR's estimates get compared against.

Usage::

    python scripts/verify_geometry.py STORE [--expect-coarse-level N]
                                            [--no-object-count]

``STORE`` is one OME-NGFF multiscale group — a ``<scene>.ome.zarr`` from the
per-scene layout, or one numbered subgroup of a bf2raw bundle / one field of a
plate. Local paths and ``gs://`` / ``s3://`` URIs both work.

Exit status is ``0`` when every check passes and ``1`` otherwise, so this can
gate a conversion job. The report on stdout is Markdown, ready to paste into
the issue the conversion was run for.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np

from zarrmony._storage import (
    _is_remote_uri,
    format_bytes,
    open_root_group,
    size_on_disk,
)
from zarrmony.geometry import DEFAULT_GEOMETRY, Geometry, plan_level_chunk_shapes
from zarrmony.writers.pyramid import coarse_level_index, compute_level_shapes

# Lucida's ``SourceCoarseConfig`` defaults, in full (lucida-store/src/coarse.rs).
# ADR-0010 adopted the first two as Geometry fields; the last two have no
# zarrmony counterpart and are checked here so a store cannot satisfy the
# policy while still failing to resolve a source coarse tier in the viewer.
LUCIDA_MAX_LONG_AXIS = 2048
LUCIDA_MAX_DECODED_BYTES_PER_TC = 64 * 1024 * 1024
LUCIDA_MAX_CHUNK_BYTES = 16 * 1024 * 1024
LUCIDA_MAX_CHUNK_COUNT_PER_TC = 4096


class Check:
    """One pass/fail line in the report, with the evidence that decided it."""

    def __init__(self, name: str, ok: bool, detail: str) -> None:
        self.name = name
        self.ok = ok
        self.detail = detail

    def line(self) -> str:
        return f"- {'PASS' if self.ok else 'FAIL'} — **{self.name}**: {self.detail}"


def _axis_letters(multiscales: dict[str, Any]) -> str:
    """Axis names as an uppercase string (``"TCZYX"``), from the OME metadata.

    The geometry rules are written over the axes that are present, so the axis
    order has to come from the store rather than be assumed.
    """
    return "".join(axis["name"].upper() for axis in multiscales["axes"])


def _level_paths(multiscales: dict[str, Any]) -> list[str]:
    return [dataset["path"] for dataset in multiscales["datasets"]]


def _level0_scale(multiscales: dict[str, Any]) -> list[float]:
    """Level 0's per-axis scale from ``coordinateTransformations``.

    This is the physical spacing the planner was originally given (µm on the
    spatial axes, 1.0 on T/C), recovered from the store so the prediction is
    rebuilt from the same inputs the conversion used.
    """
    for transform in multiscales["datasets"][0]["coordinateTransformations"]:
        if transform.get("type") == "scale":
            return [float(v) for v in transform["scale"]]
    raise SystemExit("level 0 has no scale coordinateTransformation; cannot verify")


def _geometry_from_audit(audit: dict[str, Any]) -> tuple[Geometry, str]:
    """The policy the store says it was written with, as a ``Geometry``.

    Falls back to :data:`~zarrmony.geometry.DEFAULT_GEOMETRY` for a store
    written before audit schema 9 (which is where ``config.geometry`` was
    introduced), so a pre-ADR-0010 store can still be compared against today's
    policy — that comparison is exactly what a migration wants to see.
    """
    recorded = (audit.get("config") or {}).get("geometry")
    if not recorded:
        return DEFAULT_GEOMETRY, "default policy (store predates config.geometry)"
    fields = {f for f in Geometry.__dataclass_fields__}
    updates = {k: v for k, v in recorded.items() if k in fields}
    if updates.get("chunk_shape") is not None:
        updates["chunk_shape"] = tuple(int(s) for s in updates["chunk_shape"])
    return replace(DEFAULT_GEOMETRY, **updates), "attrs.zarrmony.config.geometry"


def _scene_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-scene / per-field records, whichever this audit carries."""
    return audit.get("per_scene") or audit.get("fields") or []


def _wall_clock(audit: dict[str, Any]) -> str | None:
    """Conversion wall-clock from the audit's own start / finish timestamps.

    Read from the store rather than timed at the shell, so a job that ran
    detached on a cluster still reports its duration.
    """
    started, finished = audit.get("conversion_started_at"), audit.get(
        "conversion_finished_at"
    )
    if not (started and finished):
        return None
    try:
        elapsed = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    except ValueError:
        return None
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def _grid_shape(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.ceil(s / c) for s, c in zip(shape, chunks, strict=True))


def _spatial(values: tuple[int, ...], dims: str) -> tuple[int, ...]:
    return tuple(v for v, d in zip(values, dims, strict=True) if d in "ZYX")


def _lateral(values: tuple[int, ...], dims: str) -> tuple[int, ...]:
    return tuple(v for v, d in zip(values, dims, strict=True) if d in "YX")


def _lucida_level_fits(
    shape: tuple[int, ...], chunks: tuple[int, ...], dims: str, itemsize: int
) -> list[str]:
    """Which of Lucida's four source-coarse bounds this level fails, if any."""
    failures: list[str] = []
    lateral = _lateral(shape, dims)
    if lateral and max(lateral) > LUCIDA_MAX_LONG_AXIS:
        failures.append(f"long axis {max(lateral)} > {LUCIDA_MAX_LONG_AXIS}")
    decoded = math.prod(_spatial(shape, dims)) * itemsize
    if decoded > LUCIDA_MAX_DECODED_BYTES_PER_TC:
        failures.append(
            f"{decoded / 2**20:.1f} MiB/tc > "
            f"{LUCIDA_MAX_DECODED_BYTES_PER_TC // 2**20} MiB"
        )
    chunk_bytes = math.prod(_spatial(chunks, dims)) * itemsize
    if chunk_bytes > LUCIDA_MAX_CHUNK_BYTES:
        failures.append(
            f"chunk {chunk_bytes / 2**20:.2f} MiB > "
            f"{LUCIDA_MAX_CHUNK_BYTES // 2**20} MiB"
        )
    chunks_per_tc = math.prod(_spatial(_grid_shape(shape, chunks), dims))
    if chunks_per_tc > LUCIDA_MAX_CHUNK_COUNT_PER_TC:
        failures.append(f"{chunks_per_tc} chunks/tc > {LUCIDA_MAX_CHUNK_COUNT_PER_TC}")
    return failures


def _lucida_coarse_level(
    shapes: list[tuple[int, ...]],
    chunk_shapes: list[tuple[int, ...]],
    dims: str,
    itemsize: int,
) -> int | None:
    """The level Lucida's ``select_source_coarse_level`` would return.

    The *deepest* fitting level, where zarrmony's ``coarse_level_index`` records
    the shallowest. The two agree whenever exactly one level fits, which is the
    common case, and diverge on a pyramid with a tail of tiny levels — worth
    surfacing rather than assuming.
    """
    fitting = [
        i
        for i, (shape, chunks) in enumerate(zip(shapes, chunk_shapes, strict=True))
        if not _lucida_level_fits(shape, chunks, dims, itemsize)
    ]
    return max(fitting) if fitting else None


def count_objects(path: str) -> int:
    """Number of stored objects under ``path`` (chunks plus metadata files).

    Walks the tree, which on a multi-million-object store is minutes rather
    than seconds — hence ``--no-object-count`` for a quick re-check. The
    predicted chunk count from the level grids is reported either way.
    """
    if _is_remote_uri(path):
        import fsspec

        fs, fpath = fsspec.core.url_to_fs(path)
        return len(fs.find(fpath))
    total = 0
    for _root, _dirs, files in os.walk(path):
        total += len(files)
    return total


def verify(
    store_path: str, *, expect_coarse_level: int | None, do_count: bool
) -> tuple[list[Check], list[str]]:
    """Run every check against ``store_path``; return the checks and the report."""
    group = open_root_group(store_path, mode="r")
    attrs = dict(group.attrs)
    audit = attrs.get("zarrmony")
    if audit is None:
        raise SystemExit(
            f"{store_path} has no attrs.zarrmony — not a zarrmony-written store, "
            f"or the path points above the multiscale group"
        )
    multiscales_list = (attrs.get("ome") or {}).get("multiscales")
    if not multiscales_list:
        raise SystemExit(
            f"{store_path} has no attrs.ome.multiscales — pass one multiscale "
            f"group (a <scene>.ome.zarr, a bf2raw subgroup, or a plate field)"
        )
    multiscales = multiscales_list[0]

    dims = _axis_letters(multiscales)
    scale0 = _level0_scale(multiscales)
    shapes: list[tuple[int, ...]] = []
    chunk_shapes: list[tuple[int, ...]] = []
    for path in _level_paths(multiscales):
        array = group[path]
        shapes.append(tuple(int(s) for s in array.shape))
        chunk_shapes.append(tuple(int(c) for c in array.chunks))
    dtype = np.dtype(group[_level_paths(multiscales)[0]].dtype)
    itemsize = dtype.itemsize

    geometry, geometry_source = _geometry_from_audit(audit)
    predicted_shapes = [
        tuple(s) for s in compute_level_shapes(shapes[0], dims, scale0, dtype, geometry)
    ]
    predicted_chunks = [
        tuple(c)
        for c in plan_level_chunk_shapes(
            predicted_shapes, dims, scale0, dtype, geometry
        )
    ]

    checks: list[Check] = []

    checks.append(
        Check(
            "level shapes",
            shapes == predicted_shapes,
            (
                f"{len(shapes)} levels on disk match the policy's plan"
                if shapes == predicted_shapes
                else f"on disk {shapes}, policy plans {predicted_shapes}"
            ),
        )
    )
    checks.append(
        Check(
            "chunk shapes",
            chunk_shapes == predicted_chunks,
            (
                f"every level matches the plan; level 0 is {chunk_shapes[0]} "
                f"({math.prod(chunk_shapes[0]) * itemsize / 1024:.0f} KiB raw)"
                if chunk_shapes == predicted_chunks
                else f"on disk {chunk_shapes}, policy plans {predicted_chunks}"
            ),
        )
    )

    # Recomputed from the store rather than read from the audit, then compared
    # against the audit: the point of #86 is that the guarantee is checkable in
    # the store's metadata, so a disagreement between the two is itself a bug.
    actual_coarse = coarse_level_index(shapes, dims, dtype, geometry)
    audited = [record.get("coarse_level_index") for record in _scene_records(audit)]
    audit_agrees = (
        all(value == actual_coarse for value in audited) if audited else False
    )
    checks.append(
        Check(
            "audit coarse_level_index",
            audit_agrees,
            (
                f"`{actual_coarse}`, matching the levels on disk"
                if audit_agrees
                else f"audit says {audited}, the levels on disk give {actual_coarse}"
            ),
        )
    )
    if expect_coarse_level is not None:
        checks.append(
            Check(
                f"coarse level is {expect_coarse_level}",
                actual_coarse == expect_coarse_level,
                f"got {actual_coarse}",
            )
        )

    lucida_coarse = _lucida_coarse_level(shapes, chunk_shapes, dims, itemsize)
    if lucida_coarse is None:
        lucida_detail = (
            "no level satisfies all four SourceCoarseConfig bounds — Lucida will "
            "generate its own max-pooled coarse tier"
        )
    else:
        shape, chunks = shapes[lucida_coarse], chunk_shapes[lucida_coarse]
        lucida_detail = (
            f"level {lucida_coarse} fits all four bounds "
            f"({math.prod(_spatial(shape, dims)) * itemsize / 2**20:.1f} MiB/tc, "
            f"long axis {max(_lateral(shape, dims))}, "
            f"chunk {math.prod(_spatial(chunks, dims)) * itemsize / 1024:.0f} KiB, "
            f"{math.prod(_spatial(_grid_shape(shape, chunks), dims))} chunks/tc)"
        )
        if lucida_coarse != actual_coarse:
            # Not a defect: zarrmony records the shallowest fitting level and
            # Lucida selects the deepest, so any pyramid with a tail of small
            # levels shows two different indices for the same agreed fact —
            # that a source coarse tier exists. Reported so the two numbers are
            # not read as a disagreement.
            lucida_detail += (
                f"; zarrmony records {actual_coarse}, the shallowest fitting "
                f"level, and Lucida selects the deepest"
            )
    checks.append(
        Check("Lucida source coarse level", lucida_coarse is not None, lucida_detail)
    )

    warnings = audit.get("validation_warnings")
    checks.append(
        Check(
            "OME-NGFF validation",
            not warnings,
            (
                "no findings recorded"
                if not warnings
                else f"{len(warnings)} finding(s): {warnings}"
            ),
        )
    )

    # ---- measurements ----
    total_bytes = size_on_disk(store_path)
    predicted_chunk_objects = sum(
        math.prod(_grid_shape(shape, chunks))
        for shape, chunks in zip(shapes, chunk_shapes, strict=True)
    )
    raw_bytes = sum(math.prod(shape) for shape in shapes) * itemsize
    counted = count_objects(store_path) if do_count else None

    plugin = audit.get("reader_plugin") or {}
    reader = plugin.get("distribution") or plugin.get("name") or "?"

    report: list[str] = []
    report.append(f"## Geometry verification — `{store_path}`")
    report.append("")
    report.append(
        f"zarrmony `{audit.get('version', '?')}`, audit schema "
        f"`{audit.get('audit_schema_version', '?')}`, reader `{reader}`. "
        f"Policy read from {geometry_source}."
    )
    report.append("")
    report.append("### Checks")
    report.append("")
    report.extend(check.line() for check in checks)
    report.append("")
    report.append("### Levels")
    report.append("")
    report.append("| level | shape | chunk | chunks | MiB per (t,c) | coarse |")
    report.append("| ----- | ----- | ----- | ------ | ------------- | ------ |")
    for i, (shape, chunks) in enumerate(zip(shapes, chunk_shapes, strict=True)):
        per_tc = math.prod(_spatial(shape, dims)) * itemsize / 2**20
        fails = _lucida_level_fits(shape, chunks, dims, itemsize)
        report.append(
            f"| {i} | `{shape}` | `{chunks}` | "
            f"{math.prod(_grid_shape(shape, chunks)):,} | {per_tc:,.1f} | "
            f"{'yes' if not fails else '; '.join(fails)} |"
        )
    report.append("")
    report.append("### Measurements")
    report.append("")
    report.append(
        f"- Store size: **{format_bytes(total_bytes)}** ({total_bytes:,} bytes)"
    )
    report.append(
        f"- Chunk objects (from the level grids): **{predicted_chunk_objects:,}**"
    )
    if counted is not None:
        report.append(f"- Objects on disk (chunks + metadata): **{counted:,}**")
    report.append(
        f"- Raw (uncompressed) pyramid: {format_bytes(raw_bytes)} — "
        f"compression {raw_bytes / total_bytes:.2f}x"
        if total_bytes
        else f"- Raw (uncompressed) pyramid: {format_bytes(raw_bytes)}"
    )
    elapsed = _wall_clock(audit)
    if elapsed:
        report.append(
            f"- Conversion wall-clock (from the audit timestamps): **{elapsed}**"
        )
    report.append(f"- Axes `{dims}`, dtype `{dtype}`, level-0 scale `{scale0}`")
    report.append("")

    return checks, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "store", help="One OME-NGFF multiscale group (local path or gs:// / s3:// URI)"
    )
    parser.add_argument(
        "--expect-coarse-level",
        type=int,
        default=None,
        help="Fail unless the coarse level index is exactly this (e.g. 5 for issue #90)",
    )
    parser.add_argument(
        "--no-object-count",
        action="store_true",
        help="Skip the tree walk that counts stored objects (minutes on a large store)",
    )
    args = parser.parse_args(argv)

    checks, report = verify(
        args.store,
        expect_coarse_level=args.expect_coarse_level,
        do_count=not args.no_object_count,
    )
    print("\n".join(report))
    failed = [check for check in checks if not check.ok]
    if failed:
        print(f"{len(failed)} check(s) failed.", file=sys.stderr)
        return 1
    print("All checks passed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
