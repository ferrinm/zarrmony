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
   ADR's estimates get compared against. On a sharded store the object is the
   *shard*, so the prediction runs on the shard grid and the chunk grid is
   reported beside it as the read unit it now is — the two differ by 15.84× on
   the store measured in #124, and a prediction off the chunk grid is simply
   not comparable to what is on disk (#129). The comparison is ``objects <=
   grid``, never equality: zarr writes no object that is entirely fill value,
   so a padded trailing row legitimately comes in short (#126).

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
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np

from zarrmony._storage import (
    _is_remote_uri,
    format_bytes,
    open_root_group,
    size_on_disk,
)
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    Geometry,
    count_storage_objects,
    plan_level_chunk_shapes,
    plan_level_shard_shapes,
)
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
    # Both are tuple-typed on ``Geometry`` and both come back from JSON as
    # lists. ``Geometry.__post_init__`` normalises them too, so this is belt
    # and braces — but it keeps the coercion at the boundary where the JSON is,
    # and it applies to both fields rather than only the one that had it.
    for field in ("chunk_shape", "shard_shape"):
        if updates.get(field) is not None:
            updates[field] = tuple(int(s) for s in updates[field])
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


class ObjectCounts(NamedTuple):
    """Files under a store, split by what they are.

    ``pixels`` is the only figure comparable against a level grid: it counts
    the chunk or shard objects under the level arrays and nothing else. The
    other two are why the split exists — a multiscale group carries one
    ``zarr.json`` per level plus one for itself, and a zarrmony store also
    holds ``OME/METADATA.ome.xml`` and the source metadata beside it. Six
    files on the smallest possible store, none of them pixels, all of them
    enough to put a correct store over its own grid.
    """

    pixels: int
    metadata: int
    other: int

    @property
    def total(self) -> int:
        return self.pixels + self.metadata + self.other


def count_objects(path: str, level_paths: Sequence[str]) -> ObjectCounts:
    """Files under ``path``, classified against the level arrays' own names.

    Walks the tree, which on a multi-million-object store is minutes rather
    than seconds — hence ``--no-object-count`` for a quick re-check. The
    predicted object count from the level grids is reported either way.

    A file counts as pixels when it sits under one of ``level_paths`` and is
    not that array's ``zarr.json``. Classifying by the level names rather than
    by "everything that is not ``zarr.json``" keeps the sidecars out: the OME
    XML is a file in the store like any other, and folding it into the pixel
    figure would report a correct store as holding more objects than its
    geometry accounts for.
    """
    roots = {str(level).split("/", 1)[0] for level in level_paths}

    def classify(relative: str, name: str) -> str:
        if name == "zarr.json":
            return "metadata"
        return "pixels" if relative.split("/", 1)[0] in roots else "other"

    tally = {"pixels": 0, "metadata": 0, "other": 0}
    if _is_remote_uri(path):
        import fsspec

        fs, fpath = fsspec.core.url_to_fs(path)
        prefix = fpath.rstrip("/") + "/"
        for found in fs.find(fpath):
            relative = found[len(prefix) :] if found.startswith(prefix) else found
            tally[classify(relative, relative.rsplit("/", 1)[-1])] += 1
    else:
        for root, _dirs, files in os.walk(path):
            relative = os.path.relpath(root, path).replace(os.sep, "/")
            relative = "" if relative == "." else relative
            for name in files:
                tally[classify(f"{relative}/{name}".lstrip("/"), name)] += 1
    return ObjectCounts(tally["pixels"], tally["metadata"], tally["other"])


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
    shard_shapes: list[tuple[int, ...] | None] = []
    for path in _level_paths(multiscales):
        array = group[path]
        shapes.append(tuple(int(s) for s in array.shape))
        # ``.chunks`` on a sharded array is the *inner* unit — what a viewer
        # range-reads — and stays the chunk here. ``.shards`` is the storage
        # object, ``None`` when the array is unsharded.
        chunk_shapes.append(tuple(int(c) for c in array.chunks))
        shards = getattr(array, "shards", None)
        shard_shapes.append(tuple(int(s) for s in shards) if shards else None)
    dtype = np.dtype(group[_level_paths(multiscales)[0]].dtype)
    itemsize = dtype.itemsize
    sharded = any(shard is not None for shard in shard_shapes)
    # What one storage object covers, per level: the shard where there is one,
    # else the chunk. Everything counting objects has to go through this, or it
    # counts the read grid and disagrees with the disk by the shard factor.
    write_grids = [
        chunk if shard is None else shard
        for chunk, shard in zip(chunk_shapes, shard_shapes, strict=True)
    ]
    unit, units = ("shard", "shards") if sharded else ("chunk", "chunks")

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
    planned_shards = plan_level_shard_shapes(
        predicted_chunks, predicted_shapes, dims, scale0, dtype, geometry
    )

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

    # Only when a shard is in play at all — on an unsharded store written by an
    # unsharded policy there is nothing to say, and saying it would add a line
    # to every report that predates #117.
    if sharded or planned_shards is not None:
        planned: list[tuple[int, ...] | None] = (
            [None] * len(shapes) if planned_shards is None else list(planned_shards)
        )
        matches = shard_shapes == planned
        if matches:
            level0 = shard_shapes[0]
            assert level0 is not None  # implied by matching a non-None plan
            detail = (
                f"every level matches the plan; level 0 is {level0} "
                f"({math.prod(level0) * itemsize / 2**20:.1f} MiB raw, "
                f"{math.prod(_grid_shape(level0, chunk_shapes[0]))} chunks per shard)"
            )
        elif planned_shards is None:
            detail = (
                f"on disk {shard_shapes}, but the recorded policy plans no shards — "
                f"the store was written by a policy other than the one it records"
            )
        elif not sharded:
            detail = f"policy plans {planned}, but no level on disk is sharded"
        else:
            detail = f"on disk {shard_shapes}, policy plans {planned}"
        checks.append(Check("shard shapes", matches, detail))

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
    predicted_objects = count_storage_objects(shapes, write_grids)
    predicted_chunk_objects = count_storage_objects(shapes, chunk_shapes)
    raw_bytes = sum(math.prod(shape) for shape in shapes) * itemsize
    counted = count_objects(store_path, _level_paths(multiscales)) if do_count else None

    if counted is not None:
        on_disk = counted.pixels
        # ``<=``, never ``==``. zarr writes no object whose contents are
        # entirely fill value, so a level with a padded trailing row comes in
        # short of its own grid — 209,211 shards against 210,345 on the
        # reference volume (#126), all 1,134 absent ones on a 3-voxel sliver.
        # An excess is the failure: it means objects exist that the geometry
        # does not account for.
        within = on_disk <= predicted_objects
        missing = predicted_objects - on_disk
        if not within:
            detail = (
                f"{on_disk:,} {units} on disk against a {predicted_objects:,}-"
                f"{unit} grid — {-missing:,} more than the geometry accounts for"
            )
        elif missing:
            detail = (
                f"{on_disk:,} of {predicted_objects:,} — {missing:,} absent "
                f"({missing / predicted_objects:.2%}), which is what an all-fill "
                f"{unit} looks like: zarr writes no object that is entirely fill "
                f"value, so a padded trailing row costs nothing"
            )
        else:
            detail = f"{on_disk:,} {units}, exactly the grid — nothing all-fill"
        checks.append(Check(f"{units} on disk within the grid", within, detail))

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
    if sharded:
        # The shard is the write unit and the object-count unit, and it can
        # change partway down a pyramid — the reference volume flips its long
        # axis from X to Y at level 3 (#126). A table without it reads as
        # uniform.
        report.append(
            "| level | shape | chunk | shard | chunks | shards | "
            "MiB per (t,c) | coarse |"
        )
        report.append(
            "| ----- | ----- | ----- | ----- | ------ | ------ | "
            "------------- | ------ |"
        )
    else:
        report.append("| level | shape | chunk | chunks | MiB per (t,c) | coarse |")
        report.append("| ----- | ----- | ----- | ------ | ------------- | ------ |")
    for i, (shape, chunks) in enumerate(zip(shapes, chunk_shapes, strict=True)):
        per_tc = math.prod(_spatial(shape, dims)) * itemsize / 2**20
        fails = _lucida_level_fits(shape, chunks, dims, itemsize)
        coarse = "yes" if not fails else "; ".join(fails)
        if sharded:
            shard = shard_shapes[i]
            report.append(
                f"| {i} | `{shape}` | `{chunks}` | "
                f"{'—' if shard is None else f'`{shard}`'} | "
                f"{math.prod(_grid_shape(shape, chunks)):,} | "
                f"{'—' if shard is None else f'{math.prod(_grid_shape(shape, shard)):,}'} | "
                f"{per_tc:,.1f} | {coarse} |"
            )
        else:
            report.append(
                f"| {i} | `{shape}` | `{chunks}` | "
                f"{math.prod(_grid_shape(shape, chunks)):,} | {per_tc:,.1f} | "
                f"{coarse} |"
            )
    report.append("")
    report.append("### Measurements")
    report.append("")
    report.append(
        f"- Store size: **{format_bytes(total_bytes)}** ({total_bytes:,} bytes)"
    )
    report.append(
        f"- {unit.capitalize()} objects (from the level grids): "
        f"**{predicted_objects:,}**"
    )
    if sharded:
        report.append(
            f"- Chunks inside them (the read grid, not objects): "
            f"**{predicted_chunk_objects:,}**"
        )
    if counted is not None:
        report.append(f"- Objects on disk ({units} + metadata): **{counted.total:,}**")
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
