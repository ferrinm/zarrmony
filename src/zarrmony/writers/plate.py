"""HCS plate writer (OME-NGFF 0.5 ``plate`` layout).

Validates a :class:`PlateLayout`, writes one OME-Zarr image per FOV under
``<plate>/<row>/<column>/<seq_int>/`` (reusing :func:`writers.scene.write_scene`
so plate FOVs and per-scene stores share one pixel/metadata path), then writes
the plate-level metadata: ``attrs.ome.plate`` at the root, structural row/well
groups with ``attrs.ome.well.images``, and a single combined
``OME/METADATA.ome.xml`` (no per-FOV sidecars — the unit of provenance is the
plate). The ``bioformats2raw.layout`` marker is intentionally NOT emitted on
plate stores; modern plate-aware tools consume the ``plate`` key directly.

See ADR-0004 for the design rationale and rejected alternatives.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fsspec
from bioio_ome_zarr.writers import Channel
from ome_types.model import Image

from zarrmony._constants import NGFF_VERSION
from zarrmony._storage import open_root_group
from zarrmony.errors import LayoutDowngradeWarning, PlateLayoutError
from zarrmony.metadata._lif_scene import find_scene_xml
from zarrmony.metadata.channel_colors import colors_for_channels
from zarrmony.metadata.lif_tiles import (
    extract_tile_layout,
    grid_shape,
    reassemble_grid,
    select_auto_stitch_cascade,
)
from zarrmony.readers.plate import PlateField, PlateLayout
from zarrmony.writers.scene import _dtype_window, write_scene

_WELL_KEY_RE = re.compile(r"([A-Za-z]+)(\d+)")

# Grid-stitch audit fields (parallel to the API-level constants; duplicated
# here to keep this writer independent of ``zarrmony.api`` — a plate write can
# be driven directly for testing without importing the top-level orchestrator).
_STITCHER_GRID_NAME = "zarrmony-grid"
_GRID_OVERLAP_ASSUMPTION_PX = 0


def _grid_stitch_fov_mosaic_summary(
    tile_layout: dict | None,
    *,
    tile_count: int,
    reader: Any,
) -> dict:
    """FOV-scoped ``mosaic`` block for a grid-stitched plate field.

    Same shape as the per-scene grid-stitch audit block — records
    ``stitcher="zarrmony-grid"``, ``overlap_assumption_px=0``,
    ``placement_shape``, and the extracted tile positions when available.
    Kept parallel to ``api._grid_stitch_mosaic_summary`` on purpose: a plate
    FOV mosaic and a flat-scene mosaic get identical audit surfaces for the
    same underlying operation.
    """
    summary: dict = {
        "stitched": True,
        "stitcher": _STITCHER_GRID_NAME,
        "overlap_assumption_px": _GRID_OVERLAP_ASSUMPTION_PX,
        "tile_count": tile_count,
    }
    tile_dims = getattr(reader, "mosaic_tile_dims", None)
    if tile_dims is not None:
        y = getattr(tile_dims, "Y", None)
        x = getattr(tile_dims, "X", None)
        if y is not None and x is not None:
            summary["tile_shape"] = {"Y": int(y), "X": int(x)}
    if tile_layout is not None:
        tiles = tile_layout.get("tiles") or []
        rows, cols = grid_shape(tiles)
        if rows is not None and cols is not None:
            summary["placement_shape"] = {"rows": rows, "cols": cols}
        summary.update(tile_layout)
    return summary


def parse_well_key(key: str) -> tuple[str, str]:
    """Split a compact well key like ``"B04"`` or ``"AA01"`` into ``(row, col)``.

    Casing and zero-padding are preserved verbatim — caller validates against
    the plate's canonical row/column spellings (so ``"b04"`` or ``"B4"`` will
    fail downstream membership checks against an upper/zero-padded plate).
    """
    m = _WELL_KEY_RE.fullmatch(key)
    if m is None:
        raise ValueError(
            f"well key {key!r} is not a valid alpha+digit coordinate "
            f"(expected leading letters + trailing digits, e.g. 'B04', 'AA12')"
        )
    return m.group(1), m.group(2)


def _normalize_store_path(store_path: str | Path) -> str:
    if isinstance(store_path, Path):
        return str(store_path)
    return store_path


def _write_text(uri: str, content: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "w") as f:
        f.write(content)


def validate_plate_layout(plate_layout: PlateLayout, n_scenes: int) -> None:
    """Raise :class:`PlateLayoutError` if ``plate_layout`` is internally inconsistent.

    Runs before any pixel writes. v1 enforces single-acquisition; row/column
    references must resolve; ``scene_index`` values must be in range and unique
    (a duplicate would silently double-write the same source data into two
    different well paths); ``acquisition_id`` must point at a declared
    acquisition.
    """
    if len(plate_layout.acquisitions) > 1:
        raise PlateLayoutError(
            f"v1 plate writer supports at most 1 acquisition; "
            f"got {len(plate_layout.acquisitions)} (multi-acquisition deferred to v2)"
        )

    rows = set(plate_layout.rows)
    cols = set(plate_layout.columns)
    acq_ids = {a.id for a in plate_layout.acquisitions}
    seen_scenes: set[int] = set()
    for f in plate_layout.fields:
        if f.row not in rows:
            raise PlateLayoutError(
                f"field references unknown row {f.row!r}; plate rows are {plate_layout.rows!r}"
            )
        if f.column not in cols:
            raise PlateLayoutError(
                f"field references unknown column {f.column!r}; "
                f"plate columns are {plate_layout.columns!r}"
            )
        if not 0 <= f.scene_index < n_scenes:
            raise PlateLayoutError(
                f"field.scene_index {f.scene_index} out of range [0, {n_scenes}) for reader.scenes"
            )
        if f.scene_index in seen_scenes:
            raise PlateLayoutError(
                f"duplicate well path: scene_index {f.scene_index} is referenced "
                f"by more than one PlateField"
            )
        seen_scenes.add(f.scene_index)
        if f.acquisition_id is not None and f.acquisition_id not in acq_ids:
            raise PlateLayoutError(
                f"field.acquisition_id {f.acquisition_id} not in declared "
                f"acquisitions {sorted(acq_ids)!r}"
            )


def _group_fields_by_well(
    plate_layout: PlateLayout,
) -> list[tuple[str, str, list[PlateField]]]:
    """Group fields by (row, column), preserving first-occurrence order."""
    groups: dict[tuple[str, str], list[PlateField]] = {}
    order: list[tuple[str, str]] = []
    for f in plate_layout.fields:
        key = (f.row, f.column)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(f)
    return [(r, c, groups[(r, c)]) for (r, c) in order]


def _channels_for_current_scene(
    reader: Any, channel_colors: dict[str, str] | str | None
) -> list[Channel] | None:
    """Emission-band-colored channels for a plate FOV.

    Accepts the same ``channel_colors`` spec as ``convert()``: a dict per-channel
    override, the ``"source-file"`` sentinel (degrades to band-scheme on this
    non-LIF path — see :func:`api._channels_for_scene`), or ``None``. Passes
    ``window=`` so a plate FOV honours the dtype-range display bounds (issue
    #50) rather than falling through to bioio-ome-zarr's 0–255 default.
    """
    channel_names = (
        list(reader.channel_names) if getattr(reader, "channel_names", None) else []
    )
    if not channel_names:
        return None
    overrides = channel_colors if isinstance(channel_colors, dict) else None
    colors = colors_for_channels(channel_names, overrides=overrides)
    window = _dtype_window(reader.dtype)
    return [
        Channel(label=n, color=c, window=window)
        for n, c in zip(channel_names, colors, strict=True)
    ]


def summarize_plate_layout(plate_layout: PlateLayout) -> dict[str, Any]:
    """Return the OME-NGFF ``plate``-shaped summary of ``plate_layout`` (no writes).

    Mirrors the on-disk ``attrs.ome.plate`` and audit ``plate`` shape, derived
    purely from the layout (no pixel I/O). Used by :func:`zarrmony.inspect` to
    surface plate context before a conversion runs.
    """
    well_groups = _group_fields_by_well(plate_layout)
    well_paths = [(r, c, len(fields)) for (r, c, fields) in well_groups]
    return _build_plate_attr(plate_layout, well_paths)


def _build_plate_attr(
    plate_layout: PlateLayout, well_paths: list[tuple[str, str, int]]
) -> dict[str, Any]:
    """Build the OME-NGFF ``plate`` attribute dict from layout + well grouping."""
    row_index = {r: i for i, r in enumerate(plate_layout.rows)}
    col_index = {c: i for i, c in enumerate(plate_layout.columns)}
    wells = [
        {
            "path": f"{r}/{c}",
            "rowIndex": row_index[r],
            "columnIndex": col_index[c],
        }
        for (r, c, _n_fields) in well_paths
    ]
    acquisitions = []
    for a in plate_layout.acquisitions:
        entry: dict[str, Any] = {"id": a.id}
        if a.name is not None:
            entry["name"] = a.name
        if a.maximumfieldcount is not None:
            entry["maximumfieldcount"] = a.maximumfieldcount
        acquisitions.append(entry)

    plate_attr: dict[str, Any] = {
        "name": plate_layout.name,
        "rows": [{"name": r} for r in plate_layout.rows],
        "columns": [{"name": c} for c in plate_layout.columns],
        "wells": wells,
        "version": NGFF_VERSION,
    }
    if acquisitions:
        plate_attr["acquisitions"] = acquisitions
    if well_paths:
        plate_attr["field_count"] = max(n for (_r, _c, n) in well_paths)
    return plate_attr


def _build_well_attr(fields: list[PlateField]) -> dict[str, Any]:
    """Build a single well group's ``well`` attribute dict from its fields."""
    images = []
    for seq, f in enumerate(fields):
        entry: dict[str, Any] = {"path": str(seq)}
        if f.acquisition_id is not None:
            entry["acquisition"] = f.acquisition_id
        images.append(entry)
    return {"images": images, "version": NGFF_VERSION}


def write_plate(
    reader: Any,
    *,
    store_path: str | Path,
    plate_layout: PlateLayout,
    pyramid_min_size: int = 256,
    chunk_shape: Sequence[int] | None = None,
    channel_colors: dict[str, str] | str | None = None,
    contrast_percentile: float | None = None,
    ome_image_for_field: Any = None,
    ome_xml_builder: Any = None,
    source_xml: str | None = None,
    source_xml_filename: str | None = None,
    lif_mosaic: str = "auto-stitch",
) -> tuple[list[dict], dict]:
    """Validate ``plate_layout``, write every FOV, and emit plate-level metadata.

    Returns ``(field_records, audit_plate)`` for the audit caller.

    ``ome_image_for_field`` is an optional callable
    ``(scene_index, scene_record) -> ome_types.model.Image`` invoked once per
    FOV to assemble the combined ``OME/METADATA.ome.xml``. ``ome_xml_builder``
    is an optional callable ``list[Image] -> str`` (defaults to
    ``writers.ome_xml.build_combined_ome_xml``); injecting it keeps the writer
    free of any specific builder import-time dependency.

    ``lif_mosaic="grid-stitch"`` reassembles each mosaic FOV's canvas from
    per-tile ``FieldX``/``FieldY`` indices before writing (butt joints), so
    mosaic FOVs in a plate get correct tile arrangement while still meeting
    the "one FOV = one image" plate spec. Non-mosaic FOVs and non-LIF plates
    pass through untouched. ``lif_mosaic="per-tile"`` is not accepted here —
    per-tile output produces N sub-stores per FOV, which the plate spec cannot
    express; ``api.convert()`` raises before calling this writer.
    """
    validate_plate_layout(plate_layout, n_scenes=len(reader.scenes))

    referenced = {f.scene_index for f in plate_layout.fields}
    unreferenced = [i for i in range(len(reader.scenes)) if i not in referenced]
    if unreferenced:
        preview = unreferenced[:5]
        suffix = (
            "" if len(unreferenced) <= 5 else f", ... ({len(unreferenced) - 5} more)"
        )
        warnings.warn(
            f"{len(unreferenced)} scene(s) in reader.scenes are not referenced "
            f"by any PlateField and will not be written to the plate "
            f"(scene indices: {preview}{suffix}). To include them, add the "
            f"corresponding PlateField entries to plate_layout.fields.",
            LayoutDowngradeWarning,
            stacklevel=2,
        )

    store_str = _normalize_store_path(store_path).rstrip("/")
    well_groups = _group_fields_by_well(plate_layout)

    field_records: list[dict] = []
    images: list[Image] = []

    for row, column, well_fields in well_groups:
        for seq, f in enumerate(well_fields):
            field_path = f"{row}/{column}/{seq}"
            fov_store = f"{store_str}/{field_path}"

            reader.set_scene(f.scene_index)
            channels = _channels_for_current_scene(reader, channel_colors)
            image_name = f.field_name or reader.scenes[f.scene_index]

            # Cascade under plate mode: auto-stitch resolves per FOV to either
            # grid-stitch (when tile grid metadata is complete) or bioio-lif
            # (fallback). Stage-stitch is unreachable here — the plate writer
            # hasn't wired stage-stitch's canvas swap, and select_auto_stitch_
            # cascade() honours plate_mode=True by skipping it. Explicit
            # lif_mosaic="grid-stitch" from the caller still routes directly.
            reassembly_eligible_fov = bool(
                getattr(reader, "is_mosaic_reassembly_eligible", lambda: False)()
            )
            effective_stitcher = lif_mosaic
            cascade_selected = False
            grid_tile_layout: dict | None = None
            grid_tile_count = 0
            if reassembly_eligible_fov and lif_mosaic == "auto-stitch":
                cascade_tiles_xarr = reader.tiles_xarray_dask_data
                grid_tile_count = int(cascade_tiles_xarr.sizes["M"])
                cascade_scene_xml = find_scene_xml(reader)
                grid_tile_layout = (
                    extract_tile_layout(cascade_scene_xml)
                    if cascade_scene_xml is not None
                    else None
                )
                effective_stitcher = select_auto_stitch_cascade(
                    grid_tile_layout,
                    m_size=grid_tile_count,
                    pixel_size_x_um=None,
                    pixel_size_y_um=None,
                    plate_mode=True,
                )
                cascade_selected = True

            grid_stitch_this_fov = (
                effective_stitcher == "grid-stitch" and reassembly_eligible_fov
            )
            if grid_stitch_this_fov:
                grid_tiles_xarr = reader.tiles_xarray_dask_data
                grid_tile_count = int(grid_tiles_xarr.sizes["M"])
                if grid_tile_layout is None:
                    scene_xml = find_scene_xml(reader)
                    grid_tile_layout = (
                        extract_tile_layout(scene_xml)
                        if scene_xml is not None
                        else None
                    )
                grid_xarr = reassemble_grid(grid_tiles_xarr, grid_tile_layout)
            else:
                grid_xarr = None

            scene_record = write_scene(
                reader,
                scene_index=f.scene_index,
                store_path=fov_store,
                pyramid_min_size=pyramid_min_size,
                chunk_shape=chunk_shape,
                channels=channels,
                image_name=image_name,
                xarr_override=grid_xarr,
                record_mosaic_summary=not grid_stitch_this_fov,
                contrast_percentile=contrast_percentile,
            )
            scene_record.update(
                {
                    "row": row,
                    "column": column,
                    "field_path": field_path,
                    "field_name": f.field_name,
                    "acquisition_id": f.acquisition_id,
                }
            )
            if grid_stitch_this_fov:
                scene_record["mosaic"] = _grid_stitch_fov_mosaic_summary(
                    grid_tile_layout,
                    tile_count=grid_tile_count,
                    reader=reader,
                )
            if cascade_selected and "mosaic" in scene_record and scene_record["mosaic"]:
                scene_record["mosaic"]["cascade_selected"] = True
            field_records.append(scene_record)

            if ome_image_for_field is not None:
                images.append(ome_image_for_field(f.scene_index, scene_record))

    well_paths = [(r, c, len(fields)) for (r, c, fields) in well_groups]
    plate_attr = _build_plate_attr(plate_layout, well_paths)

    root = open_root_group(store_str, mode="a")
    root.attrs["ome"] = {"version": NGFF_VERSION, "plate": plate_attr}

    for row, column, well_fields in well_groups:
        # Row group: structural only — no attrs per spec.
        root.require_group(row)
        well_group = root.require_group(f"{row}/{column}")
        well_group.attrs["ome"] = {
            "version": NGFF_VERSION,
            "well": _build_well_attr(well_fields),
        }

    if ome_xml_builder is not None and images:
        ome_xml = ome_xml_builder(images)
        _write_text(f"{store_str}/OME/METADATA.ome.xml", ome_xml)

    if source_xml is not None:
        if source_xml_filename is None:
            raise ValueError(
                "source_xml_filename is required when source_xml is provided"
            )
        _write_text(f"{store_str}/OME/source/{source_xml_filename}", source_xml)

    return field_records, plate_attr


__all__ = [
    "parse_well_key",
    "summarize_plate_layout",
    "validate_plate_layout",
    "write_plate",
]
