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

from zarrmony._storage import open_root_group
from zarrmony.errors import LayoutDowngradeWarning, PlateLayoutError
from zarrmony.metadata.channel_colors import colors_for_channels
from zarrmony.readers.plate import PlateField, PlateLayout
from zarrmony.writers.scene import write_scene

NGFF_VERSION = "0.5"

_WELL_KEY_RE = re.compile(r"([A-Za-z]+)(\d+)")


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
    reader: Any, channel_colors: dict[str, str] | None
) -> list[Channel] | None:
    channel_names = (
        list(reader.channel_names) if getattr(reader, "channel_names", None) else []
    )
    if not channel_names:
        return None
    colors = colors_for_channels(channel_names, overrides=channel_colors)
    return [
        Channel(label=n, color=c) for n, c in zip(channel_names, colors, strict=True)
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
    channel_colors: dict[str, str] | None = None,
    ome_image_for_field: Any = None,
    ome_xml_builder: Any = None,
    source_xml: str | None = None,
    source_xml_filename: str | None = None,
) -> tuple[list[dict], dict]:
    """Validate ``plate_layout``, write every FOV, and emit plate-level metadata.

    Returns ``(field_records, audit_plate)`` for the audit caller.

    ``ome_image_for_field`` is an optional callable
    ``(scene_index, scene_record) -> ome_types.model.Image`` invoked once per
    FOV to assemble the combined ``OME/METADATA.ome.xml``. ``ome_xml_builder``
    is an optional callable ``list[Image] -> str`` (defaults to
    ``writers.ome_xml.build_combined_ome_xml``); injecting it keeps the writer
    free of any specific builder import-time dependency.
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

            scene_record = write_scene(
                reader,
                scene_index=f.scene_index,
                store_path=fov_store,
                pyramid_min_size=pyramid_min_size,
                chunk_shape=chunk_shape,
                channels=channels,
                image_name=image_name,
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
