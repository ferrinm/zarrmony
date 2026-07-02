"""Top-level convert() and inspect() — orchestrate readers, writers, audit."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import fsspec
from bioio_ome_zarr.writers import Channel
from ome_types import OME
from ome_types.model import Image, Pixels, PixelType

from zarrmony import _validate
from zarrmony._storage import size_on_disk
from zarrmony.audit import build_audit_record, write_audit_record
from zarrmony.errors import (
    ExtractorWarning,
    LayoutDowngradeWarning,
    LayoutMismatchError,
    MosaicMergedSiblingWarning,
    MosaicPlacementWarning,
    OutputExistsError,
    ValidationWarning,
    ZarrmonyError,
)
from zarrmony.metadata._lif_scene import find_scene_xml
from zarrmony.metadata.channel_colors import colors_for_channels
from zarrmony.metadata.lif_channels import (
    channels_to_ome_channels,
    channels_to_omero,
    extract_channels,
)
from zarrmony.metadata.lif_tiles import (
    compute_stage_placements,
    extract_tile_layout,
    grid_shape,
    reassemble_grid,
    reassemble_stage,
    select_auto_stitch_cascade,
    stage_overlap_discrepancy,
)
from zarrmony.naming import resolve_scene_dirnames, sanitize_scene_name
from zarrmony.readers.default import derive_bioio_distribution
from zarrmony.readers.plugin import ReaderPlugin, get_reader
from zarrmony.writers.bf2raw import write_bf2raw_wrapper
from zarrmony.writers.ome_xml import (
    attach_stage_position_plane,
    build_combined_ome_xml,
    build_ome_xml_for_scene,
)
from zarrmony.writers.per_scene import write_per_scene_metadata
from zarrmony.writers.plate import summarize_plate_layout, write_plate
from zarrmony.writers.scene import write_scene

Layout = Literal["auto", "per-scene", "bf2raw", "plate"]
ResolvedLayout = Literal["per-scene", "bf2raw", "plate"]
_VALID_LAYOUTS: tuple[Layout, ...] = ("auto", "per-scene", "bf2raw", "plate")

LifMosaic = Literal[
    "auto-stitch", "per-tile", "grid-stitch", "stage-stitch", "bioio-lif"
]
_VALID_LIF_MOSAIC: tuple[LifMosaic, ...] = (
    "auto-stitch",
    "per-tile",
    "grid-stitch",
    "stage-stitch",
    "bioio-lif",
)

# The concrete stitcher the cascade dispatches to under lif_mosaic="auto-stitch".
# "per-tile" is not a cascade output — users request it explicitly because it
# changes the on-disk shape (N sub-stores per scene, not one canvas).
_CascadeChoice = Literal["stage-stitch", "grid-stitch", "bioio-lif"]

# Meters → micrometers (OME convention for <Plane> PositionX/Y/Z units).
_METERS_TO_UM = 1_000_000.0

# Grid-stitch produces a butt-jointed canvas — no inter-tile overlap. Recorded
# in the audit alongside intended_overlap_*_pct so downstream consumers can
# diagnose missing-pixel-at-seam risk when the acquisition intended overlap.
_STITCHER_GRID_NAME = "zarrmony-grid"
_GRID_OVERLAP_ASSUMPTION_PX = 0

# Stage-stitch honours the acquisition's true intended overlap by placing each
# tile at its stage µm position (converted through the scene's pixel size).
# Recorded in the audit alongside tile_pixel_offsets + observed_overlap_pct so
# downstream consumers can compare the placement to the LIF-declared intent.
_STITCHER_STAGE_NAME = "zarrmony-stage"


def _resolve_layout(
    layout: Layout, reader: Any, plugin: ReaderPlugin
) -> ResolvedLayout:
    """Resolve a user-supplied ``layout`` against the reader's ``layout_hint``.

    Implements the ADR-0004 dispatch matrix:

    - ``auto`` + flat reader → ``per-scene``
    - ``auto`` + plate reader → ``plate``
    - ``per-scene`` / ``bf2raw`` + plate reader → that flat layout, with a
      :class:`LayoutDowngradeWarning` (plate metadata is dropped)
    - ``plate`` + flat reader → :class:`LayoutMismatchError`
    """
    layout_hint = getattr(reader, "layout_hint", "flat")
    if layout == "auto":
        return "plate" if layout_hint == "plate" else "per-scene"
    if layout == "plate" and layout_hint != "plate":
        raise LayoutMismatchError(
            f"layout='plate' requires a plate-shaped reader, but reader plugin "
            f"{plugin.name!r} reports layout_hint={layout_hint!r}. Use "
            f"layout='auto' (the default) to let zarrmony pick the matching "
            f"writer, or pick layout='per-scene' / 'bf2raw' explicitly."
        )
    if layout in ("per-scene", "bf2raw") and layout_hint == "plate":
        warnings.warn(
            f"reader plugin {plugin.name!r} is plate-shaped (layout_hint='plate') "
            f"but layout={layout!r} was requested explicitly; plate metadata "
            f"(rows, columns, wells, acquisitions) will be dropped from the "
            f"output. Use layout='auto' (the default) to keep plate structure.",
            LayoutDowngradeWarning,
            stacklevel=3,
        )
    return layout  # type: ignore[return-value]


def _check_output(output: str | Path, *, force: bool) -> None:
    """Refuse-or-clobber a single output path (file, dir, or store)."""
    s = str(output)
    fs, path = fsspec.core.url_to_fs(s)
    if not fs.exists(path):
        return
    if not force:
        raise OutputExistsError(
            f"output already exists: {s} (pass force=True to overwrite)"
        )
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


def _serialize_source_metadata(reader_metadata: Any) -> str | None:
    if reader_metadata is None:
        return None
    if isinstance(reader_metadata, str):
        return reader_metadata
    if isinstance(reader_metadata, ET.Element):
        return ET.tostring(reader_metadata, encoding="unicode")
    if isinstance(reader_metadata, OME):
        return reader_metadata.to_xml()
    return str(reader_metadata)


def _stub_image(scene_index: int, name: str, scene_record: dict) -> Image:
    """Minimal OME Image when reader.ome_metadata fails — sizes from scene_record."""
    dims = scene_record["dims"]
    shape = scene_record["level_shapes"][0]
    size_map = dict(zip(dims, shape, strict=True))
    return Image(
        id=f"Image:{scene_index}",
        name=name,
        pixels=Pixels(
            id=f"Pixels:{scene_index}",
            size_x=size_map.get("X", 1),
            size_y=size_map.get("Y", 1),
            size_z=size_map.get("Z", 1),
            size_c=size_map.get("C", 1),
            size_t=size_map.get("T", 1),
            dimension_order="XYZCT",
            type=PixelType.UINT16,
        ),
    )


def _try_get_ome_image(
    reader: Any, scene_index: int
) -> tuple[Image | None, dict | None]:
    try:
        ome = reader.ome_metadata
    except Exception as e:  # noqa: BLE001 — bioio extractors raise heterogeneous errors
        return None, {
            "scene_index": scene_index,
            "field": "ome_metadata",
            "error": f"{type(e).__name__}: {e}",
        }
    if ome is None or not getattr(ome, "images", None):
        return None, {
            "scene_index": scene_index,
            "field": "ome_metadata",
            "error": "reader.ome_metadata returned None or no images",
        }
    return ome.images[0], None


def _channels_for_scene(
    reader: Any,
    channel_colors: dict[str, str] | None,
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


def _scene_channel_count(reader: Any) -> int:
    """The current scene's C size, from the reader's public metadata surface.

    Prefers ``channel_names`` — a side-effect-free read that bioio-lif populates
    per scene — and falls back to the xarray dims read only when it's missing
    or empty. The dims read is fine for non-LIF readers but on a
    :class:`~zarrmony.readers.lif._MosaicAwareLifReader` mosaic scene it
    triggers :class:`~zarrmony.errors.MosaicStitchingWarning` for every call,
    which would fire once per mosaic scene during :func:`_lif_scene_channels`
    dispatch — even when the writer eventually routes through a
    reassembly path (stage-stitch, grid-stitch, per-tile) that never touches
    the bioio-lif stitcher. A scene with neither ``channel_names`` nor a ``C``
    dim has one implicit channel.
    """
    names = getattr(reader, "channel_names", None) or []
    if names:
        return len(names)
    xarr = reader.xarray_dask_data
    return int(xarr.sizes["C"]) if "C" in xarr.dims else 1


def _lif_scene_channels(reader: Any) -> tuple[list[dict] | None, list[Channel] | None]:
    """The LIF-vs-other decision, in one place.

    Returns ``(extracted, omero_channels)`` when ``reader`` is a LIF reader whose
    current scene yields channel identities AND that identity count matches the
    scene's C size; otherwise ``(None, None)``. ``extracted`` is the raw
    per-channel identity dicts (handed to :func:`_lif_ome_image` once scene sizes
    are known); ``omero_channels`` is the display label/color list for
    :func:`write_scene`. Both projections come from the same vetted extraction,
    so omero and the OME-XML never disagree.

    Locating the scene XML is two-tier (see :func:`find_scene_xml`):

    1. ``scene_root`` — bioio-lif's plate row/column locator. It works for plate
       wells but RAISES for ordinary non-plate **confocal** scenes, so it is only
       a *fast path* that preserves existing plate behavior.
    2. the document-order ``<Image>`` locator (``.//Image[current_scene_index]``,
       per bioio-lif PR #52) — the fallback that makes confocal scenes work.

    The count check matters: the omero block and OME-XML ``Pixels/@SizeC`` must
    describe the channels the array actually has. If extraction and data
    disagree in EITHER direction — too few or too many identities (corruption,
    partial metadata) — we decline the projection rather than mislabel the image.

    Fail-safe: any failure — not a LIF reader, no/garbage scene XML, an empty or
    count-mismatched extraction, or any unexpected reader-surface error — returns
    ``(None, None)`` so callers fall back cleanly to the existing name-based
    path. Metadata never crashes a conversion.
    """
    try:
        scene_xml = find_scene_xml(reader)
        if scene_xml is None:
            return None, None
        extracted = extract_channels(scene_xml)
        if not extracted or len(extracted) != _scene_channel_count(reader):
            return None, None
        omero_channels = [
            Channel(label=o["label"], color=o["color"])
            for o in channels_to_omero(extracted)
        ]
        return extracted, omero_channels
    except Exception:  # noqa: BLE001 — never break a conversion over metadata
        return None, None


def _lif_ome_image(
    extracted: list[dict], scene_index: int, name: str, scene_record: dict
) -> Image | None:
    """A real per-scene OME ``Image`` carrying the extracted channel identities.

    Sizes come from ``scene_record`` (exactly as :func:`_stub_image` does); the
    canonical ``<Channel>`` elements come from :func:`channels_to_ome_channels`.
    ``extracted`` was already count-checked against the scene's C size in
    :func:`_lif_scene_channels`; the ``SizeC`` assertion here is belt-and-
    suspenders. Returns ``None`` on any surprise so the caller falls back to the
    stub Image (with name-based omero, the existing behavior).
    """
    try:
        image = _stub_image(scene_index, name, scene_record)
        ome_channels = channels_to_ome_channels(extracted)
        if len(ome_channels) != image.pixels.size_c:
            return None
        image.pixels.channels = ome_channels
        return image
    except Exception:  # noqa: BLE001 — never break a conversion over metadata
        return None


def _source_xml_filename(input_path: str | Path) -> str:
    ext = Path(str(input_path)).suffix.lstrip(".").lower()
    return f"raw.{ext}.xml" if ext else "raw.xml"


def _run_validation(
    store_path: str | Path,
    layout: _validate.ResolvedLayout,
    validate: bool,
) -> list[dict[str, Any]]:
    """Validate ``store_path`` if requested and the validator is installed.

    Returns the list of findings (empty on success). Each finding is also
    surfaced as a :class:`ValidationWarning` so users see it on stderr; the
    caller threads the list into the audit record's ``validation_warnings``.
    """
    if not validate:
        return []
    if not _validate.is_available():
        warnings.warn(
            "validate=True but the ome-zarr-models extra is not installed; "
            "skipping post-conversion OME-NGFF validation. "
            "Install with `pip install zarrmony[validate]` to enable.",
            ValidationWarning,
            stacklevel=3,
        )
        return []
    findings = _validate.validate_store(store_path, layout)
    for f in findings:
        warnings.warn(
            f"OME-NGFF validation: {f['kind']} at {f.get('path', store_path)}: {f['error']}",
            ValidationWarning,
            stacklevel=3,
        )
    return findings


def _resolve_distribution(reader: Any, plugin: ReaderPlugin) -> str | None:
    """Return the effective bioio distribution name for the audit record.

    For format-specific plugins (CZI/LIF/ND2) ``plugin.distribution`` is set.
    For the catch-all default plugin it is ``None``; introspect the opened
    ``BioImage`` to recover the actual sub-package (e.g. ``bioio-ome-tiff``).
    """
    return (
        plugin.distribution
        if plugin.distribution
        else derive_bioio_distribution(reader)
    )


def convert(
    input_path: str | Path,
    output: str | Path,
    *,
    layout: Layout = "auto",
    pyramid_min_size: int = 256,
    chunk_shape: Sequence[int] | None = None,
    channel_colors: dict[str, str] | None = None,
    force: bool = False,
    checksum: bool = False,
    validate: bool = True,
    lif_mosaic: LifMosaic = "auto-stitch",
) -> dict:
    """Convert ``input_path`` to OME-Zarr v0.5.

    By default (``layout='auto'``), the writer is chosen from the reader's
    ``layout_hint``: a flat reader writes ``per-scene`` (one self-describing
    ``<sanitized_scene_name>.ome.zarr`` per scene under ``output``); a
    plate-shaped reader writes a single OME-NGFF HCS plate store at
    ``output``. Explicit overrides honor user intent but warn or error per
    ADR-0004 (forcing ``per-scene`` / ``bf2raw`` against a plate reader emits
    :class:`~zarrmony.errors.LayoutDowngradeWarning`; forcing ``plate``
    against a flat reader raises :class:`~zarrmony.errors.LayoutMismatchError`).

    ``lif_mosaic`` (LIF-specific, default ``"auto-stitch"``, ADR-0005 + #39)
    governs how mosaic LIF scenes without a vendor ``_Merged`` sibling are
    written:

    - ``"auto-stitch"`` (default) — preserves today's behavior: bioio-lif's
      1-pixel-overlap auto-stitcher writes one image per scene, with a
      :class:`~zarrmony.errors.MosaicStitchingWarning`.
    - ``"per-tile"`` — writes one OME-Zarr image per tile under
      ``<output>/<sanitized_scene>/tile_X{f:02d}Y{f:02d}.ome.zarr/``, each
      carrying its stage origin in ``<Plane>`` ``PositionX/Y/Z`` so external
      stitchers (ASHLAR, m2stitch, BigStitcher) can re-stitch correctly.
      Incompatible with ``layout="plate"`` — raises
      :class:`~zarrmony.errors.LayoutMismatchError`.
    - ``"grid-stitch"`` — reassembles a single canvas per scene by placing
      tile M=i at ``(field_y[i]*tile_H, field_x[i]*tile_W)`` from the LIF
      ``FieldX``/``FieldY`` indices (butt joints, no overlap). Fixes
      bioio-lif's M-scan-order placement bug while preserving the
      one-store-per-scene invariant. Strict: raises :class:`ValueError` with
      a clear message pointing at ``"per-tile"`` when tile metadata is
      incomplete. Composes with ``layout="plate"``.
    - ``"stage-stitch"`` — reassembles a single canvas per scene by placing
      each tile at its ``PosX``/``PosY`` stage µm position (converted to
      pixels via the scene's physical pixel size). Honours the LIF-declared
      intended overlap instead of grid-stitch's butt joints; later-placed
      tiles overwrite earlier tiles in overlap regions (deterministic — M
      order — no blending in this slice). Strict: raises :class:`ValueError`
      naming what's missing (per-tile ``PosX``/``PosY`` or scene physical
      pixel size) with ``"grid-stitch"`` named as the graceful escape. When
      the observed vs LIF-declared overlap differs by >20% on either axis,
      emits :class:`~zarrmony.errors.MosaicPlacementWarning` (placement
      proceeds — helps catch pixel-size / unit-conversion bugs).

    Non-LIF readers ignore the flag entirely.

    Per-scene return shape: ``{"input": ..., "stores": [<per-store audit>, ...]}``.
    The ``bf2raw`` and ``plate`` shapes return the single bundle's audit dict;
    plate audits use the schema-3 ``fields`` + ``plate`` keys.
    """
    if layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"layout must be one of {list(_VALID_LAYOUTS)} (got {layout!r})"
        )
    if lif_mosaic not in _VALID_LIF_MOSAIC:
        raise ValueError(
            f"lif_mosaic must be one of {list(_VALID_LIF_MOSAIC)} (got {lif_mosaic!r})"
        )

    reader, plugin, match_score = get_reader(input_path)
    if not reader.scenes:
        raise ZarrmonyError(f"reader returned no scenes for {input_path!s}")
    distribution = _resolve_distribution(reader, plugin)

    effective_layout = _resolve_layout(layout, reader, plugin)

    # Plate + per-tile is incompatible: a plate FOV is one image by spec
    # (ADR-0004). Reject before any pixels are written so users get a clean
    # signal to convert as flat to get per-tile stores (ADR-0005).
    if effective_layout == "plate" and lif_mosaic == "per-tile":
        raise LayoutMismatchError(
            "per-tile output is incompatible with plate layout — a plate FOV "
            "is one image by spec; convert as flat (layout='per-scene' or "
            "layout='auto' with a flat reader) to get per-tile stores"
        )
    # Plate + explicit stage-stitch is not yet wired end-to-end (the plate
    # writer only knows how to swap in grid-stitch's reassembly today). Reject
    # explicitly so users don't silently get bioio-lif auto-stitched plate
    # FOVs when they asked for stage-based placement; convert flat to get
    # a stage-stitched canvas per scene, or use grid-stitch under plate. The
    # auto-stitch cascade already knows to skip stage-stitch under plate mode
    # and land on grid-stitch instead, so this rejection is scoped to the
    # explicit request only.
    if effective_layout == "plate" and lif_mosaic == "stage-stitch":
        raise LayoutMismatchError(
            "stage-stitch is not yet supported under plate layout; convert "
            "flat (layout='per-scene' or layout='auto' with a flat reader) "
            "to get stage-based placement, or pass lif_mosaic='grid-stitch' "
            "to keep the plate structure with butt-jointed FOVs."
        )

    config = {
        "layout": effective_layout,
        "pyramid_min_size": pyramid_min_size,
        "chunk_shape": list(chunk_shape) if chunk_shape else None,
        "channel_colors": dict(channel_colors) if channel_colors else None,
        "force": force,
        "checksum": checksum,
        "validate": validate,
        "lif_mosaic": lif_mosaic,
    }

    if effective_layout == "plate":
        return _convert_plate(
            reader=reader,
            plugin=plugin,
            match_score=match_score,
            distribution=distribution,
            input_path=input_path,
            output=output,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channel_colors=channel_colors,
            force=force,
            checksum=checksum,
            config=config,
            validate=validate,
            lif_mosaic=lif_mosaic,
        )
    if effective_layout == "bf2raw":
        return _convert_bf2raw(
            reader=reader,
            plugin=plugin,
            match_score=match_score,
            distribution=distribution,
            input_path=input_path,
            output=output,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channel_colors=channel_colors,
            force=force,
            checksum=checksum,
            config=config,
            validate=validate,
        )
    return _convert_per_scene(
        reader=reader,
        plugin=plugin,
        match_score=match_score,
        distribution=distribution,
        input_path=input_path,
        output=output,
        pyramid_min_size=pyramid_min_size,
        chunk_shape=chunk_shape,
        channel_colors=channel_colors,
        force=force,
        checksum=checksum,
        config=config,
        validate=validate,
        lif_mosaic=lif_mosaic,
    )


def _tile_shape_from_reader(reader: Any) -> dict[str, int] | None:
    """LIF per-tile Y/X pixel dims from the reader's ``mosaic_tile_dims``, or None.

    Defensive: not every reader plugin exposes ``mosaic_tile_dims`` (only the
    LIF plugin does). Returns ``None`` when absent so the grid-stitch audit
    block simply omits ``tile_shape`` rather than crashing.
    """
    tile_dims = getattr(reader, "mosaic_tile_dims", None)
    if tile_dims is None:
        return None
    y = getattr(tile_dims, "Y", None)
    x = getattr(tile_dims, "X", None)
    if y is None or x is None:
        return None
    return {"Y": int(y), "X": int(x)}


def _stage_pixel_sizes_um(reader: Any) -> tuple[float | None, float | None]:
    """Read the current scene's Y/X physical pixel size in µm from the reader.

    Returns ``(y_um, x_um)``; either may be ``None`` when the reader can't
    report it. Callers wanting a hard error on missing values (stage-stitch)
    let :func:`compute_stage_placements` do the raising — that keeps the error
    text co-located with the escape hint pointing at ``lif_mosaic="grid-stitch"``.
    """
    px = getattr(reader, "physical_pixel_sizes", None)
    if px is None:
        return None, None
    y = getattr(px, "Y", None)
    x = getattr(px, "X", None)
    return (float(y) if y is not None else None, float(x) if x is not None else None)


def _stage_stitch_mosaic_summary(
    tile_layout: dict | None,
    *,
    tile_count: int,
    tile_shape_yx: dict[str, int] | None,
    tile_pixel_offsets: list[tuple[int, int]],
    observed_overlap_pct: dict[str, float | None],
) -> dict:
    """Audit's ``mosaic`` block for a stage-stitched scene.

    Records ``stitcher="zarrmony-stage"`` and the placement-specific fields
    ``tile_pixel_offsets`` + ``observed_overlap_pct``. Merges in the extracted
    tile positions and LIF-declared intended overlaps so downstream consumers
    can compare observed vs intended without re-parsing the source XML.
    """
    summary: dict = {
        "stitched": True,
        "stitcher": _STITCHER_STAGE_NAME,
        "tile_count": tile_count,
        "tile_pixel_offsets": [
            {"m_index": i, "y_px": int(oy), "x_px": int(ox)}
            for i, (oy, ox) in enumerate(tile_pixel_offsets)
        ],
        "observed_overlap_pct": {
            "x": observed_overlap_pct.get("x"),
            "y": observed_overlap_pct.get("y"),
        },
    }
    if tile_shape_yx is not None:
        summary["tile_shape"] = tile_shape_yx
    if tile_layout is not None:
        summary.update(tile_layout)
    return summary


def _run_stage_stitch(
    reader: Any,
    scene_index: int,
    scene_name: str,
) -> tuple[Any, dict]:
    """Reassemble the current mosaic scene via stage-µm placement.

    Returns ``(canvas_xarr, mosaic_summary)`` — the caller passes the canvas to
    :func:`write_scene` via ``xarr_override`` and attaches the mosaic summary
    to the scene record. Emits :class:`MosaicPlacementWarning` when observed
    vs intended overlap diverges past the tolerance (see
    :func:`stage_overlap_discrepancy`).

    Raises :class:`ValueError` (from :func:`compute_stage_placements`) when the
    required inputs are missing — per-tile ``PosX``/``PosY`` or scene physical
    pixel size — with ``lif_mosaic="grid-stitch"`` named as the escape hatch.
    """
    tiles_xarr = reader.tiles_xarray_dask_data
    tile_count = int(tiles_xarr.sizes["M"])
    tile_h = int(tiles_xarr.sizes["Y"])
    tile_w = int(tiles_xarr.sizes["X"])
    px_y_um, px_x_um = _stage_pixel_sizes_um(reader)
    scene_xml = find_scene_xml(reader)
    tile_layout = extract_tile_layout(scene_xml) if scene_xml is not None else None
    offsets, canvas_shape_yx, observed = compute_stage_placements(
        tile_layout,
        pixel_size_x_um=px_x_um,
        pixel_size_y_um=px_y_um,
        tile_h=tile_h,
        tile_w=tile_w,
    )
    intended_x = tile_layout.get("intended_overlap_x_pct") if tile_layout else None
    intended_y = tile_layout.get("intended_overlap_y_pct") if tile_layout else None
    discrepancies = stage_overlap_discrepancy(observed, intended_x, intended_y)
    for axis, obs, intended in discrepancies:
        warnings.warn(
            f"scene {scene_index} ({scene_name!r}): stage-stitch observed "
            f"overlap on axis {axis.upper()} is {obs:.2f}%, but LIF metadata "
            f"declares {intended:.2f}% intended overlap "
            f"(|{obs:.2f} - {intended:.2f}| / {intended:.2f} > 20%). "
            f"This usually indicates a pixel-size / unit-conversion bug — the "
            f"placement proceeded anyway but expect visibly wrong seam widths.",
            MosaicPlacementWarning,
            stacklevel=3,
        )
    canvas = reassemble_stage(tiles_xarr, offsets, canvas_shape_yx)
    summary = _stage_stitch_mosaic_summary(
        tile_layout,
        tile_count=tile_count,
        tile_shape_yx=_tile_shape_from_reader(reader),
        tile_pixel_offsets=offsets,
        observed_overlap_pct=observed,
    )
    return canvas, summary


def _grid_stitch_mosaic_summary(
    tile_layout: dict | None,
    *,
    tile_count: int,
    tile_shape_yx: dict[str, int] | None,
) -> dict:
    """Audit's ``mosaic`` block for a grid-stitched scene.

    Records ``stitcher="zarrmony-grid"``, ``overlap_assumption_px=0`` (butt
    joints — pair with ``intended_overlap_*_pct`` to diagnose missing-pixel
    seams), and ``placement_shape`` when the layout exposes contiguous grid
    coverage. Merges in the extracted tile positions + intended overlaps so
    downstream consumers get the same tile metadata surface as the auto-stitch
    path.
    """
    summary: dict = {
        "stitched": True,
        "stitcher": _STITCHER_GRID_NAME,
        "overlap_assumption_px": _GRID_OVERLAP_ASSUMPTION_PX,
        "tile_count": tile_count,
    }
    if tile_shape_yx is not None:
        summary["tile_shape"] = tile_shape_yx
    if tile_layout is not None:
        tiles = tile_layout.get("tiles") or []
        rows, cols = grid_shape(tiles)
        if rows is not None and cols is not None:
            summary["placement_shape"] = {"rows": rows, "cols": cols}
        summary.update(tile_layout)
    return summary


def _convert_per_scene(
    *,
    reader: Any,
    plugin: ReaderPlugin,
    match_score: int | None,
    distribution: str | None,
    input_path: str | Path,
    output: str | Path,
    pyramid_min_size: int,
    chunk_shape: Sequence[int] | None,
    channel_colors: dict[str, str] | None,
    force: bool,
    checksum: bool,
    config: dict,
    validate: bool,
    lif_mosaic: LifMosaic = "auto-stitch",
) -> dict:
    output_str = str(output).rstrip("/")
    dirnames = resolve_scene_dirnames(reader.scenes)
    store_paths = [f"{output_str}/{d}.ome.zarr" for d in dirnames]

    source_xml = _serialize_source_metadata(getattr(reader, "metadata", None))
    source_filename = (
        _source_xml_filename(input_path) if source_xml is not None else None
    )

    store_audits: list[dict] = []

    for scene_index, scene_name in enumerate(reader.scenes):
        reader.set_scene(scene_index)

        skip_reason = getattr(reader, "skip_reason", None)
        if skip_reason:
            warnings.warn(
                f"scene {scene_index} ({scene_name!r}): {skip_reason}",
                MosaicMergedSiblingWarning,
                stacklevel=2,
            )
            continue

        reassembly_eligible = bool(
            getattr(reader, "is_mosaic_reassembly_eligible", lambda: False)()
        )
        if lif_mosaic == "per-tile" and reassembly_eligible:
            tile_audits = _convert_per_tile_scene(
                reader=reader,
                plugin=plugin,
                match_score=match_score,
                distribution=distribution,
                input_path=input_path,
                scene_index=scene_index,
                scene_name=scene_name,
                scene_dirname=dirnames[scene_index],
                output_str=output_str,
                pyramid_min_size=pyramid_min_size,
                chunk_shape=chunk_shape,
                channel_colors=channel_colors,
                force=force,
                checksum=checksum,
                config=config,
                validate=validate,
                source_xml=source_xml,
                source_filename=source_filename,
            )
            store_audits.extend(tile_audits)
            continue

        # Resolve the effective stitcher. Under the cascade default
        # (lif_mosaic="auto-stitch"), per-scene metadata decides which of
        # stage-stitch → grid-stitch → bioio-lif runs; explicit values pass
        # through unchanged. When the scene isn't mosaic-reassembly-eligible
        # (e.g. a _Merged sibling or a scalar scene) there's nothing to
        # cascade and the reader's normal xarray_dask_data path handles it.
        effective_stitcher = lif_mosaic
        cascade_selected = False
        if reassembly_eligible and lif_mosaic == "auto-stitch":
            cascade_tiles_xarr = reader.tiles_xarray_dask_data
            cascade_m_size = int(cascade_tiles_xarr.sizes["M"])
            cascade_scene_xml = find_scene_xml(reader)
            cascade_tile_layout = (
                extract_tile_layout(cascade_scene_xml)
                if cascade_scene_xml is not None
                else None
            )
            cascade_px_y, cascade_px_x = _stage_pixel_sizes_um(reader)
            effective_stitcher = select_auto_stitch_cascade(
                cascade_tile_layout,
                m_size=cascade_m_size,
                pixel_size_x_um=cascade_px_x,
                pixel_size_y_um=cascade_px_y,
                plate_mode=False,
            )
            cascade_selected = True

        grid_stitch_mode = effective_stitcher == "grid-stitch" and reassembly_eligible
        stage_stitch_mode = effective_stitcher == "stage-stitch" and reassembly_eligible
        # Either reassembly mode consumes the raw M-intact tiles surface and
        # replaces the auto-stitch mosaic_summary with a mode-specific block;
        # `write_scene`'s auto-stitch summary would mislabel the record with
        # stitcher="bioio-lif" and the wrong overlap fields in both cases.
        reassembly_mode = grid_stitch_mode or stage_stitch_mode

        store_path = store_paths[scene_index]
        # Per-store refuse-overwrite (don't blow away the whole output dir).
        _check_output(store_path, force=force)

        started_at = datetime.now().astimezone()

        # LIF-vs-other decision in one place: for a LIF scene with extractable
        # channel identities, drive both omero (label/color) and the canonical
        # OME-XML <Channel>s from that identity; otherwise fall through to the
        # existing name-based path. (None, None) for everything non-LIF.
        lif_extracted, lif_omero_channels = _lif_scene_channels(reader)
        channels = (
            lif_omero_channels
            if lif_omero_channels is not None
            else _channels_for_scene(reader, channel_colors)
        )

        # Grid-stitch: consume raw M-intact tiles, reassemble by FieldX/FieldY,
        # pass the corrected canvas into the standard writer path via
        # xarr_override. Suppress the auto-stitch mosaic_summary — it would
        # mislabel the record with stitcher="bioio-lif" and the wrong
        # overlap_assumption_px; we attach a grid-stitch mosaic block below.
        grid_tile_layout: dict | None = None
        grid_tile_count = 0
        stage_mosaic_summary: dict | None = None
        override_xarr = None
        if grid_stitch_mode:
            grid_tiles_xarr = reader.tiles_xarray_dask_data
            grid_tile_count = int(grid_tiles_xarr.sizes["M"])
            scene_xml = find_scene_xml(reader)
            grid_tile_layout = (
                extract_tile_layout(scene_xml) if scene_xml is not None else None
            )
            override_xarr = reassemble_grid(grid_tiles_xarr, grid_tile_layout)
        elif stage_stitch_mode:
            override_xarr, stage_mosaic_summary = _run_stage_stitch(
                reader, scene_index, scene_name
            )

        scene_record = write_scene(
            reader,
            scene_index=scene_index,
            store_path=store_path,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channels=channels,
            image_name=scene_name,
            xarr_override=override_xarr,
            record_mosaic_summary=not reassembly_mode,
        )
        scene_record["store_path"] = store_path
        scene_record["dirname"] = dirnames[scene_index]
        if grid_stitch_mode:
            scene_record["mosaic"] = _grid_stitch_mosaic_summary(
                grid_tile_layout,
                tile_count=grid_tile_count,
                tile_shape_yx=_tile_shape_from_reader(reader),
            )
        elif stage_stitch_mode and stage_mosaic_summary is not None:
            scene_record["mosaic"] = stage_mosaic_summary

        # Distinguish cascade-selected from user-explicit in the audit so
        # downstream analysis can tell "grid-stitch was picked because stage
        # metadata was incomplete" apart from "user explicitly asked for
        # grid-stitch". The mosaic block already carries stitcher=... for the
        # concrete choice; this flag adds the intent.
        if cascade_selected and "mosaic" in scene_record and scene_record["mosaic"]:
            scene_record["mosaic"]["cascade_selected"] = True

        metadata_warnings: list[dict] = []
        ome_image: Image | None = None
        if lif_extracted is not None:
            ome_image = _lif_ome_image(
                lif_extracted, scene_index, scene_name, scene_record
            )
        if ome_image is None:
            # Non-LIF, or the LIF Image couldn't be built — existing behavior.
            ome_image, warning = _try_get_ome_image(reader, scene_index)
            if warning is not None:
                metadata_warnings.append(warning)
                warnings.warn(
                    f"scene {scene_index} ({scene_name}): {warning['error']}",
                    ExtractorWarning,
                    stacklevel=2,
                )
                ome_image = _stub_image(scene_index, scene_name, scene_record)

        ome_xml = build_ome_xml_for_scene(ome_image)
        write_per_scene_metadata(
            store_path,
            ome_xml=ome_xml,
            source_xml=source_xml,
            source_xml_filename=source_filename,
        )

        validation_findings = _run_validation(store_path, "per-scene", validate)

        finished_at = datetime.now().astimezone()
        audit = build_audit_record(
            input_path=input_path,
            reader_plugin=plugin,
            match_score=match_score,
            distribution=distribution,
            config=config,
            started_at=started_at,
            finished_at=finished_at,
            layout="per-scene",
            per_scene=[scene_record],
            metadata_warnings=metadata_warnings,
            checksum=checksum,
        )
        audit["validation_warnings"] = validation_findings
        audit["store_path"] = store_path
        audit["scene_index"] = scene_index
        audit["scene_name"] = scene_name
        write_audit_record(store_path, audit)
        store_audits.append(audit)

    return {
        "input": str(input_path),
        "output": output_str,
        "layout": "per-scene",
        "stores": store_audits,
    }


def _tile_dirname(field_x: int | None, field_y: int | None, fallback_index: int) -> str:
    """Filename stem for a tile sub-store.

    Per ADR-0005: zero-padded grid coordinates from the extracted ``FieldX``/
    ``FieldY``. When the extractor returns ``None`` for either (a partially-
    stamped tile in the LIF XML), fall back to the dense iteration index so two
    tiles can never collide on disk.
    """
    if field_x is None or field_y is None:
        return f"tile_{fallback_index:04d}"
    return f"tile_X{field_x:02d}Y{field_y:02d}"


def _build_tile_stores_index(
    tile_entries: list[dict], tile_count: int, scene_dir_url: str
) -> list[dict]:
    """The ``mosaic.tile_stores`` list: one entry per tile, in tile-iteration order.

    Each entry pairs the LIF grid index + stage position with the on-disk URL
    of that tile's OME-Zarr sub-store. Downstream stitchers read this to
    reconstruct the spatial layout without re-parsing the source LIF.
    """
    entries: list[dict] = []
    for m in range(tile_count):
        tile = tile_entries[m] if m < len(tile_entries) else {}
        field_x = tile.get("field_x")
        field_y = tile.get("field_y")
        dirname = _tile_dirname(field_x, field_y, m)
        entries.append(
            {
                "field_x": field_x,
                "field_y": field_y,
                "store_path": f"{scene_dir_url}/{dirname}.ome.zarr",
                "pos_x_m": tile.get("pos_x_m"),
                "pos_y_m": tile.get("pos_y_m"),
                "pos_z_m": tile.get("pos_z_m"),
            }
        )
    return entries


def _convert_per_tile_scene(
    *,
    reader: Any,
    plugin: ReaderPlugin,
    match_score: int | None,
    distribution: str | None,
    input_path: str | Path,
    scene_index: int,
    scene_name: str,
    scene_dirname: str,
    output_str: str,
    pyramid_min_size: int,
    chunk_shape: Sequence[int] | None,
    channel_colors: dict[str, str] | None,
    force: bool,
    checksum: bool,
    config: dict,
    validate: bool,
    source_xml: str | None,
    source_filename: str | None,
) -> list[dict]:
    """Write one OME-Zarr image per LIF mosaic tile under a scene-named directory.

    Iterates the raw M-intact xarray (``reader.tiles_xarray_dask_data``),
    slices one tile at a time, and dispatches each tile to
    :func:`writers.scene.write_scene` so the per-tile path reuses the single
    pixel-writing implementation maintained for per-scene/plate (the reuse
    rationale matches ADR-0004's per-FOV writer reuse).

    Per-tile OME-XML carries ``<Plane>`` ``PositionX/Y/Z`` (meters → µm) so
    external stitchers can re-stitch from the tile stores alone.
    """
    scene_dir_url = f"{output_str}/{sanitize_scene_name(scene_dirname)}"
    tiles_xarr = reader.tiles_xarray_dask_data
    tile_count = int(tiles_xarr.sizes["M"])

    # Extract tile positions / intended overlap from the scene XML once; the
    # `extract_tile_layout` extractor is fail-closed and may return None (e.g.
    # partial metadata, no <Tile> elements). When it does, tile_stores entries
    # carry None positions — the on-disk shape is preserved.
    scene_xml = find_scene_xml(reader)
    tile_layout = extract_tile_layout(scene_xml) if scene_xml is not None else None
    tile_entries = tile_layout.get("tiles", []) if tile_layout else []

    tile_stores_index = _build_tile_stores_index(
        tile_entries, tile_count, scene_dir_url
    )

    # Per-tile refuse-overwrite, mirroring per-scene mode (don't blow away the
    # entire scene directory, just collide-check each tile sub-store).
    for entry in tile_stores_index:
        _check_output(entry["store_path"], force=force)

    # Channel projection: as in per-scene mode, the LIF extracted identity wins;
    # otherwise fall through to the name-based path. Same channels for every
    # tile of the same scene.
    lif_extracted, lif_omero_channels = _lif_scene_channels(reader)
    channels = (
        lif_omero_channels
        if lif_omero_channels is not None
        else _channels_for_scene(reader, channel_colors)
    )

    tile_audits: list[dict] = []
    for m in range(tile_count):
        started_at = datetime.now().astimezone()
        entry = tile_stores_index[m]
        store_path = entry["store_path"]

        # Slice this tile out of the M-intact xarray. Drop the M index coord
        # so the writer sees clean [T?,C,Z,Y,X] dims (normalize_axes rejects
        # axes outside that set).
        tile_xarr = tiles_xarr.isel(M=m, drop=True)

        tile_image_name = (
            f"{scene_name} {_tile_dirname(entry['field_x'], entry['field_y'], m)}"
        )

        scene_record = write_scene(
            reader,
            scene_index=scene_index,
            store_path=store_path,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channels=channels,
            image_name=tile_image_name,
            xarr_override=tile_xarr,
            # Suppress the scene-level mosaic_summary on the per-tile record:
            # the audit's per-tile discriminator + tile_stores belong on the
            # audit's mosaic block, not duplicated under per_scene[0].mosaic.
            record_mosaic_summary=False,
        )
        scene_record["store_path"] = store_path
        scene_record["tile_index"] = m
        scene_record["field_x"] = entry["field_x"]
        scene_record["field_y"] = entry["field_y"]

        # OME-XML for the tile: start from the standard per-scene Image
        # construction (LIF channel identity wins; stub fallback otherwise),
        # then stamp the tile's stage position into a single <Plane>.
        metadata_warnings: list[dict] = []
        ome_image: Image | None = None
        if lif_extracted is not None:
            ome_image = _lif_ome_image(
                lif_extracted, scene_index, tile_image_name, scene_record
            )
        if ome_image is None:
            ome_image, warning = _try_get_ome_image(reader, scene_index)
            if warning is not None:
                metadata_warnings.append(warning)
                warnings.warn(
                    f"scene {scene_index} ({scene_name}) tile {m}: {warning['error']}",
                    ExtractorWarning,
                    stacklevel=2,
                )
                ome_image = _stub_image(scene_index, tile_image_name, scene_record)
            else:
                # Reuse the OME Image but override its name with the tile-scoped
                # name so each tile's per-store XML names itself, not the scene.
                ome_image.name = tile_image_name

        pos_x_um = (
            entry["pos_x_m"] * _METERS_TO_UM if entry["pos_x_m"] is not None else None
        )
        pos_y_um = (
            entry["pos_y_m"] * _METERS_TO_UM if entry["pos_y_m"] is not None else None
        )
        pos_z_um = (
            entry["pos_z_m"] * _METERS_TO_UM if entry["pos_z_m"] is not None else None
        )
        attach_stage_position_plane(
            ome_image,
            position_x_um=pos_x_um,
            position_y_um=pos_y_um,
            position_z_um=pos_z_um,
        )

        ome_xml = build_ome_xml_for_scene(ome_image)
        write_per_scene_metadata(
            store_path,
            ome_xml=ome_xml,
            source_xml=source_xml,
            source_xml_filename=source_filename,
        )

        validation_findings = _run_validation(store_path, "per-scene", validate)

        finished_at = datetime.now().astimezone()
        audit = build_audit_record(
            input_path=input_path,
            reader_plugin=plugin,
            match_score=match_score,
            distribution=distribution,
            config=config,
            started_at=started_at,
            finished_at=finished_at,
            layout="per-scene",
            per_scene=[scene_record],
            metadata_warnings=metadata_warnings,
            checksum=checksum,
        )
        audit["validation_warnings"] = validation_findings
        audit["store_path"] = store_path
        audit["scene_index"] = scene_index
        audit["scene_name"] = scene_name
        # Per-tile mosaic block: schema-5 `per_tile=true` discriminator + the
        # full tile_stores index (same list on every tile audit so any tile
        # store names its siblings). Stitcher / overlap_assumption_px do NOT
        # apply — the per-tile path skips bioio-lif's stitcher entirely.
        audit["mosaic"] = {
            "per_tile": True,
            "tile_index": m,
            "tile_count": tile_count,
            "tile_stores": tile_stores_index,
        }
        if tile_layout is not None:
            if tile_layout.get("intended_overlap_x_pct") is not None:
                audit["mosaic"]["intended_overlap_x_pct"] = tile_layout[
                    "intended_overlap_x_pct"
                ]
            if tile_layout.get("intended_overlap_y_pct") is not None:
                audit["mosaic"]["intended_overlap_y_pct"] = tile_layout[
                    "intended_overlap_y_pct"
                ]
        write_audit_record(store_path, audit)
        tile_audits.append(audit)

    return tile_audits


def _convert_bf2raw(
    *,
    reader: Any,
    plugin: ReaderPlugin,
    match_score: int | None,
    distribution: str | None,
    input_path: str | Path,
    output: str | Path,
    pyramid_min_size: int,
    chunk_shape: Sequence[int] | None,
    channel_colors: dict[str, str] | None,
    force: bool,
    checksum: bool,
    config: dict,
    validate: bool,
) -> dict:
    started_at = datetime.now().astimezone()

    _check_output(output, force=force)

    output_str = str(output).rstrip("/")
    series_paths: list[str] = []
    images: list[Image] = []
    per_scene_records: list[dict] = []
    metadata_warnings: list[dict] = []

    for scene_index, scene_name in enumerate(reader.scenes):
        scene_path = f"{output_str}/{scene_index}"

        reader.set_scene(scene_index)
        channels = _channels_for_scene(reader, channel_colors)

        scene_record = write_scene(
            reader,
            scene_index=scene_index,
            store_path=scene_path,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channels=channels,
            image_name=scene_name,
        )

        per_scene_records.append(scene_record)
        series_paths.append(str(scene_index))

        ome_image, warning = _try_get_ome_image(reader, scene_index)
        if warning is not None:
            metadata_warnings.append(warning)
            warnings.warn(
                f"scene {scene_index} ({scene_name}): {warning['error']}",
                ExtractorWarning,
                stacklevel=2,
            )
            ome_image = _stub_image(scene_index, scene_name, scene_record)
        images.append(ome_image)

    ome_xml = build_combined_ome_xml(images)
    source_xml = _serialize_source_metadata(getattr(reader, "metadata", None))
    source_filename = (
        _source_xml_filename(input_path) if source_xml is not None else None
    )

    write_bf2raw_wrapper(
        output,
        series_paths=series_paths,
        ome_xml=ome_xml,
        source_xml=source_xml,
        source_xml_filename=source_filename,
    )

    validation_findings = _run_validation(output, "bf2raw", validate)

    finished_at = datetime.now().astimezone()

    audit = build_audit_record(
        input_path=input_path,
        reader_plugin=plugin,
        match_score=match_score,
        distribution=distribution,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        layout="bf2raw",
        per_scene=per_scene_records,
        metadata_warnings=metadata_warnings,
        checksum=checksum,
    )
    audit["validation_warnings"] = validation_findings
    write_audit_record(output, audit)

    return audit


def _convert_plate(
    *,
    reader: Any,
    plugin: ReaderPlugin,
    match_score: int | None,
    distribution: str | None,
    input_path: str | Path,
    output: str | Path,
    pyramid_min_size: int,
    chunk_shape: Sequence[int] | None,
    channel_colors: dict[str, str] | None,
    force: bool,
    checksum: bool,
    config: dict,
    validate: bool,
    lif_mosaic: LifMosaic = "auto-stitch",
) -> dict:
    plate_layout = getattr(reader, "plate_layout", None)
    if plate_layout is None:
        raise ZarrmonyError(
            f"layout='plate' requires reader.plate_layout to be set; got None for {input_path!s}"
        )

    started_at = datetime.now().astimezone()
    _check_output(output, force=force)

    metadata_warnings: list[dict] = []

    def _ome_image_for_field(scene_index: int, scene_record: dict) -> Image:
        ome_image, warning = _try_get_ome_image(reader, scene_index)
        if warning is not None:
            metadata_warnings.append(warning)
            warnings.warn(
                f"scene {scene_index} ({scene_record['scene_name']}): {warning['error']}",
                ExtractorWarning,
                stacklevel=2,
            )
            return _stub_image(scene_index, scene_record["scene_name"], scene_record)
        return ome_image

    source_xml = _serialize_source_metadata(getattr(reader, "metadata", None))
    source_filename = (
        _source_xml_filename(input_path) if source_xml is not None else None
    )

    field_records, plate_attr = write_plate(
        reader,
        store_path=output,
        plate_layout=plate_layout,
        pyramid_min_size=pyramid_min_size,
        chunk_shape=chunk_shape,
        channel_colors=channel_colors,
        ome_image_for_field=_ome_image_for_field,
        ome_xml_builder=build_combined_ome_xml,
        source_xml=source_xml,
        source_xml_filename=source_filename,
        lif_mosaic=lif_mosaic,
    )

    validation_findings = _run_validation(output, "plate", validate)

    finished_at = datetime.now().astimezone()

    audit = build_audit_record(
        input_path=input_path,
        reader_plugin=plugin,
        match_score=match_score,
        distribution=distribution,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        layout="plate",
        fields=field_records,
        plate=plate_attr,
        metadata_warnings=metadata_warnings,
        checksum=checksum,
    )
    audit["validation_warnings"] = validation_findings
    write_audit_record(output, audit)

    return audit


def inspect(input_path: str | Path) -> dict:
    """Return a summary of ``input_path``'s scenes without converting.

    Used for pre-flight inspection before kicking off a slow conversion. The
    ``reader_plugin`` field mirrors the audit record's nested shape.
    """
    reader, plugin, match_score = get_reader(input_path)
    distribution = _resolve_distribution(reader, plugin)
    scenes_info = []
    for i, name in enumerate(reader.scenes):
        reader.set_scene(i)
        xarr = reader.xarray_dask_data
        px = getattr(reader, "physical_pixel_sizes", None)
        scenes_info.append(
            {
                "index": i,
                "name": name,
                "dims": list(xarr.dims),
                "shape": tuple(int(s) for s in xarr.shape),
                "dtype": str(xarr.dtype),
                "channel_names": (
                    list(reader.channel_names)
                    if "C" in xarr.dims and getattr(reader, "channel_names", None)
                    else []
                ),
                "physical_pixel_sizes": (
                    {
                        "Z": getattr(px, "Z", None),
                        "Y": getattr(px, "Y", None),
                        "X": getattr(px, "X", None),
                    }
                    if px is not None
                    else None
                ),
            }
        )
    info: dict[str, Any] = {
        "input_path": str(input_path),
        "size_bytes": size_on_disk(input_path),
        "reader_plugin": {
            "name": plugin.name,
            "source": plugin.source,
            "distribution": distribution,
            "match_score": match_score,
        },
        "n_scenes": len(reader.scenes),
        "scenes": scenes_info,
    }
    plate_layout = getattr(reader, "plate_layout", None)
    if getattr(reader, "layout_hint", "flat") == "plate" and plate_layout is not None:
        info["plate_layout"] = summarize_plate_layout(plate_layout)
    return info
