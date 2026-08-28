"""Top-level convert() and inspect() — orchestrate readers, writers, audit."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import fsspec
import numpy as np
from bioio_ome_zarr.writers import Channel
from ome_types import OME
from ome_types.model import Image, Pixels, PixelType

from zarrmony import _validate
from zarrmony._storage import format_bytes, size_on_disk
from zarrmony.audit import build_audit_record, write_audit_record
from zarrmony.errors import (
    ExtractorWarning,
    LayoutDowngradeWarning,
    LayoutMismatchError,
    MosaicMergedSiblingWarning,
    MosaicPlacementWarning,
    OutputExistsError,
    PlateSelectionError,
    ReaderKwargError,
    TileAlignmentWarning,
    ValidationWarning,
    ZarrmonyError,
)
from zarrmony.geometry import Geometry, plan_reader_tile_size, resolve_geometry
from zarrmony.metadata._lif_scene import find_scene_xml
from zarrmony.metadata.acquisition import extract_acquisition
from zarrmony.metadata.audit_channels import from_lif_extracted, from_ome_channels
from zarrmony.metadata.channel_colors import colors_for_channels
from zarrmony.metadata.czi_acquisition import extract_czi_acquisition
from zarrmony.metadata.lif_channels import (
    channels_to_ome_channels,
    channels_to_omero,
    extract_channels,
    resolve_channel_colors,
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
from zarrmony.metadata.nd2_acquisition import extract_nd2_acquisition
from zarrmony.metadata.objective import extract_objective
from zarrmony.metadata.ome_extractors import (
    extract_acquisition_from_ome,
    extract_objective_from_ome,
)
from zarrmony.naming import resolve_scene_dirnames, sanitize_scene_name
from zarrmony.readers.default import _coerce_bool, derive_bioio_distribution
from zarrmony.readers.plugin import ReaderPlugin, get_reader
from zarrmony.writers.bf2raw import write_bf2raw_wrapper
from zarrmony.writers.ome_xml import (
    attach_objective_to_image,
    attach_stage_position_plane,
    build_combined_ome_xml,
    build_instrument_from_objective,
    build_ome_xml_for_scene,
)
from zarrmony.writers.per_scene import write_per_scene_metadata
from zarrmony.writers.plate import summarize_plate_layout, write_plate
from zarrmony.writers.scene import (
    _dtype_window,
    _physical_scales_for_dims,
    write_scene,
)

Layout = Literal["auto", "per-scene", "bf2raw", "plate"]
ResolvedLayout = Literal["per-scene", "bf2raw", "plate"]
_VALID_LAYOUTS: tuple[Layout, ...] = ("auto", "per-scene", "bf2raw", "plate")

# ADR-0007: opt-in mode for ``channel_colors`` that uses the source file's own
# stored per-channel color (LIF ``<ChannelDescription LUTName>``, OME-XML
# ``<Channel Color>``) instead of the emission-band scheme. Channels without a
# stored color fall through to the emission-band scheme.
_SOURCE_FILE_MODE = "source-file"
ChannelColorSpec = dict[str, str] | Literal["source-file"] | None

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


def _require_plate_selector_if_ambiguous(
    reader: Any, reader_kwargs: dict[str, Any] | None
) -> None:
    """Raise :class:`PlateSelectionError` when the LIF has ≥2 plates and no selector.

    Per ADR-0009 / #82: a multi-plate LIF is a common shape (the user's
    ``PFF-HEK293-seeding-07172026.lif`` packs two 60-well plates in one file).
    We refuse to silently pick one — the caller must select via
    ``--plate NAME`` (CLI) or ``reader_kwargs={"plate": NAME}`` (API). Names
    surface via ``reader.available_plates`` so the error message can list
    them, and the same list is what ``inspect()`` prints under the ``plates``
    block for pre-flight discovery. Non-plate readers and single-plate LIFs
    fall through with no effect.
    """
    available = list(getattr(reader, "available_plates", []) or [])
    if len(available) < 2:
        return
    selector_passed = bool(reader_kwargs and reader_kwargs.get("plate"))
    if selector_passed:
        return
    formatted = ", ".join(repr(n) for n in available)
    raise PlateSelectionError(
        f"multi-plate LIF has {len(available)} plate templates ({formatted}); "
        f"pass --plate NAME (CLI) or reader_kwargs={{'plate': NAME}} (API) "
        f"to select one. One convert() call still produces one plate.zarr — "
        f"run convert once per plate."
    )


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


def _split_channel_colors(
    channel_colors: ChannelColorSpec,
) -> tuple[dict[str, str] | None, bool]:
    """Split the ``channel_colors`` API surface into (overrides, use_source_file).

    ``None`` and dict pass through as ``(dict_or_None, False)``. The literal
    string ``"source-file"`` — the ADR-0007 opt-in that trusts the reader's
    stored per-channel color — becomes ``(None, True)`` so downstream code can
    thread a single boolean flag without re-testing the sentinel.
    """
    if channel_colors == _SOURCE_FILE_MODE:
        return None, True
    return channel_colors, False


def _channels_for_scene(
    reader: Any,
    channel_colors: ChannelColorSpec,
) -> list[Channel] | None:
    """Name-based omero channels for readers that don't surface wavelength.

    ``channel_colors="source-file"`` degrades to the emission-band scheme here
    (the same as ``None``) — this path has no access to a reader-side stored
    color. The LIF path (:func:`_lif_scene_channels`) handles ``"source-file"``
    for real via the extracted ``LUTName``.
    """
    overrides, _use_source_file = _split_channel_colors(channel_colors)
    channel_names = (
        list(reader.channel_names) if getattr(reader, "channel_names", None) else []
    )
    if not channel_names:
        return None
    colors = colors_for_channels(channel_names, overrides=overrides)
    window = _dtype_window(reader.dtype)
    return [
        Channel(label=n, color=c, window=window)
        for n, c in zip(channel_names, colors, strict=True)
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


def _lif_scene_channels(
    reader: Any,
    channel_colors: ChannelColorSpec = None,
) -> tuple[list[dict] | None, list[Channel] | None, dict | None, dict | None]:
    """The LIF-vs-other decision, in one place.

    Returns ``(extracted, omero_channels, objective, acquisition)``:

    * ``extracted`` / ``omero_channels`` — per-channel identity dicts + the
      display label/color list. Both are populated only when ``reader`` is a
      LIF reader whose current scene yields channel identities AND that
      identity count matches the scene's C size; otherwise both are ``None``.
      Both projections come from the same vetted extraction, so omero and the
      OME-XML never disagree.
    * ``objective`` — the LIF objective-lens dict (issue #52), independently
      derived from the same scene XML. Populated whenever
      :func:`~zarrmony.metadata.objective.extract_objective` returns a
      non-empty dict; ``None`` for a non-LIF reader, missing scene XML, or a
      scene with no objective info at all. It carries only the keys the LIF
      actually surfaced — missing individual fields (magnification, NA,
      immersion, model, working distance) are omitted rather than nulled. The
      audit records it under ``per_scene[i].objective`` and the per-scene
      writer projects it into a top-level ``<Instrument><Objective/></Instrument>``.
    * ``acquisition`` — the LIF acquisition/instrument dict (issue #62 /
      ADR-0008), independently derived from the same scene XML. Populated
      whenever :func:`~zarrmony.metadata.acquisition.extract_acquisition`
      returns a non-empty dict. Carries any subset of
      ``{date, microscope, microscope_serial, imaging_method}`` — same
      omit-on-absent rule as ``objective``.

    Objective and acquisition extraction are deliberately independent of the
    channel count check: a LIF scene with a garbled channel list (mismatched
    SizeC) still has a valid objective and acquisition date, and the audit
    shouldn't drop them just because we also can't determine the fluorophores.
    All three extractors are fail-closed.

    Locating the scene XML is two-tier (see :func:`find_scene_xml`):

    1. ``scene_root`` — bioio-lif's plate row/column locator. It works for plate
       wells but RAISES for ordinary non-plate **confocal** scenes, so it is only
       a *fast path* that preserves existing plate behavior.
    2. the document-order ``<Image>`` locator (``.//Image[current_scene_index]``,
       per bioio-lif PR #52) — the fallback that makes confocal scenes work.

    The count check matters for channels: the omero block and OME-XML
    ``Pixels/@SizeC`` must describe the channels the array actually has. If
    extraction and data disagree in EITHER direction — too few or too many
    identities (corruption, partial metadata) — we decline the *channel*
    projection rather than mislabel the image. Objective extraction has no
    such coupling.

    Fail-safe: any failure — not a LIF reader, no/garbage scene XML, or any
    unexpected reader-surface error — returns ``(None, None, None, None)`` so
    callers fall back cleanly to the existing name-based path. Metadata never
    crashes a conversion.
    """
    try:
        scene_xml = find_scene_xml(reader)
        if scene_xml is None:
            return None, None, None, None
        extracted = extract_channels(scene_xml)
        objective = extract_objective(scene_xml)
        acquisition = extract_acquisition(scene_xml)
        if not extracted or len(extracted) != _scene_channel_count(reader):
            # Channel extraction fell through the count check, but the
            # objective + acquisition are decoupled from SizeC — still surface them.
            return None, None, objective, acquisition
        overrides, use_source_file = _split_channel_colors(channel_colors)
        window = _dtype_window(reader.dtype)
        omero_channels = [
            Channel(label=o["label"], color=o["color"], window=window)
            for o in channels_to_omero(
                extracted,
                overrides=overrides,
                use_source_file=use_source_file,
            )
        ]
        return extracted, omero_channels, objective, acquisition
    except Exception:  # noqa: BLE001 — never break a conversion over metadata
        return None, None, None, None


def _audit_objective_for_scene(
    reader: Any,
    lif_objective: dict | None,
    scene_index: int,
) -> dict | None:
    """LIF-extracted objective wins; else project from ``reader.ome_metadata``.

    Covers ADR-0004 (LIF) + ADR-0008 / #63–#65 (CZI, ND2, default). Fail-safe:
    any exception at either tier yields ``None`` so the scene's objective key
    is simply absent.
    """
    if lif_objective:
        return lif_objective
    try:
        ome = reader.ome_metadata
    except Exception:  # noqa: BLE001 — heterogeneous bioio errors
        return None
    return extract_objective_from_ome(ome)


def _reader_acquisition_extras(reader: Any) -> dict | None:
    """Read the soft-optional ``reader.acquisition_audit`` hook (issue #76).

    A reader plugin whose modality is known but not carried by OME's
    ``Channel.AcquisitionMode`` — e.g. a stitched TIFF from a light-sheet rig
    whose source files have no OME AcquisitionMode — can inject fields into
    the acquisition audit block by exposing ``acquisition_audit`` returning a
    dict with any subset of ``{date, microscope, microscope_serial,
    imaging_method}``. Same soft-optional shape as ``layout_hint`` /
    ``channel_names`` / ``ome_metadata`` — accessed via ``getattr`` guarded
    by ``try``/``except`` so a raising hook degrades to no extras.
    """
    try:
        extras = getattr(reader, "acquisition_audit", None)
    except Exception:  # noqa: BLE001 — never crash audit over a reader hook
        return None
    if callable(extras):
        try:
            extras = extras()
        except Exception:  # noqa: BLE001 — never crash audit over a reader hook
            return None
    if not isinstance(extras, dict) or not extras:
        return None
    return extras


def _vendor_acquisition_extras(reader: Any) -> dict | None:
    """Dispatch to a vendor-specific acquisition extractor (issues #77, #78).

    Fills gaps the OME projection can't — bioio-czi emits
    ``Microscope.manufacturer = "Zeiss"`` but leaves the model empty, and
    bioio-nd2 doesn't populate ``<Microscope>`` at all. The relevant strings
    live in the raw CZI XML (``reader.metadata``) and ND2's ``text_info()``
    respectively, so we dispatch by the reader class's module.

    Sits between LIF and OME in the ``setdefault`` layering — vendor
    extractor beats OME (so ``"Zeiss Axioscan 7"`` wins over ``"Zeiss"``),
    but LIF still beats vendor (LIF is a different reader entirely).

    Fail-safe throughout — a raising extractor or a missing module both yield
    no extras.
    """
    module = getattr(type(reader), "__module__", "") or ""
    try:
        if module.startswith("bioio_czi"):
            return extract_czi_acquisition(getattr(reader, "metadata", None))
        if module.startswith("bioio_nd2"):
            return extract_nd2_acquisition(reader)
    except Exception:  # noqa: BLE001 — never crash audit over vendor extraction
        return None
    return None


def _audit_acquisition_for_scene(
    reader: Any,
    lif_acquisition: dict | None,
    scene_index: int,
) -> dict | None:
    """Compose the per-scene acquisition dict from four layered sources.

    Precedence (uniform ``setdefault`` — first source that populates a key
    wins; later sources fill only remaining gaps):

    1. **LIF extractor** (LIF scenes only) — ``lif_acquisition`` from
       :func:`~zarrmony.metadata.acquisition.extract_acquisition`. LIF gets
       first crack because bioio-lif's OME projection is unreliable for
       Leica-native fields (``SystemTypeName``, ``SystemSerialNumber``, the
       ``HardwareSetting`` modality hints).
    2. **Vendor extractor** (CZI / ND2) — :func:`_vendor_acquisition_extras`
       dispatches to :mod:`zarrmony.metadata.czi_acquisition` or
       :mod:`zarrmony.metadata.nd2_acquisition` based on the reader's
       module. Sits above OME because bioio-czi's OME projection emits only
       ``"Zeiss"`` (no model) and bioio-nd2 omits ``<Microscope>``
       entirely — the vendor extractors fill the gap with
       ``"Zeiss Axioscan 7"`` / ``"Nikon Ti2"`` etc. (issues #77, #78).
    3. **OME projection** — :func:`extract_acquisition_from_ome` reads
       ``reader.ome_metadata`` and surfaces ``imaging_method`` from
       per-channel ``<Channel AcquisitionMode>``. For LIF scenes this fills
       gaps the LIF extractor missed; for CZI / ND2 / OME-TIFF this
       populates ``date`` and ``imaging_method`` (and ``microscope`` for
       any file the vendor tier didn't cover).
    4. **Reader hook** — :func:`_reader_acquisition_extras` calls the
       soft-optional ``reader.acquisition_audit`` (issue #76). Reserved for
       cases where a reader knows its modality (SmartSPIM = light_sheet by
       construction) but neither the LIF/vendor extractors nor OME's
       ``AcquisitionMode`` surface produces it. ``setdefault`` semantics
       mean the hook can never override a source-file-derived extraction —
       if OME says ``["confocal"]`` and the hook claims ``["light_sheet"]``,
       OME wins.

    Fail-safe throughout: any exception at any tier yields no contribution
    from that tier; the block is composed from what did succeed. Returns
    ``None`` when all sources came back empty so the audit omits the
    ``acquisition`` key entirely.
    """
    base: dict = dict(lif_acquisition) if lif_acquisition else {}
    vendor_dict = _vendor_acquisition_extras(reader) or {}
    for key, value in vendor_dict.items():
        base.setdefault(key, value)
    try:
        ome = reader.ome_metadata
    except Exception:  # noqa: BLE001 — heterogeneous bioio errors
        ome = None
    ome_dict = extract_acquisition_from_ome(ome, image_index=0) or {}
    for key, value in ome_dict.items():
        base.setdefault(key, value)
    extras = _reader_acquisition_extras(reader) or {}
    for key, value in extras.items():
        base.setdefault(key, value)
    return base or None


def _audit_channels_for_scene(
    reader: Any,
    lif_extracted: list[dict] | None,
    channel_colors: ChannelColorSpec,
) -> list[dict] | None:
    """Project the current scene's channels into the ADR-0008 audit shape (#61).

    Two branches: LIF-extracted identities (from the scene-XML extractor) take
    priority when present, threaded through the same
    :func:`resolve_channel_colors` batch the omero block uses so the audit's
    ``color`` field matches what the writer wrote. Otherwise, project from
    ``reader.ome_metadata.images[0].pixels.channels`` (the surface CZI / ND2 /
    OME-TIFF and other bioio readers already expose) and resolve colors from
    the reader's ``channel_names`` via :func:`colors_for_channels` — same
    method as :func:`_channels_for_scene`.

    Returns ``None`` when no channel identity is extractable at all (missing
    ``ome_metadata``, empty channel list, or reader-side error), so callers
    omit the ``channels`` key entirely for that scene per the ADR's
    omit-not-null rule. Never raises: metadata never breaks a conversion.
    """
    overrides, use_source_file = _split_channel_colors(channel_colors)
    if lif_extracted:
        try:
            colors = resolve_channel_colors(
                lif_extracted,
                use_source_file=use_source_file,
                overrides=overrides,
            )
        except Exception:  # noqa: BLE001 — never crash audit
            colors = None
        return from_lif_extracted(lif_extracted, colors=colors)
    try:
        ome = reader.ome_metadata
    except Exception:  # noqa: BLE001 — heterogeneous bioio errors
        return None
    if ome is None or not getattr(ome, "images", None):
        return None
    pixels = getattr(ome.images[0], "pixels", None)
    ome_channels = list(getattr(pixels, "channels", None) or []) if pixels else []
    if not ome_channels:
        return None
    channel_names = (
        list(reader.channel_names) if getattr(reader, "channel_names", None) else []
    )
    try:
        colors = (
            colors_for_channels(channel_names, overrides=overrides)
            if channel_names
            else None
        )
    except Exception:  # noqa: BLE001 — never crash audit
        colors = None
    return from_ome_channels(ome_channels, colors=colors)


def _lif_ome_image(
    extracted: list[dict],
    scene_index: int,
    name: str,
    scene_record: dict,
    channel_colors: ChannelColorSpec = None,
) -> Image | None:
    """A real per-scene OME ``Image`` carrying the extracted channel identities.

    Sizes come from ``scene_record`` (exactly as :func:`_stub_image` does); the
    canonical ``<Channel>`` elements come from :func:`channels_to_ome_channels`,
    threaded with the same ``channel_colors`` spec as :func:`_lif_scene_channels`
    so the omero block and OME-XML never disagree on per-channel color.
    ``extracted`` was already count-checked against the scene's C size in
    :func:`_lif_scene_channels`; the ``SizeC`` assertion here is belt-and-
    suspenders. Returns ``None`` on any surprise so the caller falls back to the
    stub Image (with name-based omero, the existing behavior).
    """
    try:
        image = _stub_image(scene_index, name, scene_record)
        overrides, use_source_file = _split_channel_colors(channel_colors)
        ome_channels = channels_to_ome_channels(
            extracted,
            overrides=overrides,
            use_source_file=use_source_file,
        )
        if len(ome_channels) != image.pixels.size_c:
            return None
        image.pixels.channels = ome_channels
        return image
    except Exception:  # noqa: BLE001 — never break a conversion over metadata
        return None


def _instrument_for_objective(
    objective: dict | None, image: Image | None, scene_index: int
) -> Any | None:
    """Build the per-image ``<Instrument>`` from ``objective`` and attach refs.

    Returns the ``Instrument`` (for the caller to hand to
    :func:`build_ome_xml_for_scene`) when ``objective`` is non-empty and
    ``image`` is present; otherwise ``None`` — the pre-#52 shape. Scoped IDs
    include ``scene_index`` so a bf2raw / plate consumer parsing a combined
    OME-XML sees distinct instruments per image.

    Never raises: objective is metadata, and metadata never breaks a
    conversion.
    """
    if not objective or image is None:
        return None
    try:
        instrument = build_instrument_from_objective(
            objective,
            instrument_id=f"Instrument:{scene_index}",
            objective_id=f"Objective:{scene_index}:0",
        )
        attach_objective_to_image(image, instrument=instrument)
        return instrument
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


# Backend conventions for "hand me the scene in tiles" and "this big". Both are
# coerced by the default plugin (``readers/default.py``); no other plugin takes
# them, which is why alignment is scoped to that plugin by name.
_TILED_READER_KWARG = "dask_tiles"
_TILE_SIZE_READER_KWARG = "tile_size"
_TILE_ALIGNING_PLUGIN = "bioio"

# The axes the geometry planners speak. A reader reporting a samples axis (#107)
# has it dropped here: S folds into C downstream, and the chunk planner pins C
# to 1, so it cannot move the Y/X answer this feeds.
_PLANNABLE_AXES = frozenset("TCZYX")


def _scene_planning_inputs(
    reader: Any,
) -> tuple[list[str], list[tuple[int, ...]], list[list[float]], list[Any]] | None:
    """Per-scene ``(dims, shapes, spacings, dtypes)`` read from metadata alone.

    ``None`` when the reader does not expose what the planners need, or when its
    scenes disagree about axis order — either way there is no single tile size
    to derive and the caller leaves the reader alone.

    Deliberately never touches ``xarray_dask_data``. Building that graph is the
    expensive thing this alignment exists to stop doing badly, and doing it once
    to decide how to do it again would cost more than the mismatch. Everything
    the planners need — extents, spacings, itemsize — is metadata.

    ``reader.dtype`` is read *inside* the loop, because it is a property of the
    current scene and not of the file: a whole-slide VSI reports ``uint8`` for
    its RGB ``label`` and ``>u2`` for the fluorescence scan beside it. Reading
    it once, off whichever scene the reader opened on, halves or doubles the
    itemsize the chunk planner divides the byte target by.
    """
    dims_order: list[str] | None = None
    shapes: list[tuple[int, ...]] = []
    spacings: list[list[float]] = []
    dtypes: list[Any] = []
    for index in range(len(reader.scenes)):
        try:
            reader.set_scene(index)
            order = [d for d in str(reader.dims.order) if d in _PLANNABLE_AXES]
            sizes = tuple(int(getattr(reader.dims, d)) for d in order)
            scale = _physical_scales_for_dims(order, reader)
            dtype = np.dtype(reader.dtype)
        # ADR-0001 trust model; alignment is optional, so any reader that will
        # not answer these gets left with its own blocking.
        except Exception:  # noqa: BLE001
            return None
        if dims_order is None:
            dims_order = order
        elif order != dims_order:
            return None
        shapes.append(sizes)
        spacings.append(scale)
        dtypes.append(dtype)

    if dims_order is None:
        return None
    return dims_order, shapes, spacings, dtypes


def _align_reader_tiles(
    reader: Any,
    plugin: ReaderPlugin,
    input_path: str | Path,
    reader_kwargs: dict[str, Any] | None,
    geometry: Geometry,
) -> tuple[Any, tuple[int, int] | None]:
    """Reopen ``reader`` asking for tiles that nest in the planned write grid.

    Issue #112. Nothing used to connect the reader's tile size to the geometry
    the planner picks, so ``write_pyramid`` absorbed the mismatch with a
    ``rechunk`` — and on the whole-slide path that rechunk was always a *split*,
    which is the expensive direction by an order of magnitude (see
    :class:`~zarrmony.errors.TileAlignmentWarning`). Planning the grid first and
    asking the reader for blocks that fit it removes the rechunk rather than
    optimising it.

    Returns ``(reader, tile_size)`` — the original reader and ``None`` whenever
    alignment does not apply, so every caller path stays identical to before:

    - a plugin other than the default one, since ``tile_size`` is its convention;
    - ``dask_tiles`` off or absent, where the backend is not tiling at all and
      the kwarg means nothing;
    - a caller who pinned ``tile_size``, whose choice is respected — the writer
      still warns if it turns out to split, and that check runs against the real
      dask blocks rather than a guess about what the backend did with the value;
    - a reader whose metadata will not support planning, or a reopen that fails.

    The reopen costs a second metadata parse. That is a directory scan or an
    index read against a conversion measured in hours, and it buys the whole
    difference between 831,936 dask tasks and 369,600 on the reference scene.
    """
    kwargs = dict(reader_kwargs or {})
    if plugin.name != _TILE_ALIGNING_PLUGIN:
        return reader, None
    if _TILE_SIZE_READER_KWARG in kwargs:
        return reader, None
    try:
        tiled = _coerce_bool(_TILED_READER_KWARG, kwargs.get(_TILED_READER_KWARG))
    except ReaderKwargError:
        # Let the reader's own coercion report it; this is not our error to own.
        return reader, None
    if not tiled:
        return reader, None

    planning = _scene_planning_inputs(reader)
    if planning is None:
        return reader, None
    dims, shapes, spacings, dtypes = planning
    tile = plan_reader_tile_size(shapes, dims, spacings, dtypes, geometry)
    if tile is None:
        return reader, None

    aligned = {**kwargs, _TILE_SIZE_READER_KWARG: tile}
    try:
        return plugin.open(Path(str(input_path)), **aligned), tile
    except Exception as exc:  # noqa: BLE001 — ADR-0001 trust model
        warnings.warn(
            f"could not reopen {input_path!s} with tile_size={tile} to match the "
            f"planned write grid ({type(exc).__name__}: {exc}); continuing with "
            f"the reader's own blocking, which may build a much larger dask graph",
            TileAlignmentWarning,
            stacklevel=3,
        )
        return reader, None


def convert(
    input_path: str | Path,
    output: str | Path,
    *,
    layout: Layout = "auto",
    geometry: Geometry | None = None,
    pyramid_min_size: int | None = None,
    chunk_shape: Sequence[int] | None = None,
    channel_colors: ChannelColorSpec = None,
    contrast_percentile: float | None = 99.9,
    force: bool = False,
    checksum: bool = False,
    validate: bool = True,
    lif_mosaic: LifMosaic = "auto-stitch",
    reader_kwargs: dict[str, Any] | None = None,
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

    ``geometry`` (ADR-0010) is the single frozen
    :class:`~zarrmony.geometry.Geometry` policy carrying every output-geometry
    choice — pyramid depth, the coarse-level bounds that can extend it, chunk
    shape, and the ``downsample_method`` every level above 0 is pooled with
    (``"mean"``, or ``"max"`` for sparse labels). It is resolved once here and
    threaded unchanged through per-scene, bf2raw and plate output, so all three
    layouts are planned by one rule. ``None`` (the default) uses the ADR-0010
    policy.

    ``pyramid_min_size`` and ``chunk_shape`` are retained sugar for the two
    fields that predate the policy object: each is ``None`` when unset and
    otherwise folds into the default ``Geometry``. Passing either alongside an
    explicit ``geometry`` raises :class:`ValueError` — set the field on the
    ``Geometry`` instead of saying it twice. ``chunk_shape`` (however spelled)
    bypasses the world-cubic chunk planner and is applied verbatim to every
    pyramid level; leave it unset to let ``chunk_target_bytes`` drive a
    per-level plan.

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

    ``contrast_percentile`` (issue #53) drives the omero display window's
    ``start``/``end`` fields from actual data instead of the dtype range:

    - ``99.9`` (default) — per-channel ``(min, 99.9th percentile)`` read back
      off the coarsest pyramid level once the store is written, so it touches
      no raw pixel and no more than ``geometry.coarse_max_bytes``. Fixes
      "everything opens black except the brightest pixel" on uint16
      fluorescence stores.
    - ``float`` in ``(0, 100)`` — same shape, but at a different percentile.
    - ``None`` — skip the pass; ``start``/``end`` stay pinned to the dtype
      range (the issue-#50 behavior).

    ``channel_colors`` (ADR-0007) governs per-channel display colors:

    - ``None`` (default) — emission-band colorblind scheme (cyan/green/yellow/
      magenta/white). Emission midpoint → excitation → dye-name substring.
      Residual collisions round-robin through ``UNKNOWN_PALETTE`` with a
      :class:`~zarrmony.errors.ChannelColorCollisionWarning`.
    - ``dict[str, str]`` — per-channel override map (``{name: "RRGGBB"}``).
      Unmapped channels fall through to the emission-band scheme.
    - ``"source-file"`` — use the source file's stored per-channel color
      when present (LIF ``<ChannelDescription LUTName>``, OME-XML
      ``<Channel Color>``); channels without a stored color fall through to
      the emission-band scheme.

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
    if isinstance(channel_colors, str) and channel_colors != _SOURCE_FILE_MODE:
        raise ValueError(
            f"channel_colors as a string must be {_SOURCE_FILE_MODE!r} "
            f"(got {channel_colors!r}); pass a dict for per-channel overrides "
            "or None for the emission-band scheme"
        )
    if contrast_percentile is not None and not (0.0 < contrast_percentile < 100.0):
        raise ValueError(
            f"contrast_percentile must be None or a float in (0, 100) exclusive; "
            f"got {contrast_percentile!r}"
        )
    # Fold the retained sugar into one policy object before anything opens the
    # input file — a bad geometry should fail at the call site, not after a
    # multi-minute read.
    resolved_geometry = resolve_geometry(
        geometry, chunk_shape=chunk_shape, pyramid_min_size=pyramid_min_size
    )

    reader, plugin, match_score = get_reader(input_path, reader_kwargs=reader_kwargs)
    # Multi-plate LIF ambiguity: the reader exposes every plate template via
    # `available_plates` but only auto-resolves `plate_layout` when there's
    # exactly one. Ask the user to pick with `--plate NAME` (API: `plate=`
    # in `reader_kwargs`) rather than silently routing to per-scene or
    # picking one. ADR-0009: one convert() call, one plate — the selector
    # keeps the "one convert = one output path" invariant intact.
    _require_plate_selector_if_ambiguous(reader, reader_kwargs)
    if not reader.scenes:
        raise ZarrmonyError(f"reader returned no scenes for {input_path!s}")
    # ADR-0010 (#112): the write grid is planned from geometry, so the reader
    # has to be told about it rather than left to pick a tile size the writer
    # then rechunks away. Runs after the scenes check because the derivation
    # needs them, and before `_resolve_distribution` so the audit describes the
    # reader that actually produced the pixels.
    reader, reader_tile_size = _align_reader_tiles(
        reader, plugin, input_path, reader_kwargs, resolved_geometry
    )
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

    # Preserve ``channel_colors`` verbatim in the audit config: the caller may
    # have passed a dict, ``None``, or the ADR-0007 ``"source-file"`` sentinel,
    # and downstream audit consumers should be able to distinguish all three.
    # Only defensively copy a dict so a later user mutation doesn't leak into
    # the audit record.
    audit_channel_colors: ChannelColorSpec
    if isinstance(channel_colors, dict):
        audit_channel_colors = dict(channel_colors)
    else:
        audit_channel_colors = channel_colors

    # ADR-0010: record the *resolved* geometry rather than the caller's raw
    # inputs. The old `chunk_shape: None` / `pyramid_min_size: 256` pair was
    # accurate and uninformative — the geometry defect ADR-0010 fixes was found
    # by inspecting a viewport, not the store's own metadata. Per-level shapes
    # live on each scene / field record's `level_shapes`.
    config = {
        "layout": effective_layout,
        "geometry": resolved_geometry.to_audit(),
        # #112: the tile zarrmony derived for the reader, or ``None`` when it
        # left the reader's blocking alone (a caller-pinned tile_size, an
        # untiled backend, a plugin that does not take one). Recorded because
        # "did this run's source blocks nest in its write grid?" is otherwise
        # unanswerable from the store, and it is the first thing to check when
        # a whole-slide conversion is slow.
        "reader_tile_size": list(reader_tile_size) if reader_tile_size else None,
        "channel_colors": audit_channel_colors,
        "contrast_percentile": contrast_percentile,
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
            geometry=resolved_geometry,
            channel_colors=channel_colors,
            contrast_percentile=contrast_percentile,
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
            geometry=resolved_geometry,
            channel_colors=channel_colors,
            contrast_percentile=contrast_percentile,
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
        geometry=resolved_geometry,
        channel_colors=channel_colors,
        contrast_percentile=contrast_percentile,
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
    geometry: Geometry,
    channel_colors: ChannelColorSpec,
    contrast_percentile: float | None,
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
                geometry=geometry,
                channel_colors=channel_colors,
                contrast_percentile=contrast_percentile,
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
        # existing name-based path. (None, None, None) for everything non-LIF.
        # The ``channel_colors`` spec (dict override, source-file mode, or the
        # emission-band default) is threaded into both branches so overrides
        # and source-file color hints apply uniformly. ``lif_objective`` is
        # decoupled from the channel-count check so a scene with a garbled
        # channel list still surfaces its objective.
        (
            lif_extracted,
            lif_omero_channels,
            lif_objective,
            lif_acquisition,
        ) = _lif_scene_channels(reader, channel_colors)
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
            geometry=geometry,
            channels=channels,
            image_name=scene_name,
            xarr_override=override_xarr,
            record_mosaic_summary=not reassembly_mode,
            contrast_percentile=contrast_percentile,
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
                lif_extracted,
                scene_index,
                scene_name,
                scene_record,
                channel_colors=channel_colors,
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

        # ADR-0004 (#52) + ADR-0008 (#63–#65): the objective audit dict comes
        # from the LIF extractor when available, else from
        # `reader.ome_metadata.instruments[0].objectives[0]` for CZI / ND2 /
        # OME-TIFF / default readers. The OME-XML `<Instrument><Objective/>`
        # projection stays LIF-only (`_instrument_for_objective`) — non-LIF
        # readers already carry their objective in `reader.ome_metadata` and
        # the writer's OME-XML round-trips it.
        audit_objective = _audit_objective_for_scene(reader, lif_objective, scene_index)
        if audit_objective:
            scene_record["objective"] = audit_objective
        # ADR-0008 / #62 + #63–#65: acquisition/instrument dict, same
        # LIF-then-OME fallback.
        audit_acquisition = _audit_acquisition_for_scene(
            reader, lif_acquisition, scene_index
        )
        if audit_acquisition:
            scene_record["acquisition"] = audit_acquisition
        instrument = _instrument_for_objective(lif_objective, ome_image, scene_index)

        # ADR-0008 / #61: per-scene channel identity block, uniform across
        # LIF (from the extractor) and non-LIF (from reader.ome_metadata).
        audit_channels = _audit_channels_for_scene(
            reader, lif_extracted, channel_colors
        )
        if audit_channels is not None:
            scene_record["channels"] = audit_channels

        ome_xml = build_ome_xml_for_scene(ome_image, instrument)
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
    geometry: Geometry,
    channel_colors: ChannelColorSpec,
    contrast_percentile: float | None,
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
    # tile of the same scene. ``channel_colors`` threads to both branches so
    # dict overrides and ``"source-file"`` apply per-tile as they do per-scene.
    # Objective is scene-scoped (one objective per scene, shared by all tiles)
    # and re-attached to each tile's OME-XML below.
    (
        lif_extracted,
        lif_omero_channels,
        lif_objective,
        lif_acquisition,
    ) = _lif_scene_channels(reader, channel_colors)
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
            geometry=geometry,
            channels=channels,
            image_name=tile_image_name,
            xarr_override=tile_xarr,
            # Suppress the scene-level mosaic_summary on the per-tile record:
            # the audit's per-tile discriminator + tile_stores belong on the
            # audit's mosaic block, not duplicated under per_scene[0].mosaic.
            record_mosaic_summary=False,
            contrast_percentile=contrast_percentile,
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
                lif_extracted,
                scene_index,
                tile_image_name,
                scene_record,
                channel_colors=channel_colors,
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

        # Per-tile is LIF-only in practice, but keep the same LIF-then-OME
        # fallback so any future non-LIF per-tile use has the audit surface.
        audit_objective = _audit_objective_for_scene(reader, lif_objective, scene_index)
        if audit_objective:
            scene_record["objective"] = audit_objective
        audit_acquisition = _audit_acquisition_for_scene(
            reader, lif_acquisition, scene_index
        )
        if audit_acquisition:
            scene_record["acquisition"] = audit_acquisition
        instrument = _instrument_for_objective(lif_objective, ome_image, scene_index)

        # ADR-0008 / #61: channel identity is scene-scoped; each tile store
        # carries its own copy so a stray tile remains fully self-describing.
        audit_channels = _audit_channels_for_scene(
            reader, lif_extracted, channel_colors
        )
        if audit_channels is not None:
            scene_record["channels"] = audit_channels

        ome_xml = build_ome_xml_for_scene(ome_image, instrument)
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
    geometry: Geometry,
    channel_colors: ChannelColorSpec,
    contrast_percentile: float | None,
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
            geometry=geometry,
            channels=channels,
            image_name=scene_name,
            contrast_percentile=contrast_percentile,
        )

        # ADR-0008 / #61: per-scene channel identity block. bf2raw takes the
        # non-LIF path unconditionally — the LIF plugin routes through
        # per-scene, never bf2raw — so no lif_extracted to thread through.
        audit_channels = _audit_channels_for_scene(reader, None, channel_colors)
        if audit_channels is not None:
            scene_record["channels"] = audit_channels
        # ADR-0008 / #63–#65: objective + acquisition from `reader.ome_metadata`.
        audit_objective = _audit_objective_for_scene(reader, None, scene_index)
        if audit_objective:
            scene_record["objective"] = audit_objective
        audit_acquisition = _audit_acquisition_for_scene(reader, None, scene_index)
        if audit_acquisition:
            scene_record["acquisition"] = audit_acquisition

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
    geometry: Geometry,
    channel_colors: ChannelColorSpec,
    contrast_percentile: float | None,
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

    def _audit_channels_for_field(scene_index: int) -> list[dict] | None:
        # ADR-0008 / #61: audit channels per FOV. Reader is on the current
        # scene already (write_plate calls set_scene before invoking the
        # per-FOV callbacks); we thread lif_extracted for the LIF-plate case
        # so the audit's colors match what the writer wrote.
        lif_extracted, _, _, _ = _lif_scene_channels(reader, channel_colors)
        return _audit_channels_for_scene(reader, lif_extracted, channel_colors)

    def _audit_objective_for_field(scene_index: int) -> dict | None:
        # ADR-0004 (LIF) + ADR-0008 / #63–#65 (CZI/ND2/default) objective.
        _, _, lif_objective, _ = _lif_scene_channels(reader, channel_colors)
        return _audit_objective_for_scene(reader, lif_objective, scene_index)

    def _audit_acquisition_for_field(scene_index: int) -> dict | None:
        # ADR-0008 / #62 (LIF) + #63–#65 (CZI/ND2/default) acquisition.
        _, _, _, lif_acquisition = _lif_scene_channels(reader, channel_colors)
        return _audit_acquisition_for_scene(reader, lif_acquisition, scene_index)

    source_xml = _serialize_source_metadata(getattr(reader, "metadata", None))
    source_filename = (
        _source_xml_filename(input_path) if source_xml is not None else None
    )

    field_records, plate_attr = write_plate(
        reader,
        store_path=output,
        plate_layout=plate_layout,
        geometry=geometry,
        channel_colors=channel_colors,
        contrast_percentile=contrast_percentile,
        ome_image_for_field=_ome_image_for_field,
        audit_channels_for_field=_audit_channels_for_field,
        audit_objective_for_field=_audit_objective_for_field,
        audit_acquisition_for_field=_audit_acquisition_for_field,
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


def inspect(
    input_path: str | Path,
    *,
    reader_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Return a summary of ``input_path``'s scenes without converting.

    Used for pre-flight inspection before kicking off a slow conversion. The
    ``reader_plugin`` field mirrors the audit record's nested shape.

    ``reader_kwargs`` (optional) is forwarded to the winning reader plugin's
    ``open()`` — same passthrough as :func:`convert`. Motivating case: the
    SmartSPIM reader's ``metadata_path=`` sidecar override, which lives
    elsewhere on disk than the read-only export directory.
    """
    reader, plugin, match_score = get_reader(input_path, reader_kwargs=reader_kwargs)
    distribution = _resolve_distribution(reader, plugin)
    scenes_info = []
    for i, name in enumerate(reader.scenes):
        reader.set_scene(i)
        xarr = reader.xarray_dask_data
        px = getattr(reader, "physical_pixel_sizes", None)
        scene_info: dict[str, Any] = {
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
        # ADR-0008 / #62 (+#63–#65 / #76 / #77 / #78): surface the acquisition
        # block in inspect() so pre-flight tooling sees it without paying for a
        # full convert. Same 4-tier composition as the convert path
        # (:func:`_audit_acquisition_for_scene`): LIF scene-XML extractor,
        # vendor-specific extractor (CZI / ND2 microscope model), OME
        # projection, then the soft-optional ``reader.acquisition_audit``
        # hook — each layer ``setdefault``-fills gaps left by earlier ones.
        # Same fail-safe throughout — metadata never crashes inspect().
        scene_xml = find_scene_xml(reader)
        base: dict = {}
        if scene_xml is not None:
            lif_extracted = extract_acquisition(scene_xml)
            if lif_extracted:
                base.update(lif_extracted)
        vendor_dict = _vendor_acquisition_extras(reader) or {}
        for key, value in vendor_dict.items():
            base.setdefault(key, value)
        try:
            ome = reader.ome_metadata
        except Exception:  # noqa: BLE001 — heterogeneous bioio errors
            ome = None
        ome_dict = extract_acquisition_from_ome(ome, image_index=0) or {}
        for key, value in ome_dict.items():
            base.setdefault(key, value)
        extras = _reader_acquisition_extras(reader) or {}
        for key, value in extras.items():
            base.setdefault(key, value)
        if base:
            scene_info["acquisition"] = base
        scenes_info.append(scene_info)
    input_bytes = size_on_disk(input_path)
    info: dict[str, Any] = {
        "input_path": str(input_path),
        "size_bytes": input_bytes,
        "size_human": format_bytes(input_bytes),
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
    # Multi-plate LIF discoverability (#82): surface every plate template so
    # the user can pick a `--plate NAME` selector without opening the file
    # in LAS X. Additive — non-plate readers and single-plate LIFs (where
    # `plate_layout` already carries the name) omit this key.
    available_plates = list(getattr(reader, "available_plates", []) or [])
    if len(available_plates) >= 2:
        info["plates"] = available_plates
    return info
