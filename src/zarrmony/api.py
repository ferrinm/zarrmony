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
    OutputExistsError,
    ValidationWarning,
    ZarrmonyError,
)
from zarrmony.metadata.channel_colors import colors_for_channels
from zarrmony.metadata.lif_channels import (
    channels_to_ome_channels,
    channels_to_omero,
    extract_channels,
)
from zarrmony.naming import resolve_scene_dirnames
from zarrmony.readers.default import derive_bioio_distribution
from zarrmony.readers.plugin import ReaderPlugin, get_reader
from zarrmony.writers.bf2raw import write_bf2raw_wrapper
from zarrmony.writers.ome_xml import build_combined_ome_xml, build_ome_xml_for_scene
from zarrmony.writers.per_scene import write_per_scene_metadata
from zarrmony.writers.plate import summarize_plate_layout, write_plate
from zarrmony.writers.scene import write_scene

Layout = Literal["auto", "per-scene", "bf2raw", "plate"]
ResolvedLayout = Literal["per-scene", "bf2raw", "plate"]
_VALID_LAYOUTS: tuple[Layout, ...] = ("auto", "per-scene", "bf2raw", "plate")


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
    """The current scene's C size, from the reader's public xarray surface.

    Mirrors how :func:`inspect` reads dims/sizes (no coupling to writer
    internals). A scene with no ``C`` dim has one implicit channel.
    """
    xarr = reader.xarray_dask_data
    return int(xarr.sizes["C"]) if "C" in xarr.dims else 1


def _lif_scene_root_fast(reader: Any) -> ET.Element | None:
    """The current scene's settings XML via bioio-lif's ``scene_root`` fast path.

    ``scene_root`` is bioio-lif's *plate* row/column locator: it returns the
    well's ``<Element>`` node for plate scenes but RAISES ``ValueError`` ("Row or
    column value is missing…") for ordinary non-plate confocal scenes, and may be
    absent on non-LIF readers. We read it inside a guard so either outcome —
    raise or ``None`` — is a clean miss, not a crash, letting the caller fall
    back to the document-order ``<Image>`` locator.
    """
    try:
        return getattr(reader, "scene_root", None)
    except Exception:  # noqa: BLE001 — a raise here just means "not this path"
        return None


def _lif_scene_image(reader: Any) -> ET.Element | None:
    """The current scene's ``<Image>`` element from the full LIF XML, fail-safe.

    This is the confocal locator from bioio-lif PR #52: ``reader.metadata`` is
    the whole-document LIF ``ElementTree``; ``.//Image`` returns the per-scene
    ``<Image>`` elements in scene order, and ``reader.current_scene_index``
    selects the one being converted. Each such element carries that scene's
    ``ChannelDescription`` + ``LDM_Block_Sequential_List`` — exactly what
    :func:`extract_channels` consumes.

    Every reader-surface access is guarded so no partially-readable reader can
    raise out of here: missing/``None`` metadata, a metadata object without
    ``findall``, an empty ``<Image>`` list, and an absent or out-of-range
    ``current_scene_index`` all return ``None`` (a clean miss). Returns the
    located element or ``None``; never raises.
    """
    metadata = getattr(reader, "metadata", None)
    if metadata is None or not hasattr(metadata, "findall"):
        return None
    images = metadata.findall(".//Image")
    if not images:
        return None
    index = getattr(reader, "current_scene_index", None)
    if not isinstance(index, int) or not (0 <= index < len(images)):
        return None
    return images[index]


def _lif_scene_channels(reader: Any) -> tuple[list[dict] | None, list[Channel] | None]:
    """The LIF-vs-other decision, in one place.

    Returns ``(extracted, omero_channels)`` when ``reader`` is a LIF reader whose
    current scene yields channel identities AND that identity count matches the
    scene's C size; otherwise ``(None, None)``. ``extracted`` is the raw
    per-channel identity dicts (handed to :func:`_lif_ome_image` once scene sizes
    are known); ``omero_channels`` is the display label/color list for
    :func:`write_scene`. Both projections come from the same vetted extraction,
    so omero and the OME-XML never disagree.

    Locating the scene XML is two-tier (see the two helpers above):

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
        # First-try fast path (plate wells), then the confocal fallback. Each
        # locator is internally guarded, so this never raises; an unlocatable
        # scene simply yields ``None`` here and a clean decline below. Test
        # ``is None`` explicitly rather than ``a or b`` — an ``Element`` with no
        # children is falsy, so ``or`` would wrongly skip a valid childless
        # ``scene_root`` and is deprecated besides.
        scene_element = _lif_scene_root_fast(reader)
        if scene_element is None:
            scene_element = _lif_scene_image(reader)
        if scene_element is None:
            return None, None
        scene_xml = ET.tostring(scene_element, encoding="unicode")
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

    Per-scene return shape: ``{"input": ..., "stores": [<per-store audit>, ...]}``.
    The ``bf2raw`` and ``plate`` shapes return the single bundle's audit dict;
    plate audits use the schema-3 ``fields`` + ``plate`` keys.
    """
    if layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"layout must be one of {list(_VALID_LAYOUTS)} (got {layout!r})"
        )

    reader, plugin, match_score = get_reader(input_path)
    if not reader.scenes:
        raise ZarrmonyError(f"reader returned no scenes for {input_path!s}")
    distribution = _resolve_distribution(reader, plugin)

    effective_layout = _resolve_layout(layout, reader, plugin)

    config = {
        "layout": effective_layout,
        "pyramid_min_size": pyramid_min_size,
        "chunk_shape": list(chunk_shape) if chunk_shape else None,
        "channel_colors": dict(channel_colors) if channel_colors else None,
        "force": force,
        "checksum": checksum,
        "validate": validate,
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
    )


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
) -> dict:
    output_str = str(output).rstrip("/")
    dirnames = resolve_scene_dirnames(reader.scenes)
    store_paths = [f"{output_str}/{d}.ome.zarr" for d in dirnames]

    # Per-store refuse-overwrite (don't blow away the whole output dir).
    for sp in store_paths:
        _check_output(sp, force=force)

    source_xml = _serialize_source_metadata(getattr(reader, "metadata", None))
    source_filename = (
        _source_xml_filename(input_path) if source_xml is not None else None
    )

    store_audits: list[dict] = []

    for scene_index, scene_name in enumerate(reader.scenes):
        store_path = store_paths[scene_index]
        started_at = datetime.now().astimezone()

        reader.set_scene(scene_index)

        skip_reason = getattr(reader, "skip_reason", None)
        if skip_reason:
            warnings.warn(
                f"scene {scene_index} ({scene_name!r}): {skip_reason}",
                MosaicMergedSiblingWarning,
                stacklevel=2,
            )
            continue

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

        scene_record = write_scene(
            reader,
            scene_index=scene_index,
            store_path=store_path,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channels=channels,
            image_name=scene_name,
        )
        scene_record["store_path"] = store_path
        scene_record["dirname"] = dirnames[scene_index]

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
