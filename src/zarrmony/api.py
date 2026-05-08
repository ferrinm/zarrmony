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
from pydantic import ValidationError

from zarrmony.audit import build_audit_record, write_audit_record
from zarrmony.errors import (
    ExtractorWarning,
    MetadataValidationError,
    OutputExistsError,
    ZarrmonyError,
)
from zarrmony.metadata.channel_colors import colors_for_channels
from zarrmony.metadata.model import UserMetadata
from zarrmony.naming import resolve_scene_dirnames
from zarrmony.readers.default import derive_bioio_distribution
from zarrmony.readers.plugin import ReaderPlugin, get_reader
from zarrmony.writers.bf2raw import write_bf2raw_wrapper
from zarrmony.writers.ome_xml import build_combined_ome_xml, build_ome_xml_for_scene
from zarrmony.writers.per_scene import write_per_scene_metadata
from zarrmony.writers.scene import write_scene

Layout = Literal["per-scene", "bf2raw"]


def _validate_metadata(
    metadata: UserMetadata | dict | None,
    permissive: bool,
) -> dict:
    if metadata is None:
        if permissive:
            return {}
        raise MetadataValidationError(
            "metadata is required (pass permissive=True to bypass for prototyping)"
        )
    if isinstance(metadata, UserMetadata):
        return metadata.model_dump(exclude_none=False)
    if isinstance(metadata, dict):
        if permissive:
            return dict(metadata)
        try:
            return UserMetadata(**metadata).model_dump(exclude_none=False)
        except ValidationError as e:
            raise MetadataValidationError(str(e)) from e
    raise TypeError(f"metadata must be UserMetadata, dict, or None (got {type(metadata).__name__})")


def _check_output(output: str | Path, *, force: bool) -> None:
    """Refuse-or-clobber a single output path (file, dir, or store)."""
    s = str(output)
    fs, path = fsspec.core.url_to_fs(s)
    if not fs.exists(path):
        return
    if not force:
        raise OutputExistsError(f"output already exists: {s} (pass force=True to overwrite)")
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


def _try_get_ome_image(reader: Any, scene_index: int) -> tuple[Image | None, dict | None]:
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
    channel_names = list(reader.channel_names) if getattr(reader, "channel_names", None) else []
    if not channel_names:
        return None
    colors = colors_for_channels(channel_names, overrides=channel_colors)
    return [Channel(label=n, color=c) for n, c in zip(channel_names, colors, strict=True)]


def _source_xml_filename(input_path: str | Path) -> str:
    ext = Path(str(input_path)).suffix.lstrip(".").lower()
    return f"raw.{ext}.xml" if ext else "raw.xml"


def _resolve_distribution(reader: Any, plugin: ReaderPlugin) -> str | None:
    """Return the effective bioio distribution name for the audit record.

    For format-specific plugins (CZI/LIF/ND2) ``plugin.distribution`` is set.
    For the catch-all default plugin it is ``None``; introspect the opened
    ``BioImage`` to recover the actual sub-package (e.g. ``bioio-ome-tiff``).
    """
    return plugin.distribution if plugin.distribution else derive_bioio_distribution(reader)


def convert(
    input_path: str | Path,
    output: str | Path,
    *,
    layout: Layout = "per-scene",
    metadata: UserMetadata | dict | None = None,
    per_scene_metadata: dict[str, UserMetadata | dict] | None = None,
    pyramid_min_size: int = 256,
    chunk_shape: Sequence[int] | None = None,
    channel_colors: dict[str, str] | None = None,
    force: bool = False,
    permissive: bool = False,
    checksum: bool = False,
) -> dict:
    """Convert ``input_path`` to OME-Zarr v0.5.

    By default (``layout='per-scene'``), each scene is written to its own
    self-describing store at ``<output>/<sanitized_scene_name>.ome.zarr/`` and
    the return value is ``{"input": ..., "stores": [<per-store audit>, ...]}``.

    With ``layout='bf2raw'``, a single ``bioformats2raw.layout`` store is
    written at ``output``: per-scene images live in numbered subgroups, with a
    combined ``OME/METADATA.ome.xml`` and ``OME/series`` listing. The return
    value is the bundle's audit dict (matching the v0.1.x behavior).
    """
    if layout not in ("per-scene", "bf2raw"):
        raise ValueError(f"layout must be 'per-scene' or 'bf2raw' (got {layout!r})")

    user_metadata = _validate_metadata(metadata, permissive)
    per_scene_user_metadata: dict[str, dict] = {}
    if per_scene_metadata:
        for scene_name, m in per_scene_metadata.items():
            per_scene_user_metadata[scene_name] = _validate_metadata(m, permissive)

    reader, plugin, match_score = get_reader(input_path)
    if not reader.scenes:
        raise ZarrmonyError(f"reader returned no scenes for {input_path!s}")
    distribution = _resolve_distribution(reader, plugin)

    config = {
        "layout": layout,
        "pyramid_min_size": pyramid_min_size,
        "chunk_shape": list(chunk_shape) if chunk_shape else None,
        "channel_colors": dict(channel_colors) if channel_colors else None,
        "force": force,
        "permissive": permissive,
        "checksum": checksum,
    }

    if layout == "bf2raw":
        return _convert_bf2raw(
            reader=reader,
            plugin=plugin,
            match_score=match_score,
            distribution=distribution,
            input_path=input_path,
            output=output,
            user_metadata=user_metadata,
            per_scene_user_metadata=per_scene_user_metadata,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            channel_colors=channel_colors,
            force=force,
            checksum=checksum,
            config=config,
        )
    return _convert_per_scene(
        reader=reader,
        plugin=plugin,
        match_score=match_score,
        distribution=distribution,
        input_path=input_path,
        output=output,
        user_metadata=user_metadata,
        per_scene_user_metadata=per_scene_user_metadata,
        pyramid_min_size=pyramid_min_size,
        chunk_shape=chunk_shape,
        channel_colors=channel_colors,
        force=force,
        checksum=checksum,
        config=config,
    )


def _convert_per_scene(
    *,
    reader: Any,
    plugin: ReaderPlugin,
    match_score: int | None,
    distribution: str | None,
    input_path: str | Path,
    output: str | Path,
    user_metadata: dict,
    per_scene_user_metadata: dict[str, dict],
    pyramid_min_size: int,
    chunk_shape: Sequence[int] | None,
    channel_colors: dict[str, str] | None,
    force: bool,
    checksum: bool,
    config: dict,
) -> dict:
    output_str = str(output).rstrip("/")
    dirnames = resolve_scene_dirnames(reader.scenes)
    store_paths = [f"{output_str}/{d}.ome.zarr" for d in dirnames]

    # Per-store refuse-overwrite (don't blow away the whole output dir).
    for sp in store_paths:
        _check_output(sp, force=force)

    source_xml = _serialize_source_metadata(getattr(reader, "metadata", None))
    source_filename = _source_xml_filename(input_path) if source_xml is not None else None

    store_audits: list[dict] = []

    for scene_index, scene_name in enumerate(reader.scenes):
        store_path = store_paths[scene_index]
        started_at = datetime.now().astimezone()

        reader.set_scene(scene_index)
        channels = _channels_for_scene(reader, channel_colors)

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

        scene_user_md: dict | None = None
        if scene_name in per_scene_user_metadata:
            scene_user_md = per_scene_user_metadata[scene_name]
            scene_record["user_metadata"] = scene_user_md

        ome_image, warning = _try_get_ome_image(reader, scene_index)
        metadata_warnings: list[dict] = []
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

        finished_at = datetime.now().astimezone()
        audit = build_audit_record(
            input_path=input_path,
            reader_plugin=plugin,
            match_score=match_score,
            distribution=distribution,
            config=config,
            started_at=started_at,
            finished_at=finished_at,
            per_scene=[scene_record],
            metadata_warnings=metadata_warnings,
            checksum=checksum,
        )
        # Effective user_metadata for this store: per-scene override falls back
        # to the root-level user_metadata.
        audit["user_metadata"] = scene_user_md if scene_user_md is not None else user_metadata
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
    user_metadata: dict,
    per_scene_user_metadata: dict[str, dict],
    pyramid_min_size: int,
    chunk_shape: Sequence[int] | None,
    channel_colors: dict[str, str] | None,
    force: bool,
    checksum: bool,
    config: dict,
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

        if scene_name in per_scene_user_metadata:
            scene_record["user_metadata"] = per_scene_user_metadata[scene_name]

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
    source_filename = _source_xml_filename(input_path) if source_xml is not None else None

    write_bf2raw_wrapper(
        output,
        series_paths=series_paths,
        ome_xml=ome_xml,
        source_xml=source_xml,
        source_xml_filename=source_filename,
    )

    finished_at = datetime.now().astimezone()

    audit = build_audit_record(
        input_path=input_path,
        reader_plugin=plugin,
        match_score=match_score,
        distribution=distribution,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        per_scene=per_scene_records,
        metadata_warnings=metadata_warnings,
        checksum=checksum,
    )
    audit["user_metadata"] = user_metadata
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
    return {
        "input_path": str(input_path),
        "reader_plugin": {
            "name": plugin.name,
            "source": plugin.source,
            "distribution": distribution,
            "match_score": match_score,
        },
        "n_scenes": len(reader.scenes),
        "scenes": scenes_info,
    }
