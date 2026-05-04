"""Top-level convert() and inspect() — orchestrate readers, writers, audit."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

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
from zarrmony.readers import get_reader
from zarrmony.writers.bf2raw import write_bf2raw_wrapper
from zarrmony.writers.ome_xml import build_combined_ome_xml
from zarrmony.writers.scene import write_scene


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


def convert(
    input_path: str | Path,
    output: str | Path,
    *,
    metadata: UserMetadata | dict | None = None,
    per_scene_metadata: dict[str, UserMetadata | dict] | None = None,
    pyramid_min_size: int = 256,
    chunk_shape: Sequence[int] | None = None,
    channel_colors: dict[str, str] | None = None,
    force: bool = False,
    permissive: bool = False,
    checksum: bool = False,
) -> dict:
    """Convert ``input_path`` to OME-Zarr v0.5 with bioformats2raw.layout.

    Returns the audit record that was written to ``attrs.zarrmony`` at the
    output store's root.
    """
    started_at = datetime.now().astimezone()

    user_metadata = _validate_metadata(metadata, permissive)
    per_scene_user_metadata: dict[str, dict] = {}
    if per_scene_metadata:
        for scene_name, m in per_scene_metadata.items():
            per_scene_user_metadata[scene_name] = _validate_metadata(m, permissive)

    _check_output(output, force=force)

    reader, plugin_name = get_reader(input_path)
    if not reader.scenes:
        raise ZarrmonyError(f"reader returned no scenes for {input_path!s}")

    output_str = str(output).rstrip("/")
    series_paths: list[str] = []
    images: list[Image] = []
    per_scene_records: list[dict] = []
    metadata_warnings: list[dict] = []

    for scene_index, scene_name in enumerate(reader.scenes):
        scene_path = f"{output_str}/{scene_index}"

        reader.set_scene(scene_index)
        channel_names = list(reader.channel_names) if getattr(reader, "channel_names", None) else []
        channels: list[Channel] | None = None
        if channel_names:
            colors = colors_for_channels(channel_names, overrides=channel_colors)
            channels = [
                Channel(label=n, color=c) for n, c in zip(channel_names, colors, strict=True)
            ]

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
    source_filename: str | None = None
    if source_xml is not None:
        ext = Path(str(input_path)).suffix.lstrip(".").lower()
        source_filename = f"raw.{ext}.xml" if ext else "raw.xml"

    write_bf2raw_wrapper(
        output,
        series_paths=series_paths,
        ome_xml=ome_xml,
        source_xml=source_xml,
        source_xml_filename=source_filename,
    )

    finished_at = datetime.now().astimezone()

    config = {
        "pyramid_min_size": pyramid_min_size,
        "chunk_shape": list(chunk_shape) if chunk_shape else None,
        "channel_colors": dict(channel_colors) if channel_colors else None,
        "force": force,
        "permissive": permissive,
        "checksum": checksum,
    }
    audit = build_audit_record(
        input_path=input_path,
        reader_plugin=plugin_name,
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

    Used for pre-flight inspection before kicking off a slow conversion.
    """
    reader, plugin = get_reader(input_path)
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
        "plugin": plugin,
        "n_scenes": len(reader.scenes),
        "scenes": scenes_info,
    }
