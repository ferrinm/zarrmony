"""Single-scene OME-Zarr writer.

Wraps bioio-ome-zarr's ``OMEZarrWriter`` so we can supply our own mean-pooled
pyramid instead of the parent's nearest-neighbor downsampling. The wrapping
is done as a subclass that exposes ``initialize()`` and ``write_pyramid()``
publicly; the parent's lazy initialization is otherwise opaque to callers.
"""

from collections.abc import Sequence
from typing import Any

import dask.array as da
import xarray as xr
from bioio_ome_zarr.writers import Channel, OMEZarrWriter

from zarrmony.transforms import NGFF_AXIS_TYPE, NGFF_AXIS_UNIT, normalize_axes
from zarrmony.writers.pyramid import build_pyramid, compute_level_shapes


class ZarrmonyWriter(OMEZarrWriter):
    """OMEZarrWriter subclass that lets us initialize the on-disk arrays
    separately from writing data, so we can write a pre-computed pyramid.
    """

    def initialize(self) -> None:
        """Public alias for the parent's lazy initialization."""
        if not self._initialized:
            self._initialize()

    def write_pyramid(self, level_arrays: Sequence[da.Array]) -> None:
        """Write pre-computed per-level dask arrays into the on-disk pyramid."""
        self.initialize()
        if len(level_arrays) != len(self.datasets):
            raise ValueError(
                f"level_arrays length {len(level_arrays)} does not match "
                f"declared level_shapes length {len(self.datasets)}"
            )
        ops = []
        for i, arr in enumerate(level_arrays):
            tgt_chunks = self.datasets[i].chunks
            src = arr if arr.chunks == tgt_chunks else arr.rechunk(tgt_chunks)
            if self.zarr_format == 2:
                ops.append(da.to_zarr(src, self.datasets[i], compute=False))
            else:
                ops.append(da.store(src, self.datasets[i], lock=True, compute=False))
        da.compute(*ops)


def _physical_scales_for_dims(dims: Sequence[str], reader: Any) -> list[float]:
    """Build a per-dim scale list (μm/px for spatial, 1.0 for non-spatial)."""
    px = reader.physical_pixel_sizes
    base = {
        "T": 1.0,
        "C": 1.0,
        "Z": float(px.Z) if getattr(px, "Z", None) is not None else 1.0,
        "Y": float(px.Y) if getattr(px, "Y", None) is not None else 1.0,
        "X": float(px.X) if getattr(px, "X", None) is not None else 1.0,
    }
    return [base[d] for d in dims]


def _default_channels(channel_names: Sequence[str]) -> list[Channel]:
    return [Channel(label=name, color="ffffff") for name in channel_names]


def write_scene(
    reader: Any,
    scene_index: int,
    store_path: Any,
    *,
    pyramid_min_size: int = 256,
    chunk_shape: Sequence[int] | None = None,
    channels: Sequence[Channel] | None = None,
    image_name: str | None = None,
    creator_info: dict | None = None,
    xarr_override: xr.DataArray | None = None,
    record_mosaic_summary: bool = True,
) -> dict:
    """Convert one scene to an OME-Zarr image at ``store_path``.

    Returns an audit dict (scene_index/name, dims, level_shapes,
    axis_normalization record, channel_count, physical_pixel_size).

    ``xarr_override`` substitutes a pre-built xarray for ``reader.xarray_dask_data``
    (used by ``api.convert(..., lif_mosaic="per-tile")`` to feed in one tile at
    a time without forking the writer). Physical pixel sizes still come from
    the reader. ``record_mosaic_summary=False`` suppresses the ``mosaic`` key
    in the returned audit dict — the per-tile path emits its own ``per_tile``
    discriminator at the audit caller, so attaching the scene-level mosaic
    summary to each tile's own audit would double-count and mislead.
    """
    reader.set_scene(scene_index)
    scene_name = reader.scenes[scene_index]
    name = image_name or scene_name

    mosaic_summary = (
        getattr(reader, "mosaic_summary", None) if record_mosaic_summary else None
    )
    xarr = xarr_override if xarr_override is not None else reader.xarray_dask_data
    canonical, axis_record = normalize_axes(xarr)
    dims = list(canonical.dims)
    base_shape = tuple(int(s) for s in canonical.shape)

    level_shapes = compute_level_shapes(base_shape, dims, min_size=pyramid_min_size)
    pyramid = build_pyramid(canonical.data, dims, level_shapes)

    if channels is None:
        channel_coord = canonical.coords.get("C")
        if channel_coord is not None:
            channel_names = [str(v) for v in channel_coord.values]
        elif "C" in dims:
            channel_names = [f"C:{i}" for i in range(canonical.sizes["C"])]
        else:
            channel_names = []
        channels = _default_channels(channel_names)
    channel_count = len(channels)

    axes_names = [d.lower() for d in dims]
    axes_types = [NGFF_AXIS_TYPE[d] for d in dims]
    axes_units = [NGFF_AXIS_UNIT[d] for d in dims]
    physical_pixel_size = _physical_scales_for_dims(dims, reader)

    writer = ZarrmonyWriter(
        store=store_path,
        level_shapes=level_shapes,
        dtype=canonical.dtype,
        zarr_format=3,
        image_name=name,
        channels=list(channels) if channels else None,
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
        chunk_shape=chunk_shape,
        creator_info=creator_info,
    )
    writer.write_pyramid(pyramid)

    record = {
        "scene_index": scene_index,
        "scene_name": scene_name,
        "image_name": name,
        "dims": dims,
        "level_shapes": [list(s) for s in level_shapes],
        "axis_normalization": axis_record,
        "channel_count": channel_count,
        "physical_pixel_size": dict(zip(dims, physical_pixel_size, strict=True)),
    }
    if mosaic_summary is not None:
        record["mosaic"] = mosaic_summary
    return record
