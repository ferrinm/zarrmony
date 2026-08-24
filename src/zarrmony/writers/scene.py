"""Single-scene OME-Zarr writer.

Wraps bioio-ome-zarr's ``OMEZarrWriter`` so we can supply our own mean-pooled
pyramid instead of the parent's nearest-neighbor downsampling. The wrapping
is done as a subclass that exposes ``initialize()`` and ``write_pyramid()``
publicly; the parent's lazy initialization is otherwise opaque to callers.
"""

from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from bioio_ome_zarr.writers import Channel, OMEZarrWriter

from zarrmony._storage import open_root_group
from zarrmony.geometry import DEFAULT_GEOMETRY, Geometry, plan_level_chunk_shapes
from zarrmony.transforms import NGFF_AXIS_TYPE, NGFF_AXIS_UNIT, normalize_axes
from zarrmony.writers.pyramid import (
    build_pyramid,
    coarse_level_index,
    compute_level_shapes,
)

# Approximation label recorded in the audit whenever data-driven contrast runs.
# The min + percentile are computed off the COARSEST pyramid level rather than
# the base — the coarse level is derived from the base in the same dask graph,
# so raw pixels are still read once (piggybacks on the pyramid write pass), and
# the sort/quantile stays trivially cheap even for 80+ GB inputs. Mean-pooling
# raises the observed min a hair and blurs the tail slightly; for a viewer
# auto-contrast default the difference is well below what a human eye reads.
_CONTRAST_METHOD = "coarsest-pyramid-level"


class ZarrmonyWriter(OMEZarrWriter):
    """OMEZarrWriter subclass that lets us initialize the on-disk arrays
    separately from writing data, so we can write a pre-computed pyramid.
    """

    def initialize(self) -> None:
        """Public alias for the parent's lazy initialization."""
        if not self._initialized:
            self._initialize()

    def write_pyramid(
        self,
        level_arrays: Sequence[da.Array],
        *,
        extra_ops: Sequence[da.Array] = (),
    ) -> tuple[Any, ...]:
        """Write pre-computed per-level dask arrays into the on-disk pyramid.

        ``extra_ops`` are additional lazy dask values (e.g. per-channel min /
        percentile) fused into the same ``da.compute`` call so the raw data is
        read once — the pyramid write and the extras share the underlying
        chunk reads. Returns a tuple of the computed values for ``extra_ops``
        in the order they were passed; empty when no extras were provided.
        """
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
        n_write = len(ops)
        results = da.compute(*ops, *extra_ops)
        return tuple(results[n_write:])


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


def _dtype_window(dtype: np.dtype) -> dict[str, int | float]:
    """OMERO display-window bounds spanning ``dtype``'s full range.

    Integer dtypes → ``np.iinfo(dtype).min`` / ``max``; float dtypes → ``0.0`` /
    ``1.0`` (OMERO convention for normalized floats). ``start`` / ``end`` mirror
    ``min`` / ``max`` so first-open viewers see the full range unclipped —
    percentile-based auto-contrast is a separate concern (see #50). Prevents
    the bioio-ome-zarr ``Channel`` default of ``0``–``255`` from clamping
    uint16/uint32/float32 pixels into a black-on-first-open display.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo: int | float = int(info.min)
        hi: int | float = int(info.max)
    else:
        lo, hi = 0.0, 1.0
    return {"min": lo, "max": hi, "start": lo, "end": hi}


def _default_channels(channel_names: Sequence[str], dtype: np.dtype) -> list[Channel]:
    """Emission-band-colored channels for readers that surface no wavelength.

    Reaches the ADR-0007 palette via the dye-name substring fallback in
    :func:`zarrmony.metadata.channel_colors.colors_for_channels` so a CZI/ND2/
    OME-TIFF scene named "DAPI"/"GFP"/"mCherry"/"Cy5" lands in the same
    colorblind slots as its LIF-source counterpart, and collisions are handled
    identically. ``dtype`` drives the OMERO display window (see #50) so
    uint16/uint32/float32 stores open with the full dtype range rather than
    the bioio-ome-zarr 0–255 default.
    """
    from zarrmony.metadata.channel_colors import colors_for_channels

    names = list(channel_names)
    colors = colors_for_channels(names)
    window = _dtype_window(dtype)
    return [
        Channel(label=n, color=c, window=window)
        for n, c in zip(names, colors, strict=True)
    ]


def _channel_contrast_ops(
    coarse: da.Array,
    dims: Sequence[str],
    channel_count: int,
    contrast_percentile: float,
) -> list[da.Array]:
    """Per-channel ``(min, percentile)`` lazy values on the coarsest pyramid level.

    Returns a flat list of length ``2 * channel_count``: for channel ``i`` the
    entries live at indices ``2*i`` (min) and ``2*i + 1`` (percentile). Callers
    thread this list through :meth:`ZarrmonyWriter.write_pyramid`'s
    ``extra_ops`` so the underlying chunk reads fuse with the pyramid writes,
    then re-pair the results. Emits nothing (returns ``[]``) when the array
    carries no channel dimension AND ``channel_count`` is zero, so the "no
    omero channels to update" path stays a no-op.

    Percentile is computed via :func:`dask.array.percentile`'s default
    ``internal_method`` — no ``crick`` T-digest dependency — and only on the
    coarse level, which for a typical microscopy scene is a few hundred KB per
    channel. See ``_CONTRAST_METHOD`` for the approximation trade-off.
    """
    c_axis = dims.index("C") if "C" in dims else None
    ops: list[da.Array] = []
    for i in range(channel_count):
        if c_axis is None:
            ch = coarse
        else:
            idx: list[Any] = [slice(None)] * coarse.ndim
            idx[c_axis] = i
            ch = coarse[tuple(idx)]
        flat = ch.ravel()
        ops.append(flat.min())
        ops.append(da.percentile(flat, [contrast_percentile])[0])
    return ops


def _pair_contrast_results(
    results: Sequence[Any], channel_count: int
) -> list[tuple[Any, Any]]:
    """Pair a flat ``[ch0_min, ch0_pct, ch1_min, ch1_pct, ...]`` list into tuples."""
    return [(results[2 * i], results[2 * i + 1]) for i in range(channel_count)]


def _to_json_scalar(v: Any) -> Any:
    """Cast a numpy 0-d scalar to a native Python type for JSON-serializable attrs."""
    return v.item() if hasattr(v, "item") else v


def _update_omero_window_start_end(
    store_path: Any, per_channel_stats: Sequence[tuple[Any, Any]]
) -> None:
    """Rewrite ``omero.channels[i].window.start / .end`` in-place on ``store_path``.

    Runs after :meth:`ZarrmonyWriter.write_pyramid` returns computed per-channel
    contrast stats. Preserves ``min``/``max`` (dtype-range bounds set at
    ``Channel`` construction time — see :func:`_dtype_window`) and only touches
    ``start``/``end``. No-op when the store has no ``omero`` block or fewer
    channels than stats (a defensive guard — the caller only computes stats
    when ``channel_count`` matches the omero channels).
    """
    root = open_root_group(store_path, mode="a")
    ome = dict(root.attrs.get("ome", {}))
    omero = ome.get("omero")
    if not omero:
        return
    channels = list(omero.get("channels", []))
    if not channels:
        return
    for i, (min_v, pct_v) in enumerate(per_channel_stats):
        if i >= len(channels):
            break
        window = dict(channels[i].get("window", {}))
        window["start"] = _to_json_scalar(min_v)
        window["end"] = _to_json_scalar(pct_v)
        channels[i] = {**channels[i], "window": window}
    omero = {**omero, "channels": channels}
    ome = {**ome, "omero": omero}
    root.attrs["ome"] = ome


def write_scene(
    reader: Any,
    scene_index: int,
    store_path: Any,
    *,
    geometry: Geometry = DEFAULT_GEOMETRY,
    channels: Sequence[Channel] | None = None,
    image_name: str | None = None,
    creator_info: dict | None = None,
    xarr_override: xr.DataArray | None = None,
    record_mosaic_summary: bool = True,
    contrast_percentile: float | None = None,
) -> dict:
    """Convert one scene to an OME-Zarr image at ``store_path``.

    Returns an audit dict (scene_index/name, dims, level_shapes, chunk_shapes,
    coarse_level_index, axis_normalization record, channel_count,
    physical_pixel_size).

    ``geometry`` (ADR-0010) carries every output-shape choice — pyramid depth,
    the coarse-level bounds and chunk shape — as one frozen
    :class:`~zarrmony.geometry.Geometry` value rather than as loose keywords.
    ``convert()`` resolves it once and threads the same instance through
    per-scene, bf2raw and plate output; the default is the ADR-0010 policy.
    Level shapes halve every spatial axis whose spacing is within
    ``isotropy_tolerance`` of the finest axis's
    (:func:`~zarrmony.writers.pyramid.compute_level_shapes`), so the pyramid
    moves toward isotropy rather than preserving it, and depth runs until a
    level is small enough for a viewer to hold whole; chunk shapes are then
    planned per level from that level's own physical voxel spacing
    (:func:`~zarrmony.geometry.plan_level_chunk_shapes`) and recorded under
    ``chunk_shapes``, one entry per level, alongside the ``level_shapes`` they
    were planned against. ``coarse_level_index`` names which of those levels is
    the coarse one (``None`` if none reaches the bounds), so the guarantee is
    checkable in the audit rather than in a viewport.

    ``xarr_override`` substitutes a pre-built xarray for ``reader.xarray_dask_data``
    (used by ``api.convert(..., lif_mosaic="per-tile")`` to feed in one tile at
    a time without forking the writer). Physical pixel sizes still come from
    the reader. ``record_mosaic_summary=False`` suppresses the ``mosaic`` key
    in the returned audit dict — the per-tile path emits its own ``per_tile``
    discriminator at the audit caller, so attaching the scene-level mosaic
    summary to each tile's own audit would double-count and mislead.

    ``contrast_percentile`` (issue #53) drives data-driven display contrast:
    when set (a float in ``(0, 100)``, typically ``99.9``), per-channel ``(min,
    percentile)`` values are computed off the coarsest pyramid level — fused
    into the pyramid dask graph so the raw data is read once — and written into
    the omero ``window.start`` / ``window.end`` fields, replacing the
    dtype-range placeholders (issue #50). ``None`` skips the extra ops entirely
    and leaves ``start`` / ``end`` matching ``min`` / ``max``. The audit dict
    gets a ``contrast`` block naming the percentile, the approximation method
    (see ``_CONTRAST_METHOD``), and the resolved per-channel bounds.
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

    # ADR-0010 (#85): level shapes are anisotropy-aware, so they need the
    # scene's physical spacing — an axis halves only while its spacing is within
    # `isotropy_tolerance` of the finest axis's. The same list feeds the NGFF
    # `physical_pixel_size` below and the chunk planner.
    physical_pixel_size = _physical_scales_for_dims(dims, reader)

    level_shapes = compute_level_shapes(
        base_shape, dims, physical_pixel_size, canonical.dtype, geometry
    )
    pyramid = build_pyramid(canonical.data, level_shapes)

    # ADR-0010 (#86): which level a viewer can hold whole is the property the
    # depth rule now targets, so record it rather than leaving it to be
    # rediscovered in a viewport — ``None`` when the pyramid bottomed out before
    # reaching the bounds.
    coarse_index = coarse_level_index(level_shapes, dims, canonical.dtype, geometry)

    if channels is None:
        channel_coord = canonical.coords.get("C")
        if channel_coord is not None:
            channel_names = [str(v) for v in channel_coord.values]
        elif "C" in dims:
            channel_names = [f"C:{i}" for i in range(canonical.sizes["C"])]
        else:
            channel_names = []
        channels = _default_channels(channel_names, canonical.dtype)
    channel_count = len(channels)

    axes_names = [d.lower() for d in dims]
    axes_types = [NGFF_AXIS_TYPE[d] for d in dims]
    axes_units = [NGFF_AXIS_UNIT[d] for d in dims]

    # ADR-0010 (#84): plan every level's chunk shape ourselves and hand the
    # writer an explicit per-level list. Passing ``chunk_shape=None`` would
    # delegate to bioio-ome-zarr's memory-target heuristic, which fills the
    # rightmost axis first under a 16 MiB budget — right for a 2D plane, and on
    # anything with a Z extent it converges on full-width single-plane slabs
    # that no frustum cull can trim. An explicit ``geometry.chunk_shape`` still
    # wins; the planner is what runs when the caller didn't name one.
    chunk_shapes = plan_level_chunk_shapes(
        level_shapes, dims, physical_pixel_size, canonical.dtype, geometry
    )

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
        chunk_shape=[list(c) for c in chunk_shapes],
        creator_info=creator_info,
    )

    # Only run the extra contrast ops when we actually have channels to update.
    # A scene with no C dim and no `channels` argument has no omero.channels
    # to rewrite, so there's nothing to compute; skipping keeps the pyramid
    # write graph unchanged for that case.
    run_contrast = contrast_percentile is not None and channel_count > 0
    if run_contrast:
        contrast_ops = _channel_contrast_ops(
            pyramid[-1], dims, channel_count, float(contrast_percentile)
        )
        results = writer.write_pyramid(pyramid, extra_ops=contrast_ops)
        contrast_stats = _pair_contrast_results(results, channel_count)
        _update_omero_window_start_end(store_path, contrast_stats)
    else:
        writer.write_pyramid(pyramid)
        contrast_stats = []

    record = {
        "scene_index": scene_index,
        "scene_name": scene_name,
        "image_name": name,
        "dims": dims,
        "level_shapes": [list(s) for s in level_shapes],
        "chunk_shapes": [list(c) for c in chunk_shapes],
        "coarse_level_index": coarse_index,
        "axis_normalization": axis_record,
        "channel_count": channel_count,
        "physical_pixel_size": dict(zip(dims, physical_pixel_size, strict=True)),
    }
    if mosaic_summary is not None:
        record["mosaic"] = mosaic_summary
    if run_contrast:
        record["contrast"] = {
            "percentile": float(contrast_percentile),
            "method": _CONTRAST_METHOD,
            "per_channel": [
                {
                    "channel_index": i,
                    "start": _to_json_scalar(min_v),
                    "end": _to_json_scalar(pct_v),
                }
                for i, (min_v, pct_v) in enumerate(contrast_stats)
            ],
        }
    return record
