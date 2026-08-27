"""Single-scene OME-Zarr writer.

Wraps bioio-ome-zarr's ``OMEZarrWriter`` so we can supply our own pooled
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
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    Geometry,
    plan_level_chunk_shapes,
    plan_level_shard_shapes,
)
from zarrmony.transforms import NGFF_AXIS_TYPE, NGFF_AXIS_UNIT, normalize_axes
from zarrmony.writers.pyramid import (
    coarse_level_index,
    compute_level_shapes,
    downsample_step,
)

# Approximation label recorded in the audit whenever data-driven contrast runs.
# The min + percentile are computed off the COARSEST pyramid level rather than
# the base, and read back off the store after that level has been written — so
# the pass touches no raw pixels at all, and the level it does read is bounded
# by ``geometry.coarse_max_bytes`` (64 MiB by default). Until #111 it was fused
# into the pyramid's single da.compute instead, on the theory that sharing the
# graph meant sharing the reads; measured, that fusion stalled an 80+ GB slide
# outright (#114) — 0 bytes written in 10 minutes against 12-38 chunks/min with
# contrast off. Mean-pooling raises the observed min a hair and blurs the tail
# slightly; for a viewer auto-contrast default the difference is well below what
# a human eye reads.
# Under ``downsample_method="max"`` that level is max-pooled, so the window
# opens higher — which is the right window for the pyramid actually written,
# and the audit records the method alongside the resolved bounds either way.
_CONTRAST_METHOD = "coarsest-pyramid-level"


def _write_grid(dataset: Any) -> tuple[int, ...]:
    """The block shape one write to ``dataset`` should cover — shard, else chunk.

    ``zarr.Array.chunks`` is the smallest independently *readable* unit;
    ``.shards`` is the storage object actually written. They are the same thing
    only on an unsharded array, and every place this writer reasons about "one
    unit of work" means the latter. Getting it backwards on a sharded array is
    correct but pathological: dask blocks land chunk-aligned *inside* shards,
    so each write read-modify-writes a whole object — 16× write amplification
    at the default targets, measurably slower even on a toy array.
    """
    return tuple(int(s) for s in (dataset.shards or dataset.chunks))


class ZarrmonyWriter(OMEZarrWriter):
    """OMEZarrWriter subclass that lets us initialize the on-disk arrays
    separately from writing data, so we can write a pre-computed pyramid.
    """

    def initialize(self) -> None:
        """Public alias for the parent's lazy initialization."""
        if not self._initialized:
            self._initialize()

    def read_level(self, index: int) -> da.Array:
        """Re-open one already-written level as a dask array over the store.

        Chunked on the store's own *write* grid — the shard where the level has
        one, the chunk where it does not — so a pass over a written level costs
        one task per stored object and nothing is re-derived from the source
        reader. Negative indices work as they do on any list.

        The distinction is not cosmetic. ``da.from_zarr`` left to itself adopts
        ``Array.chunks``, which on a sharded array is the *inner* chunk: a
        16-chunk shard would produce 16 tasks per object, multiplying this
        level's task count by the chunks-per-shard ratio and reintroducing
        #111's symptom on the pyramid's own read-back — the one place we
        deliberately spend a read.
        """
        self.initialize()
        dataset = self.datasets[index]
        return da.from_zarr(dataset, chunks=_write_grid(dataset))

    def write_pyramid(
        self,
        base_array: da.Array,
        *,
        geometry: Geometry = DEFAULT_GEOMETRY,
    ) -> None:
        """Write the pyramid one level at a time, each pooled from the one below.

        Level 0 is written from ``base_array``; every level above it is pooled
        (:func:`~zarrmony.writers.pyramid.downsample_step`) from the level that
        was *just written*, re-opened off the store, and written in a
        ``da.compute`` of its own.

        The alternative — build every level as one lazy graph and hand the lot
        to a single ``da.compute`` — is what this replaces (issue #111). It does
        not survive a whole-slide input: on a 141k × 172k × 4ch scene, dask spent
        hours in ``Task.__init__`` / ``blockwise.cull`` before writing a byte,
        because the graph for the coarsest level still reaches all the way back
        through every level to the reader, and every level's rechunk is
        constructed up front. Writing level-by-level bounds the graph held at any
        one moment to a single level's task count, and each level above 0 then
        reads ~a quarter of the chunks the level below wrote instead of
        re-deriving its pixels from the base — measured at 4.2× read
        amplification before the change.

        The cost is that each written level is read back once. That is the
        pyramid's own bytes, not the source's, and it buys the property the old
        path only appeared to have: raw pixels are read exactly once, by level 0.
        Correctness rests on both pooling kernels being exact over a block of any
        shape, so a level pooled from disk is bit-identical to the same level
        pooled inside one graph — see
        :data:`~zarrmony.writers.pyramid._DOWNSAMPLE_KERNELS`.

        Per-channel contrast is no longer fusable into the write, since there is
        no single compute to fuse it into; :func:`write_scene` computes it from
        :meth:`read_level` after the fact, which is cheaper anyway (issue #114).
        """
        self.initialize()
        declared = tuple(int(s) for s in self.datasets[0].shape)
        if tuple(int(s) for s in base_array.shape) != declared:
            raise ValueError(
                f"base_array shape {tuple(base_array.shape)} does not match the "
                f"declared level-0 shape {declared}"
            )
        for i, dataset in enumerate(self.datasets):
            if i == 0:
                src = base_array
            else:
                prev = self.datasets[i - 1]
                src = downsample_step(
                    self.read_level(i - 1), prev.shape, dataset.shape, geometry
                )
            tgt_chunks = _write_grid(dataset)
            if src.chunks != tgt_chunks:
                src = src.rechunk(tgt_chunks)
            if self.zarr_format == 2:
                da.to_zarr(src, dataset)
            else:
                # ``lock=True`` is load-bearing under sharding, not tidiness.
                # On an unsharded array a block maps 1:1 to an object and two
                # writers never touch the same key. A shard is one object per
                # many blocks, so where a rechunk leaves a partial shard at an
                # edge, concurrent writers to it read-modify-write the same key
                # and lose each other's updates. Do not drop this without
                # re-reading :func:`_write_grid`.
                da.store(src, dataset, lock=True)


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


def _ensure_ndarray_blocks(xarr: xr.DataArray) -> xr.DataArray:
    """Force a dask array's blocks to be real arrays before the pyramid runs.

    ``bioio-bioformats`` builds its graph out of ``LazyBioArray`` handles
    rather than materialised arrays, and reports one as the array's ``_meta``.
    Level 0 still writes, because zarr only needs ``__array__`` — but every
    level above it goes through :func:`dask.array.coarsen`, which calls
    ``.reshape`` on each block and dies with ``AttributeError: 'LazyBioArray'
    object has no attribute 'reshape'``. The same handles break the contrast
    pass, which needs ``.mean``.

    Gated on the block prototype's capabilities rather than on the reader's
    identity: a dask array whose blocks cannot reshape is broken for any
    backend, and readers that already yield ndarrays keep their graph exactly
    as built. Testing ``_meta`` costs nothing — it is dask's zero-element
    prototype, so nothing is read to decide this.
    """
    data = getattr(xarr, "data", None)
    if not isinstance(data, da.Array):
        return xarr
    meta = getattr(data, "_meta", None)
    if meta is None or (hasattr(meta, "reshape") and hasattr(meta, "mean")):
        return xarr
    coerced = data.map_blocks(
        np.asarray,
        dtype=data.dtype,
        meta=np.empty((0,) * data.ndim, dtype=data.dtype),
    )
    return xarr.copy(data=coerced)


def _rgb_sample_channels(n_samples: int, dtype: np.dtype) -> list[Channel]:
    """Channels for a folded samples axis: the primaries, not the ADR-0007 palette.

    ``_default_channels`` would route these through the emission-band mapping,
    which has no entry for "Red"/"Green"/"Blue" and would fall through to the
    colorblind-safe palette — compositing a colour photograph's red sample in
    cyan. See :func:`zarrmony.metadata.channel_colors.sample_axis_channels`.
    """
    from zarrmony.metadata.channel_colors import sample_axis_channels

    labels, colors = sample_axis_channels(n_samples)
    window = _dtype_window(dtype)
    return [
        Channel(label=label, color=color, window=window)
        for label, color in zip(labels, colors, strict=True)
    ]


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
    ``da.compute`` the list and re-pair the results with
    :func:`_pair_contrast_results`. Emits nothing (returns ``[]``) when the array
    carries no channel dimension AND ``channel_count`` is zero, so the "no
    omero channels to update" path stays a no-op.

    ``coarse`` is expected to be the *written* coarsest level, read back with
    :meth:`ZarrmonyWriter.read_level` — one task per stored chunk, over an array
    the pyramid's own depth rule caps at ``geometry.coarse_max_bytes``.

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
    shard_shapes, coarse_level_index, axis_normalization record, channel_count,
    physical_pixel_size).

    ``geometry`` (ADR-0010) carries every output-geometry choice — pyramid
    depth, the coarse-level bounds, chunk shape and the pooling kernel — as one
    frozen :class:`~zarrmony.geometry.Geometry` value rather than as keywords.
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
    were planned against. ``shard_shapes`` records the outer grid the same way
    and is ``None`` — the default — when the policy writes no shards, in which
    case the chunk is also the storage object. ``coarse_level_index`` names
    which of those levels is the coarse one (``None`` if none reaches the
    bounds), so the guarantee is checkable in the audit rather than in a
    viewport. ``downsample_method`` then
    decides how the pixels of every level above 0 are pooled — mean by default,
    ``"max"`` for sparse labels — uniformly across the pyramid.

    ``xarr_override`` substitutes a pre-built xarray for ``reader.xarray_dask_data``
    (used by ``api.convert(..., lif_mosaic="per-tile")`` to feed in one tile at
    a time without forking the writer). Physical pixel sizes still come from
    the reader. ``record_mosaic_summary=False`` suppresses the ``mosaic`` key
    in the returned audit dict — the per-tile path emits its own ``per_tile``
    discriminator at the audit caller, so attaching the scene-level mosaic
    summary to each tile's own audit would double-count and mislead.

    ``contrast_percentile`` (issue #53) drives data-driven display contrast:
    when set (a float in ``(0, 100)``, typically ``99.9``), per-channel ``(min,
    percentile)`` values are computed off the coarsest pyramid level — read back
    off the store once the pyramid is written, so no raw pixel is touched twice
    (issue #114) — and written into
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
    xarr = _ensure_ndarray_blocks(xarr)
    canonical, axis_record = normalize_axes(xarr)
    dims = list(canonical.dims)
    base_shape = tuple(int(s) for s in canonical.shape)

    # A folded samples axis invalidates whatever the caller derived: those
    # channels describe the reader's one pre-fold channel ("Channel:0:0"),
    # not the primaries C now holds. Overriding here rather than at the
    # caller keeps every entry point (convert, plate, per-tile) correct,
    # since all of them come through this function.
    if axis_record["rgb_samples_folded"]:
        channels = _rgb_sample_channels(int(canonical.sizes["C"]), canonical.dtype)

    # ADR-0010 (#85): level shapes are anisotropy-aware, so they need the
    # scene's physical spacing — an axis halves only while its spacing is within
    # `isotropy_tolerance` of the finest axis's. The same list feeds the NGFF
    # `physical_pixel_size` below and the chunk planner.
    physical_pixel_size = _physical_scales_for_dims(dims, reader)

    level_shapes = compute_level_shapes(
        base_shape, dims, physical_pixel_size, canonical.dtype, geometry
    )

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

    # ADR-0010 (#117): ``None`` unless the caller asked for shards, in which
    # case the chunk stops being an object and becomes purely a read unit. The
    # planner returns whole multiples of the chunk shapes just planned, so the
    # two grids nest by construction rather than by the writer's validation.
    shard_shapes = plan_level_shard_shapes(
        chunk_shapes, level_shapes, dims, physical_pixel_size, canonical.dtype, geometry
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
        shard_shape=(
            [list(s) for s in shard_shapes] if shard_shapes is not None else None
        ),
        creator_info=creator_info,
    )

    # ADR-0010 (#87): the writer pools each level from the one below it with
    # whichever kernel the policy names — mean by default, max for sparse
    # labels — writing them one at a time rather than as one graph (#111).
    writer.write_pyramid(canonical.data, geometry=geometry)

    # Only run the extra contrast ops when we actually have channels to update.
    # A scene with no C dim and no `channels` argument has no omero.channels
    # to rewrite, so there's nothing to compute. Runs against the written
    # coarsest level, after the pyramid is on disk (#114).
    run_contrast = contrast_percentile is not None and channel_count > 0
    if run_contrast:
        contrast_ops = _channel_contrast_ops(
            writer.read_level(-1), dims, channel_count, float(contrast_percentile)
        )
        contrast_stats = _pair_contrast_results(
            da.compute(*contrast_ops), channel_count
        )
        _update_omero_window_start_end(store_path, contrast_stats)
    else:
        contrast_stats = []

    record = {
        "scene_index": scene_index,
        "scene_name": scene_name,
        "image_name": name,
        "dims": dims,
        "level_shapes": [list(s) for s in level_shapes],
        "chunk_shapes": [list(c) for c in chunk_shapes],
        "shard_shapes": (
            [list(s) for s in shard_shapes] if shard_shapes is not None else None
        ),
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
