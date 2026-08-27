"""Integration tests for write_scene using FakeReader from conftest."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from bioio_ome_zarr.writers import Channel

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony.geometry import Geometry
from zarrmony.writers.scene import _dtype_window, write_scene


def test_write_scene_writes_pyramid_and_metadata(tmp_path) -> None:
    reader = FakeReader(
        scenes=["scene_a", "scene_b"],
        dims="TCYX",
        shape=(1, 2, 256, 256),
        pixel_sizes=FakePhysicalPixelSizes(Y=0.5, X=0.5),
        channel_names=["DAPI", "GFP"],
    )
    out = tmp_path / "scene.zarr"

    audit = write_scene(
        reader,
        scene_index=1,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=128),
    )

    assert audit["scene_index"] == 1
    assert audit["scene_name"] == "scene_b"
    assert audit["dims"] == ["T", "C", "Y", "X"]
    assert audit["channel_count"] == 2
    assert audit["physical_pixel_size"] == {"T": 1.0, "C": 1.0, "Y": 0.5, "X": 0.5}

    g = zarr.open_group(str(out), mode="r")
    assert "0" in g
    assert g["0"].shape == (1, 2, 256, 256)
    assert "1" in g
    assert g["1"].shape == (1, 2, 128, 128)
    assert "2" not in g

    assert g["0"][:].max() == 2
    assert g["0"][:].min() == 2
    assert g["1"][:].max() == 2


def test_write_scene_normalizes_non_canonical_axes(tmp_path) -> None:
    reader = FakeReader(scenes=["only"], dims="YXCT", shape=(64, 64, 2, 1))
    out = tmp_path / "norm.zarr"

    audit = write_scene(reader, scene_index=0, store_path=str(out))

    assert audit["dims"] == ["T", "C", "Y", "X"]
    assert audit["axis_normalization"]["was_transposed"] is True

    g = zarr.open_group(str(out), mode="r")
    assert g["0"].shape == (1, 2, 64, 64)


def test_write_scene_single_level_for_small_input(tmp_path) -> None:
    reader = FakeReader(scenes=["s"], dims="YX", shape=(64, 64))
    out = tmp_path / "small.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=256),
    )

    assert audit["level_shapes"] == [[64, 64]]
    g = zarr.open_group(str(out), mode="r")
    assert "0" in g
    assert "1" not in g


def test_write_scene_returns_audit_record(tmp_path) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    out = tmp_path / "audit.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=8),
    )

    assert audit["axis_normalization"]["was_transposed"] is False
    assert audit["axis_normalization"]["input_dims"] == ["T", "C", "Y", "X"]
    assert audit["axis_normalization"]["output_dims"] == ["T", "C", "Y", "X"]
    assert len(audit["level_shapes"]) == 3
    assert "mosaic" not in audit


def test_dtype_window_uint8_spans_full_byte_range() -> None:
    assert _dtype_window(np.dtype(np.uint8)) == {
        "min": 0,
        "max": 255,
        "start": 0,
        "end": 255,
    }


def test_dtype_window_uint16_spans_full_16bit_range() -> None:
    assert _dtype_window(np.dtype(np.uint16)) == {
        "min": 0,
        "max": 65535,
        "start": 0,
        "end": 65535,
    }


def test_dtype_window_uint32_spans_full_32bit_range() -> None:
    w = _dtype_window(np.dtype(np.uint32))
    assert w == {"min": 0, "max": 4294967295, "start": 0, "end": 4294967295}


def test_dtype_window_int16_covers_signed_range() -> None:
    assert _dtype_window(np.dtype(np.int16)) == {
        "min": -32768,
        "max": 32767,
        "start": -32768,
        "end": 32767,
    }


def test_dtype_window_float32_uses_normalized_range() -> None:
    # OMERO convention for normalized floats: 0.0/1.0, not the dtype extrema
    # (finfo.min/max would be ±3.4e38 and viewers can't render that).
    assert _dtype_window(np.dtype(np.float32)) == {
        "min": 0.0,
        "max": 1.0,
        "start": 0.0,
        "end": 1.0,
    }


def test_dtype_window_float64_uses_normalized_range() -> None:
    assert _dtype_window(np.dtype(np.float64)) == {
        "min": 0.0,
        "max": 1.0,
        "start": 0.0,
        "end": 1.0,
    }


@pytest.mark.parametrize(
    "dtype, expected_window",
    [
        (np.uint8, {"min": 0, "max": 255, "start": 0, "end": 255}),
        (np.uint16, {"min": 0, "max": 65535, "start": 0, "end": 65535}),
        (np.float32, {"min": 0.0, "max": 1.0, "start": 0.0, "end": 1.0}),
    ],
)
def test_write_scene_omero_window_matches_array_dtype(
    tmp_path, dtype, expected_window
) -> None:
    """Regression for #50 — the default-path OMERO window must span the array's
    dtype range, not the bioio-ome-zarr ``Channel`` fallback of 0–255. Otherwise
    every uint16/uint32/float32 store appears black-on-first-open in napari and
    OMERO because the display window clamps intensities into an 8-bit band.
    """
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32), dtype=dtype)
    out = tmp_path / f"{np.dtype(dtype).name}.zarr"

    write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=8),
    )

    g = zarr.open_group(str(out), mode="r")
    channels = g.attrs["ome"]["omero"]["channels"]
    assert len(channels) == 1
    assert channels[0]["window"] == expected_window


def test_write_scene_contrast_percentile_updates_omero_start_end(tmp_path) -> None:
    """Issue #53 — per-channel ``(min, 99.9th pct)`` should override the
    dtype-range ``start`` / ``end`` placeholder, while ``min`` / ``max`` (dtype
    range from issue #50) stay pinned.

    Uses a linear ramp so the coarse-pyramid approximation still produces a
    near-monotonic distribution: the tolerances below account for
    mean-pool-induced shift at the ends.
    """
    base = np.zeros((1, 2, 32, 32), dtype=np.uint16)
    base[0, 0] = np.arange(1024, dtype=np.uint16).reshape(32, 32)
    base[0, 1] = np.arange(2000, 3024, dtype=np.uint16).reshape(32, 32)
    xarr = xr.DataArray(da.from_array(base), dims=["T", "C", "Y", "X"])

    reader = FakeReader(
        scenes=["s"],
        dims="TCYX",
        shape=(1, 2, 32, 32),
        dtype=np.uint16,
        channel_names=["ch0", "ch1"],
    )
    out = tmp_path / "contrast.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=8),
        xarr_override=xarr,
        contrast_percentile=99.9,
    )

    g = zarr.open_group(str(out), mode="r")
    channels = g.attrs["ome"]["omero"]["channels"]

    for c in channels:
        assert c["window"]["min"] == 0
        assert c["window"]["max"] == 65535

    # Coarse (8x8) level of a 32x32 linear ramp — verified by hand: min~=49,
    # p99.9~=973. Tolerances bracket that with room for future coarsening
    # tweaks without becoming a change detector.
    ch0 = channels[0]["window"]
    assert ch0["start"] <= 60
    assert 950 <= ch0["end"] <= 1023

    ch1 = channels[1]["window"]
    assert 2000 <= ch1["start"] <= 2060
    assert 2950 <= ch1["end"] <= 3023

    contrast = audit["contrast"]
    assert contrast["percentile"] == 99.9
    assert contrast["method"] == "coarsest-pyramid-level"
    assert len(contrast["per_channel"]) == 2
    assert contrast["per_channel"][0]["channel_index"] == 0
    assert contrast["per_channel"][0]["start"] == ch0["start"]
    assert contrast["per_channel"][0]["end"] == ch0["end"]


def test_write_scene_contrast_percentile_none_leaves_dtype_window(tmp_path) -> None:
    """``contrast_percentile=None`` short-circuits the extra ops and leaves the
    dtype-range placeholder that #50 installs. Also verifies the audit dict
    omits the ``contrast`` block.
    """
    reader = FakeReader(
        scenes=["s"],
        dims="TCYX",
        shape=(1, 1, 32, 32),
        dtype=np.uint16,
        channel_names=["ch0"],
    )
    out = tmp_path / "no_contrast.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=8),
        contrast_percentile=None,
    )

    g = zarr.open_group(str(out), mode="r")
    window = g.attrs["ome"]["omero"]["channels"][0]["window"]
    assert window == {"min": 0, "max": 65535, "start": 0, "end": 65535}
    assert "contrast" not in audit


def test_write_scene_contrast_percentile_single_channel_no_c_dim(tmp_path) -> None:
    """Scenes without a C dim still have one implicit omero channel — the
    contrast code path must not crash and should record the single-channel
    bounds in the audit.
    """
    base = np.arange(1024, dtype=np.uint16).reshape(32, 32)
    xarr = xr.DataArray(da.from_array(base), dims=["Y", "X"])

    reader = FakeReader(
        scenes=["s"],
        dims="YX",
        shape=(32, 32),
        dtype=np.uint16,
    )
    out = tmp_path / "single.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=8),
        xarr_override=xarr,
        contrast_percentile=99.9,
    )

    g = zarr.open_group(str(out), mode="r")
    ome = g.attrs["ome"]
    if "omero" in ome and ome["omero"].get("channels"):
        window = ome["omero"]["channels"][0]["window"]
        assert window["min"] == 0
        assert window["max"] == 65535
        assert window["start"] <= 60
        assert 950 <= window["end"] <= 1023
    # Either way, the audit records the contrast block when at least one
    # channel is emitted; a no-channel path returns [] and is a no-op.
    assert "contrast" in audit or audit.get("channel_count", 0) == 0


def test_write_scene_records_mosaic_summary(tmp_path) -> None:
    mosaic = {"stitched": True, "tile_count": 12, "tile_shape": {"Y": 5048, "X": 5048}}
    reader = FakeReader(
        scenes=["mosaic_scene"],
        dims="TCYX",
        shape=(1, 1, 64, 64),
        mosaic_summary=mosaic,
    )
    out = tmp_path / "mosaic.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=8),
    )

    assert audit["mosaic"] == mosaic


# --- RGB scenes --------------------------------------------------------------
#
# Bio-Formats reports a colour plane as one channel of three interleaved
# samples (C=1, S=3). Whole-slide formats carry an RGB `label` and `macro`
# scene beside the fluorescence scan, and before the fold the first of them
# aborted the whole conversion with UnsupportedAxesError.


def test_write_scene_converts_an_rgb_scene(tmp_path) -> None:
    reader = FakeReader(
        scenes=["label"],
        dims="TCZYXS",
        shape=(1, 1, 1, 64, 48, 3),
        dtype=np.uint8,
        channel_names=["Channel:0:0"],
    )
    out = tmp_path / "label.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=32),
    )

    assert audit["dims"] == ["T", "C", "Z", "Y", "X"]
    assert audit["channel_count"] == 3
    assert audit["axis_normalization"]["rgb_samples_folded"] is True
    assert audit["axis_normalization"]["input_dims"] == [
        "T",
        "C",
        "Z",
        "Y",
        "X",
        "S",
    ]

    g = zarr.open_group(str(out), mode="r")
    assert g["0"].shape == (1, 3, 1, 64, 48)


def test_rgb_channels_get_the_primaries_not_the_palette(tmp_path) -> None:
    """The ADR-0007 palette encodes fluorescence emission bands and has no
    entry for "Red"/"Green"/"Blue", so routing folded samples through it would
    composite a colour photograph's red sample in cyan.
    """
    reader = FakeReader(
        scenes=["macro image"],
        dims="TCZYXS",
        shape=(1, 1, 1, 32, 32, 3),
        dtype=np.uint8,
    )
    out = tmp_path / "macro.zarr"

    write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        geometry=Geometry(pyramid_min_size=16),
    )

    g = zarr.open_group(str(out), mode="r")
    channels = g.attrs["ome"]["omero"]["channels"]
    assert [c["label"] for c in channels] == ["Red", "Green", "Blue"]
    assert [c["color"] for c in channels] == ["ff0000", "00ff00", "0000ff"]
    # uint8 RGB, so the window is the full 8-bit range rather than a placeholder.
    assert channels[0]["window"]["min"] == 0
    assert channels[0]["window"]["max"] == 255


def test_stale_caller_channels_are_replaced_after_a_fold(tmp_path) -> None:
    """``api.convert`` derives channels from ``reader.channel_names`` before
    ``write_scene`` runs, so for an RGB scene it hands us exactly one channel
    describing the pre-fold state. Writing that against a C of 3 would put a
    single mislabelled entry in omero.
    """
    reader = FakeReader(
        scenes=["label"],
        dims="TCZYXS",
        shape=(1, 1, 1, 32, 32, 3),
        dtype=np.uint8,
    )
    out = tmp_path / "stale.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        channels=[Channel(label="Channel:0:0", color="ffffff", window=None)],
        geometry=Geometry(pyramid_min_size=16),
    )

    assert audit["channel_count"] == 3
    g = zarr.open_group(str(out), mode="r")
    labels = [c["label"] for c in g.attrs["ome"]["omero"]["channels"]]
    assert labels == ["Red", "Green", "Blue"]


# --- lazy reader blocks ------------------------------------------------------
#
# bioio-bioformats builds its dask graph out of LazyBioArray handles rather
# than materialised arrays. Level 0 writes fine (zarr only needs __array__),
# but dask.array.coarsen calls .reshape on every block, so the pyramid dies.


class _LazyBlock:
    """Stand-in for bioio-bioformats' ``LazyBioArray``.

    Exposes ``__array__``, ``shape``, ``dtype``, ``ndim`` and ``__getitem__``
    — enough for dask to keep it as the array's ``_meta`` and for zarr's
    level-0 write to succeed, while ``coarsen`` and the contrast pass fail on
    the missing ``.reshape`` / ``.mean``. ``__getitem__`` matters: without it
    dask's ``meta_from_array`` cannot slice the prototype and silently
    substitutes an ndarray meta, which would hide the very thing under test.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.ndim = arr.ndim

    def __array__(self, dtype=None, copy=None):
        return self._arr if dtype is None else self._arr.astype(dtype)

    def __getitem__(self, key):
        return _LazyBlock(self._arr[key])

    def astype(self, dtype, **kwargs):
        return _LazyBlock(self._arr.astype(dtype))


def _lazy_backed_xarr(shape, dims, chunks):
    """A dask-backed DataArray whose blocks are ``_LazyBlock``, like bioio's."""
    n = int(np.prod(shape))
    arr = (np.arange(n, dtype=np.uint16) % 4096).reshape(shape)
    lazy = da.from_array(arr, chunks=chunks).map_blocks(
        _LazyBlock, dtype=arr.dtype, meta=_LazyBlock(np.empty((0,) * len(shape)))
    )
    return xr.DataArray(lazy, dims=list(dims))


def test_lazy_reader_blocks_do_not_break_the_pyramid(tmp_path) -> None:
    reader = FakeReader(scenes=["overview"], dims="TCYX", shape=(1, 1, 128, 128))
    out = tmp_path / "lazy.zarr"

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        xarr_override=_lazy_backed_xarr((1, 1, 128, 128), "TCYX", (1, 1, 64, 64)),
        geometry=Geometry(pyramid_min_size=64),
    )

    # More than one level is the whole point: level 0 wrote before this fix.
    assert len(audit["level_shapes"]) > 1
    g = zarr.open_group(str(out), mode="r")
    assert g["0"].shape == (1, 1, 128, 128)
    assert g["1"].shape == (1, 1, 64, 64)


def test_lazy_blocks_write_the_right_pixels(tmp_path) -> None:
    shape = (1, 1, 64, 64)
    expected = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 4096).reshape(shape)
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=shape)
    out = tmp_path / "lazypx.zarr"

    write_scene(
        reader,
        scene_index=0,
        store_path=str(out),
        xarr_override=_lazy_backed_xarr(shape, "TCYX", (1, 1, 32, 32)),
        geometry=Geometry(pyramid_min_size=64),
    )

    g = zarr.open_group(str(out), mode="r")
    assert np.array_equal(g["0"][:], expected)


# ---------- level-by-level writing (#111, #114) ----------


def _counting_xarr(shape, dims, chunks, tally):
    """A dask-backed DataArray that appends to ``tally`` on every block read.

    Stands in for the reader: it is the *source* graph, so counting its block
    reads counts how many times the conversion goes back to the raw pixels.
    """
    base = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 997).reshape(shape)

    def touch(block):
        tally.append(1)
        return block

    # ``meta`` given explicitly: dask would otherwise infer it by calling
    # ``touch`` on a zero-element prototype at graph-construction time, which is
    # not a read and must not be counted as one.
    data = da.from_array(base, chunks=chunks).map_blocks(
        touch, dtype=base.dtype, meta=np.empty((0,) * len(shape), dtype=base.dtype)
    )
    return xr.DataArray(data, dims=list(dims)), base


def _store_spy(monkeypatch, source_layer: str, tally: list[int]) -> list[dict]:
    """Record, per ``da.store`` call, what the writer handed the scheduler.

    ``source_layer`` is the reader graph's layer name, so ``reaches_reader``
    answers the question #111 turns on: does this level's graph still stretch
    all the way back to the raw pixels?
    """
    seen: list[dict] = []
    real_store = da.store

    def spy(sources, targets, **kwargs):
        seen.append(
            {
                "shape": tuple(int(s) for s in sources.shape),
                "reaches_reader": any(
                    str(layer).startswith(source_layer) for layer in sources.dask.layers
                ),
                "deferred": kwargs.get("compute", True) is False,
                "source_reads_before": len(tally),
            }
        )
        return real_store(sources, targets, **kwargs)

    monkeypatch.setattr(da, "store", spy)
    return seen


def test_each_level_is_written_from_the_one_below_it(tmp_path, monkeypatch) -> None:
    """The #111 restructure, asserted on the graphs the writer actually submits.

    Three properties, and the old single-``da.compute`` pyramid broke all three:
    it deferred every level (``compute=False``) into one graph, so nothing was
    written until the last level's graph had been *constructed* — hours in
    ``Task.__init__`` on a whole-slide input — and that last graph reached back
    through every intermediate level to the reader. Now each level is its own
    compute, only level 0 names the reader, and the source is fully read by the
    time level 1 is submitted.

    Note this checks the graph's *shape*, not throughput: at test scale dask
    keeps the whole pyramid in memory and reuses level 0's blocks, so the 4.2x
    read amplification measured in production does not reproduce here. The
    structure is what the amplification followed from.
    """
    shape = (1, 1, 128, 128)
    tally: list[int] = []
    xarr, _ = _counting_xarr(shape, "TCYX", (1, 1, 32, 32), tally)
    seen = _store_spy(monkeypatch, xarr.data.name, tally)
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=shape)

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(tmp_path / "onepass.zarr"),
        xarr_override=xarr,
        geometry=Geometry(pyramid_min_size=16),
    )

    assert len(audit["level_shapes"]) >= 3, "need a deep enough pyramid to matter"
    assert [call["shape"] for call in seen] == [tuple(s) for s in audit["level_shapes"]]
    assert not any(call["deferred"] for call in seen)
    assert [call["reaches_reader"] for call in seen] == [True] + [False] * (
        len(seen) - 1
    )
    # 128/32 squared source blocks, all of them consumed by level 0 alone.
    assert seen[1]["source_reads_before"] == 16
    assert len(tally) == 16


def test_contrast_does_not_send_the_writer_back_to_the_reader(tmp_path) -> None:
    """#114: contrast used to be fused into the pyramid's single compute, on the
    theory that sharing a graph meant sharing the reads. It stalled an 80+ GB
    slide outright — 0 bytes in 10 minutes. Computed off the written coarse
    level instead, it costs the source nothing: the block count is identical
    with contrast on and off.
    """
    shape = (1, 2, 128, 128)
    reader = FakeReader(
        scenes=["s"], dims="TCYX", shape=shape, channel_names=["DAPI", "GFP"]
    )
    geometry = Geometry(pyramid_min_size=16)

    off: list[int] = []
    xarr_off, _ = _counting_xarr(shape, "TCYX", (1, 1, 32, 32), off)
    write_scene(
        reader,
        scene_index=0,
        store_path=str(tmp_path / "off.zarr"),
        xarr_override=xarr_off,
        geometry=geometry,
        contrast_percentile=None,
    )

    on: list[int] = []
    xarr_on, _ = _counting_xarr(shape, "TCYX", (1, 1, 32, 32), on)
    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(tmp_path / "on.zarr"),
        xarr_override=xarr_on,
        geometry=geometry,
        contrast_percentile=99.9,
    )

    assert "contrast" in audit, "the contrast pass has to have actually run"
    assert len(on) == len(off)


def test_written_pyramid_matches_the_lazy_one(tmp_path) -> None:
    """Pixel-for-pixel: going through the store changes no value."""
    shape = (1, 1, 64, 64)
    rng = np.random.default_rng(3)
    base = rng.integers(0, 4096, size=shape, dtype=np.uint16)
    xarr = xr.DataArray(
        da.from_array(base, chunks=(1, 1, 16, 16)), dims=["T", "C", "Y", "X"]
    )
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=shape)
    geometry = Geometry(pyramid_min_size=8)

    audit = write_scene(
        reader,
        scene_index=0,
        store_path=str(tmp_path / "match.zarr"),
        xarr_override=xarr,
        geometry=geometry,
    )

    from zarrmony.writers.pyramid import build_pyramid

    expected = build_pyramid(
        da.from_array(base, chunks=(1, 1, 16, 16)), audit["level_shapes"], geometry
    )
    g = zarr.open_group(str(tmp_path / "match.zarr"), mode="r")
    for i, level in enumerate(expected):
        np.testing.assert_array_equal(g[str(i)][:], level.compute())


def test_write_pyramid_rejects_a_base_that_is_not_level_zero(tmp_path) -> None:
    """The old signature took every level and could check the list length; this
    one derives the levels itself, so level 0's shape is the whole contract."""
    from zarrmony.writers.scene import ZarrmonyWriter

    writer = ZarrmonyWriter(
        store=str(tmp_path / "mismatch.zarr"),
        level_shapes=[(1, 1, 32, 32), (1, 1, 16, 16)],
        dtype=np.uint16,
        zarr_format=3,
    )
    with pytest.raises(ValueError, match="does not match the declared level-0 shape"):
        writer.write_pyramid(da.zeros((1, 1, 16, 16), dtype=np.uint16))


def test_ndarray_backed_readers_keep_their_graph(tmp_path) -> None:
    """The coercion must be a no-op for every other reader — otherwise it adds
    a graph layer to inputs that never needed one.
    """
    from zarrmony.writers.scene import _ensure_ndarray_blocks

    plain = xr.DataArray(
        da.zeros((1, 1, 32, 32), chunks=(1, 1, 16, 16), dtype=np.uint16),
        dims=["T", "C", "Y", "X"],
    )
    assert _ensure_ndarray_blocks(plain) is plain

    numpy_backed = xr.DataArray(np.zeros((4, 4), np.uint16), dims=["Y", "X"])
    assert _ensure_ndarray_blocks(numpy_backed) is numpy_backed
