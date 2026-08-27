import numpy as np
import pytest
import xarray as xr

from zarrmony.transforms import (
    UnsupportedAxesError,
    fold_samples_axis,
    normalize_axes,
)


def _xarr(shape: tuple[int, ...], dims: str) -> xr.DataArray:
    return xr.DataArray(np.zeros(shape, dtype=np.uint16), dims=list(dims))


def test_canonical_input_unchanged() -> None:
    xa = _xarr((1, 4, 1, 100, 100), "TCZYX")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["T", "C", "Z", "Y", "X"]
    assert rec["was_transposed"] is False


def test_reorder_to_canonical() -> None:
    xa = _xarr((100, 100, 4, 1), "YXCT")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["T", "C", "Y", "X"]
    assert rec["input_dims"] == ["Y", "X", "C", "T"]
    assert rec["output_dims"] == ["T", "C", "Y", "X"]
    assert rec["was_transposed"] is True


def test_reject_unsupported_dim() -> None:
    xa = _xarr((6, 1, 4, 100, 100), "MTCYX")
    with pytest.raises(UnsupportedAxesError, match="M"):
        normalize_axes(xa)


def test_reject_duplicate_dims() -> None:
    # Build an array with duplicate dim names manually
    xa = xr.DataArray(np.zeros((2, 2)), dims=["X", "X"])
    with pytest.raises(UnsupportedAxesError, match="duplicates"):
        normalize_axes(xa)


def test_partial_dims_preserved() -> None:
    xa = _xarr((4, 100, 100), "CYX")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["C", "Y", "X"]
    assert rec["was_transposed"] is False


def test_zyx_only_canonical() -> None:
    xa = _xarr((10, 100, 100), "ZYX")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["Z", "Y", "X"]
    assert rec["was_transposed"] is False


def test_data_round_trips_through_transpose() -> None:
    arr = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    xa = xr.DataArray(arr, dims=["Y", "X", "C", "T"])
    out, _ = normalize_axes(xa)
    # Picking a known element to confirm transpose correctness
    assert out.sel(T=0, C=0, Y=1, X=2).item() == xa.sel(Y=1, X=2, C=0, T=0).item()


# --- samples axis (RGB) ------------------------------------------------------
#
# Bio-Formats reports a colour plane as C=1, S=3. This is how every whole-slide
# format's `label` and `macro` scene arrives, so it has to normalize rather than
# raise.


def test_rgb_samples_become_channels() -> None:
    xa = _xarr((1, 1, 1, 40, 50, 3), "TCZYXS")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["T", "C", "Z", "Y", "X"]
    assert out.sizes["C"] == 3
    assert rec["rgb_samples_folded"] is True
    # The reader's own dims are what the audit should show — the "S" is the
    # only evidence the input was a colour image.
    assert rec["input_dims"] == ["T", "C", "Z", "Y", "X", "S"]
    assert rec["output_dims"] == ["T", "C", "Z", "Y", "X"]


def test_folded_channels_are_named_for_the_primaries() -> None:
    xa = _xarr((1, 1, 1, 4, 5, 3), "TCZYXS")
    out, _ = normalize_axes(xa)
    assert [str(v) for v in out.coords["C"].values] == ["Red", "Green", "Blue"]


def test_rgba_folds_to_four_channels() -> None:
    xa = _xarr((1, 1, 1, 4, 5, 4), "TCZYXS")
    out, rec = normalize_axes(xa)
    assert out.sizes["C"] == 4
    assert rec["rgb_samples_folded"] is True
    assert [str(v) for v in out.coords["C"].values] == [
        "Red",
        "Green",
        "Blue",
        "Alpha",
    ]


def test_wide_samples_axis_gets_positional_labels() -> None:
    # Not a colour model we can name; better positional than a wrong mapping.
    xa = _xarr((4, 5, 6), "YXS")
    out, rec = normalize_axes(xa)
    assert rec["rgb_samples_folded"] is True
    assert list(out.dims) == ["C", "Y", "X"]
    assert [str(v) for v in out.coords["C"].values] == [f"S:{i}" for i in range(6)]


def test_samples_pixels_survive_the_fold() -> None:
    arr = np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3)
    xa = xr.DataArray(arr, dims=["Y", "X", "S"])
    out, _ = normalize_axes(xa)
    assert list(out.dims) == ["C", "Y", "X"]
    # Sample s at (y, x) must land in channel s, unshuffled.
    for s in range(3):
        assert np.array_equal(out.isel(C=s).values, arr[:, :, s])


def test_degenerate_samples_axis_is_dropped_not_folded() -> None:
    # S=1 is a greyscale image that happens to carry the axis. Folding it would
    # leave a channel axis the caller then has to explain.
    xa = _xarr((1, 1, 1, 4, 5, 1), "TCZYXS")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["T", "C", "Z", "Y", "X"]
    assert out.sizes["C"] == 1
    assert rec["rgb_samples_folded"] is False


def test_no_samples_axis_reports_no_fold() -> None:
    xa = _xarr((1, 4, 1, 10, 10), "TCZYX")
    _, rec = normalize_axes(xa)
    assert rec["rgb_samples_folded"] is False


def test_samples_axis_without_a_channel_axis_folds() -> None:
    xa = _xarr((10, 10, 3), "YXS")
    out, rec = normalize_axes(xa)
    assert list(out.dims) == ["C", "Y", "X"]
    assert rec["rgb_samples_folded"] is True


def test_multichannel_and_multisample_is_refused() -> None:
    # Interleaved samples belong to one channel, so C>1 and S>1 together mean
    # we would have to invent an ordering for the 12 resulting channels.
    xa = _xarr((1, 4, 1, 10, 10, 3), "TCZYXS")
    with pytest.raises(UnsupportedAxesError, match="without\nguessing|guessing"):
        normalize_axes(xa)


def test_fold_is_metadata_only_for_dask_input() -> None:
    da = pytest.importorskip("dask.array")
    chunked = da.zeros((1, 1, 1, 40, 50, 3), chunks=(1, 1, 1, 20, 25, 3))
    xa = xr.DataArray(chunked, dims=["T", "C", "Z", "Y", "X", "S"])
    out, _ = normalize_axes(xa)
    # A rechunk here would silently multiply the graph on a whole-slide scene:
    # the three samples stay in one chunk, exactly as they arrived.
    assert out.data.chunksize == (1, 3, 1, 20, 25)
    assert out.data.npartitions == chunked.npartitions


def test_fold_samples_axis_is_usable_on_its_own() -> None:
    xa = _xarr((4, 5, 3), "YXS")
    folded, did_fold = fold_samples_axis(xa)
    assert did_fold is True
    assert list(folded.dims) == ["Y", "X", "C"]
