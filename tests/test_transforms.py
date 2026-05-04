import numpy as np
import pytest
import xarray as xr

from zarrmony.transforms import UnsupportedAxesError, normalize_axes


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
