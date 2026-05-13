import dask.array as da
import numpy as np

from zarrmony.writers.pyramid import build_pyramid, compute_level_shapes


def test_compute_levels_2048_floor_256() -> None:
    shapes = compute_level_shapes(
        (1, 4, 2048, 2048), ["T", "C", "Y", "X"], min_size=256
    )
    assert shapes == [
        (1, 4, 2048, 2048),
        (1, 4, 1024, 1024),
        (1, 4, 512, 512),
        (1, 4, 256, 256),
    ]


def test_compute_levels_small_image_no_pyramid() -> None:
    # Halving 100 yields 50, below floor of 256, so only the base remains.
    shapes = compute_level_shapes((1, 1, 100, 100), ["T", "C", "Y", "X"], min_size=256)
    assert shapes == [(1, 1, 100, 100)]


def test_compute_levels_non_spatial_dims_preserved() -> None:
    shapes = compute_level_shapes((10, 4, 1, 1024, 1024), "TCZYX", min_size=256)
    assert shapes == [
        (10, 4, 1, 1024, 1024),
        (10, 4, 1, 512, 512),
        (10, 4, 1, 256, 256),
    ]


def test_compute_levels_no_spatial_dims() -> None:
    shapes = compute_level_shapes((10, 4), ["T", "C"], min_size=256)
    assert shapes == [(10, 4)]


def test_compute_levels_anisotropic_yx() -> None:
    # X (1500) gets to 187 first; level should stop when min(Y//2, X//2) < min_size
    shapes = compute_level_shapes(
        (1, 1, 4000, 1500), ["T", "C", "Y", "X"], min_size=256
    )
    assert shapes == [(1, 1, 4000, 1500), (1, 1, 2000, 750), (1, 1, 1000, 375)]
    # next would be (500, 187), 187 < 256, so we stop


def test_build_pyramid_mean_pool() -> None:
    base = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    base_da = da.from_array(base, chunks=(2, 2))
    levels = build_pyramid(base_da, ["Y", "X"], [(4, 4), (2, 2)])
    assert len(levels) == 2
    level1 = levels[1].compute()
    expected = np.array([[3.5, 5.5], [11.5, 13.5]], dtype=np.float32)
    np.testing.assert_array_equal(level1, expected)


def test_build_pyramid_preserves_dtype() -> None:
    base_da = da.from_array(np.full((4, 4), 100, dtype=np.uint16), chunks=(2, 2))
    levels = build_pyramid(base_da, ["Y", "X"], [(4, 4), (2, 2)])
    assert levels[1].dtype == np.uint16
    assert levels[1].compute().tolist() == [[100, 100], [100, 100]]


def test_build_pyramid_preserves_non_spatial_dims() -> None:
    # 5D input — only Y, X get coarsened
    base_da = da.from_array(
        np.zeros((1, 2, 1, 8, 8), dtype=np.uint16), chunks=(1, 2, 1, 4, 4)
    )
    levels = build_pyramid(
        base_da,
        ["T", "C", "Z", "Y", "X"],
        [(1, 2, 1, 8, 8), (1, 2, 1, 4, 4), (1, 2, 1, 2, 2)],
    )
    assert [tuple(lv.shape) for lv in levels] == [
        (1, 2, 1, 8, 8),
        (1, 2, 1, 4, 4),
        (1, 2, 1, 2, 2),
    ]
