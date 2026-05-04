"""Pyramid level shape computation and mean-pool downsampling.

Replaces bioio-ome-zarr's built-in nearest-neighbor downsampling, which produces
aliasing artifacts on intensity (fluorescence) imagery. Mean-pool is the right
default for fluorescence; for label maps a different downsampler would be needed
(out of scope for v0.1).
"""

from collections.abc import Sequence

import dask.array as da
import numpy as np

SPATIAL_DIMS: frozenset[str] = frozenset({"Y", "X"})


def compute_level_shapes(
    base_shape: Sequence[int],
    dims: Sequence[str],
    min_size: int = 256,
) -> list[tuple[int, ...]]:
    """Compute pyramid level shapes by halving Y and X until the next halving
    would put the smallest spatial dim below ``min_size``. Non-spatial dims are
    preserved unchanged across all levels. Always returns at least the base
    shape.
    """
    base = tuple(int(s) for s in base_shape)
    spatial_indices = [i for i, d in enumerate(dims) if d in SPATIAL_DIMS]

    if not spatial_indices:
        return [base]

    levels: list[tuple[int, ...]] = [base]
    while True:
        prev = levels[-1]
        next_spatial_min = min(prev[i] // 2 for i in spatial_indices)
        if next_spatial_min < min_size:
            break
        nxt = tuple(s // 2 if i in spatial_indices else s for i, s in enumerate(prev))
        levels.append(nxt)

    return levels


def build_pyramid(
    arr: da.Array,
    dims: Sequence[str],
    level_shapes: Sequence[Sequence[int]],
) -> list[da.Array]:
    """Iteratively mean-pool ``arr`` to produce one dask array per level.

    Each level is a 2x downsample of the previous level along Y and X via
    ``dask.array.coarsen(np.mean, ...)``. The result of each coarsen is cast
    back to the input dtype so all levels share dtype.

    The returned list always begins with ``arr`` itself.
    """
    spatial_indices = [i for i, d in enumerate(dims) if d in SPATIAL_DIMS]
    target_dtype = arr.dtype

    levels: list[da.Array] = [arr]
    cur = arr
    for _ in range(1, len(level_shapes)):
        cur = da.coarsen(
            np.mean,
            cur,
            {i: 2 for i in spatial_indices},
            trim_excess=True,
        ).astype(target_dtype)
        levels.append(cur)

    return levels
