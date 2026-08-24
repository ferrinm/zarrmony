"""Pyramid level shapes (anisotropy-aware) and mean-pool downsampling.

Replaces bioio-ome-zarr's built-in nearest-neighbor downsampling, which produces
aliasing artifacts on intensity (fluorescence) imagery. Mean-pool is the right
default for fluorescence; for label maps a different downsampler would be needed
(``Geometry.downsample_method="max"`` lands in a later ADR-0010 slice).

Which axes shrink from one level to the next is an ADR-0010 geometry decision
(:func:`compute_level_shapes`); how the pixels get there is this module's job
(:func:`build_pyramid`). The two meet at the level-shape list: ``build_pyramid``
derives its coarsen factors from consecutive entries of it, so uniform and
per-axis-varying downsampling are one code path.
"""

from collections.abc import Sequence

import dask.array as da
import numpy as np

from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    SPATIAL_AXES,
    Geometry,
    spacings_for_level,
)

#: The axes the *depth* rule is judged on. Pyramid depth is still the
#: pre-ADR-0010 ``pyramid_min_size`` rule — stop when the smaller of Y/X would
#: fall below it — because depth is what a viewer's zoom-out budget cares about
#: and Y/X is what it sees. Z participates in *downsampling* (:data:`SPATIAL_AXES`)
#: without participating in the depth decision; making it a fourth vote would
#: collapse a 3-plane stack's pyramid to a single level (ADR-0010, rejected
#: options).
LATERAL_DIMS: frozenset[str] = frozenset({"Y", "X"})


def _halving_floor(dim: str, geometry: Geometry) -> int:
    """The smallest post-halving extent allowed on one axis.

    :data:`~zarrmony.geometry.Geometry.axis_floor` (32) everywhere, except that
    on Y/X it is capped by ``pyramid_min_size``. The two floors overlap there:
    the depth rule already stops the pyramid before Y or X can reach
    ``axis_floor`` under any default policy (256 > 32), so the cap only bites
    when a caller has deliberately asked for a *lower* floor than 32 — and a
    caller who passes ``pyramid_min_size=8`` is entitled to the levels they
    asked for rather than to a 32-voxel default silently overriding them.
    ADR-0010 requires the change be monotone: no existing conversion loses a
    level.
    """
    if dim in LATERAL_DIMS:
        return min(geometry.axis_floor, geometry.pyramid_min_size)
    return geometry.axis_floor


def _next_level_shape(
    shape: tuple[int, ...],
    dims: Sequence[str],
    spacings: Sequence[float],
    geometry: Geometry,
) -> tuple[int, ...]:
    """Halve every spatial axis eligible at this level; leave the rest alone.

    An axis is eligible when both hold:

    1. halving it keeps it at or above its floor (:func:`_halving_floor`), so a
       thin axis is never ground away — a 3-plane stack keeps its 3 planes at
       every level; and
    2. its physical spacing at *this* level is within
       ``geometry.isotropy_tolerance`` of the finest still-halvable axis's, so
       the pyramid moves toward isotropy and the scarce axis is spent last.

    The yardstick in (2) is the finest axis *among those (1) leaves halvable*,
    not the finest axis outright. An axis pinned at its floor can no longer be
    spent, so it must not go on deciding what the others may spend: a
    single-plane Z holds its level-0 spacing forever while Y and X double
    theirs every level, and left in the reference set it would declare Y and X
    "too coarse to halve" after two levels — collapsing the pyramid of every 2D
    acquisition, which is the one shape this rule must not touch.

    Returns ``shape`` unchanged when no axis is eligible, which is the caller's
    signal that the pyramid has bottomed out.
    """
    halvable = [
        i
        for i, d in enumerate(dims)
        if d in SPATIAL_AXES and shape[i] // 2 >= _halving_floor(d, geometry)
    ]
    if not halvable:
        return shape
    finest = min(spacings[i] for i in halvable)

    out = list(shape)
    for i in halvable:
        # Rounded so an axis that is isotropic in exact arithmetic is not
        # excluded by float drift in the shape-ratio spacings (an odd extent
        # halves to a ratio a hair off 2), which matters most at
        # isotropy_tolerance=1.0.
        if round(spacings[i] / finest, 9) > geometry.isotropy_tolerance:
            continue
        out[i] = shape[i] // 2
    return tuple(out)


def compute_level_shapes(
    base_shape: Sequence[int],
    dims: Sequence[str],
    spacings_um: Sequence[float],
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> list[tuple[int, ...]]:
    """Per-level shapes for one array, halving toward isotropy (ADR-0010).

    Each level halves every spatial axis whose physical spacing is within
    ``geometry.isotropy_tolerance`` of the finest axis's *at that level*, subject
    to a per-axis floor — see :func:`_next_level_shape`. Levels therefore move
    toward isotropy: on a 10:1 confocal stack Y and X halve alone until their
    spacing has caught up with Z's, and only then does Z start halving too. On
    near-isotropic data every spatial axis halves at every level, so a level is
    ⅛ of its parent rather than ¼.

    Depth is still the pre-ADR-0010 rule: stop when the next level's smaller
    lateral (Y/X) extent would fall below ``geometry.pyramid_min_size``. Z takes
    no part in the depth decision — making it a fourth vote is the regression
    ADR-0010 rejects by name, since a 3-plane stack would then get no pyramid at
    all. Depth also stops once Y and X are both at their floor, which is what
    keeps a tall thin volume from growing a tail of levels that only thin Z.

    Non-spatial dims (T, C) are preserved unchanged across all levels, and the
    base shape is always returned as level 0.

    :param base_shape: Level 0's extent, one entry per axis.
    :param dims: Axis names in the same order (e.g. ``"TCZYX"``).
    :param spacings_um: Level 0's physical spacing per axis, ``1.0`` for the
        non-spatial ones — the list
        :func:`~zarrmony.writers.scene._physical_scales_for_dims` builds. Later
        levels' spacings are derived from it by
        :func:`~zarrmony.geometry.spacings_for_level`. Missing or nonsense
        values degrade to ``1.0``, which reads as "isotropic with every other
        unknown axis".
    :param geometry: The policy supplying ``isotropy_tolerance``, ``axis_floor``
        and ``pyramid_min_size``.
    """
    base = tuple(int(s) for s in base_shape)
    if not (len(base) == len(dims) == len(spacings_um)):
        raise ValueError(
            f"compute_level_shapes needs one entry per axis; got {len(base)} dims, "
            f"{len(dims)} axis names and {len(spacings_um)} spacings"
        )

    lateral_indices = [i for i, d in enumerate(dims) if d in LATERAL_DIMS]
    if not lateral_indices:
        # No Y/X means no depth rule to apply. Rather than invent one for a
        # shape no microscope produces, write a single level.
        return [base]

    levels: list[tuple[int, ...]] = [base]
    while True:
        prev = levels[-1]
        if all(
            prev[i] // 2 < _halving_floor(dims[i], geometry) for i in lateral_indices
        ):
            # Y and X are both at their floor, so no further level can be
            # smaller *laterally* — and a tail of levels that only thin Z is
            # not what a viewer zooming out is asking for. Note this is about
            # what the laterals can ever do, not what they do this level: an
            # oversampled-Z stack legitimately spends Z alone for a level or
            # two while Y/X wait for its spacing to catch up.
            break
        nxt = _next_level_shape(
            prev, dims, spacings_for_level(spacings_um, base, prev), geometry
        )
        if nxt == prev:
            # Every spatial axis is out of tolerance or at its floor.
            break
        if min(nxt[i] for i in lateral_indices) < geometry.pyramid_min_size:
            break
        levels.append(nxt)

    return levels


def build_pyramid(
    arr: da.Array,
    level_shapes: Sequence[Sequence[int]],
) -> list[da.Array]:
    """Iteratively mean-pool ``arr`` into one dask array per level shape.

    Coarsen factors are read off the level shapes themselves — axis ``i``'s
    factor at level ``n`` is ``level_shapes[n-1][i] // level_shapes[n][i]`` — so
    a pyramid that halves Z, Y and X and one that halves only Y and X take the
    same code path, and whatever :func:`compute_level_shapes` decided is what
    gets built. Axes with a factor of 1 (T, C, and any spatial axis held back by
    the isotropy or floor rule) are left untouched.

    Each coarsen is cast back to the input dtype so all levels share dtype, and
    trims a trailing row rather than padding when an extent is odd — which is
    exactly the floor division :func:`compute_level_shapes` predicted.

    The returned list always begins with ``arr`` itself.
    """
    levels_shapes = [tuple(int(s) for s in shape) for shape in level_shapes]
    if not levels_shapes:
        raise ValueError("build_pyramid needs at least one level shape")

    target_dtype = arr.dtype
    levels: list[da.Array] = [arr]
    cur = arr
    for prev_shape, next_shape in zip(levels_shapes, levels_shapes[1:], strict=False):
        if len(prev_shape) != len(next_shape):
            raise ValueError(
                f"level shapes must have the same number of axes; got "
                f"{prev_shape} then {next_shape}"
            )
        factors: dict[int, int] = {}
        for i, (prev_dim, next_dim) in enumerate(
            zip(prev_shape, next_shape, strict=True)
        ):
            if next_dim < 1 or next_dim > prev_dim:
                raise ValueError(
                    f"level shape {next_shape} does not downsample {prev_shape} "
                    f"on axis {i}: a level must be positive and no larger than "
                    f"its parent"
                )
            factor = prev_dim // next_dim
            if factor > 1:
                factors[i] = factor
        if factors:
            cur = da.coarsen(np.mean, cur, factors, trim_excess=True).astype(
                target_dtype
            )
        levels.append(cur)

    return levels
