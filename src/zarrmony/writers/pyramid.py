"""Pyramid level shapes (anisotropy-aware) and mean-pool downsampling.

Replaces bioio-ome-zarr's built-in nearest-neighbor downsampling, which produces
aliasing artifacts on intensity (fluorescence) imagery. Mean-pool is the right
default for fluorescence; for label maps a different downsampler would be needed
(``Geometry.downsample_method="max"`` lands in a later ADR-0010 slice).

Which axes shrink from one level to the next, and how many levels there are, is
an ADR-0010 geometry decision (:func:`compute_level_shapes`); how the pixels get
there is this module's job (:func:`build_pyramid`). The two meet at the
level-shape list: ``build_pyramid`` derives its coarsen factors from consecutive
entries of it, so uniform and per-axis-varying downsampling are one code path.

Depth answers a question about the *output*, not about the input: "does a level
exist that a viewer can hold whole?" :func:`is_coarse_level` is that question
asked of one level, and :func:`coarse_level_index` is the answer recorded in the
audit — checkable at conversion time rather than discoverable in a viewport,
which is how the defect ADR-0010 fixes was found in the first place.

Four places below are finer-grained than ADR-0010's summary paragraph — the
isotropy yardstick is the finest *still-halvable* axis, depth also stops once
Y and X are both floor-frozen, the axis floor is capped by ``pyramid_min_size``
on Y/X, and the coarse-level test is applied to the deepest level built so far.
Each is argued where it lives and recorded in that ADR's "Follow-up" sections
(issues #85 and #86); none is a local invention to be tidied away without
reading it.
"""

import math
from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np

from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    SPATIAL_AXES,
    Geometry,
    spacings_for_level,
)

#: The axes the *depth* rule is judged on. The ``pyramid_min_size`` half of that
#: rule — stop when the smaller of Y/X would fall below it — is judged here
#: because depth is what a viewer's zoom-out budget cares about and Y/X is what
#: it sees; so is the coarse level's long-axis bound. Z participates in
#: *downsampling* (:data:`SPATIAL_AXES`) and in the coarse level's byte bound,
#: but never gets a vote on the lateral floor; making it a fourth vote there
#: would collapse a 3-plane stack's pyramid to a single level (ADR-0010,
#: rejected options).
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


def is_coarse_level(
    level_shape: Sequence[int],
    dims: Sequence[str],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> bool:
    """Whether one level is small enough for a viewer to hold the whole volume.

    A **coarse level** (see ``CONTEXT.md``) is one a viewer can decode entirely
    and use as spatial context, rather than holding only the part under the
    camera. ADR-0010 makes that two bounds, both of which must hold:

    1. ``Z · Y · X · itemsize ≤ geometry.coarse_max_bytes`` — decoded bytes for
       one ``(t, c)``. T and C are excluded because a viewer holds one timepoint
       of one channel at a time; a 40-timepoint store does not need a level 40×
       smaller to be navigable.
    2. ``max(Y, X) ≤ geometry.coarse_max_long_axis`` — the lateral extent, in
       voxels, of the texture the level becomes.

    Both defaults are Lucida's ``SourceCoarseConfig`` values, adopted knowingly
    (ADR-0010 rejects picking an independent target "for margin": the bound is a
    ``>`` comparison, so a level at exactly 64 MiB passes, and diverging would
    risk planning a store that satisfies a self-imposed number while missing the
    real one). They are :class:`~zarrmony.geometry.Geometry` fields so a store
    can be planned for a different consumer.

    The property is monotone down the pyramid — every level is no larger than
    its parent on every axis — so the deepest level built so far is the only one
    the depth rule has to test.

    :param level_shape: This level's extent, one entry per axis.
    :param dims: Axis names in the same order (e.g. ``"TCZYX"``).
    :param dtype: Anything :func:`numpy.dtype` accepts; only its ``itemsize`` is
        used. Decoded bytes, not compressed — what the viewer has to hold.
    :param geometry: The policy supplying the two bounds.
    """
    shape = tuple(int(s) for s in level_shape)
    if len(shape) != len(dims):
        raise ValueError(
            f"is_coarse_level needs one entry per axis; got {len(shape)} dims "
            f"and {len(dims)} axis names"
        )
    itemsize = max(1, np.dtype(dtype).itemsize)
    voxels = math.prod(
        [extent for extent, d in zip(shape, dims, strict=True) if d in SPATIAL_AXES]
    )
    if voxels * itemsize > geometry.coarse_max_bytes:
        return False
    # ``default=0`` covers an array with no Y/X at all: there is no lateral
    # extent to exceed the bound, so the bound is vacuously satisfied.
    long_axis = max(
        (extent for extent, d in zip(shape, dims, strict=True) if d in LATERAL_DIMS),
        default=0,
    )
    return long_axis <= geometry.coarse_max_long_axis


def coarse_level_index(
    level_shapes: Sequence[Sequence[int]],
    dims: Sequence[str],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> int | None:
    """Index of the shallowest coarse level, or ``None`` if there is none.

    The *shallowest* (largest) qualifying level, because coarseness is monotone
    down the pyramid: every deeper level also fits the bounds, and the one a
    viewer wants for spatial context is the most detailed of them.

    ``None`` is a fact about the pyramid, not a failure — ``CONTEXT.md`` says a
    pyramid may contain no coarse level, and one that bottoms out on the axis
    floor while still too large simply has none. Recording it either way is the
    point of :func:`is_coarse_level` existing: the guarantee becomes checkable
    in the store's own metadata instead of in a viewport.
    """
    for index, shape in enumerate(level_shapes):
        if is_coarse_level(shape, dims, dtype, geometry):
            return index
    return None


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
    dtype: Any,
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

    Depth is the **greater** of two rules:

    1. the pre-ADR-0010 one — stop when the next level's smaller lateral (Y/X)
       extent would fall below ``geometry.pyramid_min_size``; and
    2. keep going until a level is a **coarse level** (:func:`is_coarse_level`),
       i.e. one a viewer can hold whole.

    Taking the greater rather than replacing (1) with (2) is what makes the
    change monotone: no conversion loses a level it had before. Rule 2 is why
    ``dtype`` is a parameter — "can a viewer hold this level?" is a question
    about decoded bytes, and a uint8 volume reaches the bound a level earlier
    than the same shape in uint16. On the ADR-0010 reference volume rule 1 stops
    at ``(226, 552, 465)``, still 110 MiB per ``(t, c)``; rule 2 buys the one
    further level, ``(113, 276, 232)`` at 13.8 MiB, that a 3D camera can
    actually use as context.

    Z takes no part in rule 1 — making it a fourth vote there is the regression
    ADR-0010 rejects by name, since a 3-plane stack would then get no pyramid at
    all — though it does count toward the coarse level's byte bound, where a
    deep stack is exactly what makes a level too big to hold.

    Two things bound rule 2's reach, both pre-existing stops rather than special
    cases: depth still ends once Y and X are both at their floor (what keeps a
    tall thin volume from growing a tail of levels that only thin Z), and once
    no spatial axis is eligible at all. A pyramid that bottoms out on either
    while still too large simply has no coarse level, and
    :func:`coarse_level_index` reports ``None`` — a fact for the audit, not a
    failure.

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
    :param dtype: Anything :func:`numpy.dtype` accepts; only its ``itemsize`` is
        used, by the coarse-level byte bound.
    :param geometry: The policy supplying ``isotropy_tolerance``, ``axis_floor``,
        ``pyramid_min_size`` and the two ``coarse_max_*`` bounds.
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
        if min(
            nxt[i] for i in lateral_indices
        ) < geometry.pyramid_min_size and is_coarse_level(prev, dims, dtype, geometry):
            # The Y/X rule is done, *and* the pyramid has reached a level a
            # viewer can hold whole. Depth is the greater of the two, so the
            # first half of that condition alone is not enough to stop: a volume
            # still too large for a viewer keeps halving past pyramid_min_size,
            # down to the axis floor if that is what it takes. Testing ``prev``
            # rather than every level is sound because coarseness is monotone
            # down the pyramid.
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
