"""Output geometry policy — every shape choice for a store, in one frozen object.

Geometry (see ``CONTEXT.md``) is the set of choices that fix an output store's
*shape* rather than its content: how many pyramid levels there are, what each
level's extent is, and how each level is divided into chunks. ADR-0010 makes
those choices a single frozen :class:`Geometry` value that ``convert()``
resolves once and threads through the per-scene, bf2raw and plate write paths,
rather than a growing set of loose keyword arguments — each of which previously
touched ~14 sites between the public signature and the writers.

``chunk_shape`` and ``pyramid_min_size`` are retained on ``convert()`` as sugar
that folds into a :class:`Geometry` (see :func:`resolve_geometry`), so callers
written against the pre-ADR-0010 API keep working unchanged.

This module also owns the **chunk planner** (:func:`plan_chunk_shape` /
:func:`plan_level_chunk_shapes`) that consumes ``chunk_target_bytes``, and the
per-level spacing helper (:func:`spacings_for_level`) it is built on. Chunking
is a geometry choice — "how each level is divided into chunks" — so it lives
beside the policy that governs it rather than in a writer.

``isotropy_tolerance``, ``axis_floor``, ``pyramid_min_size`` and the two
``coarse_max_*`` bounds are consumed by the anisotropy-aware pyramid rule
(:func:`~zarrmony.writers.pyramid.compute_level_shapes`), which lives beside the
downsampler (:func:`~zarrmony.writers.pyramid.build_pyramid`) that executes it
and that reads ``downsample_method``.

Every field affects written output. ``downsample_method`` is the one that
changes *pixels* rather than shapes, which is why the audit records it: the same
source now produces two different pyramids depending on it.

Every rule here is written over *the axes that are present*, never over
dimensionality: nothing asks whether the array is 2D, and ADR-0010 rejects
gating the policy on ``Z > 1`` by name. A singleton Z contributes one candidate
length to the chunk search and a constant µm extent to the cubeness score, so
it moves neither the winning chunk nor the pyramid depth — a 2160² plate field
is planned by exactly the rule a whole-brain volume is. That is stated as a
property in ``test_geometry_parity.py``; do not add a 2D or a plate shortcut
without reading the "Follow-up (issue #88)" section of that ADR first.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

DownsampleMethod = Literal["mean", "max"]

# ADR-0010 defaults. Named constants rather than bare literals in the field
# defaults so the CLI, the docs and the tests can all cite one source.
#
# 512 KiB raw chunk target: Lucida's transport study puts interactive source
# reads at p50 325 KiB against a flat ~110 ms TTFB, and blosc-zstd on tissue
# gives 2.3–4.2x, so 512 KiB raw lands at ~150–220 KiB compressed. On the other
# side an 8 MB decoded per-frame upload budget admits 16 such chunks.
DEFAULT_CHUNK_TARGET_BYTES = 512 * 1024
# An axis halves only when its physical spacing is within this factor of the
# finest axis's, so the pyramid moves toward isotropy and the scarce axis
# (usually Z) is spent last.
DEFAULT_ISOTROPY_TOLERANCE = 1.5
# No axis halves below this many voxels, and an axis already below it never
# halves — a 3-plane stack keeps its 3 planes at every level.
DEFAULT_AXIS_FLOOR = 32
# Coarse-level bounds, adopted from Lucida's ``SourceCoarseConfig`` defaults
# (ADR-0010 records the coupling deliberately): a level is a coarse level when
# its decoded Z*Y*X*itemsize per (t, c) fits in 64 MiB and its long lateral
# axis is at most 2048.
DEFAULT_COARSE_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_COARSE_MAX_LONG_AXIS = 2048
# Mean-pool is right for intensity imagery and is what the OME-Zarr ecosystem
# assumes; "max" exists for sparse-label acquisitions where mean-pooling
# dissolves small objects into the background.
DEFAULT_DOWNSAMPLE_METHOD: DownsampleMethod = "mean"
# The pre-ADR-0010 depth rule: stop halving when the smallest of Y/X would fall
# below this. Retained — ADR-0010 makes depth the *greater* of this and the
# coarse-level rule, so no existing conversion loses a level.
DEFAULT_PYRAMID_MIN_SIZE = 256

_VALID_DOWNSAMPLE_METHODS: tuple[DownsampleMethod, ...] = ("mean", "max")

#: The axes a chunk is planned over. T and C are never chunked — a chunk spans
#: one timepoint of one channel, so a viewer fetching one channel at one
#: timepoint never pays for the others.
SPATIAL_AXES: frozenset[str] = frozenset({"Z", "Y", "X"})

# Physical spacings arrive from reader metadata, which is allowed to be absent
# or nonsense. A non-positive or non-finite spacing carries no information
# about which axis is scarce, so it degrades to 1.0 — the planner then treats
# that axis as isotropic with any other unknown axis, which is the same answer
# a caller with no pixel-size metadata at all would get.
_FALLBACK_SPACING = 1.0


def _safe_spacing(value: Any) -> float:
    """Coerce one reader-supplied µm spacing to a usable positive float."""
    try:
        spacing = float(value)
    except (TypeError, ValueError):
        return _FALLBACK_SPACING
    if not math.isfinite(spacing) or spacing <= 0.0:
        return _FALLBACK_SPACING
    return spacing


@dataclass(frozen=True, slots=True)
class Geometry:
    """The complete output-geometry policy for one conversion.

    Immutable by construction: pass a new instance (or :func:`dataclasses.replace`)
    to change a field. Being frozen is what lets one instance be shared as a
    module-level default (:data:`DEFAULT_GEOMETRY`) and threaded through every
    writer without defensive copying.

    :param chunk_target_bytes: Raw (uncompressed) byte target for a single
        chunk. :func:`plan_chunk_shape` picks the largest power-of-two shape
        that fits and is closest to cubic in micrometres.
    :param isotropy_tolerance: A spatial axis is downsampled at a level only
        when its physical spacing is within this factor of the finest axis's.
        ``1.0`` halves only exactly-isotropic axes; a very large value halves
        every spatial axis at every level.
    :param axis_floor: Minimum voxels on any axis; an axis at or below the
        floor is never halved. On Y/X it is capped by ``pyramid_min_size`` so
        an explicitly lowered depth floor is not overridden by this default.
    :param coarse_max_bytes: Upper bound on a coarse level's decoded bytes per
        ``(t, c)``. With ``coarse_max_long_axis``, this extends pyramid depth
        until a level fits both — see
        :func:`~zarrmony.writers.pyramid.is_coarse_level`.
    :param coarse_max_long_axis: Upper bound on a coarse level's longest
        lateral axis, in voxels.
    :param downsample_method: The pooling kernel every pyramid level above 0 is
        built with — ``"mean"`` (default) or ``"max"``, applied uniformly.
        Max-pool biases every level above 0 high and makes the pyramid useless
        for measurement; it exists for sparse-label acquisitions, where
        mean-pooling dissolves small objects into the background. See
        :func:`~zarrmony.writers.pyramid.build_pyramid`.
    :param pyramid_min_size: Stop halving when the smallest of Y/X would fall
        below this — unless the two ``coarse_max_*`` bounds ask for more depth,
        which they win, so no conversion loses a level.
    :param chunk_shape: Explicit per-axis chunk shape that bypasses the
        planner entirely. ``None`` (default) means "plan it".
    """

    chunk_target_bytes: int = DEFAULT_CHUNK_TARGET_BYTES
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE
    axis_floor: int = DEFAULT_AXIS_FLOOR
    coarse_max_bytes: int = DEFAULT_COARSE_MAX_BYTES
    coarse_max_long_axis: int = DEFAULT_COARSE_MAX_LONG_AXIS
    downsample_method: DownsampleMethod = DEFAULT_DOWNSAMPLE_METHOD
    pyramid_min_size: int = DEFAULT_PYRAMID_MIN_SIZE
    chunk_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Normalize before validating so a list/tuple/generator all land as one
        # canonical, hashable, JSON-friendly type. ``object.__setattr__`` is the
        # sanctioned frozen-dataclass escape hatch during __post_init__.
        if self.chunk_shape is not None and not isinstance(self.chunk_shape, tuple):
            object.__setattr__(
                self, "chunk_shape", tuple(int(s) for s in self.chunk_shape)
            )

        for name in (
            "chunk_target_bytes",
            "axis_floor",
            "coarse_max_bytes",
            "coarse_max_long_axis",
            "pyramid_min_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"Geometry.{name} must be a positive int; got {value!r}"
                )
        if (
            not isinstance(self.isotropy_tolerance, int | float)
            or isinstance(self.isotropy_tolerance, bool)
            or self.isotropy_tolerance < 1.0
        ):
            raise ValueError(
                "Geometry.isotropy_tolerance must be a float >= 1.0 (1.0 means "
                f"'halve only exactly-isotropic axes'); got {self.isotropy_tolerance!r}"
            )
        if self.downsample_method not in _VALID_DOWNSAMPLE_METHODS:
            raise ValueError(
                f"Geometry.downsample_method must be one of "
                f"{list(_VALID_DOWNSAMPLE_METHODS)}; got {self.downsample_method!r}"
            )
        if self.chunk_shape is not None and (
            not self.chunk_shape or any(s < 1 for s in self.chunk_shape)
        ):
            raise ValueError(
                "Geometry.chunk_shape must be a non-empty sequence of positive "
                f"ints (e.g. (1, 1, 64, 64, 64)); got {self.chunk_shape!r}"
            )

    def to_audit(self) -> dict[str, Any]:
        """The resolved policy as a JSON-serializable dict for the audit record.

        Recorded under ``attrs.zarrmony.config.geometry``, replacing the
        pre-ADR-0010 ``chunk_shape`` / ``pyramid_min_size`` input echo. This is
        the *policy*, not its result: what it produced for a given array lives
        on each scene / field record as ``level_shapes``, ``chunk_shapes`` and
        ``coarse_level_index``.
        """
        record = asdict(self)
        record["chunk_shape"] = (
            list(self.chunk_shape) if self.chunk_shape is not None else None
        )
        return record


#: Shared default policy. Safe to share because :class:`Geometry` is frozen.
DEFAULT_GEOMETRY = Geometry()


def resolve_geometry(
    geometry: Geometry | None,
    *,
    chunk_shape: Sequence[int] | None = None,
    pyramid_min_size: int | None = None,
) -> Geometry:
    """Fold ``convert()``'s retained sugar into one :class:`Geometry`.

    ``chunk_shape`` and ``pyramid_min_size`` survive on ``convert()`` per
    ADR-0010 so no pre-existing caller breaks; each is ``None`` when the caller
    did not pass it, and otherwise overrides the corresponding field of a
    default policy.

    Passing ``geometry`` *and* either sugar argument raises :class:`ValueError`
    rather than picking a precedence: the two spellings would be saying
    different things about the same field, and silently letting one win is the
    kind of surprise that only surfaces once the store is on disk.
    """
    if geometry is not None:
        conflicting = [
            name
            for name, value in (
                ("chunk_shape", chunk_shape),
                ("pyramid_min_size", pyramid_min_size),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                f"geometry= was passed together with {' and '.join(conflicting)}=; "
                f"these set the same policy field two ways. Set the field on the "
                f"Geometry instead, e.g. "
                f"dataclasses.replace(geometry, {conflicting[0]}=...)."
            )
        return geometry

    updates: dict[str, Any] = {}
    if chunk_shape is not None:
        updates["chunk_shape"] = tuple(int(s) for s in chunk_shape)
    if pyramid_min_size is not None:
        updates["pyramid_min_size"] = pyramid_min_size
    return replace(DEFAULT_GEOMETRY, **updates) if updates else DEFAULT_GEOMETRY


def spacings_for_level(
    base_spacings: Sequence[float],
    base_shape: Sequence[int],
    level_shape: Sequence[int],
) -> tuple[float, ...]:
    """Per-axis µm spacing at a pyramid level, derived from its shape.

    ``spacing_level_i = spacing_0 × (shape_0 / shape_i)`` per axis. A level is
    a downsample of level 0, so an axis that lost half its voxels covers the
    same physical distance with voxels twice as long — which is the whole
    reason chunk planning is per level rather than once per store. Axes that
    did not downsample keep their level-0 spacing, so this works unchanged for
    the per-axis-varying factors ADR-0010's pyramid rule produces.

    Non-spatial axes (T, C) have no physical spacing; callers pass ``1.0`` for
    them (see :func:`~zarrmony.writers.scene._physical_scales_for_dims`) and
    the identity ratio leaves that ``1.0`` untouched.

    Spacings that a reader could not report are normalized to ``1.0`` rather
    than propagated as ``0``/``NaN`` — see :data:`_FALLBACK_SPACING`.
    """
    if not (len(base_spacings) == len(base_shape) == len(level_shape)):
        raise ValueError(
            f"spacings_for_level needs one entry per axis; got "
            f"{len(base_spacings)} spacings, {len(base_shape)} base dims and "
            f"{len(level_shape)} level dims"
        )
    out: list[float] = []
    for spacing, base_dim, level_dim in zip(
        base_spacings, base_shape, level_shape, strict=True
    ):
        safe = _safe_spacing(spacing)
        # A zero-length axis is not a real level; leave the spacing alone
        # rather than dividing by it.
        out.append(safe * (int(base_dim) / int(level_dim)) if level_dim > 0 else safe)
    return tuple(out)


def _axis_candidates(extent: int) -> list[int]:
    """Every power-of-two chunk length for one axis, clamped to ``extent``.

    ``1, 2, 4, …`` up to the first power of two at or above ``extent``, each
    clamped. The clamp is why the list can end in a non-power-of-two (extent
    ``3`` gives ``[1, 2, 3]``): ADR-0010 asks for power-of-two chunks *and* for
    chunks that never exceed the level, and where the two disagree the level
    wins — a chunk longer than the axis buys nothing but padding.
    """
    extent = max(1, int(extent))
    values: list[int] = []
    size = 1
    while True:
        values.append(min(size, extent))
        if size >= extent:
            return values
        size *= 2


def plan_chunk_shape(
    level_shape: Sequence[int],
    dims: Sequence[str],
    spacings_um: Sequence[float],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> tuple[int, ...]:
    """The largest chunk under the byte target that is closest to cubic in µm.

    ADR-0010's chunk rule, for one pyramid level. Candidate lengths per spatial
    axis are the powers of two clamped to the level extent
    (:func:`_axis_candidates`); T and C are pinned to ``1``. Among all
    candidate shapes whose raw size (``voxels × itemsize``) fits
    ``geometry.chunk_target_bytes``, the winner is:

    1. the one with the most voxels — the byte target is a transport sweet
       spot, not a ceiling to stay well under, and a chunk at half the target
       costs a round trip to deliver half the data; then
    2. among those, the one whose *physical* extents are closest to cubic,
       scored as ``max(extent_µm) / min(extent_µm)``; then
    3. a positional tie-break preferring length on the inner (fastest-varying)
       axis, so C-order neighbours stay together and the answer is
       deterministic.

    Cubic in **micrometres**, not voxels: on a 10:1 confocal stack (Z 5 µm, XY
    0.5 µm) a fixed 64³ would span 320 × 32 × 32 µm, making culling ten times
    coarser in Z than laterally — the defect ADR-0010 exists to remove, one
    order of magnitude down. The rule degenerates to exactly 64³ on
    near-isotropic uint16 data at the default target, so nothing is lost on the
    common case.

    :param level_shape: This level's extent, one entry per axis.
    :param dims: Axis names in the same order (e.g. ``"TCZYX"``); membership in
        :data:`SPATIAL_AXES` decides which axes are planned.
    :param spacings_um: This level's physical spacing per axis — from
        :func:`spacings_for_level`, not level 0's.
    :param dtype: Anything :func:`numpy.dtype` accepts; only its ``itemsize``
        is used.
    :param geometry: The policy supplying ``chunk_target_bytes``.
    """
    shape = tuple(int(s) for s in level_shape)
    if not (len(shape) == len(dims) == len(spacings_um)):
        raise ValueError(
            f"plan_chunk_shape needs one entry per axis; got {len(shape)} dims, "
            f"{len(dims)} axis names and {len(spacings_um)} spacings"
        )
    itemsize = max(1, np.dtype(dtype).itemsize)
    max_voxels = max(1, geometry.chunk_target_bytes // itemsize)

    spatial = [i for i, d in enumerate(dims) if d in SPATIAL_AXES]
    chunk = [1] * len(shape)
    if not spatial:
        # No Z/Y/X at all (a degenerate T/C-only array). Nothing to plan.
        return tuple(chunk)

    spacings = [_safe_spacing(spacings_um[i]) for i in spatial]
    candidates = [_axis_candidates(shape[i]) for i in spatial]

    best_key: tuple[int, float, tuple[int, ...]] | None = None
    best_combo: tuple[int, ...] = tuple(1 for _ in spatial)
    for combo in itertools.product(*candidates):
        voxels = math.prod(combo)
        if voxels > max_voxels:
            continue
        extents = [
            length * spacing for length, spacing in zip(combo, spacings, strict=True)
        ]
        # Rounded so that two shapes that are equally cubic in exact arithmetic
        # tie here too, and the positional tie-break — not float noise — picks
        # between them.
        cubeness = round(max(extents) / min(extents), 9)
        key = (voxels, -cubeness, tuple(reversed(combo)))
        if best_key is None or key > best_key:
            best_key, best_combo = key, combo

    for axis, length in zip(spatial, best_combo, strict=True):
        chunk[axis] = length
    return tuple(chunk)


def plan_level_chunk_shapes(
    level_shapes: Sequence[Sequence[int]],
    dims: Sequence[str],
    base_spacings: Sequence[float],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> list[tuple[int, ...]]:
    """Plan one chunk shape per pyramid level, in level order.

    Each level is planned against *its own* spacing
    (:func:`spacings_for_level`), so a level that halved Y and X but not Z gets
    a chunk that is still cubic in µm at that level rather than inheriting
    level 0's answer.

    An explicit ``geometry.chunk_shape`` bypasses planning entirely and is
    replicated across levels — the caller said what they wanted, on every
    level, and second-guessing it per level would make the override mean
    something different at level 3 than at level 0.
    """
    levels = [tuple(int(s) for s in shape) for shape in level_shapes]
    if not levels:
        raise ValueError("plan_level_chunk_shapes needs at least one level shape")
    if geometry.chunk_shape is not None:
        return [tuple(geometry.chunk_shape)] * len(levels)
    base_shape = levels[0]
    return [
        plan_chunk_shape(
            shape,
            dims,
            spacings_for_level(base_spacings, base_shape, shape),
            dtype,
            geometry,
        )
        for shape in levels
    ]


__all__ = [
    "DEFAULT_AXIS_FLOOR",
    "DEFAULT_CHUNK_TARGET_BYTES",
    "DEFAULT_COARSE_MAX_BYTES",
    "DEFAULT_COARSE_MAX_LONG_AXIS",
    "DEFAULT_DOWNSAMPLE_METHOD",
    "DEFAULT_GEOMETRY",
    "DEFAULT_ISOTROPY_TOLERANCE",
    "DEFAULT_PYRAMID_MIN_SIZE",
    "SPATIAL_AXES",
    "DownsampleMethod",
    "Geometry",
    "plan_chunk_shape",
    "plan_level_chunk_shapes",
    "resolve_geometry",
    "spacings_for_level",
]
