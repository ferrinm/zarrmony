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
:func:`plan_level_chunk_shapes`) that consumes ``chunk_target_bytes``, the
**shard planner** (:func:`plan_shard_shape` / :func:`plan_level_shard_shapes`)
that consumes ``shard_target_bytes``, and the per-level spacing helper
(:func:`spacings_for_level`) both are built on. Chunking is a geometry choice —
"how each level is divided into chunks" — so it lives beside the policy that
governs it rather than in a writer.

The two planners run the same rule (:func:`_best_spatial_combo`) over different
candidate sets, which is the whole point: a shard is the shape a chunk would
have been at the larger target, and it is *how many objects exist* rather than
*how finely the array can be read*. Sharding is off unless asked for
(``shard_target_bytes=None``), so an unsharded conversion is byte-identical to
a pre-#117 one.

:func:`plan_write_grid` names the one those two collapse to — what a single
write touches, which is the shard where the policy plans one and the chunk
where it does not. :func:`plan_reader_tile_size` runs it *backwards*, deriving
the tile a reader should produce so its blocks nest in the grid they will be
written on, and :func:`split_axes` is the predicate saying when they do not
(issue #112).

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
# Raising the target past this is supported but warned about, because the cost
# lands in a viewer rather than in the conversion. A consumer that sizes a
# fixed-byte residency pool in units of the chunk shape holds fewer chunks as
# the chunk grows, and not merely proportionally: Lucida's 2D slice atlas packs
# a square slot grid, ``floor(sqrt(budget / chunk_bytes))**2``, so its 64 MB
# budget holds 121 resident chunks at the 512 KiB default and 4 at 8 MiB. The
# square floor is what makes the fall-off abrupt rather than gradual. See
# ADR-0010, "Follow-up (issue #113)".
CHUNK_TARGET_WARN_BYTES = 2 * 1024 * 1024
# What ``--shard-target-bytes`` resolves to when asked for without a value. NOT
# a field default: ``Geometry.shard_target_bytes`` defaults to ``None`` and
# sharding is off unless a caller says otherwise (ADR-0010, issue #117). 8 MiB
# is the write unit that took a whole-slide scene from a projected nine days to
# 3 h 02 m, and it holds 16 chunks of the 512 KiB default — so a sharded store
# writes in the units that made that run finish while still being read in the
# units a viewer budgets by.
DEFAULT_SHARD_TARGET_BYTES = 8 * 1024 * 1024
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
    :param shard_target_bytes: Raw byte target for a single *shard* — one
        storage object holding many chunks. ``None`` (default) writes no
        shards, and the store is byte-identical to a pre-#117 conversion.
        Setting it turns the chunk into a pure read unit: the shard becomes
        what is written and counted as an object, while a viewer still range
        -reads one ``chunk_target_bytes`` chunk out of it. See
        :func:`plan_shard_shape`.
    :param shard_shape: Explicit per-axis shard shape that bypasses the shard
        planner, and independently enables sharding. Must be a whole multiple
        of each level's chunk shape on every axis.
    """

    chunk_target_bytes: int = DEFAULT_CHUNK_TARGET_BYTES
    isotropy_tolerance: float = DEFAULT_ISOTROPY_TOLERANCE
    axis_floor: int = DEFAULT_AXIS_FLOOR
    coarse_max_bytes: int = DEFAULT_COARSE_MAX_BYTES
    coarse_max_long_axis: int = DEFAULT_COARSE_MAX_LONG_AXIS
    downsample_method: DownsampleMethod = DEFAULT_DOWNSAMPLE_METHOD
    pyramid_min_size: int = DEFAULT_PYRAMID_MIN_SIZE
    chunk_shape: tuple[int, ...] | None = None
    shard_target_bytes: int | None = None
    shard_shape: tuple[int, ...] | None = None

    @property
    def sharding_enabled(self) -> bool:
        """Whether this policy writes shards at all.

        Either spelling switches it on, so callers test this rather than one
        field: ``shard_shape`` alone is a complete instruction and does not
        need a redundant ``shard_target_bytes`` beside it to take effect.
        """
        return self.shard_target_bytes is not None or self.shard_shape is not None

    def __post_init__(self) -> None:
        # Normalize before validating so a list/tuple/generator all land as one
        # canonical, hashable, JSON-friendly type. ``object.__setattr__`` is the
        # sanctioned frozen-dataclass escape hatch during __post_init__.
        for field in ("chunk_shape", "shard_shape"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, field, tuple(int(s) for s in value))

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
        for field in ("chunk_shape", "shard_shape"):
            value = getattr(self, field)
            if value is not None and (not value or any(s < 1 for s in value)):
                raise ValueError(
                    f"Geometry.{field} must be a non-empty sequence of positive "
                    f"ints (e.g. (1, 1, 64, 64, 64)); got {value!r}"
                )
        if self.shard_target_bytes is not None and (
            not isinstance(self.shard_target_bytes, int)
            or isinstance(self.shard_target_bytes, bool)
            or self.shard_target_bytes < 1
        ):
            raise ValueError(
                "Geometry.shard_target_bytes must be None or a positive int; "
                f"got {self.shard_target_bytes!r}"
            )
        # A shard smaller than the chunk it is meant to contain cannot hold one,
        # so the planner would return a shard equal to the chunk: a
        # sharding_indexed array with one chunk per shard, which costs the codec
        # and loses every consumer that cannot read it while cutting the object
        # count by nothing. Caught here rather than shrugged off at plan time.
        # Skipped when chunk_shape is explicit, since chunk_target_bytes is then
        # never consulted and comparing against it would reject valid pairs;
        # divisibility still checks that case (see :func:`plan_shard_shape`).
        if (
            self.shard_target_bytes is not None
            and self.chunk_shape is None
            and self.shard_target_bytes < self.chunk_target_bytes
        ):
            raise ValueError(
                f"Geometry.shard_target_bytes ({self.shard_target_bytes}) is below "
                f"chunk_target_bytes ({self.chunk_target_bytes}); a shard holds "
                f"whole chunks, so it cannot be smaller than one"
            )

    def to_audit(self) -> dict[str, Any]:
        """The resolved policy as a JSON-serializable dict for the audit record.

        Recorded under ``attrs.zarrmony.config.geometry``, replacing the
        pre-ADR-0010 ``chunk_shape`` / ``pyramid_min_size`` input echo. This is
        the *policy*, not its result: what it produced for a given array lives
        on each scene / field record as ``level_shapes``, ``chunk_shapes``,
        ``shard_shapes`` and ``coarse_level_index``.
        """
        record = asdict(self)
        for field in ("chunk_shape", "shard_shape"):
            value = getattr(self, field)
            record[field] = list(value) if value is not None else None
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


def _best_spatial_combo(
    candidates: Sequence[Sequence[int]],
    spacings: Sequence[float],
    max_voxels: int,
) -> tuple[int, ...]:
    """The largest candidate combination under ``max_voxels``, closest to cubic.

    ADR-0010's shape rule, factored out because both grids obey it: pick the
    most voxels that fit, break ties toward cubic *in micrometres*, then break
    remaining ties positionally toward length on the inner (fastest-varying)
    axis so the answer is deterministic. Only the candidate lists differ — the
    chunk planner offers powers of two from ``1``, the shard planner offers
    whole multiples of the chunk — so keeping one scorer is what makes a shard
    the same shape a chunk would have been at the larger target.

    Falls back to the smallest candidate on every axis when nothing fits, which
    is ``1`` for chunks and one chunk for shards.
    """
    best_key: tuple[int, float, tuple[int, ...]] | None = None
    best_combo = tuple(axis[0] for axis in candidates)
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
    return best_combo


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

    best_combo = _best_spatial_combo(candidates, spacings, max_voxels)
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


def _shard_axis_candidates(chunk_length: int, extent: int) -> list[int]:
    """Whole multiples of ``chunk_length``, doubling, up to covering ``extent``.

    A shard must contain a whole number of chunks on every axis — that is what
    lets its index address them — so the candidate set is the chunk length
    doubled rather than the powers of two from ``1``. The list stops at the
    first multiple that covers the axis: a shard longer than the chunks that
    exist buys nothing, since the trailing chunks are absent rather than
    padded.
    """
    chunk_length = max(1, int(chunk_length))
    n_chunks = max(1, math.ceil(max(1, int(extent)) / chunk_length))
    values: list[int] = []
    multiple = 1
    while True:
        values.append(min(multiple, n_chunks) * chunk_length)
        if multiple >= n_chunks:
            return values
        multiple *= 2


def plan_shard_shape(
    chunk_shape: Sequence[int],
    level_shape: Sequence[int],
    dims: Sequence[str],
    spacings_um: Sequence[float],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> tuple[int, ...]:
    """The largest whole-chunk shard under ``shard_target_bytes``, closest to cubic.

    The chunk rule (:func:`plan_chunk_shape`) applied one level up, against a
    candidate set of whole chunk multiples rather than powers of two from ``1``
    — same byte-target-then-cubeness scoring, same positional tie-break
    (:func:`_best_spatial_combo`). On near-isotropic uint16 data at the default
    targets a 64³ chunk lands in a ``128 × 128 × 256`` shard: 16 chunks per
    object, 8 MiB.

    That shard is 2:1 rather than cubic, and deliberately so. Filling the byte
    target comes before cubeness in the shared rule, and 8 MiB of uint16 is
    4.19 M voxels — 1.6 M short of a 128³ cube and far short of a 256³ one, so
    the only way to spend the target is to double one axis. Cubeness is what
    makes a *chunk* cull well against a camera; nothing culls a shard, which is
    a write unit and an object count. The tie-break then puts the long axis on
    X, so a shard is also the most contiguous of the equally-good options.

    T and C stay at the chunk's own length, which the chunk rule pins to ``1``.
    A shard spanning channels would make one write object depend on pixels from
    two channels, coupling writes that the whole store is otherwise careful to
    keep independent — and it buys nothing, because a shard's index already
    lets a reader take one chunk out without the rest.

    An explicit ``geometry.shard_shape`` bypasses this and is validated for
    divisibility against ``chunk_shape`` instead: a shard that is not a whole
    multiple of the chunk on every axis is not a shard zarr can write, and
    saying so here beats a codec-layer error once pixels are moving.

    :param chunk_shape: This level's chunk shape, from :func:`plan_chunk_shape`
        — the unit the returned shard is a whole multiple of.
    :param level_shape: This level's extent, one entry per axis.
    :param dims: Axis names in the same order (e.g. ``"TCZYX"``).
    :param spacings_um: This level's physical spacing per axis, from
        :func:`spacings_for_level`.
    :param dtype: Anything :func:`numpy.dtype` accepts; only ``itemsize`` is used.
    :param geometry: The policy supplying ``shard_target_bytes`` / ``shard_shape``.
    """
    chunk = tuple(int(s) for s in chunk_shape)
    shape = tuple(int(s) for s in level_shape)
    if not (len(shape) == len(dims) == len(spacings_um) == len(chunk)):
        raise ValueError(
            f"plan_shard_shape needs one entry per axis; got {len(shape)} dims, "
            f"{len(dims)} axis names, {len(spacings_um)} spacings and "
            f"{len(chunk)} chunk lengths"
        )

    if geometry.shard_shape is not None:
        shard = tuple(int(s) for s in geometry.shard_shape)
        if len(shard) != len(chunk):
            raise ValueError(
                f"Geometry.shard_shape has {len(shard)} axes but the level has "
                f"{len(chunk)}; got {shard!r} against chunk {chunk!r}"
            )
        bad = [
            (axis, s, c)
            for axis, (s, c) in enumerate(zip(shard, chunk, strict=True))
            if s % c
        ]
        if bad:
            detail = ", ".join(
                f"axis {axis} ({dims[axis]}): shard {s} is not a multiple of chunk {c}"
                for axis, s, c in bad
            )
            raise ValueError(
                f"Geometry.shard_shape {shard!r} must be a whole multiple of the "
                f"chunk shape {chunk!r} on every axis — {detail}"
            )
        return shard

    if geometry.shard_target_bytes is None:
        raise ValueError(
            "plan_shard_shape called with sharding disabled; check "
            "Geometry.sharding_enabled first"
        )

    itemsize = max(1, np.dtype(dtype).itemsize)
    max_voxels = max(1, geometry.shard_target_bytes // itemsize)

    spatial = [i for i, d in enumerate(dims) if d in SPATIAL_AXES]
    shard = list(chunk)
    if not spatial:
        return tuple(shard)

    spacings = [_safe_spacing(spacings_um[i]) for i in spatial]
    candidates = [_shard_axis_candidates(chunk[i], shape[i]) for i in spatial]

    best_combo = _best_spatial_combo(candidates, spacings, max_voxels)
    for axis, length in zip(spatial, best_combo, strict=True):
        shard[axis] = length
    return tuple(shard)


def plan_level_shard_shapes(
    chunk_shapes: Sequence[Sequence[int]],
    level_shapes: Sequence[Sequence[int]],
    dims: Sequence[str],
    base_spacings: Sequence[float],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> list[tuple[int, ...]] | None:
    """Plan one shard shape per pyramid level, or ``None`` when sharding is off.

    ``None`` rather than a list of "no shard" sentinels, so the caller passes it
    straight to the writer's ``shard_shape=`` and an unsharded conversion stays
    byte-identical to a pre-#117 one. Each level is planned against its own
    spacing and its own chunk, for the same reason chunk planning is per level:
    a level that halved Y and X but not Z needs a shard that is still cubic in
    µm *there*.
    """
    if not geometry.sharding_enabled:
        return None
    levels = [tuple(int(s) for s in shape) for shape in level_shapes]
    chunks = [tuple(int(s) for s in shape) for shape in chunk_shapes]
    if len(levels) != len(chunks):
        raise ValueError(
            f"plan_level_shard_shapes needs one chunk shape per level; got "
            f"{len(chunks)} chunk shapes for {len(levels)} levels"
        )
    if not levels:
        raise ValueError("plan_level_shard_shapes needs at least one level shape")
    base_shape = levels[0]
    return [
        plan_shard_shape(
            chunk,
            shape,
            dims,
            spacings_for_level(base_spacings, base_shape, shape),
            dtype,
            geometry,
        )
        for chunk, shape in zip(chunks, levels, strict=True)
    ]


def plan_write_grid(
    level_shape: Sequence[int],
    dims: Sequence[str],
    spacings_um: Sequence[float],
    dtype: Any,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> tuple[int, ...]:
    """The block one write to this level covers — the shard, else the chunk.

    The two planners answer different questions and only one of them is "what
    does a single write touch". Under sharding the chunk stops being a storage
    object and becomes purely a read unit, so the write unit is the shard; with
    sharding off the two coincide. Anything reasoning about *units of work* —
    the writer's rechunk target, the reader tile that should feed it — wants
    this, not :func:`plan_chunk_shape`.

    An explicit ``geometry.chunk_shape`` bypasses the chunk planner here for the
    same reason :func:`plan_level_chunk_shapes` honours it: the caller said what
    they wanted.
    """
    chunk = (
        tuple(int(s) for s in geometry.chunk_shape)
        if geometry.chunk_shape is not None
        else plan_chunk_shape(level_shape, dims, spacings_um, dtype, geometry)
    )
    if not geometry.sharding_enabled:
        return chunk
    return plan_shard_shape(chunk, level_shape, dims, spacings_um, dtype, geometry)


def split_axes(
    source_lengths: Sequence[int],
    write_grid: Sequence[int],
    extents: Sequence[int],
    dims: Sequence[str],
) -> list[tuple[str, int, int]]:
    """Axes where blocks of ``source_lengths`` must be *split* to fill ``write_grid``.

    Returns ``(axis_name, source_length, grid_length)`` per offending axis, empty
    when every write can be assembled from whole source blocks. This is the
    predicate behind issue #112, and the asymmetry it encodes is the measured
    one: on the reference whole-slide scene, feeding 1024² blocks to a 512²
    write grid costs 9.0x the source partition count in dask tasks, while
    feeding 512² blocks to a 2048² grid costs 1.06x. Merging whole blocks is
    nearly free; splitting one is not, and it re-reads the source block once per
    output block on top of that.

    An axis is safe when every write-grid boundary is also a source boundary.
    Grid boundaries fall at multiples of ``grid``, source boundaries at multiples
    of ``source``, so the test is ``grid % source == 0`` — *except* where the
    grid already spans the whole extent. Such an axis holds exactly one block,
    so it has no interior boundary for a source block to straddle and no
    ``source`` can split it. That exemption is load-bearing rather than a
    micro-optimisation: :func:`_axis_candidates` clamps its last candidate to
    the extent, so a small scene legitimately plans a non-power-of-two length
    (a 375-row thumbnail plans ``375``) that would otherwise look unsafe against
    every tile.
    """
    if not (len(source_lengths) == len(write_grid) == len(extents) == len(dims)):
        raise ValueError(
            f"split_axes needs one entry per axis; got {len(source_lengths)} "
            f"source lengths, {len(write_grid)} grid lengths, {len(extents)} "
            f"extents and {len(dims)} axis names"
        )
    offenders: list[tuple[str, int, int]] = []
    for name, source, grid, extent in zip(
        dims, source_lengths, write_grid, extents, strict=True
    ):
        source, grid, extent = int(source), int(grid), int(extent)
        if source < 1 or grid < 1 or grid >= extent:
            continue
        if grid % source:
            offenders.append((str(name), source, grid))
    return offenders


def plan_reader_tile_size(
    scene_shapes: Sequence[Sequence[int]],
    dims: Sequence[str],
    scene_spacings: Sequence[Sequence[float]],
    scene_dtypes: Sequence[Any],
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> tuple[int, int] | None:
    """The ``(Y, X)`` tile a reader should produce so no scene's blocks get split.

    Issue #112. Nothing connected the reader's tile size to the geometry the
    planner picks, so the writer's rechunk absorbed the mismatch — and on the
    common whole-slide path that rechunk was always a *split*, the expensive
    direction. Planning the write grid first and asking the reader for blocks
    that nest inside it removes the rechunk instead of optimising it.

    ``None`` when the axes carry no Y or X, which is the only case where a tile
    size means nothing.

    **Why the minimum across scenes, and why it is safe.** ``tile_size`` is a
    reader-constructor argument — one value for every scene — while the write
    grid is planned per scene, so a multi-scene file cannot have them all match
    exactly. Taking the element-wise minimum is provably split-free rather than
    merely a good guess: every non-single-chunk grid length is a power of two
    (:func:`_axis_candidates`), so the smallest divides all the others, and the
    scenes it does *not* divide are exactly the ones whose grid spans their
    whole extent — which :func:`split_axes` shows cannot be split at all. Scenes
    larger than the tile then pay a cheap merge. The alternative, taking the
    dominant scene's grid, is what a reader tuned by hand would do and it splits
    every smaller scene: a whole-slide file is a gigapixel scene plus a
    ``label`` and a ``macro`` thumbnail, and those thumbnails are the ones that
    would pay.

    **Why the dtype is per scene too.** It is the one planning input that looks
    like a property of the file and is not. A whole-slide VSI carries `>u2`
    fluorescence beside `uint8` RGB thumbnails, and itemsize sets how many
    voxels fit the byte target — so reading the dtype once, off whichever scene
    the reader happens to be pointing at, plans half the file against the wrong
    budget. The failure is one-directional and silent: too small an itemsize
    plans too *large* a grid, which derives a tile the real grid then has to
    split. That is the pathology this function exists to remove.

    :param scene_shapes: One level-0 shape per scene, each with one entry per
        axis in ``dims`` order.
    :param dims: Axis names shared by every scene (e.g. ``"TCZYX"``).
    :param scene_spacings: One physical-spacing list per scene, matching
        ``scene_shapes``.
    :param scene_dtypes: One dtype per scene, matching ``scene_shapes``.
        Anything :func:`numpy.dtype` accepts; only ``itemsize`` is used.
    :param geometry: The policy supplying the chunk and shard targets.
    """
    axis_index = {name: i for i, name in enumerate(dims)}
    if "Y" not in axis_index or "X" not in axis_index:
        return None
    if not (len(scene_shapes) == len(scene_spacings) == len(scene_dtypes)):
        raise ValueError(
            f"plan_reader_tile_size needs one spacing list and one dtype per "
            f"scene; got {len(scene_spacings)} spacings and "
            f"{len(scene_dtypes)} dtypes for {len(scene_shapes)} scenes"
        )
    if not scene_shapes:
        raise ValueError("plan_reader_tile_size needs at least one scene shape")

    best: dict[str, int] = {}
    fallback: dict[str, int] = {}
    for shape, spacings, dtype in zip(
        scene_shapes, scene_spacings, scene_dtypes, strict=True
    ):
        extents = tuple(int(s) for s in shape)
        grid = plan_write_grid(extents, dims, spacings, dtype, geometry)
        for name in ("Y", "X"):
            axis = axis_index[name]
            length = int(grid[axis])
            # A grid spanning the whole axis constrains nothing (one block, no
            # interior boundary), so it must not drag the minimum down — a
            # 375-row thumbnail would otherwise pin every scene's tile to 375.
            # Kept as a fallback for the all-single-chunk case, where any value
            # is split-free and the largest is the least wasteful.
            if length >= extents[axis]:
                fallback[name] = max(fallback.get(name, 0), length)
            else:
                best[name] = min(best.get(name, length), length)

    tile_y = best.get("Y", fallback.get("Y", 1))
    tile_x = best.get("X", fallback.get("X", 1))
    return (max(1, tile_y), max(1, tile_x))


__all__ = [
    "CHUNK_TARGET_WARN_BYTES",
    "DEFAULT_AXIS_FLOOR",
    "DEFAULT_CHUNK_TARGET_BYTES",
    "DEFAULT_COARSE_MAX_BYTES",
    "DEFAULT_COARSE_MAX_LONG_AXIS",
    "DEFAULT_DOWNSAMPLE_METHOD",
    "DEFAULT_GEOMETRY",
    "DEFAULT_ISOTROPY_TOLERANCE",
    "DEFAULT_PYRAMID_MIN_SIZE",
    "DEFAULT_SHARD_TARGET_BYTES",
    "SPATIAL_AXES",
    "DownsampleMethod",
    "Geometry",
    "plan_chunk_shape",
    "plan_level_chunk_shapes",
    "plan_level_shard_shapes",
    "plan_reader_tile_size",
    "plan_shard_shape",
    "plan_write_grid",
    "resolve_geometry",
    "spacings_for_level",
    "split_axes",
]
