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

**Inert fields.** ``chunk_target_bytes``, ``isotropy_tolerance``,
``axis_floor``, ``coarse_max_bytes``, ``coarse_max_long_axis`` and
``downsample_method`` are carried and audited but have no behaviour yet — the
planner and the anisotropy-aware pyramid that consume them arrive in later
slices of ADR-0010. Until then, setting them changes only the audit record.
Values are still validated at construction so a typo fails at the call site
rather than silently at some later release.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

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


@dataclass(frozen=True, slots=True)
class Geometry:
    """The complete output-geometry policy for one conversion.

    Immutable by construction: pass a new instance (or :func:`dataclasses.replace`)
    to change a field. Being frozen is what lets one instance be shared as a
    module-level default (:data:`DEFAULT_GEOMETRY`) and threaded through every
    writer without defensive copying.

    :param chunk_target_bytes: Raw (uncompressed) byte target for a single
        chunk. The planner picks the largest power-of-two shape that fits.
    :param isotropy_tolerance: A spatial axis is downsampled at a level only
        when its physical spacing is within this factor of the finest axis's.
    :param axis_floor: Minimum voxels on any axis; an axis at or below the
        floor is never halved.
    :param coarse_max_bytes: Upper bound on a coarse level's decoded bytes per
        ``(t, c)``.
    :param coarse_max_long_axis: Upper bound on a coarse level's longest
        lateral axis, in voxels.
    :param downsample_method: ``"mean"`` (default) or ``"max"``. Max-pool
        biases every level above 0 high; it exists for sparse labels.
    :param pyramid_min_size: Stop halving when the smallest of Y/X would fall
        below this.
    :param chunk_shape: Explicit per-axis chunk shape that bypasses the
        planner entirely. ``None`` (default) means "plan it".

    Only ``pyramid_min_size`` and ``chunk_shape`` affect written output today;
    see the module docstring on the inert fields.
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
        pre-ADR-0010 ``chunk_shape`` / ``pyramid_min_size`` input echo. Per-level
        shapes live on each scene / field record's ``level_shapes``; per-level
        chunk shapes and the coarse level index join them in a later slice.
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


__all__ = [
    "DEFAULT_AXIS_FLOOR",
    "DEFAULT_CHUNK_TARGET_BYTES",
    "DEFAULT_COARSE_MAX_BYTES",
    "DEFAULT_COARSE_MAX_LONG_AXIS",
    "DEFAULT_DOWNSAMPLE_METHOD",
    "DEFAULT_GEOMETRY",
    "DEFAULT_ISOTROPY_TOLERANCE",
    "DEFAULT_PYRAMID_MIN_SIZE",
    "DownsampleMethod",
    "Geometry",
    "resolve_geometry",
]
