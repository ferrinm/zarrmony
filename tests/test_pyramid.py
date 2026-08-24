"""Tests for the ADR-0010 anisotropy-aware pyramid (issues #85, #86).

Four concerns, in order:

1. :func:`compute_level_shapes` — the shape rule: halve every spatial axis
   whose µm spacing is within ``isotropy_tolerance`` of the finest still-
   halvable axis's, subject to a per-axis voxel floor.
2. The depth rule — the greater of the Y/X ``pyramid_min_size`` floor and the
   depth at which a level becomes a coarse level (:func:`is_coarse_level`), so
   depth is chosen for the property a viewer actually needs and no conversion
   loses a level it had before (#86).
3. :func:`build_pyramid` — mean-pooling with the coarsen factors read off
   consecutive level shapes, so per-axis-varying and uniform downsampling are
   one code path.
4. That per-axis factors reach the store's ``coordinateTransformations``, which
   ``OMEZarrWriter`` derives from the level shapes we hand it, and that the
   coarse level's index reaches the audit record.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import Geometry, spacings_for_level
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin
from zarrmony.writers.pyramid import (
    build_pyramid,
    coarse_level_index,
    compute_level_shapes,
    is_coarse_level,
)

# The ADR-0010 reference acquisition: a SmartSPIM whole-brain export, uint16 at
# Z 2.0 / Y 1.8 / X 1.8 µm. Near-isotropic (Z is within 1.12x of the lateral
# spacing), so every spatial axis halves at every level.
REFERENCE_SHAPE = (1, 3, 3627, 8835, 7452)
REFERENCE_SPACINGS = (1.0, 1.0, 2.0, 1.8, 1.8)

# A 10:1 confocal stack: Z is the scarce axis by an order of magnitude.
CONFOCAL_SPACINGS = (1.0, 1.0, 5.0, 0.5, 0.5)


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch ``zarrmony.api.get_reader`` to return a configurable FakeReader."""

    def installer(reader: FakeReader) -> None:
        plugin = ReaderPlugin(
            name="bioio-fake",
            match=lambda _p: 100,
            open=lambda _p: object(),
            distribution="bioio-fake",
            source="builtin",
        )
        monkeypatch.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, plugin, 100),
        )

    return installer


def _anisotropy(spacings: tuple[float, ...], dims: str) -> float:
    """Ratio of coarsest to finest spatial spacing — 1.0 when isotropic."""
    spatial = [s for s, d in zip(spacings, dims, strict=True) if d in "ZYX"]
    return max(spatial) / min(spatial)


# ---------- level shapes: the pre-#85 behaviour that must not change ----------


def test_isotropic_lateral_stack_halves_to_the_depth_floor() -> None:
    shapes = compute_level_shapes(
        (1, 4, 2048, 2048),
        "TCYX",
        (1.0, 1.0, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert shapes == [
        (1, 4, 2048, 2048),
        (1, 4, 1024, 1024),
        (1, 4, 512, 512),
        (1, 4, 256, 256),
    ]


def test_small_image_gets_no_pyramid() -> None:
    # Halving 100 yields 50, below the floor of 256, so only the base remains.
    shapes = compute_level_shapes(
        (1, 1, 100, 100),
        "TCYX",
        (1.0, 1.0, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert shapes == [(1, 1, 100, 100)]


def test_non_spatial_dims_are_preserved() -> None:
    shapes = compute_level_shapes(
        (10, 4, 1, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 1.0, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert shapes == [
        (10, 4, 1, 1024, 1024),
        (10, 4, 1, 512, 512),
        (10, 4, 1, 256, 256),
    ]


def test_no_lateral_dims_means_a_single_level() -> None:
    # No Y/X, so no depth rule to apply; inventing one for a shape no
    # microscope produces would be worse than writing level 0 alone.
    assert compute_level_shapes((10, 4), "TC", (1.0, 1.0), np.uint16) == [(10, 4)]


def test_depth_is_still_decided_by_the_lateral_axes() -> None:
    # X (1500) reaches 375 first; the next level's min(500, 375) is below 256.
    shapes = compute_level_shapes(
        (1, 1, 4000, 1500),
        "TCYX",
        (1.0, 1.0, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert shapes == [(1, 1, 4000, 1500), (1, 1, 2000, 750), (1, 1, 1000, 375)]


def test_a_single_plane_does_not_freeze_the_lateral_axes() -> None:
    """A frozen axis must not become the isotropy yardstick.

    Z=1 can never halve, so it holds its level-0 spacing while Y and X double
    theirs every level. Left in the reference set it would declare Y and X "too
    coarse" after two levels and collapse the pyramid of every 2D acquisition.
    """
    shapes = compute_level_shapes(
        (1, 1, 1, 256, 256),
        "TCZYX",
        (1.0, 1.0, 1.0, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=32),
    )
    assert shapes == [
        (1, 1, 1, 256, 256),
        (1, 1, 1, 128, 128),
        (1, 1, 1, 64, 64),
        (1, 1, 1, 32, 32),
    ]


# ---------- level shapes: the new rule ----------


def test_reference_volume_halves_every_axis() -> None:
    """Z at 2.0 µm is within 1.5x of 1.8 µm laterally, so it halves too."""
    shapes = compute_level_shapes(
        REFERENCE_SHAPE,
        "TCZYX",
        REFERENCE_SPACINGS,
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [
        (3627, 8835, 7452),
        (1813, 4417, 3726),
        (906, 2208, 1863),
        (453, 1104, 931),
        (226, 552, 465),
        # Level 5 is the coarse level, bought by the #86 stopping rule: the
        # Y/X floor would have stopped at level 4, still 110 MiB per (t, c).
        (113, 276, 232),
    ]
    assert all(s[:2] == (1, 3) for s in shapes)


def test_a_three_plane_stack_keeps_its_planes() -> None:
    """The floor is per axis, not on ``min(Z, Y, X)``.

    Applying the depth floor to the smallest of all three would see ``min = 3``
    and write a single level — the regression ADR-0010 rejects by name.
    """
    shapes = compute_level_shapes(
        (1, 2, 3, 2048, 2048),
        "TCZYX",
        CONFOCAL_SPACINGS,
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert shapes == [
        (1, 2, 3, 2048, 2048),
        (1, 2, 3, 1024, 1024),
        (1, 2, 3, 512, 512),
        (1, 2, 3, 256, 256),
    ]


def test_the_pyramid_moves_toward_isotropy() -> None:
    """A 10:1 stack is less anisotropic at its coarsest level than at level 0.

    Y and X are spent first, so their spacing climbs toward Z's instead of the
    ratio being preserved as an invariant — which is exactly what the rejected
    uniform-halving policy would have done.
    """
    base = (1, 1, 64, 2048, 2048)

    shapes = compute_level_shapes(
        base, "TCZYX", CONFOCAL_SPACINGS, np.uint16, Geometry()
    )

    assert [s[2:] for s in shapes] == [
        (64, 2048, 2048),
        (64, 1024, 1024),
        (64, 512, 512),
        (64, 256, 256),
    ]
    level_0 = spacings_for_level(CONFOCAL_SPACINGS, base, shapes[0])
    coarsest = spacings_for_level(CONFOCAL_SPACINGS, base, shapes[-1])
    assert _anisotropy(level_0, "TCZYX") == 10.0
    assert _anisotropy(coarsest, "TCZYX") == 1.25


def test_the_scarce_axis_is_spent_last() -> None:
    """The same 10:1 stack, level by level: Z holds until Y/X catch up."""
    shapes = compute_level_shapes(
        (1, 1, 64, 2048, 2048),
        "TCZYX",
        CONFOCAL_SPACINGS,
        np.uint16,
        Geometry(pyramid_min_size=8, axis_floor=8),
    )
    assert [s[2:] for s in shapes] == [
        (64, 2048, 2048),
        (64, 1024, 1024),  # Z 5.0 µm vs Y/X 1.0 — 5x, out of tolerance
        (64, 512, 512),  # 2.5x
        (64, 256, 256),  # 2x
        (32, 128, 128),  # Y/X now 4.0 µm, within 1.5x of Z — Z joins in
        (16, 64, 64),
        (8, 32, 32),
        (8, 16, 16),  # Z is at the floor: 8 // 2 is below 8, so it holds
        (8, 8, 8),
    ]


def test_an_axis_at_the_floor_never_halves_again() -> None:
    """40 planes halve once to 20 — except 20 is below the 32-voxel floor."""
    shapes = compute_level_shapes(
        (1, 1, 40, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 0.5, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [(40, 1024, 1024), (40, 512, 512), (40, 256, 256)]


def test_an_axis_halves_down_to_the_floor_but_not_past_it() -> None:
    shapes = compute_level_shapes(
        (1, 1, 64, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 0.5, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [(64, 1024, 1024), (32, 512, 512), (32, 256, 256)]


def test_a_tall_volume_grows_no_tail_of_z_only_levels() -> None:
    """Once Y and X are at their floor the pyramid stops, Z or no Z.

    A 512-plane column over a 128² field would otherwise keep halving Z alone
    for four more levels, each a new resolution that is not new laterally —
    nothing a viewer zooming out can use.
    """
    shapes = compute_level_shapes(
        (1, 1, 512, 128, 128),
        "TCZYX",
        (1.0, 1.0, 0.5, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=32),
    )
    assert [s[2:] for s in shapes] == [(512, 128, 128), (256, 64, 64), (128, 32, 32)]


def test_an_oversampled_z_is_spent_while_the_lateral_axes_wait() -> None:
    """The scarce axis can be lateral: Z at 0.2 µm against 1.0 µm laterally.

    Levels 1 and 2 shrink Z alone — no lateral progress, but not a stall
    either: they are what brings Z's spacing up to Y/X's, after which all three
    halve together. The depth rule has to tell this apart from the exhausted
    laterals above, so it asks what Y/X can *ever* do, not what they did here.
    """
    shapes = compute_level_shapes(
        (1, 1, 256, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 0.2, 1.0, 1.0),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [
        (256, 1024, 1024),
        (128, 1024, 1024),  # Z 0.2 µm is 5x finer than Y/X — spend it first
        (64, 1024, 1024),  # 2.5x
        (32, 512, 512),  # Z is now 0.8 µm, within 1.5x of 1.0 — all three go
        (32, 256, 256),  # Z has hit the 32-voxel floor
    ]


def test_the_axis_floor_does_not_override_a_lower_depth_floor() -> None:
    """``pyramid_min_size=8`` still buys levels down to 8 on Y/X.

    ADR-0010 requires the change be monotone — no existing conversion loses a
    level — so the 32-voxel default cannot quietly outvote a caller who asked
    for a smaller one.
    """
    shapes = compute_level_shapes(
        (1, 1, 64, 64),
        "TCYX",
        (1.0, 1.0, 0.5, 0.5),
        np.uint16,
        Geometry(pyramid_min_size=8),
    )
    assert shapes == [(1, 1, 64, 64), (1, 1, 32, 32), (1, 1, 16, 16), (1, 1, 8, 8)]


def test_tolerance_of_one_halves_only_exactly_isotropic_axes() -> None:
    """Z at 0.55 µm against 0.5 laterally: inside 1.5x, outside exactly 1.0x.

    Only level 1 is compared. Past it the two pyramids diverge in depth as well
    as in shape — under a tolerance of 1.0 the finest axis is always the one
    that halves, so the levels descend one axis at a time.
    """
    base = (1, 1, 512, 1024, 1024)
    spacings = (1.0, 1.0, 0.55, 0.5, 0.5)

    strict = compute_level_shapes(
        base,
        "TCZYX",
        spacings,
        np.uint16,
        Geometry(pyramid_min_size=256, isotropy_tolerance=1.0),
    )
    default = compute_level_shapes(
        base, "TCZYX", spacings, np.uint16, Geometry(pyramid_min_size=256)
    )

    assert strict[1][2:] == (512, 512, 512)
    assert default[1][2:] == (256, 512, 512)


def test_a_large_tolerance_halves_every_spatial_axis() -> None:
    """The rejected uniform-halving policy stays reachable as a setting."""
    shapes = compute_level_shapes(
        (1, 1, 512, 1024, 1024),
        "TCZYX",
        CONFOCAL_SPACINGS,
        np.uint16,
        Geometry(pyramid_min_size=256, isotropy_tolerance=1e9),
    )
    assert [s[2:] for s in shapes] == [
        (512, 1024, 1024),
        (256, 512, 512),
        (128, 256, 256),
    ]


def test_unusable_spacings_read_as_isotropic() -> None:
    """A reader that reports nothing usable should not pin the scarce axis.

    Zero, negative and non-finite spacings carry no information about which
    axis is scarce, so they degrade to 1.0 and every spatial axis halves —
    the same answer a caller with no pixel-size metadata at all would get.
    """
    shapes = compute_level_shapes(
        (1, 1, 512, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 0.0, float("nan"), -1.0),
        np.uint16,
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [
        (512, 1024, 1024),
        (256, 512, 512),
        (128, 256, 256),
    ]


def test_compute_level_shapes_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        compute_level_shapes((1, 1, 64, 64), "TCZYX", (1.0, 1.0, 1.0, 1.0), np.uint16)


# ---------- which levels are coarse ----------


def test_a_level_is_coarse_only_when_both_bounds_hold() -> None:
    """64 MiB per (t, c) *and* a long axis of 2048 — neither alone is enough."""
    # 128 * 512 * 512 uint16 voxels is exactly 64 MiB, and the bound is a `>`
    # comparison, so exactly-at-the-bound passes (ADR-0010 rejects picking a
    # target below Lucida's "for margin").
    assert is_coarse_level((1, 1, 128, 512, 512), "TCZYX", np.uint16)
    assert not is_coarse_level((1, 1, 129, 512, 512), "TCZYX", np.uint16)
    # 32 MiB, comfortably inside the byte bound — but a 4096-voxel long axis is
    # not a texture a viewer holds as context.
    assert not is_coarse_level((1, 1, 1, 4096, 4096), "TCZYX", np.uint16)


def test_the_byte_bound_is_per_timepoint_and_channel() -> None:
    """A 40-timepoint, 4-channel store is not 160x too big to navigate.

    A viewer holds one timepoint of one channel at a time, so T and C take no
    part in the byte bound — otherwise a time series would be driven to a
    pointlessly tiny coarse level by axes nobody decodes at once.
    """
    assert is_coarse_level((40, 4, 128, 512, 512), "TCZYX", np.uint16)


def test_the_byte_bound_reads_the_dtype() -> None:
    """Decoded bytes, so the same shape lands differently per itemsize."""
    shape = (1, 1, 128, 512, 512)
    assert is_coarse_level(shape, "TCZYX", np.uint8)  # 32 MiB
    assert is_coarse_level(shape, "TCZYX", np.uint16)  # 64 MiB, exactly at it
    assert not is_coarse_level(shape, "TCZYX", np.float32)  # 128 MiB


def test_the_bounds_are_geometry_fields() -> None:
    """Both are knobs, so a store can be planned for a different consumer."""
    tight = Geometry(coarse_max_bytes=1024 * 1024, coarse_max_long_axis=128)
    assert is_coarse_level((1, 1, 8, 128, 128), "TCZYX", np.uint16, tight)
    assert not is_coarse_level((1, 1, 8, 256, 256), "TCZYX", np.uint16, tight)


def test_coarse_level_index_names_the_shallowest_qualifying_level() -> None:
    """The largest level that fits — the deeper ones fit too, and are worse.

    Coarseness is monotone down the pyramid, so "is there one?" and "which one?"
    are answered by the same scan; the level a viewer wants for context is the
    most detailed of those it can hold whole.
    """
    shapes = [
        (1, 1, 64, 2048, 2048),  # 512 MiB
        (1, 1, 64, 1024, 1024),  # 128 MiB
        (1, 1, 64, 512, 512),  # 32 MiB — the first that fits
        (1, 1, 64, 256, 256),  # 8 MiB
    ]
    assert coarse_level_index(shapes, "TCZYX", np.uint16) == 2


def test_a_level_zero_that_already_fits_is_the_coarse_level() -> None:
    """A small field needs no extra depth — level 0 is already holdable."""
    geometry = Geometry(pyramid_min_size=256)
    shapes = compute_level_shapes(
        (1, 1, 8, 512, 512),
        "TCZYX",
        (1.0, 1.0, 0.5, 0.5, 0.5),
        np.uint16,
        geometry,
    )
    assert [s[2:] for s in shapes] == [(8, 512, 512), (8, 256, 256)]
    assert coarse_level_index(shapes, "TCZYX", np.uint16, geometry) == 0


# ---------- depth: the coarse-level stopping rule ----------


def test_the_reference_volume_gains_a_coarse_level() -> None:
    """The motivating case: depth runs one level past the Y/X rule.

    ``pyramid_min_size`` stops at level 4, which is 110 MiB per (t, c) — a
    level no viewer can hold as context, and the whole defect ADR-0010 exists
    to fix. The coarse-level rule buys level 5 at 13.8 MiB, 276 long axis.
    """
    geometry = Geometry()
    shapes = compute_level_shapes(
        REFERENCE_SHAPE, "TCZYX", REFERENCE_SPACINGS, np.uint16, geometry
    )

    assert shapes[-1][2:] == (113, 276, 232)
    assert not is_coarse_level(shapes[4], "TCZYX", np.uint16, geometry)
    assert coarse_level_index(shapes, "TCZYX", np.uint16, geometry) == 5


def test_the_dtype_decides_where_depth_stops() -> None:
    """Same shape and spacings, one level apart, because bytes are the bound.

    Level 4 of the reference volume is 58 M voxels: 110 MiB in uint16, past the
    bound, but 55 MiB in uint8, inside it. A Y/X voxel-count floor cannot see
    this difference at all — which is ADR-0010's argument for not using one.
    """
    narrow = compute_level_shapes(
        REFERENCE_SHAPE, "TCZYX", REFERENCE_SPACINGS, np.uint8, Geometry()
    )
    wide = compute_level_shapes(
        REFERENCE_SHAPE, "TCZYX", REFERENCE_SPACINGS, np.uint16, Geometry()
    )

    assert narrow[-1][2:] == (226, 552, 465)
    assert wide[-1][2:] == (113, 276, 232)


def test_the_axis_floor_still_bounds_the_extended_depth() -> None:
    """The coarse rule extends depth; it does not lift the 32-voxel floor.

    Bounds this shape can never reach, so the extension runs as far as it is
    ever allowed to: every axis lands exactly on the floor and stops there,
    rather than grinding an axis away in pursuit of a target.
    """
    geometry = Geometry(coarse_max_bytes=1024)
    shapes = compute_level_shapes(
        (1, 1, 256, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 0.5, 0.5, 0.5),
        np.uint16,
        geometry,
    )

    # The Y/X rule alone would have stopped at (64, 256, 256).
    assert shapes[-1][2:] == (32, 32, 32)
    assert all(extent >= 32 for extent in shapes[-1][2:])
    assert coarse_level_index(shapes, "TCZYX", np.uint16, geometry) is None


def test_bounds_no_level_can_reach_record_a_null_coarse_level() -> None:
    """A pyramid may contain no coarse level, and that is a fact to record.

    ``coarse_max_long_axis=16`` is below the 32-voxel axis floor, so the two
    floors are in conflict and no level can ever satisfy the bound. Depth still
    terminates on the floor-frozen laterals and the conversion still writes;
    only the recorded index is null.
    """
    geometry = Geometry(coarse_max_long_axis=16)
    shapes = compute_level_shapes(
        (1, 1, 4, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 2.0, 0.5, 0.5),
        np.uint16,
        geometry,
    )

    assert [s[2:] for s in shapes] == [
        (4, 1024, 1024),
        (4, 512, 512),
        (4, 256, 256),
        (4, 128, 128),
        (4, 64, 64),
        (4, 32, 32),
    ]
    assert coarse_level_index(shapes, "TCZYX", np.uint16, geometry) is None


#: The pre-#86 depth rule, expressed in the post-#86 policy: bounds so wide that
#: level 0 always qualifies, so the coarse rule never asks for a level the
#: ``pyramid_min_size`` rule did not already grant.
_PRE_COARSE_RULE = {"coarse_max_bytes": 2**62, "coarse_max_long_axis": 2**31}


@pytest.mark.parametrize(
    "base, dims, spacings, geometry",
    [
        (REFERENCE_SHAPE, "TCZYX", REFERENCE_SPACINGS, Geometry()),
        ((1, 1, 64, 2048, 2048), "TCZYX", CONFOCAL_SPACINGS, Geometry()),
        ((1, 2, 3, 2048, 2048), "TCZYX", CONFOCAL_SPACINGS, Geometry()),
        ((1, 4, 2048, 2048), "TCYX", (1.0, 1.0, 0.5, 0.5), Geometry()),
        ((1, 1, 512, 128, 128), "TCZYX", (1.0, 1.0, 0.5, 0.5, 0.5), Geometry()),
        (
            (1, 1, 16, 256, 256),
            "TCZYX",
            (1.0, 1.0, 4.0, 0.5, 0.5),
            Geometry(pyramid_min_size=8, axis_floor=4),
        ),
    ],
)
def test_no_shape_loses_a_level_to_the_new_rule(
    base: tuple[int, ...],
    dims: str,
    spacings: tuple[float, ...],
    geometry: Geometry,
) -> None:
    """Depth is a ``max()`` of two rules, so it can only grow.

    ADR-0010 commits to the change being monotone. The stronger property holds
    too and is what is asserted: the new pyramid *extends* the old one level for
    level rather than reshaping it, so a re-conversion's level ``n`` is the same
    array it was before.
    """
    before = compute_level_shapes(
        base,
        dims,
        spacings,
        np.uint16,
        dataclasses.replace(geometry, **_PRE_COARSE_RULE),
    )
    after = compute_level_shapes(base, dims, spacings, np.uint16, geometry)

    assert len(after) >= len(before)
    assert after[: len(before)] == before


# ---------- building the levels ----------


def test_build_pyramid_mean_pools() -> None:
    base = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    levels = build_pyramid(da.from_array(base, chunks=(2, 2)), [(4, 4), (2, 2)])
    assert len(levels) == 2
    np.testing.assert_array_equal(
        levels[1].compute(), np.array([[3.5, 5.5], [11.5, 13.5]], dtype=np.float32)
    )


def test_build_pyramid_preserves_dtype() -> None:
    base_da = da.from_array(np.full((4, 4), 100, dtype=np.uint16), chunks=(2, 2))
    levels = build_pyramid(base_da, [(4, 4), (2, 2)])
    assert levels[1].dtype == np.uint16
    assert levels[1].compute().tolist() == [[100, 100], [100, 100]]


def test_build_pyramid_reads_its_factors_off_the_level_shapes() -> None:
    """Per-axis-varying factors need no special case — Z holds, then halves."""
    base_da = da.from_array(np.zeros((1, 2, 8, 8, 8), dtype=np.uint16))
    level_shapes = [(1, 2, 8, 8, 8), (1, 2, 8, 4, 4), (1, 2, 4, 2, 2)]

    levels = build_pyramid(base_da, level_shapes)

    assert [tuple(lv.shape) for lv in levels] == level_shapes


def test_build_pyramid_trims_an_odd_extent_rather_than_padding() -> None:
    # 9 // 2 == 4 is what compute_level_shapes predicted; coarsen must agree.
    base_da = da.from_array(np.arange(9 * 4, dtype=np.uint16).reshape(9, 4))
    levels = build_pyramid(base_da, [(9, 4), (4, 2)])
    assert tuple(levels[1].shape) == (4, 2)


def test_build_pyramid_rejects_a_level_larger_than_its_parent() -> None:
    base_da = da.from_array(np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="does not downsample"):
        build_pyramid(base_da, [(4, 4), (8, 8)])


def test_build_pyramid_needs_at_least_one_level_shape() -> None:
    with pytest.raises(ValueError, match="at least one level"):
        build_pyramid(da.from_array(np.zeros((4, 4), dtype=np.uint16)), [])


# ---------- end to end ----------


def _scale_transforms(store: Path) -> list[list[float]]:
    """Each dataset's ``coordinateTransformations`` scale, in level order."""
    root = json.loads((store / "zarr.json").read_text())
    datasets = root["attributes"]["ome"]["multiscales"][0]["datasets"]
    return [d["coordinateTransformations"][0]["scale"] for d in datasets]


def test_coordinate_transformations_carry_the_per_axis_factors(
    tmp_path: Path, patched_reader
) -> None:
    """NGFF scale metadata needs no separate handling.

    ``OMEZarrWriter`` derives each dataset's scale as its shape divided by
    level 0's, per axis, so a Z that holds for three levels and then halves
    twice shows up in ``coordinateTransformations`` without the writer knowing
    anything about the isotropy rule.
    """
    patched_reader(
        FakeReader(
            scenes=["s"],
            dims="TCZYX",
            shape=(1, 1, 16, 256, 256),
            pixel_sizes=FakePhysicalPixelSizes(Z=4.0, Y=0.5, X=0.5),
        )
    )
    out = tmp_path / "out"

    result = convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(pyramid_min_size=8, axis_floor=4),
        contrast_percentile=None,
        validate=False,
    )

    assert [s[2:] for s in result["stores"][0]["per_scene"][0]["level_shapes"]] == [
        [16, 256, 256],
        [16, 128, 128],
        [16, 64, 64],
        [16, 32, 32],
        [8, 16, 16],
        [4, 8, 8],
    ]
    # Z holds at 4.0 µm while Y/X double, then joins — and the last level is
    # isotropic at 16 µm on every axis, which is the point of the rule.
    assert _scale_transforms(out / "s.ome.zarr") == [
        [1.0, 1.0, 4.0, 0.5, 0.5],
        [1.0, 1.0, 4.0, 1.0, 1.0],
        [1.0, 1.0, 4.0, 2.0, 2.0],
        [1.0, 1.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 8.0, 8.0, 8.0],
        [1.0, 1.0, 16.0, 16.0, 16.0],
    ]


def test_the_written_pyramid_reaches_the_coarse_bounds_and_says_so(
    tmp_path: Path, patched_reader
) -> None:
    """End to end: the extra level is written, and the audit names it.

    A 64 KiB byte bound stands in for the 64 MiB one so the fixture stays a
    megabyte rather than a terabyte; the mechanism is the same. The Y/X rule
    alone would stop at level 1 (128 KiB per (t, c)) — one level short of
    anything a viewer could hold whole.
    """
    patched_reader(
        FakeReader(
            scenes=["s"],
            dims="TCZYX",
            shape=(1, 1, 32, 128, 128),
            pixel_sizes=FakePhysicalPixelSizes(Z=0.5, Y=0.5, X=0.5),
        )
    )
    out = tmp_path / "out"

    result = convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(
            pyramid_min_size=64, axis_floor=4, coarse_max_bytes=64 * 1024
        ),
        contrast_percentile=None,
        validate=False,
    )

    scene = result["stores"][0]["per_scene"][0]
    assert scene["level_shapes"] == [
        [1, 1, 32, 128, 128],
        [1, 1, 16, 64, 64],
        [1, 1, 8, 32, 32],  # 16 KiB per (t, c) — the coarse level
    ]
    assert scene["coarse_level_index"] == 2
    # And it is on disk, not just in the returned dict: the guarantee has to be
    # checkable from the store itself.
    root = json.loads((out / "s.ome.zarr" / "zarr.json").read_text())
    on_disk = root["attributes"]["zarrmony"]["per_scene"][0]
    assert on_disk["coarse_level_index"] == 2


def test_a_pyramid_with_no_coarse_level_records_null(
    tmp_path: Path, patched_reader
) -> None:
    """Bounds the axis floor cannot satisfy still convert — with a null index."""
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 128, 128)))
    out = tmp_path / "out"

    result = convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(pyramid_min_size=64, coarse_max_long_axis=16),
        contrast_percentile=None,
        validate=False,
    )

    scene = result["stores"][0]["per_scene"][0]
    assert scene["level_shapes"] == [[1, 1, 128, 128], [1, 1, 64, 64], [1, 1, 32, 32]]
    assert scene["coarse_level_index"] is None


def test_plate_fields_carry_the_coarse_level_index(
    tmp_path: Path, patched_reader
) -> None:
    """Plate audits get the key too — one ``write_scene``, one record shape.

    A 64² field is inside both bounds at level 0, which is the whole answer:
    there is nothing to zoom out to, and the audit says so with a 0 rather than
    leaving a consumer to infer it from the shapes.
    """
    layout = PlateLayout(
        name="coarse-plate",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1, name="acq")],
        fields=[PlateField(scene_index=0, row="A", column="01", acquisition_id=1)],
    )
    patched_reader(
        FakeReader(
            scenes=["s0"],
            dims="TCYX",
            shape=(1, 1, 64, 64),
            layout_hint="plate",
            plate_layout=layout,
            channel_names=["DAPI"],
        )
    )

    audit = convert(
        "/tmp/x.czi",
        tmp_path / "plate.ome.zarr",
        geometry=Geometry(pyramid_min_size=32),
        contrast_percentile=None,
        validate=False,
    )

    field = audit["fields"][0]
    assert field["level_shapes"] == [[1, 1, 64, 64], [1, 1, 32, 32]]
    assert field["coarse_level_index"] == 0
