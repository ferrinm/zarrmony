"""Tests for the ADR-0010 anisotropy-aware pyramid (issue #85).

Three concerns, in order:

1. :func:`compute_level_shapes` — the rule itself: halve every spatial axis
   whose µm spacing is within ``isotropy_tolerance`` of the finest still-
   halvable axis's, subject to a per-axis voxel floor, with depth still decided
   by the Y/X ``pyramid_min_size`` rule.
2. :func:`build_pyramid` — mean-pooling with the coarsen factors read off
   consecutive level shapes, so per-axis-varying and uniform downsampling are
   one code path.
3. That per-axis factors reach the store's ``coordinateTransformations``, which
   ``OMEZarrWriter`` derives from the level shapes we hand it.
"""

from __future__ import annotations

import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import Geometry, spacings_for_level
from zarrmony.readers.plugin import ReaderPlugin
from zarrmony.writers.pyramid import build_pyramid, compute_level_shapes

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
        (1, 1, 100, 100), "TCYX", (1.0, 1.0, 0.5, 0.5), Geometry(pyramid_min_size=256)
    )
    assert shapes == [(1, 1, 100, 100)]


def test_non_spatial_dims_are_preserved() -> None:
    shapes = compute_level_shapes(
        (10, 4, 1, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 1.0, 0.5, 0.5),
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
    assert compute_level_shapes((10, 4), "TC", (1.0, 1.0)) == [(10, 4)]


def test_depth_is_still_decided_by_the_lateral_axes() -> None:
    # X (1500) reaches 375 first; the next level's min(500, 375) is below 256.
    shapes = compute_level_shapes(
        (1, 1, 4000, 1500), "TCYX", (1.0, 1.0, 0.5, 0.5), Geometry(pyramid_min_size=256)
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
        REFERENCE_SHAPE, "TCZYX", REFERENCE_SPACINGS, Geometry(pyramid_min_size=256)
    )
    assert [s[2:] for s in shapes] == [
        (3627, 8835, 7452),
        (1813, 4417, 3726),
        (906, 2208, 1863),
        (453, 1104, 931),
        (226, 552, 465),
    ]
    # Depth is unchanged: the next level's 232 lateral extent is below 256.
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

    shapes = compute_level_shapes(base, "TCZYX", CONFOCAL_SPACINGS, Geometry())

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
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [(40, 1024, 1024), (40, 512, 512), (40, 256, 256)]


def test_an_axis_halves_down_to_the_floor_but_not_past_it() -> None:
    shapes = compute_level_shapes(
        (1, 1, 64, 1024, 1024),
        "TCZYX",
        (1.0, 1.0, 0.5, 0.5, 0.5),
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
        (1, 1, 64, 64), "TCYX", (1.0, 1.0, 0.5, 0.5), Geometry(pyramid_min_size=8)
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
        base, "TCZYX", spacings, Geometry(pyramid_min_size=256, isotropy_tolerance=1.0)
    )
    default = compute_level_shapes(
        base, "TCZYX", spacings, Geometry(pyramid_min_size=256)
    )

    assert strict[1][2:] == (512, 512, 512)
    assert default[1][2:] == (256, 512, 512)


def test_a_large_tolerance_halves_every_spatial_axis() -> None:
    """The rejected uniform-halving policy stays reachable as a setting."""
    shapes = compute_level_shapes(
        (1, 1, 512, 1024, 1024),
        "TCZYX",
        CONFOCAL_SPACINGS,
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
        Geometry(pyramid_min_size=256),
    )
    assert [s[2:] for s in shapes] == [
        (512, 1024, 1024),
        (256, 512, 512),
        (128, 256, 256),
    ]


def test_compute_level_shapes_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        compute_level_shapes((1, 1, 64, 64), "TCZYX", (1.0, 1.0, 1.0, 1.0))


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
