"""The planner says how many objects a run will write, at plan time (issue #122).

Every input to the object count is known before a byte moves: the level shapes,
each level's chunk shape, and each level's shard shape or its absence. Nothing
computed it, so a gigapixel 2D scene at the 512 KiB default planned ~493,000
objects and roughly six days (#113) without saying so — the user found out by
watching the run not finish.

Two layers, in order:

1. :func:`count_storage_objects` — the arithmetic, over whichever grid is the
   storage object at each level. Pure, and shared with the acceptance tool that
   currently predicts the count off the chunk grid on sharded stores (#124).
2. The scene writer's :class:`ObjectCountWarning` — emitted once per scene,
   with sharding off, above :data:`STORAGE_OBJECT_WARN_COUNT`, before the
   arrays exist, and naming ``--shard-target-bytes`` together with the catch
   that a sharded store needs ``sharding_indexed`` support to open.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.errors import ObjectCountWarning
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    DEFAULT_SHARD_TARGET_BYTES,
    STORAGE_OBJECT_WARN_COUNT,
    Geometry,
    count_storage_objects,
    plan_level_chunk_shapes,
    plan_level_shard_shapes,
)
from zarrmony.readers.plugin import ReaderPlugin
from zarrmony.writers import scene as scene_module
from zarrmony.writers.pyramid import compute_level_shapes

DIMS = list("TCZYX")

# The #113 reference scene and its measured counts: 493,484 chunk objects
# unsharded against 31,156 shard objects at 512²-in-2048², the two runs the
# threshold was chosen between (ADR-0010, follow-ups #113 and #124).
WSI = (1, 4, 1, 140757, 171855)
WSI_SPACING = [1.0, 1.0, 1.0, 0.25, 0.25]
WSI_CHUNK_OBJECTS = 493_484
WSI_SHARD_OBJECTS = 31_156

# A scene small enough to actually write in a test. Stock geometry plans three
# levels and six objects; a 64² chunk override plans 336 of them.
SMALL = (1, 1, 1, 1024, 1024)
SMALL_PIXEL_SIZES = FakePhysicalPixelSizes(Z=1.0, Y=0.5, X=0.5)
TINY_CHUNK = Geometry(chunk_shape=(1, 1, 1, 64, 64))
TINY_CHUNK_OBJECTS = 336
TINY_CHUNK_SHARD_OBJECTS = 3


# --------------------------------------------------------------------------
# 1. The arithmetic
# --------------------------------------------------------------------------


def test_counts_whole_and_partial_objects_over_a_pyramid() -> None:
    """A trailing partial object is still an object, on every axis it happens on.

    Hand-computed: level 0 is a 3 x 3 grid of 4-voxel chunks over 10 voxels on
    both lateral axes (``ceil(10 / 4)``), times 2 channels, so 18; level 1 is
    2 channels x ``ceil(5 / 4)`` twice, so 8. Flooring instead of ceiling would
    silently drop the edge of every level.
    """
    levels = [(1, 2, 1, 10, 10), (1, 2, 1, 5, 5)]
    grids = [(1, 1, 1, 4, 4), (1, 1, 1, 4, 4)]
    assert count_storage_objects(levels, grids) == 18 + 8


def test_counts_shards_when_handed_shard_grids() -> None:
    """The grid is whatever one storage object is, which under sharding is not the chunk.

    Handing the chunk grid to a sharded store is exactly the defect #124 found
    in ``verify_geometry.py``: a 512²-in-2048² level reports the 2048² count.
    """
    level = [(1, 1, 1, 4096, 4096)]
    assert count_storage_objects(level, [(1, 1, 1, 512, 512)]) == 64
    assert count_storage_objects(level, [(1, 1, 1, 2048, 2048)]) == 4


def test_a_grid_larger_than_the_level_is_one_object() -> None:
    assert count_storage_objects([(1, 1, 1, 300, 300)], [(1, 1, 1, 512, 512)]) == 1


@pytest.mark.parametrize(
    ("levels", "grids"),
    [
        ([(1, 1, 8, 8)], [(1, 1, 4, 4), (1, 1, 4, 4)]),  # a grid per level
        ([(1, 1, 8, 8)], [(1, 4, 4)]),  # a length per axis
    ],
)
def test_a_mismatched_grid_is_an_error_not_a_guess(levels, grids) -> None:
    with pytest.raises(ValueError, match="count_storage_objects needs one"):
        count_storage_objects(levels, grids)


def test_the_reference_scene_reproduces_both_measured_counts() -> None:
    """The threshold has to separate a run that was fine from one that was not.

    Both counts come from the planners rather than from the ADR's text, so this
    fails if a geometry change moves either — which is the point: 100,000 is
    only meaningful relative to the two runs it sits between.
    """
    levels = compute_level_shapes(WSI, DIMS, WSI_SPACING, np.uint16, DEFAULT_GEOMETRY)
    chunks = plan_level_chunk_shapes(
        levels, DIMS, WSI_SPACING, np.uint16, DEFAULT_GEOMETRY
    )
    shards = plan_level_shard_shapes(
        chunks,
        levels,
        DIMS,
        WSI_SPACING,
        np.uint16,
        Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES),
    )
    assert count_storage_objects(levels, chunks) == WSI_CHUNK_OBJECTS
    assert count_storage_objects(levels, shards) == WSI_SHARD_OBJECTS
    assert WSI_SHARD_OBJECTS < STORAGE_OBJECT_WARN_COUNT < WSI_CHUNK_OBJECTS


# --------------------------------------------------------------------------
# 2. The warning
# --------------------------------------------------------------------------


def test_it_is_an_ordinary_user_warning() -> None:
    """Which is what makes ``warnings.simplefilter`` the whole override story."""
    assert issubclass(ObjectCountWarning, UserWarning)


@pytest.fixture
def small_scene(monkeypatch: pytest.MonkeyPatch):
    """Patch ``get_reader`` with a one-scene reader ``convert()`` can write fast."""
    reader = FakeReader(
        scenes=["slide"],
        dims="TCZYX",
        shape=SMALL,
        pixel_sizes=SMALL_PIXEL_SIZES,
    )
    plugin = ReaderPlugin(
        name="bioio-fake",
        match=lambda _p: 100,
        open=lambda _p: reader,
        distribution="bioio-fake",
        source="builtin",
    )
    monkeypatch.setattr(
        api_module,
        "get_reader",
        lambda _path, *, reader_kwargs=None: (reader, plugin, 100),
    )
    return reader


@pytest.fixture
def low_threshold(monkeypatch: pytest.MonkeyPatch) -> int:
    """Lower the threshold to the writer, so a test scene can cross it.

    The alternative is a fixture that plans 100,000 objects, which means
    writing 100,000 files. The threshold's real value is pinned against the two
    measured runs in ``test_the_reference_scene_reproduces_both_measured_counts``;
    what these tests are about is the behaviour on either side of it.
    """
    monkeypatch.setattr(scene_module, "STORAGE_OBJECT_WARN_COUNT", 100)
    return 100


def _convert(tmp_path: Path, **kwargs) -> None:
    convert(
        "/tmp/x.tif",
        tmp_path / "out",
        contrast_percentile=None,
        validate=False,
        **kwargs,
    )


def test_warns_once_with_the_count_the_flag_and_the_catch(
    tmp_path: Path, small_scene, low_threshold
) -> None:
    """One warning per scene, carrying all four things it fails without."""
    with pytest.warns(ObjectCountWarning) as records:
        _convert(tmp_path, geometry=TINY_CHUNK)
    # The 64² override also trips the tile-alignment warning, which is a
    # separate true thing about the same conversion; one *object count* warning
    # is what "exactly one" means.
    emitted = [w for w in records if issubclass(w.category, ObjectCountWarning)]
    assert len(emitted) == 1

    message = str(emitted[0].message)
    # 1. the count, and which scene it is about
    assert f"{TINY_CHUNK_OBJECTS:,} storage objects" in message
    assert "scene 0 ('slide'" in message
    assert "3 levels" in message
    assert "level-0 chunk (1, 1, 1, 64, 64)" in message
    # 2. what it costs, anchored to the run that was measured
    assert "slow to write" in message and "slow to list" in message
    assert "six days" in message
    # 3. the lever, and what it buys
    assert f"--shard-target-bytes {DEFAULT_SHARD_TARGET_BYTES}" in message
    assert f"({TINY_CHUNK_SHARD_OBJECTS:,} objects)" in message
    # 4. the catch, without which the advice is worse than the problem
    assert "sharding_indexed" in message


def test_the_message_cites_nothing_the_reader_cannot_act_on(
    tmp_path: Path, small_scene, low_threshold
) -> None:
    """No issue numbers, no ADR numbers, no downstream project names.

    This is read at a terminal by someone converting a file. The measurements
    behind the threshold belong in the class docstring and the ADR, where
    someone reading this repository will look for them.
    """
    with pytest.warns(ObjectCountWarning) as records:
        _convert(tmp_path, geometry=TINY_CHUNK)
    message = str(records.pop(ObjectCountWarning).message).lower()
    for citation in ("adr-", "issue #", "#1", "lucida"):
        assert citation not in message


def test_the_warning_lands_before_the_arrays_are_created(
    tmp_path: Path, small_scene, low_threshold
) -> None:
    """Plan time is the only time the answer is still useful.

    Turning the warning into an error is the cheap way to ask *when* it fires:
    a store with no arrays in it means nothing had been created yet.
    """
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ObjectCountWarning)
        with pytest.raises(ObjectCountWarning):
            convert(
                "/tmp/x.tif",
                out,
                geometry=TINY_CHUNK,
                contrast_percentile=None,
                validate=False,
            )
    assert list(out.rglob("*.ome.zarr/0")) == []


def test_sharding_silences_it(tmp_path: Path, small_scene, low_threshold, recwarn):
    """With shards the object count *is* the shard count, and the trade is made."""
    _convert(
        tmp_path,
        geometry=Geometry(
            chunk_shape=(1, 1, 1, 64, 64),
            shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES,
        ),
    )
    assert [w for w in recwarn if issubclass(w.category, ObjectCountWarning)] == []


def test_a_pyramid_under_the_threshold_is_quiet(
    tmp_path: Path, small_scene, recwarn
) -> None:
    """Stock threshold, stock geometry: six objects is not worth a sentence."""
    _convert(tmp_path)
    assert [w for w in recwarn if issubclass(w.category, ObjectCountWarning)] == []


def test_it_does_not_offer_a_lever_that_would_not_move() -> None:
    """At an 8 MiB chunk a shard holds one chunk, so packing them cuts nothing.

    Called directly: the only way to reach this branch through ``convert()`` is
    a store of some hundreds of gigabytes. The count still gets reported — what
    changes is that the message stops recommending a flag that would not help.
    """
    huge_chunk = (1, 1, 1, 2048, 2048)  # 8 MiB of uint16, the shard target
    geometry = Geometry(chunk_shape=huge_chunk, chunk_target_bytes=8 * 1024 * 1024)
    with pytest.warns(ObjectCountWarning) as records:
        scene_module._warn_on_object_count(
            [(1, 1, 1, 1_048_576, 1_048_576)],
            [huge_chunk],
            DIMS,
            [1.0, 1.0, 1.0, 0.5, 0.5],
            np.uint16,
            geometry,
            0,
            "slide",
        )
    message = str(records[0].message)
    assert "262,144 storage objects" in message
    assert "--shard-target-bytes cannot cut this" in message


def test_an_explicit_chunk_shape_is_counted_the_same_way(
    tmp_path: Path, small_scene, low_threshold, recwarn
) -> None:
    """The override changes the grid, not whether the count is reported.

    Same scene, same threshold, both directions: the 64² override plans 336
    objects and warns, and the planner's own 512² chunk plans 6 and does not.
    ``plan_chunk_shape`` is never consulted on the first, so a count taken off
    the planner rather than off the resolved shapes would miss it entirely.
    """
    with pytest.warns(ObjectCountWarning, match=f"{TINY_CHUNK_OBJECTS:,}"):
        _convert(tmp_path, geometry=TINY_CHUNK)

    recwarn.clear()
    _convert(tmp_path / "planned", geometry=DEFAULT_GEOMETRY)
    assert [w for w in recwarn if issubclass(w.category, ObjectCountWarning)] == []
