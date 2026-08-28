"""Reader tiles nest in the planned write grid (issue #112).

Nothing used to connect the reader's tile size to the geometry the planner
picks, so ``write_pyramid`` absorbed the mismatch with a ``rechunk``. On the
common whole-slide path that rechunk was always a *split* — and the cost of a
split is not the cost of a merge:

===========================  ==========  ================
source tile -> write grid    dask tasks  vs source blocks
===========================  ==========  ================
1024 -> 512 (split)             831,936             9.00x
512 -> 512 (exact)              369,600             1.00x
1024 -> 2048 (merge)            115,920             1.25x
512 -> 2048 (merge)             392,784             1.06x
===========================  ==========  ================

Measured on the reference scene ``[1, 4, 1, 140757, 171855]`` uint16; the split
row is the configuration zarrmony's own runbook used to recommend. That
asymmetry is the whole design: a tile that *divides* the write grid is nearly
free, a tile that exceeds or straddles it is not, and the fix is to tell the
reader the grid rather than to optimise the rechunk.

Three layers, in order:

1. :func:`plan_write_grid` and :func:`split_axes` — what a write covers, and
   when source blocks fail to nest in it.
2. :func:`plan_reader_tile_size` — the tile derived back out of the grid,
   including why the minimum across scenes is provably split-free.
3. ``convert()`` reopening the reader with it, respecting a pinned
   ``tile_size``, and the writer warning when blocks split anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.errors import TileAlignmentWarning
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    DEFAULT_SHARD_TARGET_BYTES,
    Geometry,
    plan_reader_tile_size,
    plan_write_grid,
    split_axes,
)
from zarrmony.readers.plugin import ReaderPlugin

DIMS = list("TCZYX")
SHARDED = Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES)

# The #112 reference scene, and the label/macro thumbnails a whole-slide file
# carries beside it. Their planned grids differ, which is the multi-scene case.
WSI = (1, 4, 1, 140757, 171855)
THUMB = (1, 3, 1, 375, 504)
SPACING = [1.0, 1.0, 1.0, 0.25, 0.25]


# --------------------------------------------------------------------------
# 1. The write grid, and when blocks fail to nest in it
# --------------------------------------------------------------------------


def test_write_grid_is_the_chunk_when_sharding_is_off() -> None:
    grid = plan_write_grid(WSI, DIMS, SPACING, np.uint16, DEFAULT_GEOMETRY)
    assert grid == (1, 1, 1, 512, 512)


def test_write_grid_is_the_shard_when_sharding_is_on() -> None:
    """The write unit follows the shard, because that is what one write covers.

    Reaching for the chunk here is the #117 trap one level up: it would derive a
    reader tile matching the read unit, leaving every write to assemble 16 of
    them.
    """
    grid = plan_write_grid(WSI, DIMS, SPACING, np.uint16, SHARDED)
    assert grid == (1, 1, 1, 2048, 2048)


def test_write_grid_honours_an_explicit_chunk_shape() -> None:
    explicit = (1, 1, 1, 128, 128)
    grid = plan_write_grid(
        WSI, DIMS, SPACING, np.uint16, Geometry(chunk_shape=explicit)
    )
    assert grid == explicit


@pytest.mark.parametrize(
    ("tile", "expected_split"),
    [
        (2048, True),  # larger than the grid — every write splits one block
        (1024, True),  # the trap: bigger tiles, bigger graph
        (512, False),  # exact
        (256, False),  # divides — a cheap merge
        (64, False),
        (300, True),  # straddles: boundaries land inside source blocks
    ],
)
def test_split_axes_flags_only_the_split_direction(
    tile: int, expected_split: bool
) -> None:
    offenders = split_axes((1, 1, 1, tile, tile), (1, 1, 1, 512, 512), WSI, DIMS)
    assert bool(offenders) is expected_split
    if expected_split:
        assert {name for name, _, _ in offenders} == {"Y", "X"}


def test_split_axes_exempts_an_axis_the_grid_already_spans() -> None:
    """A single-chunk axis has no interior boundary, so nothing can split it.

    Load-bearing rather than a micro-optimisation: ``_axis_candidates`` clamps
    its last candidate to the extent, so a 375-row thumbnail legitimately plans
    a grid of 375 — which divides no power-of-two tile and would otherwise make
    every tile look unsafe against it.
    """
    thumb_grid = plan_write_grid(THUMB, DIMS, SPACING, np.uint16, DEFAULT_GEOMETRY)
    assert thumb_grid[3:] == (375, 504)  # clamped, not powers of two
    assert split_axes((1, 1, 1, 512, 512), thumb_grid, THUMB, DIMS) == []


# --------------------------------------------------------------------------
# 2. Deriving the tile back out of the grid
# --------------------------------------------------------------------------


def test_derived_tile_matches_the_write_grid() -> None:
    assert plan_reader_tile_size(
        [WSI], DIMS, [SPACING], np.uint16, DEFAULT_GEOMETRY
    ) == (512, 512)
    assert plan_reader_tile_size([WSI], DIMS, [SPACING], np.uint16, SHARDED) == (
        2048,
        2048,
    )


def test_derived_tile_is_the_minimum_across_scenes() -> None:
    """One reader, one tile size, many scenes — so the tile must fit them all.

    The minimum is safe in the direction that matters: every scene either
    matches it or merges whole blocks into a larger grid. Taking the dominant
    scene's grid instead is what a hand-tuned reader does, and it splits every
    smaller scene.
    """
    # A flat slide next to a deep isotropic stack: the world-cubic planner
    # spends the same byte budget on two lateral axes for one and three axes for
    # the other, so their grids genuinely disagree.
    stack = (1, 2, 256, 4096, 4096)
    stack_spacing = [1.0, 1.0, 1.0, 1.0, 1.0]
    flat_grid = plan_write_grid(WSI, DIMS, SPACING, np.uint16)
    stack_grid = plan_write_grid(stack, DIMS, stack_spacing, np.uint16)
    assert flat_grid[3:] == (512, 512)
    assert stack_grid[3:] == (64, 64)

    tile = plan_reader_tile_size(
        [WSI, stack], DIMS, [SPACING, stack_spacing], np.uint16
    )
    assert tile == (64, 64)

    # And the point of taking the minimum: neither scene splits.
    for shape, grid in ((WSI, flat_grid), (stack, stack_grid)):
        source = (1, 1, 1, tile[0], tile[1])
        assert split_axes(source, grid, shape, DIMS) == []


def test_a_thumbnail_does_not_drag_the_tile_down() -> None:
    """A scene whose grid spans its whole extent constrains nothing.

    Without the exemption the 375-row ``label`` thumbnail would pin the tile to
    375 and make the gigapixel scene straddle on every write — the failure this
    change exists to remove, reintroduced by the fix for it.
    """
    tile = plan_reader_tile_size(
        [WSI, THUMB], DIMS, [SPACING, SPACING], np.uint16, DEFAULT_GEOMETRY
    )
    assert tile == (512, 512)


def test_derived_tile_is_none_without_lateral_axes() -> None:
    assert plan_reader_tile_size([(1, 4)], ["T", "C"], [[1.0, 1.0]], np.uint16) is None


# --------------------------------------------------------------------------
# 3. The property the numbers in the module docstring are about
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tile", "grid", "ratio_ceiling"),
    [
        (512, 512, 1.0),  # exact: rechunk is a no-op
        (256, 512, 1.3),  # merge: a concat task per output block
        (1024, 512, 9.5),  # split: the pathology
    ],
)
def test_split_costs_an_order_of_magnitude_more_tasks_than_a_merge(
    tile: int, grid: int, ratio_ceiling: float
) -> None:
    """Pins the asymmetry the whole design rests on, at a size that runs fast.

    Ratios rather than absolute counts, since dask's graph construction is free
    to change: what must not change is that splitting is an order of magnitude
    worse than merging, because that is why the tile is derived from the grid
    rather than the other way round.
    """
    shape = (1, 1, 1, 8192, 8192)
    src = da.zeros(shape, dtype=np.uint16, chunks=(1, 1, 1, tile, tile))
    target = (1, 1, 1, grid, grid)
    out = src if src.chunksize == target else src.rechunk(target)
    ratio = len(out.__dask_graph__()) / src.npartitions
    assert ratio <= ratio_ceiling

    # And the predicate agrees with the measurement about which case is which.
    assert bool(split_axes((1, 1, 1, tile, tile), target, shape, DIMS)) is (ratio > 1.5)


# --------------------------------------------------------------------------
# 4. convert() wiring
# --------------------------------------------------------------------------


SCENE = (1, 1, 1, 2048, 2048)
SCENE_TILE = (512, 512)  # what the planner asks for, at 512 KiB and 0.5 um/px


class TiledFakeReader(FakeReader):
    """A ``FakeReader`` whose dask blocking follows ``tile_size``, as a real one's does.

    Without this the fake would report the same blocking however it was opened,
    and every assertion about the reopen would be about the kwarg rather than
    about the thing the kwarg exists to change.
    """

    def __init__(self, tile_size: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tile = tile_size

    @property
    def xarray_dask_data(self):
        import xarray as xr

        order = self.dims.order
        lateral = dict(zip("YX", self._tile, strict=True)) if self._tile else {}
        chunks = tuple(
            lateral.get(d, size if d in "YX" else 1)
            for d, size in zip(order, self.dims.shape, strict=True)
        )
        arr = np.zeros(self.dims.shape, dtype=self._dtype)
        return xr.DataArray(da.from_array(arr, chunks=chunks), dims=list(order))


@pytest.fixture
def recording_plugin(monkeypatch: pytest.MonkeyPatch):
    """Patch ``get_reader`` with a plugin recording every ``open()`` kwarg set.

    Named ``bioio`` because tile alignment is scoped to the default plugin —
    ``tile_size`` is its convention and no other plugin takes one.
    """

    def installer(name: str = "bioio") -> list[dict[str, Any]]:
        opens: list[dict[str, Any]] = []

        def _open(_path: Path, **kwargs: Any) -> TiledFakeReader:
            opens.append(dict(kwargs))
            return TiledFakeReader(
                tile_size=kwargs.get("tile_size"),
                scenes=["slide"],
                dims="TCZYX",
                shape=SCENE,
                pixel_sizes=FakePhysicalPixelSizes(Z=1.0, Y=0.5, X=0.5),
            )

        plugin = ReaderPlugin(
            name=name,
            match=lambda _p: 100,
            open=_open,
            distribution="bioio-fake",
            source="builtin",
        )

        def _get_reader(_path: Any, *, reader_kwargs: dict | None = None):
            return _open(Path("/tmp/x.tif"), **(reader_kwargs or {})), plugin, 100

        monkeypatch.setattr(api_module, "get_reader", _get_reader)
        return opens

    return installer


def test_convert_reopens_the_reader_with_the_derived_tile(
    tmp_path: Path, recording_plugin, recwarn
) -> None:
    opens = recording_plugin()
    convert(
        "/tmp/x.tif",
        tmp_path / "out",
        reader_kwargs={"dask_tiles": "true"},
        contrast_percentile=None,
        validate=False,
    )
    assert len(opens) == 2, "expected a metadata open then an aligned reopen"
    assert "tile_size" not in opens[0]
    assert opens[1]["tile_size"] == SCENE_TILE
    assert opens[1]["dask_tiles"] == "true", "other kwargs survive the reopen"
    # The reopen is the point, not the kwarg: the writer has nothing to split.
    assert [w for w in recwarn if issubclass(w.category, TileAlignmentWarning)] == []


def test_a_pinned_tile_size_is_respected(tmp_path: Path, recording_plugin) -> None:
    """The caller's choice wins; the writer still says so when it splits."""
    opens = recording_plugin()
    with pytest.warns(TileAlignmentWarning, match="do not nest"):
        convert(
            "/tmp/x.tif",
            tmp_path / "out",
            reader_kwargs={"dask_tiles": "true", "tile_size": (1024, 1024)},
            contrast_percentile=None,
            validate=False,
        )
    assert len(opens) == 1, "a pinned tile_size must not trigger a reopen"
    assert opens[0]["tile_size"] == (1024, 1024)


def test_an_untiled_reader_is_left_alone(tmp_path: Path, recording_plugin) -> None:
    """No ``dask_tiles``, no tiling — ``tile_size`` would mean nothing to pass."""
    opens = recording_plugin()
    convert("/tmp/x.tif", tmp_path / "out", contrast_percentile=None, validate=False)
    assert len(opens) == 1


def test_a_non_default_plugin_is_left_alone(tmp_path: Path, recording_plugin) -> None:
    """``tile_size`` is the default plugin's convention, not a universal one."""
    opens = recording_plugin(name="zarrmony-smartspim")
    convert(
        "/tmp/x.tif",
        tmp_path / "out",
        reader_kwargs={"dask_tiles": "true"},
        contrast_percentile=None,
        validate=False,
    )
    assert len(opens) == 1


def test_audit_records_the_derived_tile(tmp_path: Path, recording_plugin) -> None:
    """Two runs of the same file differ only in cost, so the store cannot show it."""
    recording_plugin()
    out = tmp_path / "out"
    result = convert(
        "/tmp/x.tif",
        out,
        reader_kwargs={"dask_tiles": "true"},
        contrast_percentile=None,
        validate=False,
    )
    assert result["stores"][0]["config"]["reader_tile_size"] == [512, 512]
    on_disk = json.loads((out / "slide.ome.zarr" / "zarr.json").read_text())
    recorded = on_disk["attributes"]["zarrmony"]["config"]["reader_tile_size"]
    assert recorded == [512, 512]


def test_audit_records_null_when_the_reader_was_left_alone(
    tmp_path: Path, recording_plugin
) -> None:
    recording_plugin()
    result = convert(
        "/tmp/x.tif", tmp_path / "out", contrast_percentile=None, validate=False
    )
    assert result["stores"][0]["config"]["reader_tile_size"] is None


def test_a_failed_reopen_warns_and_keeps_converting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alignment is an optimisation; a backend that rejects the kwarg is not fatal."""

    def _open(_path: Path, **kwargs: Any) -> TiledFakeReader:
        if "tile_size" in kwargs:
            raise TypeError("this backend takes no tile_size")
        return TiledFakeReader(
            scenes=["slide"],
            dims="TCZYX",
            shape=SCENE,
            pixel_sizes=FakePhysicalPixelSizes(Z=1.0, Y=0.5, X=0.5),
        )

    plugin = ReaderPlugin(
        name="bioio",
        match=lambda _p: 100,
        open=_open,
        distribution="bioio-fake",
        source="builtin",
    )
    monkeypatch.setattr(
        api_module,
        "get_reader",
        lambda _p, *, reader_kwargs=None: (
            _open(Path("/tmp/x.tif"), **(reader_kwargs or {})),
            plugin,
            100,
        ),
    )

    with pytest.warns(TileAlignmentWarning, match="could not reopen"):
        result = convert(
            "/tmp/x.tif",
            tmp_path / "out",
            reader_kwargs={"dask_tiles": "true"},
            contrast_percentile=None,
            validate=False,
        )
    assert result["stores"][0]["config"]["reader_tile_size"] is None
    assert len(result["stores"]) == 1


# --------------------------------------------------------------------------
# 5. The writer's own warning — the backstop for readers convert() cannot align
# --------------------------------------------------------------------------


def _convert_with_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tile: tuple[int, int]
) -> None:
    """Convert through a plugin ``convert()`` will not align, blocked at ``tile``."""
    reader = TiledFakeReader(
        tile_size=tile,
        scenes=["s"],
        dims="TCZYX",
        shape=SCENE,
        pixel_sizes=FakePhysicalPixelSizes(Z=1.0, Y=0.5, X=0.5),
    )
    plugin = ReaderPlugin(
        name="zarrmony-smartspim",
        match=lambda _p: 100,
        open=lambda _p, **_k: reader,
        distribution=None,
        source="builtin",
    )
    monkeypatch.setattr(
        api_module,
        "get_reader",
        lambda _p, *, reader_kwargs=None: (reader, plugin, 100),
    )
    convert("/tmp/x.tif", tmp_path / "out", contrast_percentile=None, validate=False)


def test_writer_warns_when_source_blocks_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the tile that would have worked, since guessing it is the hard part."""
    with pytest.warns(TileAlignmentWarning, match=r"tile_size=512,512"):
        _convert_with_blocking(tmp_path, monkeypatch, (1024, 1024))


def test_writer_is_quiet_when_blocks_nest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recwarn
) -> None:
    """Blocks smaller than the grid merge, which is the cheap direction."""
    _convert_with_blocking(tmp_path, monkeypatch, (256, 256))
    assert [w for w in recwarn if issubclass(w.category, TileAlignmentWarning)] == []
