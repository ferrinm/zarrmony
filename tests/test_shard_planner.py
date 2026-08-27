"""Tests for the ADR-0010 shard planner (issue #117).

A shard is a container holding many chunks in one storage object, with an index
that lets any single chunk be range-read back out (``CONTEXT.md``). It changes
how many objects exist, never how finely the array can be read — which is the
whole reason it is worth having, and the property most of these tests exist to
pin.

Four concerns, in order:

1. The policy fields — sharding is **off** unless asked for, and either
   spelling turns it on.
2. :func:`plan_shard_shape` / :func:`plan_level_shard_shapes` — the rule: the
   chunk rule one level up, over whole chunk multiples, T and C pinned, never
   longer than the chunks that exist.
3. That the plan reaches the writer, lands on disk as a ``sharding_indexed``
   codec with the chunk plan *inside* it, cuts the object count, and is
   recorded in the audit.
4. That the writer works in shards rather than chunks — the three places
   (issue #117) where reaching for ``.chunks`` would be correct but
   pathological.

The chunk planner these build on lives in ``test_chunk_planner.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import (
    DEFAULT_CHUNK_TARGET_BYTES,
    DEFAULT_SHARD_TARGET_BYTES,
    Geometry,
    plan_chunk_shape,
    plan_level_shard_shapes,
    plan_shard_shape,
)
from zarrmony.readers.plugin import ReaderPlugin

DIMS = "TCZYX"
SHARDED = Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES)


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


def _arrays(store: Path) -> dict[str, dict[str, Any]]:
    """Every array's on-disk grid, keyed by path within ``store``.

    ``chunk_grid`` on a sharded array is the *outer* grid — the object — and
    the read unit is buried in the ``sharding_indexed`` codec's own
    ``chunk_shape``. Reading both out here is the point: a test that only
    looked at ``chunk_grid`` would report a sharded store as having 2048²
    chunks and miss that it reads in 512² ones.
    """
    out: dict[str, dict[str, Any]] = {}
    for zj in sorted(store.rglob("zarr.json")):
        node = json.loads(zj.read_text())
        if node.get("node_type") != "array":
            continue
        outer = node["chunk_grid"]["configuration"]["chunk_shape"]
        codecs = [c["name"] for c in node["codecs"]]
        inner = outer
        shard = None
        if codecs[:1] == ["sharding_indexed"]:
            shard = outer
            inner = node["codecs"][0]["configuration"]["chunk_shape"]
        out[str(zj.parent.relative_to(store))] = {
            "shape": node["shape"],
            "read_unit": inner,
            "shard": shard,
            "codecs": codecs,
        }
    return out


def _objects(store: Path) -> int:
    """Files under ``store`` that hold pixels rather than metadata."""
    return sum(1 for p in store.rglob("*") if p.is_file() and p.name != "zarr.json")


# ---------- the policy fields ----------


def test_sharding_is_off_by_default() -> None:
    # The whole opt-in premise: a store written without asking for shards is
    # the store zarrmony wrote before #117 existed.
    g = Geometry()
    assert g.shard_target_bytes is None
    assert g.shard_shape is None
    assert g.sharding_enabled is False


@pytest.mark.parametrize(
    "policy",
    [
        Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES),
        Geometry(shard_shape=(1, 1, 128, 128, 128)),
        Geometry(shard_target_bytes=1 << 24, shard_shape=(1, 1, 128, 128, 128)),
    ],
)
def test_either_spelling_enables_sharding(policy: Geometry) -> None:
    # shard_shape alone is a complete instruction; it does not need a redundant
    # target beside it to take effect.
    assert policy.sharding_enabled is True


def test_shard_target_below_chunk_target_is_rejected() -> None:
    # A shard holds whole chunks, so one smaller than a chunk would plan out as
    # shard == chunk: the codec's cost and none of its benefit.
    with pytest.raises(ValueError, match="below chunk_target_bytes"):
        Geometry(shard_target_bytes=DEFAULT_CHUNK_TARGET_BYTES // 2)
    assert Geometry(shard_target_bytes=DEFAULT_CHUNK_TARGET_BYTES).sharding_enabled


def test_shard_target_is_not_compared_against_an_unused_chunk_target() -> None:
    # With an explicit chunk_shape the byte target is never consulted, so
    # comparing against it would reject a perfectly valid pair.
    g = Geometry(chunk_shape=(1, 1, 8, 8, 8), shard_target_bytes=4096)
    assert g.sharding_enabled


@pytest.mark.parametrize("bad", [(), (1, 1, 0, 64, 64), (1, -1, 64)])
def test_invalid_shard_shape_is_rejected(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shard_shape"):
        Geometry(shard_shape=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "8388608"])
def test_invalid_shard_target_bytes_is_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="shard_target_bytes"):
        Geometry(shard_target_bytes=bad)


def test_shard_shape_is_normalized_to_a_tuple() -> None:
    assert Geometry(shard_shape=[1, 1, 128, 128, 128]).shard_shape == (
        1,
        1,
        128,
        128,
        128,
    )


# ---------- the rule ----------


@pytest.mark.parametrize(
    ("shape", "spacings"),
    [
        # Near-isotropic volume — the ADR-0010 reference acquisition's shape.
        ((1, 3, 3627, 8835, 7452), (1.0, 1.0, 2.0, 1.8, 1.8)),
        # 10:1 confocal stack, where cubic-in-voxels and cubic-in-µm diverge.
        ((1, 2, 64, 2048, 2048), (1.0, 1.0, 5.0, 0.5, 0.5)),
        # Whole-slide 2D — the shape that opened issue #113.
        ((1, 4, 1, 140757, 171855), (1.0, 1.0, 1.0, 0.325, 0.325)),
        # A level small enough that the shard cannot grow at all.
        ((1, 2, 1, 300, 300), (1.0, 1.0, 1.0, 1.0, 1.0)),
    ],
)
def test_shard_contains_whole_chunks_and_fits_the_target(
    shape: tuple[int, ...], spacings: tuple[float, ...]
) -> None:
    chunk = plan_chunk_shape(shape, DIMS, spacings, np.uint16, SHARDED)
    shard = plan_shard_shape(chunk, shape, DIMS, spacings, np.uint16, SHARDED)

    assert all(
        s % c == 0 for s, c in zip(shard, chunk, strict=True)
    ), f"shard {shard} does not tile chunk {chunk}"
    assert math.prod(shard) * 2 <= DEFAULT_SHARD_TARGET_BYTES
    assert all(s >= c for s, c in zip(shard, chunk, strict=True))


def test_shard_pins_t_and_c_to_the_chunk() -> None:
    # A shard spanning channels would make one write object depend on two
    # channels' pixels, coupling writes the store otherwise keeps independent —
    # and buys nothing, since the index already yields one chunk at a time.
    shape = (4, 6, 64, 1024, 1024)
    spacings = (1.0, 1.0, 1.0, 1.0, 1.0)
    chunk = plan_chunk_shape(shape, DIMS, spacings, np.uint16, SHARDED)
    shard = plan_shard_shape(chunk, shape, DIMS, spacings, np.uint16, SHARDED)
    assert shard[:2] == (1, 1) == chunk[:2]


def test_shard_never_exceeds_the_chunks_that_exist() -> None:
    # A shard longer than the axis buys nothing: the trailing chunks are absent
    # rather than padded, so the object is the same size either way.
    shape = (1, 1, 1, 300, 300)
    spacings = (1.0, 1.0, 1.0, 1.0, 1.0)
    chunk = plan_chunk_shape(shape, DIMS, spacings, np.uint16, SHARDED)
    shard = plan_shard_shape(chunk, shape, DIMS, spacings, np.uint16, SHARDED)
    assert chunk == (1, 1, 1, 300, 300)
    assert shard == chunk


def test_isotropic_defaults_land_on_sixteen_chunks_per_object() -> None:
    # The pinned answer at both default targets. 8 MiB of uint16 is 4.19 M
    # voxels — short of a 128³ cube by 1.6 M and nowhere near 256³ — so filling
    # the target means doubling one axis, and the positional tie-break puts the
    # long one on X where it is most contiguous. Change this deliberately.
    shape = (1, 2, 512, 2048, 2048)
    spacings = (1.0, 1.0, 1.0, 1.0, 1.0)
    chunk = plan_chunk_shape(shape, DIMS, spacings, np.uint16, SHARDED)
    shard = plan_shard_shape(chunk, shape, DIMS, spacings, np.uint16, SHARDED)
    assert chunk == (1, 1, 64, 64, 64)
    assert shard == (1, 1, 128, 128, 256)
    assert math.prod(shard) // math.prod(chunk) == 16


def test_whole_slide_shard_cuts_objects_sixteenfold() -> None:
    # The #113 arithmetic: the chunk stays the 512 KiB read unit a viewer
    # budgets by, while the object count falls to what the 8 MiB conversion
    # achieved.
    shape = (1, 4, 1, 140757, 171855)
    spacings = (1.0, 1.0, 1.0, 0.325, 0.325)
    chunk = plan_chunk_shape(shape, DIMS, spacings, np.uint16, SHARDED)
    shard = plan_shard_shape(chunk, shape, DIMS, spacings, np.uint16, SHARDED)
    assert chunk == (1, 1, 1, 512, 512)
    assert shard == (1, 1, 1, 2048, 2048)

    def grid(unit: tuple[int, ...]) -> int:
        return math.ceil(shape[3] / unit[3]) * math.ceil(shape[4] / unit[4]) * shape[1]

    assert grid(chunk) == 369_600
    assert grid(shard) == 23_184


def test_explicit_shard_shape_bypasses_the_planner() -> None:
    g = Geometry(shard_shape=(1, 1, 128, 128, 128))
    shape = (1, 2, 512, 2048, 2048)
    spacings = (1.0, 1.0, 1.0, 1.0, 1.0)
    chunk = plan_chunk_shape(shape, DIMS, spacings, np.uint16, g)
    assert plan_shard_shape(chunk, shape, DIMS, spacings, np.uint16, g) == (
        1,
        1,
        128,
        128,
        128,
    )


def test_explicit_shard_shape_that_does_not_tile_the_chunk_is_rejected() -> None:
    # Not a shard zarr can write. Caught by name and axis here, rather than as
    # a codec-layer error once pixels are already moving.
    g = Geometry(chunk_shape=(1, 1, 64, 64, 64), shard_shape=(1, 1, 128, 100, 128))
    with pytest.raises(ValueError, match=r"axis 3 \(Y\): shard 100 is not a multiple"):
        plan_shard_shape(
            (1, 1, 64, 64, 64),
            (1, 2, 512, 2048, 2048),
            DIMS,
            (1.0, 1.0, 1.0, 1.0, 1.0),
            np.uint16,
            g,
        )


def test_explicit_shard_shape_with_the_wrong_rank_is_rejected() -> None:
    g = Geometry(shard_shape=(128, 128, 128))
    with pytest.raises(ValueError, match="has 3 axes but the level has 5"):
        plan_shard_shape(
            (1, 1, 64, 64, 64),
            (1, 2, 512, 2048, 2048),
            DIMS,
            (1.0, 1.0, 1.0, 1.0, 1.0),
            np.uint16,
            g,
        )


def test_plan_level_shard_shapes_is_none_when_sharding_is_off() -> None:
    # None rather than a list of sentinels, so it goes straight to the writer's
    # shard_shape= and an unsharded conversion is unchanged.
    assert (
        plan_level_shard_shapes(
            [(1, 1, 64, 64, 64)],
            [(1, 2, 512, 512, 512)],
            DIMS,
            (1.0, 1.0, 1.0, 1.0, 1.0),
            np.uint16,
            Geometry(),
        )
        is None
    )


def test_plan_shard_shape_refuses_to_guess_when_sharding_is_off() -> None:
    with pytest.raises(ValueError, match="sharding disabled"):
        plan_shard_shape(
            (1, 1, 64, 64, 64),
            (1, 2, 512, 512, 512),
            DIMS,
            (1.0, 1.0, 1.0, 1.0, 1.0),
            np.uint16,
            Geometry(),
        )


def test_each_level_is_sharded_against_its_own_spacing() -> None:
    """A shard is re-planned per level, not planned once and replicated.

    On a 8:1 stack that halves Y and X but not Z, every level is *less*
    anisotropic than the one below it, so the shard that is closest to cubic
    moves: level 0 spends its budget laterally where the voxels are small,
    level 1 lands on a perfect 256 µm cube, and by level 2 Z is the only axis
    with room left. A plan replicated from level 0 would show as three
    identical shapes.
    """
    levels = [
        (1, 1, 64, 2048, 2048),
        (1, 1, 64, 1024, 1024),
        (1, 1, 64, 512, 512),
    ]
    base_spacings = (1.0, 1.0, 4.0, 0.5, 0.5)
    chunks = [
        plan_chunk_shape(
            lvl,
            DIMS,
            (1.0, 1.0, 4.0, 0.5 * 2**i, 0.5 * 2**i),
            np.uint16,
            SHARDED,
        )
        for i, lvl in enumerate(levels)
    ]
    shards = plan_level_shard_shapes(
        chunks, levels, DIMS, base_spacings, np.uint16, SHARDED
    )
    assert shards is not None and len(shards) == 3

    for chunk, shard in zip(chunks, shards, strict=True):
        assert all(s % c == 0 for s, c in zip(shard, chunk, strict=True))
        assert math.prod(shard) * 2 <= DEFAULT_SHARD_TARGET_BYTES

    assert shards[0] != shards[1], "the plan was replicated, not re-derived"
    # Level 1's spacing is 4.0 / 1.0 / 1.0 µm, on which the rule can hit an
    # exact cube inside the budget — and does.
    assert shards[1] == (1, 1, 64, 256, 256)
    assert len({length * 1.0 for length in shards[1][3:]}) == 1


def test_plan_level_shard_shapes_rejects_a_chunk_list_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="one chunk shape per level"):
        plan_level_shard_shapes(
            [(1, 1, 64, 64, 64)],
            [(1, 1, 512, 512, 512), (1, 1, 256, 256, 256)],
            DIMS,
            (1.0, 1.0, 1.0, 1.0, 1.0),
            np.uint16,
            SHARDED,
        )


# ---------- end to end ----------


#: Level 0 of the test scene. 1024² uint16 lands a 512² chunk at the default
#: target, so a 1024² shard holds a proper 2×2 of them and the two grids are
#: genuinely distinct — a scene one level smaller would plan them equal and
#: prove nothing.
E2E_SHAPE = (1, 2, 1, 1024, 1024)
E2E_SHARD = (1, 1, 1, 1024, 1024)
E2E_PIXELS = (
    (np.arange(math.prod(E2E_SHAPE)) % 65_521).astype(np.uint16).reshape(E2E_SHAPE)
)


def _wsi_reader() -> FakeReader:
    """A 2D scene small enough to convert in a test, sharded 4 chunks to an object."""
    return FakeReader(
        scenes=["slide"],
        dims="TCZYX",
        shape=E2E_SHAPE,
        pixel_sizes=FakePhysicalPixelSizes(Z=1.0, Y=0.325, X=0.325),
        channel_names=["DAPI", "GFP"],
        data=E2E_PIXELS,
    )


def test_default_conversion_writes_no_shards(tmp_path: Path, patched_reader) -> None:
    patched_reader(_wsi_reader())
    out = tmp_path / "out"
    result = convert("/tmp/x.czi", out, contrast_percentile=None, validate=False)

    arrays = _arrays(out / "slide.ome.zarr")
    assert all(a["shard"] is None for a in arrays.values())
    assert all(a["codecs"][0] == "bytes" for a in arrays.values())
    assert result["stores"][0]["per_scene"][0]["shard_shapes"] is None


def test_sharded_conversion_keeps_the_read_unit_and_cuts_objects(
    tmp_path: Path, patched_reader
) -> None:
    """The claim the whole feature rests on, checked on one store against another.

    Same pixels, same read unit, same level shapes; fewer objects. If any of
    those four move together, the shard has stopped being a pure storage
    concern and become a geometry change.
    """
    patched_reader(_wsi_reader())
    plain = tmp_path / "plain"
    convert("/tmp/x.czi", plain, contrast_percentile=None, validate=False)

    patched_reader(_wsi_reader())
    sharded = tmp_path / "sharded"
    result = convert(
        "/tmp/x.czi",
        sharded,
        geometry=Geometry(shard_shape=E2E_SHARD),
        contrast_percentile=None,
        validate=False,
    )

    plain_arrays = _arrays(plain / "slide.ome.zarr")
    shard_arrays = _arrays(sharded / "slide.ome.zarr")

    # The read unit is untouched — that is the difference between a shard and a
    # bigger chunk.
    assert [a["read_unit"] for a in shard_arrays.values()] == [
        a["read_unit"] for a in plain_arrays.values()
    ]
    assert [a["shape"] for a in shard_arrays.values()] == [
        a["shape"] for a in plain_arrays.values()
    ]
    assert all(a["codecs"] == ["sharding_indexed"] for a in shard_arrays.values())
    assert all(a["shard"] == list(E2E_SHARD) for a in shard_arrays.values())

    assert _objects(sharded / "slide.ome.zarr") < _objects(plain / "slide.ome.zarr")

    # Pixels survive the round trip through the codec, at every level.
    for level in ("0", "1"):
        got = da.from_zarr(str(sharded / "slide.ome.zarr"), component=level).compute()
        want = da.from_zarr(str(plain / "slide.ome.zarr"), component=level).compute()
        np.testing.assert_array_equal(got, want)

    record = result["stores"][0]["per_scene"][0]
    assert record["shard_shapes"] == [list(E2E_SHARD)] * len(record["level_shapes"])
    assert record["chunk_shapes"][0] == [1, 1, 1, 512, 512] != record["shard_shapes"][0]


def test_a_subset_straddling_both_grids_reads_correctly(
    tmp_path: Path, patched_reader
) -> None:
    # The consumer case that matters most in practice: someone slicing a region
    # into numpy for analysis, on no grid in particular.
    patched_reader(_wsi_reader())
    out = tmp_path / "out"
    convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(shard_shape=E2E_SHARD),
        contrast_percentile=None,
        validate=False,
    )
    import zarr

    level0 = zarr.open_group(str(out / "slide.ome.zarr"), mode="r")["0"]
    # Aligned to neither the 512² chunk grid nor the 1024² shard grid.
    got = level0[0, 1, 0, 300:800, 90:970]
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, E2E_PIXELS[0, 1, 0, 300:800, 90:970])


def test_coarse_levels_may_hold_one_chunk_per_shard_and_stay_sharded(
    tmp_path: Path, patched_reader
) -> None:
    """Sharding is uniform across levels even where it buys nothing.

    A coarse level is often a single chunk already, so the planner has no
    multiple to reach for and the shard comes out equal to the chunk. Leaving
    that level unsharded would be marginally smaller — a shard index is ~16
    bytes — but it would make "can my consumer read this store" a per-level
    question, and a consumer that opened level 2 fine would still fail on
    level 0. One answer per store is worth the index.
    """
    patched_reader(_wsi_reader())
    out = tmp_path / "out"
    convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES),
        contrast_percentile=None,
        validate=False,
    )
    arrays = _arrays(out / "slide.ome.zarr")
    assert all(a["codecs"] == ["sharding_indexed"] for a in arrays.values())
    assert arrays["0"]["shard"] == [1, 1, 1, 1024, 1024] != arrays["0"]["read_unit"]
    degenerate = [k for k, a in arrays.items() if a["shard"] == a["read_unit"]]
    assert degenerate, "expected at least one level with nothing left to pack"


def test_geometry_audit_records_the_shard_policy(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(_wsi_reader())
    out = tmp_path / "out"
    convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES),
        contrast_percentile=None,
        validate=False,
    )
    attrs = json.loads((out / "slide.ome.zarr" / "zarr.json").read_text())["attributes"]
    recorded = attrs["zarrmony"]["config"]["geometry"]
    assert recorded["shard_target_bytes"] == DEFAULT_SHARD_TARGET_BYTES
    assert recorded["shard_shape"] is None


# ---------- the writer works in shards, not chunks ----------


def test_write_grid_prefers_the_shard(tmp_path: Path) -> None:
    from zarrmony.writers.scene import _write_grid

    @dataclasses.dataclass
    class FakeArray:
        chunks: tuple[int, ...]
        shards: tuple[int, ...] | None

    assert _write_grid(FakeArray((1, 1, 64, 64, 64), None)) == (1, 1, 64, 64, 64)
    assert _write_grid(FakeArray((1, 1, 64, 64, 64), (1, 1, 256, 256, 256))) == (
        1,
        1,
        256,
        256,
        256,
    )


def test_pyramid_writes_are_shard_aligned_not_chunk_aligned(
    tmp_path: Path, patched_reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ``da.store`` block covers a whole shard.

    Chunk-aligned blocks into a sharded array are *correct* — which is why this
    needs asserting rather than trusting a passing conversion. They are also a
    read-modify-write of the whole object per block: 16× write amplification at
    the default targets, and the reason issue #117 lists this as a fix rather
    than a nicety.
    """
    from zarrmony.writers import scene as scene_module

    seen: list[tuple[tuple[int, ...], Any, Any]] = []
    real_store = scene_module.da.store

    def spy(src, target, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((src.chunksize, target.shards, target.shape))
        return real_store(src, target, **kwargs)

    monkeypatch.setattr(scene_module.da, "store", spy)

    patched_reader(_wsi_reader())
    convert(
        "/tmp/x.czi",
        tmp_path / "out",
        geometry=Geometry(shard_shape=E2E_SHARD),
        contrast_percentile=None,
        validate=False,
    )

    assert seen, "no level was written through da.store"
    for blocksize, shards, shape in seen:
        assert shards is not None
        # A shard wider than a coarse level clamps to it — which is still one
        # whole object per block, the property being asserted.
        assert blocksize == tuple(min(s, n) for s, n in zip(shards, shape, strict=True))
        assert blocksize != tuple(min(64, n) for n in shape)


def test_read_level_chunks_on_the_shard_grid(tmp_path: Path) -> None:
    """One task per stored object, not per chunk inside it.

    ``da.from_zarr`` left to itself adopts the inner chunk, which would
    multiply the pyramid's own read-back task count by the chunks-per-shard
    ratio — reintroducing #111's symptom in the one place the writer
    deliberately spends a read.
    """
    from zarrmony.writers.scene import ZarrmonyWriter

    writer = ZarrmonyWriter(
        store=str(tmp_path / "s.zarr"),
        level_shapes=[[1, 1, 1, 512, 512]],
        dtype=np.uint16,
        zarr_format=3,
        chunk_shape=[[1, 1, 1, 64, 64]],
        shard_shape=[[1, 1, 1, 256, 256]],
    )
    writer.initialize()
    level = writer.read_level(0)
    assert writer.datasets[0].chunks == (1, 1, 1, 64, 64)
    assert level.chunksize == (1, 1, 1, 256, 256)
    assert level.npartitions == 4
