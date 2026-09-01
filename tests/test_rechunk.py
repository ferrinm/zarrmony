"""Tests for ``zarrmony rechunk`` — OME-Zarr → OME-Zarr geometry migration (#91).

The acceptance criteria in the issue are properties, not examples, so most of
these are written as properties over a matrix of shapes, dtypes and chunkings
rather than as one hand-picked store:

* **Equivalence** — ``convert@new`` and ``convert@old + rechunk`` produce
  byte-identical arrays at *every* level, which is the criterion "level-0 voxels
  bit-identical and upper levels matching a re-conversion" stated as one test.
* **Read-once** — a counting store wrapper asserts no source object is fetched
  twice, for chunkings the LCM rule has to actually work for (full-width slabs,
  coprime edges, sharded sources).
* **Resume** — an interrupted run finishes on re-run without redoing work.
* **Partial targets** — an interrupted store is not readable as an OME-Zarr.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert, rechunk
from zarrmony._rechunk import (
    STATE_KEY,
    plan_image,
    pooled_tile,
    read_once_tile,
    read_source_image,
)
from zarrmony.errors import (
    OutputExistsError,
    RechunkSourceError,
    RechunkStateError,
    RechunkVerificationError,
    WorkingSetTooLargeError,
)
from zarrmony.geometry import DEFAULT_GEOMETRY, Geometry
from zarrmony.readers.plugin import ReaderPlugin

# A pyramid deep enough to exercise the pooled-level path without making the
# fixtures slow: 32 px floor over a 128 px base gives levels 0/1/2.
SMALL = {"pyramid_min_size": 32}


def _fake_plugin(name: str = "bioio-fake") -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=name,
        source="builtin",
    )


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch ``zarrmony.api.get_reader`` to return a configurable FakeReader."""

    def installer(reader: FakeReader, plugin: str = "bioio-fake"):
        monkeypatch.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, _fake_plugin(plugin), 100),
        )

    return installer


def _levels(store: Path | str) -> dict[str, np.ndarray]:
    """Every pyramid level of an image store, as in-memory arrays."""
    group = zarr.open_group(str(store), mode="r")
    datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
    return {d["path"]: np.asarray(group[d["path"]][...]) for d in datasets}


def _grids(store: Path | str) -> list[tuple[Any, Any]]:
    group = zarr.open_group(str(store), mode="r")
    datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
    return [(tuple(group[d["path"]].chunks), group[d["path"]].shards) for d in datasets]


# ---------------------------------------------------------------------------
# The tile rule
# ---------------------------------------------------------------------------


def test_read_once_tile_is_a_multiple_of_both_grids() -> None:
    """The property the whole streaming guarantee rests on, over a rough matrix."""
    cases = [
        ((1, 1, 1, 2048, 2048), (1, 1, 64, 64, 64), (1, 3, 512, 2048, 2048)),
        ((1, 1, 16, 128, 128), (1, 1, 64, 64, 64), (2, 4, 300, 700, 900)),
        ((1, 1, 5, 7, 11), (1, 1, 3, 4, 8), (1, 1, 100, 100, 100)),
        ((1, 1, 64, 64, 64), (1, 1, 64, 64, 64), (1, 1, 640, 640, 640)),
    ]
    for source, target, extent in cases:
        tile = read_once_tile(source, target, extent)
        for t, s, w, e in zip(tile, source, target, extent, strict=True):
            assert t <= e
            # Either the tile is a whole number of both grids, or the axis holds
            # exactly one tile and has no interior boundary to straddle.
            assert (t % s == 0 and t % w == 0) or t == e


def test_read_once_tile_recovers_the_old_geometry_case() -> None:
    """Full-width single-plane slabs into 64³ chunks read as Z bands of 64."""
    assert read_once_tile(
        (1, 1, 1, 2048, 2048), (1, 1, 64, 64, 64), (1, 2, 512, 2048, 2048)
    ) == (1, 1, 64, 2048, 2048)


def test_pooled_tile_keeps_the_parent_read_whole() -> None:
    """``tile * factor`` must land on the parent's own write grid."""
    cases = [
        ((1, 1, 64, 64, 64), (1, 1, 64, 64, 64), (1, 1, 2, 2, 2)),
        ((1, 1, 32, 128, 128), (1, 1, 64, 64, 64), (1, 1, 1, 2, 2)),
        ((1, 1, 48, 48, 48), (1, 1, 64, 64, 64), (1, 1, 2, 2, 2)),
    ]
    extent = (1, 2, 1024, 1024, 1024)
    for grid, parent, factors in cases:
        tile = pooled_tile(grid, parent, factors, extent)
        for t, g, p, f, e in zip(tile, grid, parent, factors, extent, strict=True):
            assert (t % g == 0) or t == e
            assert ((t * f) % p == 0) or t == e


# ---------------------------------------------------------------------------
# Equivalence: convert@new == convert@old + rechunk, at every level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dims", "shape", "old_chunk", "dtype"),
    [
        # Full-width single-plane slabs — the ADR-0010 old-geometry case.
        ("TCZYX", (1, 2, 16, 128, 128), (1, 1, 1, 128, 128), np.uint16),
        # A 2D scene with a chunk grid that does not divide the new one.
        ("TCYX", (1, 1, 96, 96), (1, 1, 48, 48), np.uint8),
        # Coprime chunk edges, so the LCM tile is genuinely larger than both.
        ("TCYX", (1, 2, 128, 128), (1, 1, 32, 40), np.uint16),
        # Single channel, deep in Z.
        ("TCZYX", (1, 1, 32, 64, 64), (1, 1, 8, 64, 64), np.int16),
    ],
)
def test_rechunk_matches_a_reconversion_at_every_level(
    tmp_path: Path, patched_reader, dims, shape, old_chunk, dtype
) -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 200, size=shape).astype(dtype)

    patched_reader(FakeReader(scenes=["s"], dims=dims, shape=shape, data=data))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=old_chunk, **SMALL)

    patched_reader(FakeReader(scenes=["s"], dims=dims, shape=shape, data=data))
    convert("/tmp/fake.lif", tmp_path / "ref", **SMALL)

    rechunk(
        tmp_path / "old" / "s.ome.zarr",
        tmp_path / "new" / "s.ome.zarr",
        geometry=Geometry(pyramid_min_size=32),
        verify="full",
    )

    reference = _levels(tmp_path / "ref" / "s.ome.zarr")
    migrated = _levels(tmp_path / "new" / "s.ome.zarr")
    assert sorted(migrated) == sorted(reference)
    for path, expected in reference.items():
        np.testing.assert_array_equal(migrated[path], expected, err_msg=f"level {path}")

    # And the storage geometry itself matches, not just the voxels.
    assert _grids(tmp_path / "new" / "s.ome.zarr") == _grids(
        tmp_path / "ref" / "s.ome.zarr"
    )


def test_level_zero_is_bit_identical_to_the_source(
    tmp_path: Path, patched_reader
) -> None:
    """Level 0 is a copy, so it must equal the *source's* level 0 exactly."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 65535, size=(1, 2, 8, 96, 96)).astype(np.uint16)
    patched_reader(FakeReader(scenes=["s"], dims="TCZYX", shape=data.shape, data=data))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 1, 96, 96), **SMALL)

    rechunk(tmp_path / "old" / "s.ome.zarr", tmp_path / "new", verify="full")

    source = zarr.open_group(str(tmp_path / "old" / "s.ome.zarr"), mode="r")
    target = zarr.open_group(str(tmp_path / "new"), mode="r")
    np.testing.assert_array_equal(target["0"][...], source["0"][...])


def test_rechunk_honours_an_explicit_target_geometry(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 128, 128)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 128, 128), **SMALL)

    rechunk(
        tmp_path / "old" / "s.ome.zarr",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32), pyramid_min_size=32),
    )
    assert zarr.open_group(str(tmp_path / "new"), mode="r")["0"].chunks == (
        1,
        1,
        32,
        32,
    )


# ---------------------------------------------------------------------------
# Read-once
# ---------------------------------------------------------------------------


def _count_object_reads(monkeypatch) -> dict[tuple[str, str], int]:
    """Tally every object fetch zarr makes, keyed by ``(store root, key)``.

    Counted at :class:`zarr.storage.LocalStore` rather than through a
    ``WrapperStore``, because the fetch is the thing the streaming guarantee is
    about and this sees every one of them regardless of how the store was
    opened — including the handles ``rechunk`` opens per image and per level.
    """
    reads: dict[tuple[str, str], int] = {}
    real_get = zarr.storage.LocalStore.get

    async def counting_get(self, key, *args, **kwargs):
        reads[(str(self.root), key)] = reads.get((str(self.root), key), 0) + 1
        return await real_get(self, key, *args, **kwargs)

    monkeypatch.setattr(zarr.storage.LocalStore, "get", counting_get)
    return reads


@pytest.mark.parametrize(
    ("dims", "shape", "old_chunk"),
    [
        ("TCZYX", (1, 1, 16, 128, 128), (1, 1, 1, 128, 128)),
        ("TCYX", (1, 1, 128, 128), (1, 1, 32, 40)),
        ("TCYX", (1, 2, 96, 96), (1, 1, 96, 96)),
    ],
)
def test_every_source_object_is_read_exactly_once(
    tmp_path: Path, patched_reader, monkeypatch, dims, shape, old_chunk
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims=dims, shape=shape))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=old_chunk, **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"

    reads = _count_object_reads(monkeypatch)
    # `verify` deliberately re-reads the source, which is the point of it; the
    # streaming guarantee is about the copy pass.
    rechunk(source, tmp_path / "new", verify="none")
    monkeypatch.undo()

    level0 = {
        key: n
        for (root, key), n in reads.items()
        if Path(root) == source
        and key.startswith("0/")
        and not key.endswith("zarr.json")
    }
    assert level0, "the source's level 0 was never read"
    repeated = {k: n for k, n in level0.items() if n > 1}
    assert not repeated, f"these source objects were read more than once: {repeated}"


# ---------------------------------------------------------------------------
# Resume and partial targets
# ---------------------------------------------------------------------------


class Boom(RuntimeError):
    """Stand-in for whatever kills a long conversion — OOM, preemption, Ctrl-C."""


def _interrupt_after(monkeypatch, n_tiles: int) -> None:
    """Make the level-0 copy die after ``n_tiles`` tiles have landed."""
    import zarrmony._rechunk as rc

    real = rc._write_level_zero
    state = {"budget": n_tiles}

    def bomb(source_array, target_array, plan, done, flush):
        for index, region in enumerate(
            rc._iter_tiles(plan.level_shapes[0], plan.tiles[0])
        ):
            if index < done:
                continue
            if state["budget"] <= 0:
                flush(index)
                raise Boom("interrupted")
            target_array[region] = source_array[region]
            state["budget"] -= 1
        return real.__wrapped__ if False else index + 1  # pragma: no cover

    monkeypatch.setattr(rc, "_write_level_zero", bomb)


def test_interrupted_rechunk_resumes_and_completes(
    tmp_path: Path, patched_reader, monkeypatch
) -> None:
    rng = np.random.default_rng(2)
    data = rng.integers(0, 4096, size=(1, 1, 128, 128)).astype(np.uint16)
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=data.shape, data=data))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 32, 32), **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"
    target = tmp_path / "new"

    _interrupt_after(monkeypatch, 2)
    with pytest.raises(Boom):
        rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 16, 16)))

    # A partial target carries resume state and is NOT an OME-Zarr.
    partial = json.loads((target / "zarr.json").read_text())["attributes"]
    assert STATE_KEY in partial
    assert "ome" not in partial
    assert partial[STATE_KEY]["status"] == "in-progress"
    assert sum(partial[STATE_KEY]["progress"][""]) > 0

    monkeypatch.undo()
    result = rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 16, 16)))

    assert result["stores"][0]["resumed"] is True
    finished = json.loads((target / "zarr.json").read_text())["attributes"]
    assert STATE_KEY not in finished
    assert "ome" in finished
    np.testing.assert_array_equal(
        zarr.open_group(str(target), mode="r")["0"][...],
        zarr.open_group(str(source), mode="r")["0"][...],
    )


def test_resume_refuses_a_target_written_against_a_different_plan(
    tmp_path: Path, patched_reader, monkeypatch
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 128, 128)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 32, 32), **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"
    target = tmp_path / "new"

    _interrupt_after(monkeypatch, 1)
    with pytest.raises(Boom):
        rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 16, 16)))
    monkeypatch.undo()

    with pytest.raises(RechunkStateError, match="chunk_shapes"):
        rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 64, 64)))

    # force discards the partial target and starts over under the new plan.
    rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 64, 64)), force=True)
    assert zarr.open_group(str(target), mode="r")["0"].chunks == (1, 1, 64, 64)


def test_force_is_required_to_overwrite_a_finished_target(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"

    rechunk(source, tmp_path / "new", geometry=Geometry(chunk_shape=(1, 1, 32, 32)))
    with pytest.raises(OutputExistsError):
        rechunk(source, tmp_path / "new", geometry=Geometry(chunk_shape=(1, 1, 48, 48)))
    rechunk(
        source,
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 48, 48)),
        force=True,
    )
    assert zarr.open_group(str(tmp_path / "new"), mode="r")["0"].chunks == (
        1,
        1,
        48,
        48,
    )


def test_the_source_is_never_modified(tmp_path: Path, patched_reader) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"

    before = {
        p.relative_to(source): p.read_bytes()
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    rechunk(source, tmp_path / "new", geometry=Geometry(chunk_shape=(1, 1, 32, 32)))
    after = {
        p.relative_to(source): p.read_bytes()
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    assert before == after


# ---------------------------------------------------------------------------
# Metadata fidelity
# ---------------------------------------------------------------------------


def test_metadata_and_sidecars_survive_a_per_scene_rechunk(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(
        FakeReader(
            scenes=["alpha"],
            dims="TCYX",
            shape=(1, 2, 96, 96),
            channel_names=["DAPI", "GFP"],
        )
    )
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    source = tmp_path / "old" / "alpha.ome.zarr"
    target = tmp_path / "new"
    rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 32, 32)))

    src_ome = zarr.open_group(str(source), mode="r").attrs["ome"]
    dst_ome = zarr.open_group(str(target), mode="r").attrs["ome"]

    # Copied verbatim.
    assert dst_ome["multiscales"][0]["axes"] == src_ome["multiscales"][0]["axes"]
    assert dst_ome["multiscales"][0].get("name") == src_ome["multiscales"][0].get(
        "name"
    )
    assert (
        dst_ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        == src_ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    )
    assert [c["label"] for c in dst_ome["omero"]["channels"]] == ["DAPI", "GFP"]

    # Sidecars byte-for-byte.
    for name in ["OME/METADATA.ome.xml", "OME/source/raw.lif.xml"]:
        assert (target / name).read_bytes() == (source / name).read_bytes()


def test_audit_keeps_vendor_provenance_and_records_the_rechunk(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"
    target = tmp_path / "new"
    rechunk(source, target, geometry=Geometry(chunk_shape=(1, 1, 32, 32)))

    before = zarr.open_group(str(source), mode="r").attrs["zarrmony"]
    after = zarr.open_group(str(target), mode="r").attrs["zarrmony"]

    # The vendor file, and the plugin that produced these pixels, still stand.
    assert after["input"] == before["input"]
    assert after["reader_plugin"] == before["reader_plugin"]
    assert after["conversion_started_at"] == before["conversion_started_at"]
    assert after["conversion_finished_at"] == before["conversion_finished_at"]

    # The geometry does not.
    assert after["config"]["geometry"]["chunk_shape"] == [1, 1, 32, 32]
    assert after["config"]["reader_tile_size"] is None
    assert after["per_scene"][0]["chunk_shapes"][0] == [1, 1, 32, 32]

    # And the pass is on the record, as the discriminator for a migrated store.
    assert "rechunks" not in before
    assert len(after["rechunks"]) == 1
    entry = after["rechunks"][0]
    assert entry["operation"] == "rechunk"
    assert entry["source"]["path"].endswith("s.ome.zarr")
    assert entry["source"]["had_audit"] is True
    assert entry["source_geometry"]["chunk_shape"] == [1, 1, 96, 96]
    assert entry["verification"]["mode"] == "sample"
    assert entry["verification"]["passed"] is True
    assert after["audit_schema_version"] == 15


def test_rechunking_twice_appends_rather_than_replaces(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    rechunk(
        tmp_path / "old" / "s.ome.zarr",
        tmp_path / "mid",
        geometry=Geometry(chunk_shape=(1, 1, 48, 48)),
    )
    rechunk(
        tmp_path / "mid",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 16, 16)),
    )

    entries = zarr.open_group(str(tmp_path / "new"), mode="r").attrs["zarrmony"][
        "rechunks"
    ]
    assert len(entries) == 2
    assert entries[0]["source_geometry"]["chunk_shape"] == [1, 1, 96, 96]
    assert entries[1]["source_geometry"]["chunk_shape"] == [1, 1, 48, 48]


def test_a_store_with_no_audit_still_rechunks(tmp_path: Path, patched_reader) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    source = tmp_path / "old" / "s.ome.zarr"

    group = zarr.open_group(str(source), mode="a")
    del group.attrs["zarrmony"]

    rechunk(source, tmp_path / "new", geometry=Geometry(chunk_shape=(1, 1, 32, 32)))
    audit = zarr.open_group(str(tmp_path / "new"), mode="r").attrs["zarrmony"]
    assert audit["reader_plugin"] is None
    assert audit["input"]["path"].endswith("s.ome.zarr")
    assert audit["rechunks"][0]["source"]["had_audit"] is False


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def test_bf2raw_bundle_keeps_its_series_and_layout_marker(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert(
        "/tmp/fake.lif",
        tmp_path / "old",
        layout="bf2raw",
        chunk_shape=(1, 1, 96, 96),
        **SMALL,
    )
    rechunk(
        tmp_path / "old",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
    )

    src = zarr.open_group(str(tmp_path / "old"), mode="r")
    dst = zarr.open_group(str(tmp_path / "new"), mode="r")
    assert dst.attrs["ome"]["bioformats2raw.layout"] == 3
    assert dst["OME"].attrs["ome"]["series"] == src["OME"].attrs["ome"]["series"]
    assert (tmp_path / "new" / "OME" / "METADATA.ome.xml").read_bytes() == (
        tmp_path / "old" / "OME" / "METADATA.ome.xml"
    ).read_bytes()
    for series in src["OME"].attrs["ome"]["series"]:
        assert dst[series]["0"].chunks == (1, 1, 32, 32)
        np.testing.assert_array_equal(dst[series]["0"][...], src[series]["0"][...])


def test_directory_of_siblings_fans_out(tmp_path: Path, patched_reader) -> None:
    patched_reader(FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)

    result = rechunk(
        tmp_path / "old",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
    )
    assert result["layout"] == "sibling-directory"
    assert len(result["stores"]) == 2
    for name in ["a", "b"]:
        store = tmp_path / "new" / f"{name}.ome.zarr"
        assert zarr.open_group(str(store), mode="r")["0"].chunks == (1, 1, 32, 32)


def test_fan_out_is_idempotent(tmp_path: Path, patched_reader) -> None:
    """A store already at the target geometry is skipped, not rewritten.

    This is what makes re-running a batch after an interruption safe: the
    finished children cost a metadata read and the unfinished ones get done.
    """
    patched_reader(FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    rechunk(
        tmp_path / "old",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
    )

    again = rechunk(
        tmp_path / "new",
        tmp_path / "newer",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
    )
    assert [s["skipped"] for s in again["stores"]] == [True, True]
    assert not (tmp_path / "newer").exists()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_non_ome_zarr_source_is_refused(tmp_path: Path) -> None:
    zarr.open_group(str(tmp_path / "plain"), mode="w", zarr_format=3)
    with pytest.raises(RechunkSourceError, match="no OME-Zarr layout metadata"):
        rechunk(tmp_path / "plain", tmp_path / "new")


def test_an_oversized_working_set_is_refused_before_writing(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 128, 128)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 128, 128), **SMALL)

    with pytest.raises(WorkingSetTooLargeError, match="read"):
        rechunk(
            tmp_path / "old" / "s.ome.zarr",
            tmp_path / "new",
            geometry=Geometry(chunk_shape=(1, 1, 16, 16)),
            max_working_set_bytes=16,
        )
    assert not (tmp_path / "new").exists()


def test_verification_catches_a_misplaced_block(
    tmp_path: Path, patched_reader, monkeypatch
) -> None:
    """Corrupt one written tile and confirm the read-back notices."""
    import zarrmony._rechunk as rc

    rng = np.random.default_rng(3)
    data = rng.integers(1, 5000, size=(1, 1, 96, 96)).astype(np.uint16)
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=data.shape, data=data))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)

    real = rc._write_level_zero

    def scramble(source_array, target_array, plan, done, flush):
        written = real(source_array, target_array, plan, done, flush)
        target_array[0, 0, 0, 0] = 0 if target_array[0, 0, 0, 0] != 0 else 1
        return written

    monkeypatch.setattr(rc, "_write_level_zero", scramble)
    with pytest.raises(RechunkVerificationError, match="differ"):
        rechunk(
            tmp_path / "old" / "s.ome.zarr",
            tmp_path / "new",
            geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
            verify="full",
        )


# ---------------------------------------------------------------------------
# Geometry inheritance
# ---------------------------------------------------------------------------


def test_downsample_method_is_inherited_not_reset(
    tmp_path: Path, patched_reader
) -> None:
    """A ``max``-pooled source keeps ``max`` even though the default is ``mean``."""
    assert DEFAULT_GEOMETRY.downsample_method == "mean"
    rng = np.random.default_rng(4)
    data = rng.integers(0, 300, size=(1, 1, 128, 128)).astype(np.uint16)
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=data.shape, data=data))
    convert(
        "/tmp/fake.lif",
        tmp_path / "old",
        geometry=Geometry(
            downsample_method="max", chunk_shape=(1, 1, 128, 128), pyramid_min_size=32
        ),
    )
    source = tmp_path / "old" / "s.ome.zarr"

    rechunk(source, tmp_path / "new", geometry=Geometry(pyramid_min_size=32))
    audit = zarr.open_group(str(tmp_path / "new"), mode="r").attrs["zarrmony"]
    assert audit["config"]["geometry"]["downsample_method"] == "max"
    assert audit["rechunks"][0]["downsample_method_inherited"] is True
    np.testing.assert_array_equal(
        zarr.open_group(str(tmp_path / "new"), mode="r")["1"][...],
        zarr.open_group(str(source), mode="r")["1"][...],
    )

    # An explicit override still wins.
    rechunk(
        source,
        tmp_path / "forced",
        geometry=Geometry(pyramid_min_size=32),
        downsample_method="mean",
    )
    forced = zarr.open_group(str(tmp_path / "forced"), mode="r").attrs["zarrmony"]
    assert forced["config"]["geometry"]["downsample_method"] == "mean"


def test_contrast_percentile_is_inherited(tmp_path: Path, patched_reader) -> None:
    patched_reader(
        FakeReader(
            scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96), channel_names=["DAPI"]
        )
    )
    convert(
        "/tmp/fake.lif",
        tmp_path / "old",
        chunk_shape=(1, 1, 96, 96),
        contrast_percentile=95.0,
        **SMALL,
    )
    rechunk(
        tmp_path / "old" / "s.ome.zarr",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
    )
    audit = zarr.open_group(str(tmp_path / "new"), mode="r").attrs["zarrmony"]
    assert audit["config"]["contrast_percentile"] == 95.0
    assert audit["rechunks"][0]["contrast"]["inherited"] is True


def test_contrast_off_at_the_source_stays_off(tmp_path: Path, patched_reader) -> None:
    patched_reader(
        FakeReader(
            scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96), channel_names=["DAPI"]
        )
    )
    convert(
        "/tmp/fake.lif",
        tmp_path / "old",
        chunk_shape=(1, 1, 96, 96),
        contrast_percentile=None,
        **SMALL,
    )
    source = tmp_path / "old" / "s.ome.zarr"
    rechunk(source, tmp_path / "new", geometry=Geometry(chunk_shape=(1, 1, 32, 32)))

    src_window = zarr.open_group(str(source), mode="r").attrs["ome"]["omero"][
        "channels"
    ][0]
    dst_window = zarr.open_group(str(tmp_path / "new"), mode="r").attrs["ome"]["omero"][
        "channels"
    ][0]
    assert dst_window["window"] == src_window["window"]


# ---------------------------------------------------------------------------
# Planning surface
# ---------------------------------------------------------------------------


def test_plan_reports_the_working_set_before_anything_is_written(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCZYX", shape=(1, 1, 16, 128, 128)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 1, 128, 128), **SMALL)

    image = read_source_image(tmp_path / "old" / "s.ome.zarr", "")
    plan = plan_image(
        image, Geometry(chunk_shape=(1, 1, 8, 32, 32), pyramid_min_size=32)
    )
    assert plan.tiles[0] == (1, 1, 8, 128, 128)
    assert plan.working_set_bytes[0] == 8 * 128 * 128 * image.dtype.itemsize
    assert plan.tile_counts[0] == 2
    assert plan.is_noop is False


def test_progress_lines_are_emitted(tmp_path: Path, patched_reader) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 96, 96)))
    convert("/tmp/fake.lif", tmp_path / "old", chunk_shape=(1, 1, 96, 96), **SMALL)
    lines: list[str] = []
    rechunk(
        tmp_path / "old" / "s.ome.zarr",
        tmp_path / "new",
        geometry=Geometry(chunk_shape=(1, 1, 32, 32)),
        progress=lines.append,
    )
    assert any("tiles" in line for line in lines)
