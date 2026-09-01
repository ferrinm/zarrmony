"""Tests for ``scripts/verify_geometry.py``, the ADR-0010 acceptance verifier.

The script is what both acceptance runbooks tell an operator to run, and its
output gets pasted onto an issue as the record of a multi-hour conversion — so
a number it gets wrong is a number that outlives the run. Until #129 it had no
concept of a shard: it predicted the object count off the *chunk* grid while
the disk holds *shards*, a 15.84x disagreement on the store measured in #124,
with nothing in the report to explain it.

Three things are pinned here, in order:

1. An unsharded store reads exactly as it did before, minus nothing.
2. A sharded store is predicted, counted and reported in shards, with the
   chunk grid still shown as the read unit it has become.
3. The comparison against disk is ``objects <= grid``. Zarr writes no object
   that is entirely fill value, so a correct store can come in short — #126's
   did, by 1,134 shards — and a verifier demanding equality would have failed
   the very run that found it.

The stores are written through the normal ``convert()`` path rather than
assembled by hand: the point of the script is to check what the writer does,
so a fixture that agreed with the planner by construction would prove nothing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import DEFAULT_SHARD_TARGET_BYTES, Geometry
from zarrmony.readers.plugin import ReaderPlugin

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_geometry.py"
_spec = importlib.util.spec_from_file_location("verify_geometry", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vg = importlib.util.module_from_spec(_spec)
sys.modules["verify_geometry"] = vg
_spec.loader.exec_module(vg)

#: 1024² uint16 lands a 512² chunk at the default target, so the default shard
#: target packs a 2x2 of them into a 1024² object: the two grids are genuinely
#: distinct, which is the whole thing under test.
SHAPE = (1, 2, 1, 1024, 1024)
PIXELS = (np.arange(math.prod(SHAPE)) % 65_521).astype(np.uint16).reshape(SHAPE)
SHARDED = Geometry(shard_target_bytes=DEFAULT_SHARD_TARGET_BYTES)


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch ``zarrmony.api.get_reader`` to hand back a fixed fake scene."""

    def installer() -> None:
        reader = FakeReader(
            scenes=["slide"],
            dims="TCZYX",
            shape=SHAPE,
            pixel_sizes=FakePhysicalPixelSizes(Z=1.0, Y=0.325, X=0.325),
            channel_names=["DAPI", "GFP"],
            data=PIXELS,
        )
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


def _write(out: Path, installer, geometry: Geometry | None = None) -> str:
    installer()
    convert(
        "/tmp/x.czi",
        out,
        geometry=geometry,
        contrast_percentile=None,
        validate=False,
    )
    return str(out / "slide.ome.zarr")


def _run(store: str, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
    """Verify ``store``; return the checks keyed by name, plus the report."""
    kwargs.setdefault("expect_coarse_level", None)
    kwargs.setdefault("do_count", True)
    checks, report = vg.verify(store, **kwargs)
    return {check.name: check for check in checks}, report


def _row(report: list[str], prefix: str) -> str:
    return next(line for line in report if line.startswith(prefix))


# ---------- the unsharded store is untouched ----------


def test_an_unsharded_report_never_mentions_a_shard(tmp_path: Path, patched_reader):
    """The store that predates #117 reads exactly as it did.

    A shard column full of ``—``, or a "0 shards" line, would be noise in every
    report zarrmony has produced so far. The one addition on this path is the
    objects-within-grid check, which applies to a chunk store just as much.
    """
    store = _write(tmp_path / "plain", patched_reader)
    checks, report = _run(store)

    # The heading echoes the store path, which is a tmp dir named after this
    # test — and so says "unsharded".
    body = "\n".join(line for line in report if store not in line)
    assert "shard" not in body.lower()
    assert "shard shapes" not in checks
    assert (
        "| level | shape | chunk | chunks | MiB per (t,c) | coarse |" in report
    ), "the levels table gained a column on a store that has no shards"
    assert _row(report, "- Chunk objects").endswith("**12**")
    assert _row(report, "- Objects on disk (chunks + metadata)").endswith("**18**")


# ---------- the sharded store is reported in shards ----------


def test_the_levels_table_shows_the_shard_shape_per_level(
    tmp_path: Path, patched_reader
):
    _, report = _run(_write(tmp_path / "sharded", patched_reader, SHARDED))

    assert (
        "| level | shape | chunk | shard | chunks | shards | MiB per (t,c) | coarse |"
        in report
    )
    assert (
        "| 0 | `(1, 2, 1, 1024, 1024)` | `(1, 1, 1, 512, 512)` | "
        "`(1, 1, 1, 1024, 1024)` | 8 | 2 | 2.0 | yes |" in report
    )
    # Level 1 is a single 512² chunk in a 512² shard — the shard shape differs
    # from level 0's, and the table has to show that rather than imply one
    # geometry for the pyramid.
    assert "| 1 | `(1, 2, 1, 512, 512)` | `(1, 1, 1, 512, 512)` | " in _row(
        report, "| 1 |"
    )


def test_the_prediction_counts_shards_and_says_so(tmp_path: Path, patched_reader):
    """The bug from #124: 493,484 predicted against 31,634 on disk.

    Predicting off the chunk grid puts the headline object count out by the
    chunks-per-shard factor and leaves the report unable to explain the gap.
    Here that factor is 2x, and both figures have to appear — the shard because
    it is the object, the chunk because it is still what a viewer reads.
    """
    _, report = _run(_write(tmp_path / "sharded", patched_reader, SHARDED))

    assert _row(report, "- Shard objects (from the level grids)").endswith("**6**")
    assert _row(report, "- Chunks inside them").endswith("**12**")
    assert _row(report, "- Objects on disk (shards + metadata)").endswith("**12**")
    assert not [line for line in report if line.startswith("- Chunk objects")]


def test_the_planned_shard_shapes_are_checked_against_disk(
    tmp_path: Path, patched_reader
):
    checks, _ = _run(_write(tmp_path / "sharded", patched_reader, SHARDED))

    check = checks["shard shapes"]
    assert check.ok, check.detail
    assert "(1, 1, 1, 1024, 1024)" in check.detail
    assert "4 chunks per shard" in check.detail


def test_a_store_sharded_against_a_policy_that_is_not_fails(
    tmp_path: Path, patched_reader
):
    """Disk and the recorded policy disagreeing is a finding, not a shrug.

    Same spirit as the existing coarse-level check, which recomputes from the
    store and compares against the audit: the two are independent records of
    one decision, so a divergence means one of them is lying about the store.
    """
    store = _write(tmp_path / "sharded", patched_reader, SHARDED)
    group = vg.open_root_group(store, mode="a")
    attrs = dict(group.attrs)
    attrs["zarrmony"]["config"]["geometry"]["shard_target_bytes"] = None
    attrs["zarrmony"]["config"]["geometry"]["shard_shape"] = None
    group.attrs.update(attrs)

    checks, _ = _run(store)

    assert checks["shard shapes"].ok is False
    assert "the recorded policy plans no shards" in checks["shard shapes"].detail


def test_an_explicit_shard_shape_survives_the_audit_round_trip(
    tmp_path: Path, patched_reader
):
    """JSON has no tuples, and ``Geometry``'s shape fields are tuple-typed.

    ``--shard-shape`` is the spelling that puts a list into the audit and hands
    it back on the next read, so it is the one that would find a coercion the
    verifier had only applied to ``chunk_shape``.
    """
    explicit = Geometry(
        chunk_shape=(1, 1, 1, 256, 256), shard_shape=(1, 1, 1, 512, 512)
    )
    store = _write(tmp_path / "explicit", patched_reader, explicit)

    checks, report = _run(store)

    assert checks["shard shapes"].ok, checks["shard shapes"].detail
    assert "`(1, 1, 1, 512, 512)`" in _row(report, "| 0 |")


def test_geometry_from_audit_coerces_both_shape_fields() -> None:
    geometry, source = vg._geometry_from_audit(
        {
            "config": {
                "geometry": {
                    "chunk_shape": [1, 1, 64, 64, 64],
                    "shard_shape": [1, 1, 128, 128, 128],
                }
            }
        }
    )

    assert geometry.chunk_shape == (1, 1, 64, 64, 64)
    assert geometry.shard_shape == (1, 1, 128, 128, 128)
    assert geometry.sharding_enabled
    assert source == "attrs.zarrmony.config.geometry"


# ---------- objects <= grid, never == ----------


def test_the_grid_is_an_upper_bound_and_a_shortfall_passes(
    tmp_path: Path, patched_reader, monkeypatch: pytest.MonkeyPatch
):
    """#126's store was 1,134 shards under its grid and entirely correct.

    Every absent object sat on a trailing row covering a 3-voxel sliver of a
    padded axis — all fill value, and zarr writes no such object. An equality
    assertion would have failed the run that produced the finding, so the
    shortfall has to pass *and* be quantified: a reader has to be able to tell
    "some all-fill objects were skipped" from "the geometry is wrong".
    """
    store = _write(tmp_path / "sharded", patched_reader, SHARDED)
    monkeypatch.setattr(vg, "count_objects", lambda *_a, **_k: vg.ObjectCounts(4, 4, 2))

    checks, _ = _run(store)

    check = checks["shards on disk within the grid"]
    assert check.ok
    assert "4 of 6" in check.detail
    assert "2 absent (33.33%)" in check.detail


def test_more_objects_than_the_grid_is_the_failure(
    tmp_path: Path, patched_reader, monkeypatch: pytest.MonkeyPatch
):
    store = _write(tmp_path / "sharded", patched_reader, SHARDED)
    monkeypatch.setattr(vg, "count_objects", lambda *_a, **_k: vg.ObjectCounts(7, 4, 2))

    checks, _ = _run(store)

    check = checks["shards on disk within the grid"]
    assert check.ok is False
    assert "1 more than the geometry accounts for" in check.detail


def test_a_real_store_sits_exactly_on_its_grid(tmp_path: Path, patched_reader):
    # No padded axis in this scene, so nothing is all-fill and the two figures
    # meet. Worth pinning beside the shortfall case: `<=` must not be a licence
    # to pass a store that wrote half of what it planned.
    for name, geometry in (("plain", None), ("sharded", SHARDED)):
        checks, _ = _run(_write(tmp_path / name, patched_reader, geometry))
        unit = "shards" if geometry else "chunks"
        assert checks[f"{unit} on disk within the grid"].ok
        assert "exactly the grid" in checks[f"{unit} on disk within the grid"].detail


def test_sidecars_and_metadata_are_not_counted_against_the_grid(
    tmp_path: Path, patched_reader
):
    """The store holds files that are not pixels, and they are not objects.

    One ``zarr.json`` per level plus one for the group, and zarrmony also
    writes ``OME/METADATA.ome.xml`` and the source metadata beside it. Counting
    those as chunks reports a correct store as 6 objects over its own grid.
    """
    store = _write(tmp_path / "plain", patched_reader)

    counted = vg.count_objects(store, ["0", "1", "2"])

    assert counted.pixels == 12
    assert counted.metadata == 4  # three levels plus the group
    assert counted.other == 2  # METADATA.ome.xml and the source XML
    assert counted.total == sum(1 for p in Path(store).rglob("*") if p.is_file())


def test_no_object_count_skips_the_walk_and_the_check(tmp_path: Path, patched_reader):
    checks, report = _run(_write(tmp_path / "plain", patched_reader), do_count=False)

    assert "chunks on disk within the grid" not in checks
    assert not [line for line in report if line.startswith("- Objects on disk")]
    assert _row(report, "- Chunk objects").endswith("**12**")
