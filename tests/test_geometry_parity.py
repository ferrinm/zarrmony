"""ADR-0010 (#88): one geometry policy, applied uniformly — including 2D.

``test_geometry.py`` checks that a :class:`~zarrmony.geometry.Geometry` *reaches*
each of the three write paths. This module checks the stronger property the ADR
actually promises: that the three paths **agree**. Per-scene, bf2raw and plate
output for the same FOV geometry must plan the same level shapes, the same
per-level chunk shapes and the same coarse level, and must put the same arrays
on disk — no ``Z > 1`` gate, no plate exemption, no 2D special case.

The fixture deliberately uses a 2160² single-plane field, the shape ADR-0010
names when it rejects "gate the new policy on ``Z > 1``". Under the pre-ADR-0010
default that field became one ``(1,1,1,2160,2160)`` chunk of 8.9 MiB — more than
a viewer's whole 8 MB per-frame decoded upload budget in a single object. It is
also the shape a dimensionality-dependent shortcut would most plausibly be
written for, so it is the one worth pinning.

The last two tests state "no dimensionality-dependent special case" as a
property rather than as a code inspection: a singleton Z must not move the
lateral plan, and the planner's maximality invariant must hold identically for
2D and 3D inputs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    plan_chunk_shape,
    plan_level_chunk_shapes,
)
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin
from zarrmony.writers.pyramid import compute_level_shapes

# A PerkinElmer Phenix field: 2160² sCMOS, single plane, 40x water (0.1494 µm/px).
# Named in ADR-0010's "gate the new policy on Z > 1" rejection.
FIELD_SHAPE = (1, 1, 1, 2160, 2160)
FIELD_DIMS = "TCZYX"
FIELD_PX = FakePhysicalPixelSizes(Z=None, Y=0.1494, X=0.1494)
FIELD_DTYPE = np.dtype("uint16")

# What the field used to be: one chunk spanning the whole plane.
_UNCHUNKED_FIELD_BYTES = 2160 * 2160 * FIELD_DTYPE.itemsize  # 8.9 MiB

_PLUGIN = ReaderPlugin(
    name="bioio-fake",
    match=lambda _p: 100,
    open=lambda _p: object(),
    distribution="bioio-fake",
    source="builtin",
)


def _field_reader(**kwargs) -> FakeReader:
    """A fresh reader over one Phenix-shaped field.

    Fresh per conversion because ``FakeReader`` carries the current scene index;
    every instance describes identical geometry, which is the whole point — any
    difference in the output has to come from the write path, not the input.
    """
    return FakeReader(
        scenes=["field"],
        dims=FIELD_DIMS,
        shape=FIELD_SHAPE,
        pixel_sizes=FIELD_PX,
        channel_names=["DAPI"],
        dtype=FIELD_DTYPE,
        **kwargs,
    )


def _plate_layout(fields: int = 1) -> PlateLayout:
    return PlateLayout(
        name="parity-plate",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1, name="acq")],
        fields=[
            PlateField(scene_index=i, row="A", column="01", acquisition_id=1)
            for i in range(fields)
        ],
    )


def _array_geometry(store: Path) -> dict[str, dict]:
    """Every array's on-disk shape + chunk shape, keyed by path within ``store``."""
    out: dict[str, dict] = {}
    for zj in sorted(store.rglob("zarr.json")):
        node = json.loads(zj.read_text())
        if node.get("node_type") != "array":
            continue
        out[str(zj.parent.relative_to(store))] = {
            "shape": node["shape"],
            "chunks": node["chunk_grid"]["configuration"]["chunk_shape"],
        }
    return out


def _on_disk_audit(store: Path) -> dict:
    return json.loads((store / "zarr.json").read_text())["attributes"]["zarrmony"]


def _levels_by_index(arrays: dict[str, dict]) -> dict[str, dict]:
    """Re-key ``_array_geometry`` output by level, dropping the layout's prefix.

    ``0`` per-scene, ``0/0`` under a bf2raw series, ``A/01/0/0`` under a plate
    well — the same array, three paths. Comparing the three layouts means
    comparing what is left once the path is stripped.
    """
    return {path.rsplit("/", 1)[-1]: geom for path, geom in arrays.items()}


@pytest.fixture(scope="module")
def field_stores(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """One Phenix field converted three ways, with validation on.

    Module-scoped: the three conversions write ~12 MiB each, and every test
    below reads the same result. ``pytest.MonkeyPatch()`` rather than the
    function-scoped ``monkeypatch`` fixture, for the same reason.
    """
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("geometry-parity")

    def install(reader: FakeReader) -> None:
        mp.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, _PLUGIN, 100),
        )

    try:
        install(_field_reader())
        per_scene_out = tmp / "per-scene"
        per_scene = convert("/tmp/x.czi", per_scene_out, contrast_percentile=None)

        install(_field_reader())
        bf2raw_out = tmp / "bf2raw"
        bf2raw = convert(
            "/tmp/x.czi", bf2raw_out, layout="bf2raw", contrast_percentile=None
        )

        install(_field_reader(layout_hint="plate", plate_layout=_plate_layout()))
        plate_out = tmp / "plate.ome.zarr"
        plate = convert("/tmp/x.czi", plate_out, contrast_percentile=None)

        yield {
            "per-scene": {
                "audit": per_scene["stores"][0],
                "record": per_scene["stores"][0]["per_scene"][0],
                "root": per_scene_out / "field.ome.zarr",
            },
            "bf2raw": {
                "audit": bf2raw,
                "record": bf2raw["per_scene"][0],
                "root": bf2raw_out,
            },
            "plate": {
                "audit": plate,
                "record": plate["fields"][0],
                "root": plate_out,
            },
        }
    finally:
        mp.undo()


# ---------- the three paths agree ----------

_PLANNED_KEYS = ("level_shapes", "chunk_shapes", "coarse_level_index")


@pytest.mark.parametrize("layout", ["bf2raw", "plate"])
def test_every_layout_plans_the_geometry_the_per_scene_path_would(
    field_stores: dict, layout: str
) -> None:
    """The audit-recorded plan is identical across write paths.

    Per-scene is the reference because it is the path the geometry rules were
    developed against; bf2raw and plate reuse ``write_scene``, and this is the
    assertion that the reuse is total rather than approximate.
    """
    reference = field_stores["per-scene"]["record"]
    candidate = field_stores[layout]["record"]
    assert {k: candidate[k] for k in _PLANNED_KEYS} == {
        k: reference[k] for k in _PLANNED_KEYS
    }


@pytest.mark.parametrize("layout", ["bf2raw", "plate"])
def test_every_layout_writes_the_same_arrays_on_disk(
    field_stores: dict, layout: str
) -> None:
    """And the plan is what actually reached the store, level for level."""
    reference = _levels_by_index(_array_geometry(field_stores["per-scene"]["root"]))
    candidate = _levels_by_index(_array_geometry(field_stores[layout]["root"]))
    assert candidate == reference


@pytest.mark.parametrize("layout", ["per-scene", "bf2raw", "plate"])
def test_the_plan_matches_what_landed_on_disk(field_stores: dict, layout: str) -> None:
    """``chunk_shapes`` / ``level_shapes`` in the audit are not aspirational."""
    record = field_stores[layout]["record"]
    arrays = _levels_by_index(_array_geometry(field_stores[layout]["root"]))
    for level, (shape, chunks) in enumerate(
        zip(record["level_shapes"], record["chunk_shapes"], strict=True)
    ):
        assert arrays[str(level)] == {"shape": shape, "chunks": chunks}


@pytest.mark.parametrize("layout", ["per-scene", "bf2raw", "plate"])
def test_coarse_level_index_is_recorded_on_disk_for_every_layout(
    field_stores: dict, layout: str
) -> None:
    """Audit schema 11's ``coarse_level_index``, in the plate and bf2raw audits too.

    Level 0 is not coarse here: 8.9 MiB per (t, c) is well inside the 64 MiB
    byte bound, but its 2160-voxel long axis exceeds ``coarse_max_long_axis``.
    So for a 2D field the *lateral* bound is what decides coarseness — the byte
    bound is essentially never binding on a single plane.
    """
    audit = _on_disk_audit(field_stores[layout]["root"])
    records = audit["fields"] if layout == "plate" else audit["per_scene"]
    assert records[0]["coarse_level_index"] == 1
    assert records[0]["level_shapes"][1] == [1, 1, 1, 1080, 1080]


@pytest.mark.parametrize("layout", ["per-scene", "bf2raw", "plate"])
def test_2d_output_is_valid_ome_zarr_in_every_layout(
    field_stores: dict, layout: str
) -> None:
    """Splitting a plane into chunks does not cost spec conformance."""
    assert field_stores[layout]["audit"]["validation_warnings"] == []


# ---------- the 2D field itself ----------


def test_the_2d_field_is_split_into_chunks_under_the_byte_target(
    field_stores: dict,
) -> None:
    """The behaviour ADR-0010 declined to exempt, pinned.

    Level 0 is 25 chunks of exactly 512 KiB rather than one 8.9 MiB object. The
    eight-extra-round-trips objection is answered by concurrency: at 16 in
    flight these issue in parallel for one TTFB, and each sits nearer the
    measured p50 325 KiB interactive read than one whole-plane object does.
    """
    record = field_stores["per-scene"]["record"]
    level0 = record["chunk_shapes"][0]

    assert level0 == [1, 1, 1, 512, 512]
    assert level0 != record["level_shapes"][0]
    assert (
        math.prod(level0) * FIELD_DTYPE.itemsize == DEFAULT_GEOMETRY.chunk_target_bytes
    )
    assert math.prod(level0) * FIELD_DTYPE.itemsize < _UNCHUNKED_FIELD_BYTES

    # And that is 25 real objects on disk, not a declared chunk grid the writer
    # then filled with one object.
    level0_dir = field_stores["per-scene"]["root"] / "0"
    objects = [
        p for p in level0_dir.rglob("*") if p.is_file() and p.name != "zarr.json"
    ]
    assert len(objects) == math.prod(
        math.ceil(s / c) for s, c in zip(record["level_shapes"][0], level0, strict=True)
    )
    assert len(objects) == 25


def test_no_level_of_the_2d_field_exceeds_the_byte_target(field_stores: dict) -> None:
    for chunk in field_stores["per-scene"]["record"]["chunk_shapes"]:
        assert (
            math.prod(chunk) * FIELD_DTYPE.itemsize
            <= DEFAULT_GEOMETRY.chunk_target_bytes
        )


def test_the_2d_fields_object_count_is_the_one_the_adr_records(
    field_stores: dict,
) -> None:
    """The cost side of the 2D decision, pinned next to the benefit.

    39 chunk objects where the pre-planner default wrote 4 — one whole plane per
    level, since 8.9 MiB fit its 16 MiB budget outright. Irrelevant on local
    disk; on object storage this is the listing and per-object metadata cost
    ADR-0010 accepts, and multiplying it by fields and channels is how the plate
    figure in that ADR's Consequences is arrived at. Here so that a change to
    the byte target has to restate the bill rather than quietly revise it.
    """
    root = field_stores["per-scene"]["root"]
    per_level = [
        len(
            [
                p
                for p in (root / str(level)).rglob("*")
                if p.is_file() and p.name != "zarr.json"
            ]
        )
        for level in range(len(field_stores["per-scene"]["record"]["level_shapes"]))
    ]
    assert per_level == [25, 9, 4, 1]
    assert sum(per_level) == 39


def test_the_2d_pyramid_halves_only_the_laterals(field_stores: dict) -> None:
    """Depth behaves with no Z axis to spend.

    A single-plane Z is below the axis floor, so it never halves and — per #85 —
    it is also excluded from the isotropy yardstick, which is what keeps the
    laterals halving all the way down. Depth is then the plain Y/X rule: 2160 →
    1080 → 540 → 270, stopping because the next level's 135 would fall below
    ``pyramid_min_size`` and 270² is already a level a viewer can hold whole.
    """
    levels = field_stores["per-scene"]["record"]["level_shapes"]
    assert levels == [
        [1, 1, 1, 2160, 2160],
        [1, 1, 1, 1080, 1080],
        [1, 1, 1, 540, 540],
        [1, 1, 1, 270, 270],
    ]
    assert {level[2] for level in levels} == {1}


def test_a_1080_square_field_becomes_nine_chunks() -> None:
    """ADR-0010's own worked example for the 2D objection.

    "Splitting a 1080² field into nine chunks adds eight round trips" is how the
    objection is phrased there, and nine is what the planner gives — so the
    answer the ADR argues against on concurrency grounds is the one shipped.
    """
    shape = (1, 1, 1, 1080, 1080)
    spacings = [1.0, 1.0, 1.0, 0.1494, 0.1494]

    chunk = plan_chunk_shape(shape, list(FIELD_DIMS), spacings, FIELD_DTYPE)

    assert chunk == (1, 1, 1, 512, 512)
    grid = math.prod(math.ceil(s / c) for s, c in zip(shape, chunk, strict=True))
    assert grid == 9


# ---------- every FOV of a plate gets the same plan ----------


def test_every_fov_of_a_plate_shares_one_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four FOVs, one geometry — the plate writer forwards, it does not decide.

    Small (256²) on purpose: this is about the per-FOV loop, and the 2160 field
    above already pins the numbers.
    """
    reader = FakeReader(
        scenes=[f"f{i}" for i in range(4)],
        dims=FIELD_DIMS,
        shape=(1, 1, 1, 256, 256),
        pixel_sizes=FIELD_PX,
        channel_names=["DAPI"],
        layout_hint="plate",
        plate_layout=PlateLayout(
            name="multi",
            rows=["A", "B"],
            columns=["01", "02"],
            acquisitions=[Acquisition(id=1, name="acq")],
            fields=[
                PlateField(scene_index=i, row=row, column=col, acquisition_id=1)
                for i, (row, col) in enumerate(
                    [("A", "01"), ("A", "02"), ("B", "01"), ("B", "02")]
                )
            ],
        ),
    )
    monkeypatch.setattr(
        api_module,
        "get_reader",
        lambda _path, *, reader_kwargs=None: (reader, _PLUGIN, 100),
    )
    out = tmp_path / "plate.ome.zarr"

    audit = convert("/tmp/x.czi", out, contrast_percentile=None)

    plans = [{k: f[k] for k in _PLANNED_KEYS} for f in audit["fields"]]
    assert len(plans) == 4
    assert all(plan == plans[0] for plan in plans[1:])
    # Every FOV array on disk, too — four wells, one shape each.
    assert (
        len({json.dumps(g, sort_keys=True) for g in _array_geometry(out).values()}) == 1
    )


# ---------- no dimensionality-dependent special case ----------


def test_a_singleton_z_axis_does_not_change_the_lateral_plan() -> None:
    """``TCZYX`` with ``Z=1`` and ``TCYX`` plan the same laterals.

    The two spellings of a 2D acquisition — readers disagree about whether to
    surface a degenerate Z — must not produce two different stores. A singleton
    Z contributes exactly one candidate length to the chunk search and a
    constant µm extent to the cubeness score, so it can change neither the
    winner nor the depth rule.
    """
    with_z = compute_level_shapes(
        (1, 1, 1, 2160, 2160),
        list("TCZYX"),
        [1.0, 1.0, 1.0, 0.1494, 0.1494],
        FIELD_DTYPE,
    )
    without_z = compute_level_shapes(
        (1, 1, 2160, 2160), list("TCYX"), [1.0, 1.0, 0.1494, 0.1494], FIELD_DTYPE
    )
    assert [s[:2] + s[3:] for s in with_z] == list(without_z)

    chunks_with_z = plan_level_chunk_shapes(
        with_z, list("TCZYX"), [1.0, 1.0, 1.0, 0.1494, 0.1494], FIELD_DTYPE
    )
    chunks_without_z = plan_level_chunk_shapes(
        without_z, list("TCYX"), [1.0, 1.0, 0.1494, 0.1494], FIELD_DTYPE
    )
    assert [c[:2] + c[3:] for c in chunks_with_z] == list(chunks_without_z)


@pytest.mark.parametrize(
    ("shape", "dims", "spacings", "dtype"),
    [
        # 2D: a Phenix field, and the same plane with no Z axis at all.
        ((1, 1, 1, 2160, 2160), "TCZYX", [1.0, 1.0, 1.0, 0.1494, 0.1494], "uint16"),
        ((1, 1, 2160, 2160), "TCYX", [1.0, 1.0, 0.1494, 0.1494], "uint16"),
        ((1, 1, 4096, 4096), "TCYX", [1.0, 1.0, 0.65, 0.65], "uint8"),
        # 3D: near-isotropic light-sheet, and a 10:1 confocal stack.
        ((1, 2, 512, 1024, 1024), "TCZYX", [1.0, 1.0, 1.8, 1.8, 1.8], "uint16"),
        ((1, 1, 64, 2048, 2048), "TCZYX", [1.0, 1.0, 5.0, 0.5, 0.5], "uint16"),
        # A 3-plane stack: the shape the axis floor exists for.
        ((1, 1, 3, 2048, 2048), "TCZYX", [1.0, 1.0, 1.0, 0.325, 0.325], "float32"),
    ],
)
def test_the_planner_is_maximal_on_every_dimensionality(
    shape: tuple[int, ...], dims: str, spacings: list[float], dtype: str
) -> None:
    """One rule, stated as the property that would break if 2D were special-cased.

    The planner takes the largest chunk that fits the byte target, so for every
    spatial axis short of its extent, the next candidate length up must overflow
    the target. A dimensionality-dependent shortcut — "a single plane gets one
    chunk", "plate FOVs keep the whole field" — fails this immediately, because
    such a chunk is either over the target or under-grown. Holding for 2D and 3D
    alike is what "no special case" means operationally.
    """
    itemsize = np.dtype(dtype).itemsize
    levels = compute_level_shapes(shape, list(dims), spacings, dtype)
    chunks = plan_level_chunk_shapes(levels, list(dims), spacings, dtype)

    for level_shape, chunk in zip(levels, chunks, strict=True):
        assert math.prod(chunk) * itemsize <= DEFAULT_GEOMETRY.chunk_target_bytes
        for axis, (extent, length) in enumerate(zip(level_shape, chunk, strict=True)):
            if dims[axis] not in ("Z", "Y", "X") or length == extent:
                continue
            grown = list(chunk)
            # Candidate lengths are powers of two clamped to the extent, so the
            # next one up is either double or the extent itself.
            grown[axis] = min(2 * length, extent)
            assert (
                math.prod(grown) * itemsize > DEFAULT_GEOMETRY.chunk_target_bytes
            ), f"{dims} level {level_shape}: chunk {chunk} could have grown on {dims[axis]}"
