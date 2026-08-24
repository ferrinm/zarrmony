"""Tests for the ADR-0010 world-cubic chunk planner (issue #84).

Three concerns, in order:

1. :func:`spacings_for_level` — the per-level µm spacing the planner reasons
   about, derived from a level's shape rather than re-measured.
2. :func:`plan_chunk_shape` / :func:`plan_level_chunk_shapes` — the rule itself:
   largest chunk under the byte target, closest to cubic in micrometres, never
   longer than the level, T and C pinned to 1.
3. That the plan reaches the writer as an explicit per-level list, lands on
   disk, and is recorded in the audit — and that an explicit ``chunk_shape``
   still bypasses the whole thing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import (
    DEFAULT_CHUNK_TARGET_BYTES,
    Geometry,
    plan_chunk_shape,
    plan_level_chunk_shapes,
    spacings_for_level,
)
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin

# The ADR-0010 reference acquisition: a SmartSPIM export at Z 2.0 / Y 1.8 /
# X 1.8 µm, uint16. Near-isotropic, so the world-cubic rule should land on the
# plain 64³ that a voxel-space rule would have produced.
REFERENCE_SHAPE = (1, 3, 3627, 8835, 7452)
REFERENCE_SPACINGS = (1.0, 1.0, 2.0, 1.8, 1.8)


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


def _array_chunks(store: Path) -> dict[str, list[int]]:
    """Every array's on-disk chunk shape, keyed by path within ``store``."""
    out: dict[str, list[int]] = {}
    for zj in sorted(store.rglob("zarr.json")):
        node = json.loads(zj.read_text())
        if node.get("node_type") != "array":
            continue
        out[str(zj.parent.relative_to(store))] = node["chunk_grid"]["configuration"][
            "chunk_shape"
        ]
    return out


def _extents(
    chunk: tuple[int, ...], dims: str, spacings: tuple[float, ...]
) -> list[float]:
    """The chunk's physical size in µm, one entry per spatial axis."""
    return [
        length * spacing
        for length, spacing, dim in zip(chunk, spacings, dims, strict=True)
        if dim in "ZYX"
    ]


def _cubeness(chunk: tuple[int, ...], dims: str, spacings: tuple[float, ...]) -> float:
    """``max/min`` of the chunk's physical extents; 1.0 is a perfect cube."""
    e = _extents(chunk, dims, spacings)
    return max(e) / min(e)


# ---------- per-level spacing ----------


def test_spacing_scales_with_the_downsample_factor() -> None:
    # Half the voxels on an axis means each voxel covers twice the distance.
    assert spacings_for_level(
        (1.0, 1.0, 2.0, 0.325, 0.325),
        (1, 2, 8, 2048, 1536),
        (1, 2, 8, 512, 384),
    ) == (1.0, 1.0, 2.0, 1.3, 1.3)


def test_axes_that_did_not_downsample_keep_their_spacing() -> None:
    # The Z-preserving pyramid is the common case today, and the per-axis
    # factors ADR-0010's later slices produce are handled by the same identity.
    got = spacings_for_level((1.0, 5.0, 0.5, 0.5), (1, 64, 512, 256), (1, 64, 128, 64))
    assert got == (1.0, 5.0, 2.0, 2.0)


def test_level_0_spacing_is_the_base_spacing() -> None:
    base = (1.0, 1.0, 2.0, 1.8, 1.8)
    assert spacings_for_level(base, REFERENCE_SHAPE, REFERENCE_SHAPE) == base


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), None, "n/a"])
def test_unusable_spacings_degrade_to_one_micrometre(bad: object) -> None:
    # Reader metadata is allowed to be absent or nonsense; a spacing that
    # carries no information must not poison the ratio the planner scores on.
    got = spacings_for_level((bad, 1.0), (100, 100), (50, 100))
    assert got == (2.0, 1.0)


def test_spacings_for_level_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        spacings_for_level((1.0, 1.0), (1, 2, 3), (1, 2, 3))


# ---------- the rule ----------


def test_reference_fixture_plans_64_cubed() -> None:
    """ADR-0010's headline case: near-isotropic uint16 lands on 64³."""
    chunk = plan_chunk_shape(REFERENCE_SHAPE, "TCZYX", REFERENCE_SPACINGS, np.uint16)
    assert chunk == (1, 1, 64, 64, 64)
    assert math.prod(chunk) * 2 == DEFAULT_CHUNK_TARGET_BYTES


def test_ten_to_one_anisotropy_is_cubic_in_um_not_in_voxels() -> None:
    """A 10:1 confocal stack: Z 5 µm, XY 0.5 µm.

    A voxel-cubic 64³ would span 320 × 32 × 32 µm — ten times coarser in Z than
    laterally, which is the defect ADR-0010 exists to remove. The planner
    instead spends its voxels on the cheap axes.
    """
    dims, spacings = "TCZYX", (1.0, 1.0, 5.0, 0.5, 0.5)
    chunk = plan_chunk_shape((1, 1, 200, 2048, 2048), dims, spacings, np.uint16)

    assert chunk == (1, 1, 16, 128, 128)
    # Non-cubic in voxels...
    assert chunk[2] != chunk[3]
    # ...precisely so that it is near-cubic in micrometres.
    assert _extents(chunk, dims, spacings) == [80.0, 64.0, 64.0]
    assert _cubeness(chunk, dims, spacings) == 1.25
    # Far better than the voxel-cubic answer it replaces.
    assert _cubeness((1, 1, 64, 64, 64), dims, spacings) == 10.0


def test_isotropic_spacing_gives_a_voxel_cube() -> None:
    spacings = (1.0, 1.0, 0.5, 0.5, 0.5)
    chunk = plan_chunk_shape((1, 1, 512, 512, 512), "TCZYX", spacings, np.uint16)
    assert chunk == (1, 1, 64, 64, 64)


def test_t_and_c_are_never_chunked() -> None:
    # A chunk spans one timepoint of one channel: a viewer showing DAPI at t=0
    # never pays to transport GFP or t=1.
    chunk = plan_chunk_shape(
        (10, 4, 256, 256, 256), "TCZYX", (1.0, 1.0, 1.0, 1.0, 1.0), np.uint16
    )
    assert chunk[0] == 1
    assert chunk[1] == 1


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3, 100, 100),
        (1, 1, 1, 17, 4096),
        (1, 2, 5, 7, 11),
        (1, 1, 2160, 2160),
    ],
)
def test_chunk_never_exceeds_the_level_extent(shape: tuple[int, ...]) -> None:
    dims = "TCZYX" if len(shape) == 5 else "TCYX"
    spacings = tuple(1.0 for _ in shape)
    chunk = plan_chunk_shape(shape, dims, spacings, np.uint16)
    assert all(c <= s for c, s in zip(chunk, shape, strict=True))
    assert all(c >= 1 for c in chunk)


def test_a_level_smaller_than_the_target_is_one_chunk() -> None:
    # 4 × 100 × 100 uint16 is 80 KB — well under 512 KiB, so nothing is gained
    # by splitting it.
    shape = (1, 1, 4, 100, 100)
    chunk = plan_chunk_shape(shape, "TCZYX", (1.0, 1.0, 1.0, 1.0, 1.0), np.uint16)
    assert chunk == shape[:1] + (1,) + shape[2:]


def test_a_two_dimensional_level_plans_a_square_tile() -> None:
    # No Z axis at all: the rule reduces to the largest square tile that fits,
    # not the full-width strip bioio-ome-zarr's heuristic produced.
    chunk = plan_chunk_shape(
        (1, 1, 2160, 2160), "TCYX", (1.0, 1.0, 0.325, 0.325), np.uint16
    )
    assert chunk == (1, 1, 512, 512)


def test_a_level_with_no_spatial_axes_plans_all_ones() -> None:
    assert plan_chunk_shape((4, 3), "TC", (1.0, 1.0), np.uint16) == (1, 1)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.uint8, (1, 1, 64, 64, 128)),
        (np.uint16, (1, 1, 64, 64, 64)),
        (np.float32, (1, 1, 32, 64, 64)),
    ],
)
def test_the_byte_target_is_met_exactly_for_every_dtype(
    dtype: np.typing.DTypeLike, expected: tuple[int, ...]
) -> None:
    """Itemsize is spent, not ignored: a float32 chunk holds a quarter the voxels."""
    chunk = plan_chunk_shape(
        (1, 1, 512, 512, 512), "TCZYX", (1.0, 1.0, 1.0, 1.0, 1.0), dtype
    )
    assert chunk == expected
    assert math.prod(chunk) * np.dtype(dtype).itemsize == DEFAULT_CHUNK_TARGET_BYTES


def test_chunk_target_bytes_scales_the_plan() -> None:
    """The knob is honoured, and the target is a sweet spot rather than a cap."""
    args = ((1, 1, 512, 512, 512), "TCZYX", (1.0, 1.0, 1.0, 1.0, 1.0), np.uint16)
    small = plan_chunk_shape(*args, Geometry(chunk_target_bytes=64 * 1024))
    large = plan_chunk_shape(*args, Geometry(chunk_target_bytes=2 * 1024 * 1024))

    assert math.prod(small) * 2 == 64 * 1024
    assert math.prod(large) * 2 == 2 * 1024 * 1024
    assert small == (1, 1, 32, 32, 32)
    assert large == (1, 1, 64, 128, 128)


def test_a_target_smaller_than_one_voxel_still_plans_a_chunk() -> None:
    # Degenerate, but a chunk of zero voxels is not a thing; floor at one voxel
    # rather than raising on a policy the user is free to set.
    chunk = plan_chunk_shape(
        (1, 1, 8, 8, 8),
        "TCZYX",
        (1.0, 1.0, 1.0, 1.0, 1.0),
        np.float64,
        Geometry(chunk_target_bytes=1),
    )
    assert chunk == (1, 1, 1, 1, 1)


def test_plan_chunk_shape_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        plan_chunk_shape((1, 1, 64, 64), "TCZYX", (1.0, 1.0, 1.0, 1.0), np.uint16)


# ---------- per level ----------


def test_each_level_is_planned_against_its_own_spacing() -> None:
    """A level that halved Y and X does not inherit level 0's chunk.

    Y and X coarsen by 2 each level while Z holds, so the level's voxels get
    progressively less anisotropic and the planner moves budget from the
    lateral axes into Z — level 2 comes out exactly cubic (32 × 32 × 32 µm).
    """
    level_shapes = [(1, 1, 8, 128, 128), (1, 1, 8, 64, 64), (1, 1, 8, 32, 32)]
    geometry = Geometry(chunk_target_bytes=4096)

    plans = plan_level_chunk_shapes(
        level_shapes, "TCZYX", (1.0, 1.0, 4.0, 0.5, 0.5), np.uint16, geometry
    )

    assert plans == [(1, 1, 2, 32, 32), (1, 1, 4, 16, 32), (1, 1, 8, 16, 16)]
    assert _extents(plans[2], "TCZYX", (1.0, 1.0, 4.0, 2.0, 2.0)) == [32.0, 32.0, 32.0]


def test_an_explicit_chunk_shape_bypasses_the_planner() -> None:
    # The caller said what they wanted; second-guessing it per level would make
    # the override mean something different at level 2 than at level 0.
    plans = plan_level_chunk_shapes(
        [(1, 1, 8, 128, 128), (1, 1, 8, 64, 64)],
        "TCZYX",
        (1.0, 1.0, 4.0, 0.5, 0.5),
        np.uint16,
        Geometry(chunk_shape=(1, 1, 4, 16, 16)),
    )
    assert plans == [(1, 1, 4, 16, 16), (1, 1, 4, 16, 16)]


def test_plan_level_chunk_shapes_needs_at_least_one_level() -> None:
    with pytest.raises(ValueError, match="at least one level"):
        plan_level_chunk_shapes([], "TCZYX", (1.0,) * 5, np.uint16)


# ---------- end to end ----------


def _anisotropic_reader() -> FakeReader:
    """8 planes at Z 4.0 µm over a 128² field at 0.5 µm — 256 KB of pixels."""
    return FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 8, 128, 128),
        pixel_sizes=FakePhysicalPixelSizes(Z=4.0, Y=0.5, X=0.5),
    )


def test_the_writer_receives_an_explicit_per_level_chunk_shape(
    tmp_path: Path, patched_reader
) -> None:
    """Each level lands on disk with its own planned chunk, not a replicated one."""
    patched_reader(_anisotropic_reader())
    out = tmp_path / "out"

    convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(chunk_target_bytes=4096, pyramid_min_size=32),
        contrast_percentile=None,
        validate=False,
    )

    assert _array_chunks(out / "s.ome.zarr") == {
        "0": [1, 1, 2, 32, 32],
        "1": [1, 1, 4, 16, 32],
        "2": [1, 1, 8, 16, 16],
    }


def test_the_audit_records_per_level_chunk_shapes(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(_anisotropic_reader())
    out = tmp_path / "out"

    result = convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(chunk_target_bytes=4096, pyramid_min_size=32),
        contrast_percentile=None,
        validate=False,
    )

    record = result["stores"][0]["per_scene"][0]
    # Positionally aligned with the level_shapes they were planned against.
    assert len(record["chunk_shapes"]) == len(record["level_shapes"])
    assert record["chunk_shapes"] == [
        [1, 1, 2, 32, 32],
        [1, 1, 4, 16, 32],
        [1, 1, 8, 16, 16],
    ]
    # And it is what actually got written, not an aspiration.
    assert record["chunk_shapes"] == list(_array_chunks(out / "s.ome.zarr").values())

    root = json.loads((out / "s.ome.zarr" / "zarr.json").read_text())
    on_disk = root["attributes"]["zarrmony"]["per_scene"][0]["chunk_shapes"]
    assert on_disk == record["chunk_shapes"]


def test_default_policy_no_longer_writes_a_full_width_slab(
    tmp_path: Path, patched_reader
) -> None:
    """The regression this slice exists to fix, on a plausible acquisition.

    The pre-#84 heuristic filled the rightmost axis first under a 16 MiB
    budget, producing single-plane full-width slabs; a viewer fetching a 512³
    region paid for the whole width of every plane it touched.
    """
    reader = FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 16, 1024, 1024),
        pixel_sizes=FakePhysicalPixelSizes(Z=1.0, Y=1.0, X=1.0),
    )
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.czi", out, contrast_percentile=None, validate=False)

    level_0 = _array_chunks(out / "s.ome.zarr")["0"]
    assert level_0 == [1, 1, 16, 128, 128]
    # Never the full lateral width, and never a single plane.
    assert level_0[4] < 1024
    assert level_0[2] > 1
    assert math.prod(level_0) * 2 == DEFAULT_CHUNK_TARGET_BYTES


def test_explicit_chunk_shape_still_overrides_the_planner_end_to_end(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(_anisotropic_reader())
    out = tmp_path / "out"

    result = convert(
        "/tmp/x.czi",
        out,
        chunk_shape=(1, 1, 4, 64, 64),
        pyramid_min_size=32,
        contrast_percentile=None,
        validate=False,
    )

    # Applied verbatim to every level.
    assert _array_chunks(out / "s.ome.zarr") == {
        "0": [1, 1, 4, 64, 64],
        "1": [1, 1, 4, 64, 64],
        "2": [1, 1, 4, 64, 64],
    }
    assert result["stores"][0]["per_scene"][0]["chunk_shapes"] == [
        [1, 1, 4, 64, 64],
        [1, 1, 4, 64, 64],
        [1, 1, 4, 64, 64],
    ]


def test_bf2raw_audit_carries_chunk_shapes(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s0", "s1"], dims="TCYX", shape=(1, 1, 64, 64))
    patched_reader(reader)
    out = tmp_path / "out"

    audit = convert(
        "/tmp/x.czi",
        out,
        layout="bf2raw",
        pyramid_min_size=32,
        contrast_percentile=None,
        validate=False,
    )

    for record in audit["per_scene"]:
        assert len(record["chunk_shapes"]) == len(record["level_shapes"])
    assert audit["per_scene"][0]["chunk_shapes"] == [[1, 1, 64, 64], [1, 1, 32, 32]]


def test_plate_audit_carries_chunk_shapes(tmp_path: Path, patched_reader) -> None:
    layout = PlateLayout(
        name="chunk-plate",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1, name="acq")],
        fields=[PlateField(scene_index=0, row="A", column="01", acquisition_id=1)],
    )
    reader = FakeReader(
        scenes=["s0"],
        dims="TCYX",
        shape=(1, 1, 64, 64),
        layout_hint="plate",
        plate_layout=layout,
        channel_names=["DAPI"],
    )
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert(
        "/tmp/x.czi",
        out,
        pyramid_min_size=32,
        contrast_percentile=None,
        validate=False,
    )

    field = audit["fields"][0]
    assert len(field["chunk_shapes"]) == len(field["level_shapes"])
    assert field["chunk_shapes"] == [[1, 1, 64, 64], [1, 1, 32, 32]]
