"""Tests for the ADR-0010 ``Geometry`` policy object and its end-to-end wiring.

Three concerns, in order:

1. The value object itself — defaults, immutability, normalization, validation.
2. The sugar fold (:func:`resolve_geometry`) that keeps ``convert()``'s
   ``chunk_shape`` / ``pyramid_min_size`` working.
3. That one resolved policy reaches all three write paths (per-scene, bf2raw,
   plate), and a pinned end-to-end geometry that every later ADR-0010 slice
   has to change deliberately.

The chunk planner the policy feeds lives in ``test_chunk_planner.py``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    Geometry,
    resolve_geometry,
)
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin


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


# ---------- the value object ----------


def test_defaults_are_the_adr_0010_policy() -> None:
    g = Geometry()
    assert g.chunk_target_bytes == 512 * 1024
    assert g.isotropy_tolerance == 1.5
    assert g.axis_floor == 32
    assert g.coarse_max_bytes == 64 * 1024 * 1024
    assert g.coarse_max_long_axis == 2048
    assert g.downsample_method == "mean"
    assert g.pyramid_min_size == 256
    assert g.chunk_shape is None


def test_geometry_is_frozen() -> None:
    g = Geometry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.pyramid_min_size = 64  # type: ignore[misc]


def test_replace_produces_an_independent_policy() -> None:
    """The intended way to vary one field — the default instance is shared."""
    g = dataclasses.replace(DEFAULT_GEOMETRY, downsample_method="max")
    assert g.downsample_method == "max"
    assert DEFAULT_GEOMETRY.downsample_method == "mean"
    # Every other field carries over untouched.
    assert g.pyramid_min_size == DEFAULT_GEOMETRY.pyramid_min_size


def test_chunk_shape_normalizes_to_a_tuple() -> None:
    g = Geometry(chunk_shape=[1, 1, 64, 64, 64])
    assert g.chunk_shape == (1, 1, 64, 64, 64)
    # Normalized to a hashable type, so the whole policy stays hashable.
    assert hash(g) == hash(Geometry(chunk_shape=(1, 1, 64, 64, 64)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_target_bytes": 0},
        {"chunk_target_bytes": -1},
        {"axis_floor": 0},
        {"coarse_max_bytes": 0},
        {"coarse_max_long_axis": 0},
        {"pyramid_min_size": 0},
        {"pyramid_min_size": 8.0},
        {"pyramid_min_size": True},
    ],
)
def test_positive_int_fields_are_validated(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="positive int"):
        Geometry(**kwargs)


def test_isotropy_tolerance_below_one_is_rejected() -> None:
    # < 1.0 is meaningless: no axis's spacing is less than the finest axis's.
    with pytest.raises(ValueError, match="isotropy_tolerance"):
        Geometry(isotropy_tolerance=0.5)
    assert Geometry(isotropy_tolerance=1.0).isotropy_tolerance == 1.0


def test_unknown_downsample_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="downsample_method"):
        Geometry(downsample_method="median")  # type: ignore[arg-type]
    assert Geometry(downsample_method="max").downsample_method == "max"


@pytest.mark.parametrize("bad", [(), (1, 1, 0, 64, 64), (1, -1, 64)])
def test_invalid_chunk_shape_is_rejected(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="chunk_shape"):
        Geometry(chunk_shape=bad)


def test_to_audit_is_json_serializable_and_complete() -> None:
    g = Geometry(chunk_shape=(1, 1, 64, 64, 64), pyramid_min_size=64)
    record = g.to_audit()
    assert record == {
        "chunk_target_bytes": 524288,
        "isotropy_tolerance": 1.5,
        "axis_floor": 32,
        "coarse_max_bytes": 67108864,
        "coarse_max_long_axis": 2048,
        "downsample_method": "mean",
        "pyramid_min_size": 64,
        # JSON has no tuples — the audit records a list.
        "chunk_shape": [1, 1, 64, 64, 64],
    }
    assert json.loads(json.dumps(record)) == record


# ---------- the sugar fold ----------


def test_resolve_geometry_with_nothing_returns_the_default_policy() -> None:
    assert resolve_geometry(None) is DEFAULT_GEOMETRY


def test_resolve_geometry_folds_the_retained_sugar() -> None:
    g = resolve_geometry(None, chunk_shape=[1, 1, 32, 32], pyramid_min_size=64)
    assert g.chunk_shape == (1, 1, 32, 32)
    assert g.pyramid_min_size == 64
    # Everything the caller didn't mention stays at the ADR-0010 default.
    assert g.chunk_target_bytes == DEFAULT_GEOMETRY.chunk_target_bytes


def test_resolve_geometry_passes_an_explicit_policy_through() -> None:
    g = Geometry(pyramid_min_size=64)
    assert resolve_geometry(g) is g


@pytest.mark.parametrize(
    "sugar", [{"chunk_shape": (1, 1, 8, 8)}, {"pyramid_min_size": 64}]
)
def test_geometry_plus_sugar_is_an_error(sugar: dict) -> None:
    # Two spellings of the same field; picking a winner silently would only
    # surface once the store was on disk.
    with pytest.raises(ValueError, match="same policy field two ways"):
        resolve_geometry(Geometry(), **sugar)


def test_convert_rejects_geometry_plus_sugar(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    with pytest.raises(ValueError, match="same policy field two ways"):
        convert(
            "/tmp/x.czi",
            tmp_path / "out",
            geometry=Geometry(),
            pyramid_min_size=8,
        )
    # Rejected before the reader is even opened, so nothing was written.
    assert not (tmp_path / "out").exists()


# ---------- the policy reaches every write path ----------


def test_geometry_reaches_the_per_scene_writer(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 64, 64))
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert(
        "/tmp/x.czi",
        out,
        geometry=Geometry(pyramid_min_size=16, chunk_shape=(1, 1, 32, 32)),
        contrast_percentile=None,
    )

    # pyramid_min_size=16 on a 64x64 base gives levels 64 -> 32 -> 16.
    assert result["stores"][0]["per_scene"][0]["level_shapes"] == [
        [1, 1, 64, 64],
        [1, 1, 32, 32],
        [1, 1, 16, 16],
    ]
    arrays = _array_geometry(out / "s.ome.zarr")
    assert arrays["0"]["chunks"] == [1, 1, 32, 32]


def test_geometry_reaches_the_bf2raw_writer(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s0", "s1"], dims="TCYX", shape=(1, 1, 64, 64))
    patched_reader(reader)
    out = tmp_path / "out"

    audit = convert(
        "/tmp/x.czi",
        out,
        layout="bf2raw",
        geometry=Geometry(pyramid_min_size=32, chunk_shape=(1, 1, 16, 16)),
        contrast_percentile=None,
    )

    assert [r["level_shapes"] for r in audit["per_scene"]] == [
        [[1, 1, 64, 64], [1, 1, 32, 32]],
        [[1, 1, 64, 64], [1, 1, 32, 32]],
    ]
    arrays = _array_geometry(out)
    assert arrays["0/0"]["chunks"] == [1, 1, 16, 16]
    assert arrays["1/0"]["chunks"] == [1, 1, 16, 16]


def test_geometry_reaches_the_plate_writer(tmp_path: Path, patched_reader) -> None:
    layout = PlateLayout(
        name="geo-plate",
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
        geometry=Geometry(pyramid_min_size=16, chunk_shape=(1, 1, 8, 8)),
        contrast_percentile=None,
    )

    assert audit["fields"][0]["level_shapes"] == [
        [1, 1, 64, 64],
        [1, 1, 32, 32],
        [1, 1, 16, 16],
    ]
    arrays = _array_geometry(out)
    assert arrays["A/01/0/0"]["chunks"] == [1, 1, 8, 8]


# ---------- the audit surface ----------


def test_audit_records_the_resolved_policy(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    result = convert("/tmp/x.czi", tmp_path / "out", contrast_percentile=None)

    config = result["stores"][0]["config"]
    assert config["geometry"] == DEFAULT_GEOMETRY.to_audit()
    # The uninformative input echo is gone — resolved policy replaces it.
    assert "pyramid_min_size" not in config
    assert "chunk_shape" not in config


def test_audit_geometry_reflects_the_sugar(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    result = convert(
        "/tmp/x.czi",
        tmp_path / "out",
        pyramid_min_size=8,
        chunk_shape=(1, 1, 16, 16),
        contrast_percentile=None,
    )

    assert result["stores"][0]["config"]["geometry"] == {
        **DEFAULT_GEOMETRY.to_audit(),
        "pyramid_min_size": 8,
        "chunk_shape": [1, 1, 16, 16],
    }


def test_audit_geometry_round_trips_to_disk(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.czi", out, contrast_percentile=None)

    root = json.loads((out / "s.ome.zarr" / "zarr.json").read_text())
    on_disk = root["attributes"]["zarrmony"]["config"]["geometry"]
    assert on_disk == DEFAULT_GEOMETRY.to_audit()


# ---------- pinned on-disk geometry ----------


def test_default_geometry_output_for_a_fixed_input(
    tmp_path: Path, patched_reader
) -> None:
    """Pinned on-disk geometry for a fixed input under the default policy.

    The whole-store canary for the ADR-0010 series: any slice that moves a
    level boundary or a chunk boundary has to change these numbers on purpose.

    **Level shapes** are unchanged since v0.14.0, and both pyramid slices left
    them that way on purpose: #85's isotropy rule leaves Z alone here (2.0 µm is
    6.2x the lateral spacing, well outside the 1.5 tolerance), and #86's
    stopping rule buys no extra level because level 0 is *already* a coarse
    level — 48 MiB per ``(t, c)`` with a 2048-voxel long axis, inside both
    bounds. Hence ``coarse_level_index == 0``.

    **Chunk shapes** changed in #84. Before the planner these were
    bioio-ome-zarr's memory-target heuristic filling the rightmost axis first
    under a 16 MiB budget — level 0 came out ``[1, 1, 2, 2048, 1536]``, a
    12 MiB full-width slab that no frustum cull could trim. Each level now gets
    its own 512 KiB world-cubic plan instead: Z is pinned at its full 8 planes
    (16 µm, the most Z on offer) and the remaining budget goes to Y and X, split
    as evenly as powers of two allow.
    """
    reader = FakeReader(
        scenes=["big"],
        dims="TCZYX",
        shape=(1, 2, 8, 2048, 1536),
        pixel_sizes=FakePhysicalPixelSizes(Z=2.0, Y=0.325, X=0.325),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.czi", out, contrast_percentile=None, validate=False)

    scene = result["stores"][0]["per_scene"][0]
    assert scene["level_shapes"] == [
        [1, 2, 8, 2048, 1536],
        [1, 2, 8, 1024, 768],
        [1, 2, 8, 512, 384],
    ]
    assert scene["coarse_level_index"] == 0
    assert _array_geometry(out / "big.ome.zarr") == {
        "0": {"shape": [1, 2, 8, 2048, 1536], "chunks": [1, 1, 8, 128, 256]},
        "1": {"shape": [1, 2, 8, 1024, 768], "chunks": [1, 1, 8, 128, 256]},
        "2": {"shape": [1, 2, 8, 512, 384], "chunks": [1, 1, 8, 128, 256]},
    }
