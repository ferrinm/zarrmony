"""Integration tests for write_scene using FakeReader from conftest."""

import zarr

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony.writers.scene import write_scene


def test_write_scene_writes_pyramid_and_metadata(tmp_path) -> None:
    reader = FakeReader(
        scenes=["scene_a", "scene_b"],
        dims="TCYX",
        shape=(1, 2, 256, 256),
        pixel_sizes=FakePhysicalPixelSizes(Y=0.5, X=0.5),
        channel_names=["DAPI", "GFP"],
    )
    out = tmp_path / "scene.zarr"

    audit = write_scene(
        reader, scene_index=1, store_path=str(out), pyramid_min_size=128
    )

    assert audit["scene_index"] == 1
    assert audit["scene_name"] == "scene_b"
    assert audit["dims"] == ["T", "C", "Y", "X"]
    assert audit["channel_count"] == 2
    assert audit["physical_pixel_size"] == {"T": 1.0, "C": 1.0, "Y": 0.5, "X": 0.5}

    g = zarr.open_group(str(out), mode="r")
    assert "0" in g
    assert g["0"].shape == (1, 2, 256, 256)
    assert "1" in g
    assert g["1"].shape == (1, 2, 128, 128)
    assert "2" not in g

    assert g["0"][:].max() == 2
    assert g["0"][:].min() == 2
    assert g["1"][:].max() == 2


def test_write_scene_normalizes_non_canonical_axes(tmp_path) -> None:
    reader = FakeReader(scenes=["only"], dims="YXCT", shape=(64, 64, 2, 1))
    out = tmp_path / "norm.zarr"

    audit = write_scene(reader, scene_index=0, store_path=str(out))

    assert audit["dims"] == ["T", "C", "Y", "X"]
    assert audit["axis_normalization"]["was_transposed"] is True

    g = zarr.open_group(str(out), mode="r")
    assert g["0"].shape == (1, 2, 64, 64)


def test_write_scene_single_level_for_small_input(tmp_path) -> None:
    reader = FakeReader(scenes=["s"], dims="YX", shape=(64, 64))
    out = tmp_path / "small.zarr"

    audit = write_scene(
        reader, scene_index=0, store_path=str(out), pyramid_min_size=256
    )

    assert audit["level_shapes"] == [[64, 64]]
    g = zarr.open_group(str(out), mode="r")
    assert "0" in g
    assert "1" not in g


def test_write_scene_returns_audit_record(tmp_path) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    out = tmp_path / "audit.zarr"

    audit = write_scene(reader, scene_index=0, store_path=str(out), pyramid_min_size=8)

    assert audit["axis_normalization"]["was_transposed"] is False
    assert audit["axis_normalization"]["input_dims"] == ["T", "C", "Y", "X"]
    assert audit["axis_normalization"]["output_dims"] == ["T", "C", "Y", "X"]
    assert len(audit["level_shapes"]) == 3
