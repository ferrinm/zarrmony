"""Integration tests for write_scene using FakeReader from conftest."""

import numpy as np
import pytest
import zarr

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony.writers.scene import _dtype_window, write_scene


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
    assert "mosaic" not in audit


def test_dtype_window_uint8_spans_full_byte_range() -> None:
    assert _dtype_window(np.dtype(np.uint8)) == {
        "min": 0,
        "max": 255,
        "start": 0,
        "end": 255,
    }


def test_dtype_window_uint16_spans_full_16bit_range() -> None:
    assert _dtype_window(np.dtype(np.uint16)) == {
        "min": 0,
        "max": 65535,
        "start": 0,
        "end": 65535,
    }


def test_dtype_window_uint32_spans_full_32bit_range() -> None:
    w = _dtype_window(np.dtype(np.uint32))
    assert w == {"min": 0, "max": 4294967295, "start": 0, "end": 4294967295}


def test_dtype_window_int16_covers_signed_range() -> None:
    assert _dtype_window(np.dtype(np.int16)) == {
        "min": -32768,
        "max": 32767,
        "start": -32768,
        "end": 32767,
    }


def test_dtype_window_float32_uses_normalized_range() -> None:
    # OMERO convention for normalized floats: 0.0/1.0, not the dtype extrema
    # (finfo.min/max would be ±3.4e38 and viewers can't render that).
    assert _dtype_window(np.dtype(np.float32)) == {
        "min": 0.0,
        "max": 1.0,
        "start": 0.0,
        "end": 1.0,
    }


def test_dtype_window_float64_uses_normalized_range() -> None:
    assert _dtype_window(np.dtype(np.float64)) == {
        "min": 0.0,
        "max": 1.0,
        "start": 0.0,
        "end": 1.0,
    }


@pytest.mark.parametrize(
    "dtype, expected_window",
    [
        (np.uint8, {"min": 0, "max": 255, "start": 0, "end": 255}),
        (np.uint16, {"min": 0, "max": 65535, "start": 0, "end": 65535}),
        (np.float32, {"min": 0.0, "max": 1.0, "start": 0.0, "end": 1.0}),
    ],
)
def test_write_scene_omero_window_matches_array_dtype(
    tmp_path, dtype, expected_window
) -> None:
    """Regression for #50 — the default-path OMERO window must span the array's
    dtype range, not the bioio-ome-zarr ``Channel`` fallback of 0–255. Otherwise
    every uint16/uint32/float32 store appears black-on-first-open in napari and
    OMERO because the display window clamps intensities into an 8-bit band.
    """
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32), dtype=dtype)
    out = tmp_path / f"{np.dtype(dtype).name}.zarr"

    write_scene(reader, scene_index=0, store_path=str(out), pyramid_min_size=8)

    g = zarr.open_group(str(out), mode="r")
    channels = g.attrs["ome"]["omero"]["channels"]
    assert len(channels) == 1
    assert channels[0]["window"] == expected_window


def test_write_scene_records_mosaic_summary(tmp_path) -> None:
    mosaic = {"stitched": True, "tile_count": 12, "tile_shape": {"Y": 5048, "X": 5048}}
    reader = FakeReader(
        scenes=["mosaic_scene"],
        dims="TCYX",
        shape=(1, 1, 64, 64),
        mosaic_summary=mosaic,
    )
    out = tmp_path / "mosaic.zarr"

    audit = write_scene(reader, scene_index=0, store_path=str(out), pyramid_min_size=8)

    assert audit["mosaic"] == mosaic
