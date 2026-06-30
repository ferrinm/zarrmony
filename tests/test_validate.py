"""Tests for the optional OME-NGFF post-conversion validator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import zarr

from tests.conftest import FakeReader
from zarrmony import _validate
from zarrmony import api as zm_api
from zarrmony.errors import ValidationWarning
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin

pytest.importorskip("ome_zarr_models")


@pytest.fixture
def fake_plugin() -> ReaderPlugin:
    return ReaderPlugin(
        name="fake",
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=None,
        source="builtin",
    )


def _patched_get_reader(reader: FakeReader, plugin: ReaderPlugin):
    return patch(
        "zarrmony.api.get_reader",
        return_value=(reader, plugin, 100),
    )


def test_is_available_true_in_dev_env() -> None:
    assert _validate.is_available() is True


def test_validate_store_passes_for_per_scene(
    tmp_path: Path, fake_plugin: ReaderPlugin
) -> None:
    reader = FakeReader(scenes=["a"], dims="TCYX", shape=(1, 1, 16, 16))
    with _patched_get_reader(reader, fake_plugin):
        result = zm_api.convert(
            input_path=str(tmp_path / "in.fake"),
            output=str(tmp_path / "out"),
            pyramid_min_size=8,
            validate=False,
        )
    sp = result["stores"][0]["store_path"]
    assert _validate.validate_store(sp, "per-scene") == []


def test_validate_store_passes_for_bf2raw(
    tmp_path: Path, fake_plugin: ReaderPlugin
) -> None:
    reader = FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 16, 16))
    with _patched_get_reader(reader, fake_plugin):
        zm_api.convert(
            input_path=str(tmp_path / "in.fake"),
            output=str(tmp_path / "bundle.ome.zarr"),
            pyramid_min_size=8,
            layout="bf2raw",
            validate=False,
        )
    assert _validate.validate_store(tmp_path / "bundle.ome.zarr", "bf2raw") == []


def test_validate_store_passes_for_plate(
    tmp_path: Path, fake_plugin: ReaderPlugin
) -> None:
    plate_layout = PlateLayout(
        name="p1",
        rows=["A"],
        columns=["01", "02"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(
                scene_index=0,
                row="A",
                column="01",
                field_name="A01-f0",
                acquisition_id=1,
            ),
            PlateField(
                scene_index=1,
                row="A",
                column="02",
                field_name="A02-f0",
                acquisition_id=1,
            ),
        ],
    )
    reader = FakeReader(
        scenes=["s0", "s1"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=plate_layout,
    )
    with _patched_get_reader(reader, fake_plugin):
        zm_api.convert(
            input_path=str(tmp_path / "in.fake"),
            output=str(tmp_path / "plate.ome.zarr"),
            pyramid_min_size=8,
            validate=False,
        )
    assert _validate.validate_store(tmp_path / "plate.ome.zarr", "plate") == []


def test_validate_store_flags_known_malformed_image(tmp_path: Path) -> None:
    sp = tmp_path / "bad.ome.zarr"
    g = zarr.create_group(str(sp), zarr_format=3)
    g.attrs["ome"] = {"version": "0.5"}  # missing required `multiscales`
    findings = _validate.validate_store(sp, "per-scene")
    assert len(findings) == 1
    assert findings[0]["kind"] == "image"
    assert "multiscales" in findings[0]["error"]


def test_convert_default_validate_writes_clean_audit(
    tmp_path: Path, fake_plugin: ReaderPlugin
) -> None:
    reader = FakeReader(scenes=["a"], dims="TCYX", shape=(1, 1, 16, 16))
    with _patched_get_reader(reader, fake_plugin):
        result = zm_api.convert(
            input_path=str(tmp_path / "in.fake"),
            output=str(tmp_path / "out"),
            pyramid_min_size=8,
        )
    audit = result["stores"][0]
    assert audit["validation_warnings"] == []
    assert audit["config"]["validate"] is True


def test_convert_validate_false_skips_validator(
    tmp_path: Path, fake_plugin: ReaderPlugin
) -> None:
    reader = FakeReader(scenes=["a"], dims="TCYX", shape=(1, 1, 16, 16))
    with _patched_get_reader(reader, fake_plugin):
        with patch(
            "zarrmony.api._validate.validate_store",
            side_effect=AssertionError("validator should not be called"),
        ):
            result = zm_api.convert(
                input_path=str(tmp_path / "in.fake"),
                output=str(tmp_path / "out"),
                pyramid_min_size=8,
                validate=False,
            )
    assert result["stores"][0]["validation_warnings"] == []


def test_convert_validate_true_but_extra_missing_warns_and_continues(
    tmp_path: Path, fake_plugin: ReaderPlugin
) -> None:
    reader = FakeReader(scenes=["a"], dims="TCYX", shape=(1, 1, 16, 16))
    with _patched_get_reader(reader, fake_plugin):
        with patch("zarrmony.api._validate.is_available", return_value=False):
            with pytest.warns(ValidationWarning, match="ome-zarr-models extra"):
                result = zm_api.convert(
                    input_path=str(tmp_path / "in.fake"),
                    output=str(tmp_path / "out"),
                    pyramid_min_size=8,
                )
    assert result["stores"][0]["validation_warnings"] == []
