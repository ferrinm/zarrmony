"""End-to-end tests for convert() and inspect() against FakeReader."""

import json
import warnings
from pathlib import Path

import pytest
import zarr
from ome_types import from_xml

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert, inspect
from zarrmony.errors import (
    ExtractorWarning,
    MetadataValidationError,
    OutputExistsError,
)


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch ``zarrmony.api.get_reader`` to return a configurable FakeReader.

    Tests call ``patched_reader(reader_instance, plugin='bioio-fake')`` to
    install the reader; subsequent calls to ``convert``/``inspect`` will use it.
    """

    def installer(reader: FakeReader, plugin: str = "bioio-fake"):
        monkeypatch.setattr(api_module, "get_reader", lambda _path: (reader, plugin))

    return installer


def _good_metadata() -> dict:
    return {"microscope": "Axioscan", "modality": "fluorescence"}


def test_convert_minimal_lifecycle(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(
        scenes=["alpha", "beta"],
        dims="TCYX",
        shape=(1, 2, 64, 64),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader, plugin="bioio-fake")
    out = tmp_path / "x.ome.zarr"

    audit = convert("/tmp/fake.lif", out, metadata=_good_metadata(), pyramid_min_size=32)

    # Top-level structure
    with open(out / "zarr.json") as f:
        root = json.load(f)
    assert root["attributes"]["ome"]["bioformats2raw.layout"] == 3
    assert root["attributes"]["zarrmony"] == audit

    # OME group
    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    assert ome_zj["attributes"]["ome"]["series"] == ["0", "1"]

    # Per-scene images present
    g0 = zarr.open_group(str(out / "0"), mode="r")
    assert g0["0"].shape == (1, 2, 64, 64)

    # OME-XML has 2 images
    parsed = from_xml((out / "OME" / "METADATA.ome.xml").read_text())
    assert [img.name for img in parsed.images] == ["alpha", "beta"]

    # Source XML written, named after input extension
    assert (out / "OME" / "source" / "raw.lif.xml").exists()


def test_convert_writes_audit_with_user_metadata(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    audit = convert(
        "/tmp/x.czi",
        out,
        metadata={"microscope": "Axioscan", "modality": "multiplex", "objective": "20x"},
        pyramid_min_size=8,
    )

    assert audit["version"]
    assert audit["reader_plugin"] == "bioio-fake"
    assert len(audit["per_scene"]) == 1
    assert audit["user_metadata"]["microscope"] == "Axioscan"
    assert audit["user_metadata"]["objective"] == "20x"
    assert audit["config"]["pyramid_min_size"] == 8


def test_convert_metadata_gate_raises(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    with pytest.raises(MetadataValidationError):
        convert("/tmp/x.lif", tmp_path / "x.zarr", metadata=None)

    with pytest.raises(MetadataValidationError):
        convert("/tmp/x.lif", tmp_path / "y.zarr", metadata={"microscope": "Axioscan"})


def test_convert_permissive_bypasses_gate(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    audit = convert(
        "/tmp/x.lif",
        tmp_path / "x.zarr",
        metadata=None,
        permissive=True,
        pyramid_min_size=8,
    )
    assert audit["user_metadata"] == {}


def test_convert_refuses_overwrite(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    convert("/tmp/x.lif", out, metadata=_good_metadata(), pyramid_min_size=8)
    with pytest.raises(OutputExistsError):
        convert("/tmp/x.lif", out, metadata=_good_metadata(), pyramid_min_size=8)


def test_convert_force_overwrites(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    convert("/tmp/x.lif", out, metadata=_good_metadata(), pyramid_min_size=8)
    audit = convert("/tmp/x.lif", out, metadata=_good_metadata(), pyramid_min_size=8, force=True)
    assert "version" in audit  # second conversion succeeded


def test_convert_per_scene_metadata_routes_correctly(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    audit = convert(
        "/tmp/x.lif",
        out,
        metadata=_good_metadata(),
        per_scene_metadata={
            "b": {"microscope": "Axioscan", "modality": "multiplex", "objective": "63x"},
        },
        pyramid_min_size=8,
    )

    scenes_by_name = {r["scene_name"]: r for r in audit["per_scene"]}
    assert "user_metadata" in scenes_by_name["b"]
    assert scenes_by_name["b"]["user_metadata"]["objective"] == "63x"
    assert "user_metadata" not in scenes_by_name["a"]


def test_convert_warns_and_records_on_extractor_failure(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32), ome_metadata_fails=True)
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit = convert("/tmp/x.lif", out, metadata=_good_metadata(), pyramid_min_size=8)

    extractor_warns = [w for w in caught if issubclass(w.category, ExtractorWarning)]
    assert len(extractor_warns) >= 1
    assert audit["metadata_warnings"]
    assert audit["metadata_warnings"][0]["field"] == "ome_metadata"

    # Conversion still succeeded — stub Image was used
    assert (out / "OME" / "METADATA.ome.xml").exists()


def test_convert_writes_source_xml_named_by_input_ext(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    convert("/tmp/sample.czi", out, metadata=_good_metadata(), pyramid_min_size=8)

    assert (out / "OME" / "source" / "raw.czi.xml").exists()
    assert "fake source xml" in (out / "OME" / "source" / "raw.czi.xml").read_text()


def test_convert_omits_source_xml_when_reader_metadata_none(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32), raw_xml=None)
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    convert("/tmp/x.lif", out, metadata=_good_metadata(), pyramid_min_size=8)

    assert not (out / "OME" / "source").exists()


def test_inspect_returns_scene_summary(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(
        scenes=["alpha", "beta"],
        dims="TCYX",
        shape=(1, 2, 256, 256),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader, plugin="bioio-fake")

    info = inspect("/tmp/x.lif")

    assert info["plugin"] == "bioio-fake"
    assert info["n_scenes"] == 2
    assert [s["name"] for s in info["scenes"]] == ["alpha", "beta"]
    assert info["scenes"][0]["dims"] == ["T", "C", "Y", "X"]
    assert info["scenes"][0]["channel_names"] == ["DAPI", "GFP"]
    assert info["scenes"][0]["shape"] == (1, 2, 256, 256)
    assert info["scenes"][0]["dtype"] == "uint16"
