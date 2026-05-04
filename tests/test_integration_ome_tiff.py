"""Integration test: synthesize an OME-TIFF, convert it through the real
default-reader (bioio.BioImage) path, and verify the produced bf2raw layout.

This is the only test that exercises the full pipeline against a real bioio
reader (FakeReader-based tests cover the orchestration logic, but not the
real-reader interface). LIF and CZI can't be synthesized in pure Python; this
test catches integration bugs that would otherwise only show up against pilot
data.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("bioio")
pytest.importorskip("bioio_ome_tiff")

import zarr  # noqa: E402
from ome_types import from_xml  # noqa: E402

from tests.fixtures.synthesize import make_synth_ome_tiff  # noqa: E402
from zarrmony import convert  # noqa: E402


def _good_metadata() -> dict:
    return {"microscope": "FakeScope", "modality": "fluorescence"}


def test_single_scene_ome_tiff_round_trip(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "in.ome.tif",
        n_scenes=1,
        dims="TCYX",
        shape=(1, 2, 64, 64),
        channel_names=["DAPI", "GFP"],
        pixel_size_um=0.5,
    )
    out = tmp_path / "out.ome.zarr"

    audit = convert(str(src), out, metadata=_good_metadata(), pyramid_min_size=32)

    # Audit attrs
    assert audit["reader_plugin"] == "bioio-ome-tiff"
    assert audit["input"]["size_bytes"] > 0
    assert len(audit["per_scene"]) == 1
    assert audit["user_metadata"]["microscope"] == "FakeScope"

    # bf2raw structure
    with open(out / "zarr.json") as f:
        root = json.load(f)
    assert root["attributes"]["ome"]["bioformats2raw.layout"] == 3
    assert root["attributes"]["ome"]["version"] == "0.5"
    assert root["attributes"]["zarrmony"] == audit

    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    assert ome_zj["attributes"]["ome"]["series"] == ["0"]

    # OME-XML
    parsed = from_xml((out / "OME" / "METADATA.ome.xml").read_text())
    assert len(parsed.images) == 1

    # Source XML named after input extension
    assert (out / "OME" / "source" / "raw.tif.xml").exists()

    # Per-scene image has multiscales metadata + correct shape and channels.
    # bioio normalizes all OME-TIFF input to canonical TCZYX, inserting a
    # singleton Z even though our synth wrote TCYX — so the on-disk shape is
    # 5D, not 4D.
    g = zarr.open_group(str(out / "0"), mode="r")
    multiscales = g.attrs["ome"]["multiscales"]
    assert multiscales[0]["axes"] == [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    assert g["0"].shape == (1, 2, 1, 64, 64)
    omero_channels = g.attrs["ome"]["omero"]["channels"]
    assert [c["label"] for c in omero_channels] == ["DAPI", "GFP"]


def test_multi_scene_ome_tiff_round_trip(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "multi.ome.tif",
        n_scenes=3,
        dims="TCYX",
        shape=(1, 1, 32, 32),
        channel_names=["DAPI"],
    )
    out = tmp_path / "multi.ome.zarr"

    audit = convert(str(src), out, metadata=_good_metadata(), pyramid_min_size=8)

    assert len(audit["per_scene"]) == 3

    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    assert ome_zj["attributes"]["ome"]["series"] == ["0", "1", "2"]

    parsed = from_xml((out / "OME" / "METADATA.ome.xml").read_text())
    assert len(parsed.images) == 3

    # bioio normalizes to 5D TCZYX (singleton Z added).
    # Each scene's content reflects the scene_index+1 fill value
    for scene_idx in range(3):
        g = zarr.open_group(str(out / str(scene_idx)), mode="r")
        assert g["0"].shape == (1, 1, 1, 32, 32)
        assert int(g["0"][:].flat[0]) == scene_idx + 1


def test_pyramid_levels_match_expected(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "pyr.ome.tif",
        n_scenes=1,
        dims="TCYX",
        shape=(1, 1, 256, 256),
        channel_names=["DAPI"],
    )
    out = tmp_path / "pyr.ome.zarr"

    audit = convert(str(src), out, metadata=_good_metadata(), pyramid_min_size=32)

    # bioio normalizes to 5D TCZYX. 256 → 128 → 64 → 32; next 16 < 32, stop.
    assert audit["per_scene"][0]["level_shapes"] == [
        [1, 1, 1, 256, 256],
        [1, 1, 1, 128, 128],
        [1, 1, 1, 64, 64],
        [1, 1, 1, 32, 32],
    ]
