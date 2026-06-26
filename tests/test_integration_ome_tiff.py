"""Integration test: synthesize an OME-TIFF, convert it through the real
default-reader (bioio.BioImage) path, and verify both the per-scene default
layout and the opt-in bf2raw layout against a real reader.

LIF and CZI can't be synthesized in pure Python; this test catches integration
bugs that would otherwise only show up against pilot data.
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


# ---------- per-scene (default) ----------


def test_per_scene_single_scene_ome_tiff_round_trip(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "in.ome.tif",
        n_scenes=1,
        dims="TCYX",
        shape=(1, 2, 64, 64),
        channel_names=["DAPI", "GFP"],
        pixel_size_um=0.5,
    )
    out = tmp_path / "out"

    result = convert(str(src), out, metadata=_good_metadata(), pyramid_min_size=32)

    assert result["layout"] == "per-scene"
    assert len(result["stores"]) == 1
    audit = result["stores"][0]
    assert audit["reader_plugin"]["name"] == "bioio"
    assert audit["reader_plugin"]["distribution"] == "bioio-ome-tiff"
    assert audit["reader_plugin"]["source"] == "builtin"
    assert audit["reader_plugin"]["match_score"] == 0
    assert audit["audit_schema_version"] == 4
    assert audit["input"]["size_bytes"] > 0
    assert audit["user_metadata"]["microscope"] == "FakeScope"

    # bioio's synth scene name is "Image:0".
    store = out / f"{audit['per_scene'][0]['scene_name'].replace(':', '_')}.ome.zarr"
    assert store.is_dir()

    # Per-store root carries its own attrs.zarrmony audit.
    with open(store / "zarr.json") as f:
        root = json.load(f)
    assert root["attributes"]["zarrmony"] == audit

    # Single-Image OME-XML inside the store.
    parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    assert len(parsed.images) == 1

    # Source XML named after input extension, sitting inside the store.
    assert (store / "OME" / "source" / "raw.tif.xml").exists()

    # Pyramid + axes look right.
    g = zarr.open_group(str(store), mode="r")
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


def test_per_scene_multi_scene_ome_tiff_round_trip(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "multi.ome.tif",
        n_scenes=3,
        dims="TCYX",
        shape=(1, 1, 32, 32),
        channel_names=["DAPI"],
    )
    out = tmp_path / "out"

    result = convert(str(src), out, metadata=_good_metadata(), pyramid_min_size=8)

    assert len(result["stores"]) == 3

    for store_audit in result["stores"]:
        store = Path(store_audit["store_path"])
        assert store.is_dir()
        # Each per-scene store is independently openable + has its own OME-XML.
        parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
        assert len(parsed.images) == 1

    # Each store reflects its own scene_index+1 fill value.
    for store_audit in result["stores"]:
        scene_idx = store_audit["scene_index"]
        store = Path(store_audit["store_path"])
        g = zarr.open_group(str(store), mode="r")
        assert g["0"].shape == (1, 1, 1, 32, 32)
        assert int(g["0"][:].flat[0]) == scene_idx + 1


def test_per_scene_pyramid_levels_match_expected(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "pyr.ome.tif",
        n_scenes=1,
        dims="TCYX",
        shape=(1, 1, 256, 256),
        channel_names=["DAPI"],
    )
    out = tmp_path / "out"

    result = convert(str(src), out, metadata=_good_metadata(), pyramid_min_size=32)

    audit = result["stores"][0]
    # bioio normalizes to 5D TCZYX. 256 → 128 → 64 → 32; next 16 < 32, stop.
    assert audit["per_scene"][0]["level_shapes"] == [
        [1, 1, 1, 256, 256],
        [1, 1, 1, 128, 128],
        [1, 1, 1, 64, 64],
        [1, 1, 1, 32, 32],
    ]


# ---------- bf2raw (opt-in) ----------


def test_bf2raw_single_scene_ome_tiff_round_trip(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "in.ome.tif",
        n_scenes=1,
        dims="TCYX",
        shape=(1, 2, 64, 64),
        channel_names=["DAPI", "GFP"],
        pixel_size_um=0.5,
    )
    out = tmp_path / "out.ome.zarr"

    audit = convert(
        str(src), out, layout="bf2raw", metadata=_good_metadata(), pyramid_min_size=32
    )

    assert audit["reader_plugin"]["name"] == "bioio"
    assert audit["reader_plugin"]["distribution"] == "bioio-ome-tiff"
    assert audit["audit_schema_version"] == 4
    assert len(audit["per_scene"]) == 1

    with open(out / "zarr.json") as f:
        root = json.load(f)
    assert root["attributes"]["ome"]["bioformats2raw.layout"] == 3
    assert root["attributes"]["ome"]["version"] == "0.5"
    assert root["attributes"]["zarrmony"] == audit

    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    assert ome_zj["attributes"]["ome"]["series"] == ["0"]

    parsed = from_xml((out / "OME" / "METADATA.ome.xml").read_text())
    assert len(parsed.images) == 1

    assert (out / "OME" / "source" / "raw.tif.xml").exists()

    g = zarr.open_group(str(out / "0"), mode="r")
    assert g["0"].shape == (1, 2, 1, 64, 64)


def test_bf2raw_multi_scene_ome_tiff_round_trip(tmp_path: Path) -> None:
    src = make_synth_ome_tiff(
        tmp_path / "multi.ome.tif",
        n_scenes=3,
        dims="TCYX",
        shape=(1, 1, 32, 32),
        channel_names=["DAPI"],
    )
    out = tmp_path / "multi.ome.zarr"

    audit = convert(
        str(src), out, layout="bf2raw", metadata=_good_metadata(), pyramid_min_size=8
    )

    assert len(audit["per_scene"]) == 3

    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    assert ome_zj["attributes"]["ome"]["series"] == ["0", "1", "2"]

    parsed = from_xml((out / "OME" / "METADATA.ome.xml").read_text())
    assert len(parsed.images) == 3

    for scene_idx in range(3):
        g = zarr.open_group(str(out / str(scene_idx)), mode="r")
        assert g["0"].shape == (1, 1, 1, 32, 32)
        assert int(g["0"][:].flat[0]) == scene_idx + 1
