"""End-to-end tests for convert() and inspect() against FakeReader.

Default tests cover ``layout='per-scene'`` (the new default). A small set of
parity tests pin the opt-in ``layout='bf2raw'`` path.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr
from ome_types import from_xml

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert, inspect
from zarrmony.errors import (
    ExtractorWarning,
    MosaicMergedSiblingWarning,
    OutputExistsError,
)
from zarrmony.readers.plugin import ReaderPlugin


def _fake_plugin(name: str = "bioio-fake") -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=name,
        source="builtin",
    )


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch ``zarrmony.api.get_reader`` to return a configurable FakeReader."""

    def installer(reader: FakeReader, plugin: str = "bioio-fake"):
        plugin_obj = _fake_plugin(plugin)
        monkeypatch.setattr(
            api_module, "get_reader", lambda _path: (reader, plugin_obj, 100)
        )

    return installer


# ---------- per-scene mode (default) ----------


def test_per_scene_minimal_lifecycle(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(
        scenes=["alpha", "beta"],
        dims="TCYX",
        shape=(1, 2, 64, 64),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader, plugin="bioio-fake")
    out = tmp_path / "out"

    result = convert("/tmp/fake.lif", out, pyramid_min_size=32)

    assert result["layout"] == "per-scene"
    assert len(result["stores"]) == 2

    for scene_name in ["alpha", "beta"]:
        store = out / f"{scene_name}.ome.zarr"
        assert store.is_dir()

        # Per-store root attrs.zarrmony audit
        with open(store / "zarr.json") as f:
            root = json.load(f)
        assert "zarrmony" in root["attributes"]
        assert root["attributes"]["zarrmony"]["scene_name"] == scene_name

        # Single-Image OME-XML
        parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
        assert len(parsed.images) == 1
        assert parsed.images[0].name == scene_name

        # Source XML duplicated per store
        assert (store / "OME" / "source" / "raw.lif.xml").exists()

        # Pyramid array openable
        g = zarr.open_group(str(store), mode="r")
        assert g["0"].shape == (1, 2, 64, 64)


def test_per_scene_writes_audit(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.czi", out, pyramid_min_size=8)

    assert len(result["stores"]) == 1
    audit = result["stores"][0]
    assert audit["version"]
    assert audit["reader_plugin"]["name"] == "bioio-fake"
    assert audit["reader_plugin"]["distribution"] == "bioio-fake"
    assert audit["reader_plugin"]["source"] == "builtin"
    assert audit["reader_plugin"]["match_score"] == 100
    assert audit["config"]["pyramid_min_size"] == 8
    assert audit["config"]["layout"] == "per-scene"


def test_per_scene_refuses_existing_store(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)
    with pytest.raises(OutputExistsError):
        convert("/tmp/x.lif", out, pyramid_min_size=8)


def test_per_scene_force_overwrites_existing_store(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)
    result = convert("/tmp/x.lif", out, pyramid_min_size=8, force=True)
    assert result["stores"][0]["version"]


def test_per_scene_does_not_clobber_unrelated_sibling(
    tmp_path: Path, patched_reader
) -> None:
    """A pre-existing sibling store under OUTPUT must survive a fresh per-scene run."""
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"
    out.mkdir()
    sibling = out / "preexisting.ome.zarr"
    sibling.mkdir()
    (sibling / "marker.txt").write_text("keep me")

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    assert (sibling / "marker.txt").read_text() == "keep me"


def test_per_scene_warns_and_records_on_extractor_failure(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(
        scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32), ome_metadata_fails=True
    )
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = convert("/tmp/x.lif", out, pyramid_min_size=8)

    extractor_warns = [w for w in caught if issubclass(w.category, ExtractorWarning)]
    assert len(extractor_warns) >= 1

    audit = result["stores"][0]
    assert audit["metadata_warnings"]
    assert audit["metadata_warnings"][0]["field"] == "ome_metadata"
    # Stub Image was used, so per-store OME-XML still exists.
    store = out / "s.ome.zarr"
    assert (store / "OME" / "METADATA.ome.xml").exists()


def test_per_scene_writes_source_xml_named_by_input_ext(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/sample.czi", out, pyramid_min_size=8)

    # Each per-scene store carries its own copy of the raw vendor XML.
    for scene_name in ["a", "b"]:
        src = out / f"{scene_name}.ome.zarr" / "OME" / "source" / "raw.czi.xml"
        assert src.exists()
        assert "fake source xml" in src.read_text()


def test_per_scene_omits_source_xml_when_reader_metadata_none(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32), raw_xml=None)
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    assert not (out / "s.ome.zarr" / "OME" / "source").exists()


def test_per_scene_sanitizes_scene_names_in_dirnames(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(scenes=["a/b", "c d"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=8)

    assert (out / "a_b.ome.zarr").is_dir()
    assert (out / "c_d.ome.zarr").is_dir()
    # Original (unsanitized) scene_name preserved in audit + OME-XML.
    by_name = {s["scene_name"]: s for s in result["stores"]}
    assert set(by_name) == {"a/b", "c d"}


# ---------- omero display window matches array dtype (#50) ----------


@pytest.mark.parametrize(
    "dtype, expected_window",
    [
        (np.uint8, {"min": 0, "max": 255, "start": 0, "end": 255}),
        (np.uint16, {"min": 0, "max": 65535, "start": 0, "end": 65535}),
        (np.float32, {"min": 0.0, "max": 1.0, "start": 0.0, "end": 1.0}),
    ],
)
def test_per_scene_omero_window_matches_reader_dtype(
    tmp_path: Path, patched_reader, dtype, expected_window
) -> None:
    """The ``_channels_for_scene`` api path (named channels, non-LIF) must ship
    an OMERO display window spanning the reader's dtype range — not the
    bioio-ome-zarr ``Channel`` default of 0–255 that would make uint16 stores
    open black. Regression guard for #50.
    """
    reader = FakeReader(
        scenes=["s"],
        dims="TCYX",
        shape=(1, 2, 32, 32),
        channel_names=["DAPI", "GFP"],
        dtype=dtype,
    )
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.czi", out, pyramid_min_size=8)

    g = zarr.open_group(str(out / "s.ome.zarr"), mode="r")
    channels = g.attrs["ome"]["omero"]["channels"]
    assert len(channels) == 2
    for c in channels:
        assert c["window"] == expected_window


# ---------- bf2raw mode (opt-in) ----------


def test_bf2raw_opt_in_round_trip(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(
        scenes=["alpha", "beta"],
        dims="TCYX",
        shape=(1, 2, 64, 64),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader, plugin="bioio-fake")
    out = tmp_path / "x.ome.zarr"

    audit = convert("/tmp/fake.lif", out, layout="bf2raw", pyramid_min_size=32)

    with open(out / "zarr.json") as f:
        root = json.load(f)
    assert root["attributes"]["ome"]["bioformats2raw.layout"] == 3
    assert root["attributes"]["zarrmony"] == audit
    assert audit["config"]["layout"] == "bf2raw"

    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    assert ome_zj["attributes"]["ome"]["series"] == ["0", "1"]

    g0 = zarr.open_group(str(out / "0"), mode="r")
    assert g0["0"].shape == (1, 2, 64, 64)

    parsed = from_xml((out / "OME" / "METADATA.ome.xml").read_text())
    assert [img.name for img in parsed.images] == ["alpha", "beta"]
    assert (out / "OME" / "source" / "raw.lif.xml").exists()


def test_bf2raw_refuses_overwrite(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.zarr"

    convert("/tmp/x.lif", out, layout="bf2raw", pyramid_min_size=8)
    with pytest.raises(OutputExistsError):
        convert("/tmp/x.lif", out, layout="bf2raw", pyramid_min_size=8)


def test_unknown_layout_raises(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    with pytest.raises(ValueError, match="layout"):
        convert("/tmp/x.lif", tmp_path / "x", layout="bogus")


# ---------- inspect ----------


def test_inspect_returns_scene_summary(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(
        scenes=["alpha", "beta"],
        dims="TCYX",
        shape=(1, 2, 256, 256),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader, plugin="bioio-fake")

    info = inspect("/tmp/x.lif")

    assert info["reader_plugin"]["name"] == "bioio-fake"
    assert info["reader_plugin"]["distribution"] == "bioio-fake"
    assert info["reader_plugin"]["source"] == "builtin"
    assert info["reader_plugin"]["match_score"] == 100
    assert info["n_scenes"] == 2
    assert [s["name"] for s in info["scenes"]] == ["alpha", "beta"]
    assert info["scenes"][0]["dims"] == ["T", "C", "Y", "X"]
    assert info["scenes"][0]["channel_names"] == ["DAPI", "GFP"]
    assert info["scenes"][0]["shape"] == (1, 2, 256, 256)
    assert info["scenes"][0]["dtype"] == "uint16"
    # Flat reader: no plate_layout key (additive, non-breaking).
    assert "plate_layout" not in info


def test_inspect_returns_size_bytes_for_file_input(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)
    src = tmp_path / "in.lif"
    src.write_bytes(b"\x00" * 2048)

    info = inspect(str(src))
    assert info["size_bytes"] == 2048
    assert info["size_human"] == "2.0 KB"


def test_inspect_returns_recursive_size_bytes_for_directory_input(
    tmp_path: Path, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)
    src = tmp_path / "input.czi"
    (src / "sub").mkdir(parents=True)
    (src / "top.bin").write_bytes(b"x" * 100)
    (src / "sub" / "leaf.bin").write_bytes(b"y" * 900)

    info = inspect(str(src))
    assert info["size_bytes"] == 1000
    assert info["size_human"] == "1000 B"


def test_inspect_includes_plate_layout_for_plate_reader(
    tmp_path: Path, patched_reader
) -> None:
    from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

    plate_layout = PlateLayout(
        name="synthetic-2x2",
        rows=["A", "B"],
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
            PlateField(
                scene_index=2,
                row="B",
                column="01",
                field_name="B01-f0",
                acquisition_id=1,
            ),
        ],
    )
    reader = FakeReader(
        scenes=["s0", "s1", "s2"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=plate_layout,
    )
    patched_reader(reader, plugin="bioio-fake-plate")

    info = inspect("/tmp/x.czi")

    assert "plate_layout" in info
    pl = info["plate_layout"]
    assert pl["name"] == "synthetic-2x2"
    assert pl["rows"] == [{"name": "A"}, {"name": "B"}]
    assert pl["columns"] == [{"name": "01"}, {"name": "02"}]
    assert pl["acquisitions"] == [{"id": 1, "name": "acq", "maximumfieldcount": 1}]
    assert pl["field_count"] == 1
    # 3 wells imaged out of a 2x2 = 4-well plate.
    assert len(pl["wells"]) == 3
    assert sorted(w["path"] for w in pl["wells"]) == ["A/01", "A/02", "B/01"]


# ---------- skip_reason: mosaic scene with vendor _Merged sibling ----------


def test_per_scene_skips_scene_with_skip_reason(tmp_path: Path, patched_reader) -> None:
    # Scene 0 advertises a skip_reason (e.g. its '_Merged' sibling is scene 1);
    # only scene 1 should be written, and a MosaicMergedSiblingWarning should fire.
    reader = FakeReader(
        scenes=["Position 1", "Position 1_Merged"],
        dims="TCYX",
        shape=(1, 1, 32, 32),
        skip_reasons={0: "vendor-merged sibling 'Position 1_Merged' is present"},
    )
    patched_reader(reader, plugin="bioio-fake")
    out = tmp_path / "out"

    with pytest.warns(MosaicMergedSiblingWarning, match="Position 1_Merged"):
        result = convert("/tmp/fake.lif", out, pyramid_min_size=32)

    assert result["layout"] == "per-scene"
    assert len(result["stores"]) == 1
    assert result["stores"][0]["scene_name"] == "Position 1_Merged"

    # The skipped scene's store directory must NOT exist.
    assert not (out / "Position_1.ome.zarr").exists()
    assert (out / "Position_1_Merged.ome.zarr").is_dir()


# ---------- mosaic audit surfaces tile positions + intended overlap --------


def test_per_scene_mosaic_tiles_round_trip_to_audit_attrs(
    tmp_path: Path, patched_reader
) -> None:
    """Issue #34 acceptance: a mosaic conversion's per_scene[0].mosaic carries
    the extracted ``tiles`` list and ``intended_overlap_*_pct`` values in the
    on-disk audit attrs (``attrs.zarrmony``).
    """
    mosaic_summary = {
        "stitched": True,
        "stitcher": "bioio-lif",
        "overlap_assumption_px": 1,
        "tile_count": 2,
        "tile_shape": {"Y": 512, "X": 512},
        "tiles": [
            {
                "field_x": 0,
                "field_y": 0,
                "pos_x_m": 0.04,
                "pos_y_m": 0.017,
                "pos_z_m": 0.0117,
            },
            {
                "field_x": 1,
                "field_y": 0,
                "pos_x_m": 0.0405,
                "pos_y_m": 0.017,
                "pos_z_m": 0.0117,
            },
        ],
        "intended_overlap_x_pct": 10.0,
        "intended_overlap_y_pct": 15.0,
    }
    reader = FakeReader(
        scenes=["mosaic"],
        dims="TCYX",
        shape=(1, 1, 64, 64),
        mosaic_summary=mosaic_summary,
    )
    patched_reader(reader, plugin="bioio-lif")
    out = tmp_path / "out"

    convert("/tmp/fake.lif", out, pyramid_min_size=8)

    with open(out / "mosaic.ome.zarr" / "zarr.json") as f:
        root = json.load(f)
    mosaic = root["attributes"]["zarrmony"]["per_scene"][0]["mosaic"]
    assert mosaic["intended_overlap_x_pct"] == 10.0
    assert mosaic["intended_overlap_y_pct"] == 15.0
    assert mosaic["tiles"] == mosaic_summary["tiles"]
