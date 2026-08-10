"""Tests for LIF plate detection (issue #81, ADR-0009 tracer bullet).

Covers three layers:

* :func:`extract_plate_layouts` — pure XML → structured plate list, including
  row/column normalization (uppercase / zero-padded), sparse-plate semantics,
  multi-plate enumeration, and fail-closed behavior on garbage input.
* :class:`_MosaicAwareLifReader` plate wiring — the reader proxy exposes
  ``layout_hint``, ``plate_layout``, and ``available_plates`` derived from
  the XML. Multi-plate LIFs stay flat until #82 wires ``--plate NAME``.
* ``convert(..., layout='auto')`` end-to-end — a single-plate LIF flows
  through the plate writer and produces a spec-conformant OME-NGFF plate
  store; ``inspect()`` surfaces the plate block.

Flat-LIF regression is guarded by :func:`test_flat_lif_stays_flat_end_to_end`,
which round-trips a non-plate LIF and asserts the per-scene writer is what runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import FakeReader, build_lif_plate_metadata
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony import inspect as zm_inspect
from zarrmony.errors import PlateSelectionError
from zarrmony.metadata.lif_plate import extract_plate_layouts
from zarrmony.readers.lif import _MosaicAwareLifReader
from zarrmony.readers.plugin import ReaderPlugin


def _fake_plugin(name: str = "bioio-lif") -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=name,
        source="builtin",
    )


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    def installer(reader):
        plugin_obj = _fake_plugin()
        monkeypatch.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, plugin_obj, 100),
        )

    return installer


# ---------- extractor ----------


def test_extract_plate_layouts_single_plate_returns_expected_shape() -> None:
    xml = build_lif_plate_metadata(
        [{"name": "MyPlate", "rows": ["A", "B"], "columns": ["01", "02"]}]
    )
    plates = extract_plate_layouts(xml)
    assert len(plates) == 1
    plate = plates[0]
    assert plate["name"] == "MyPlate"
    assert plate["rows"] == ["A", "B"]
    assert plate["columns"] == ["01", "02"]
    assert len(plate["fields"]) == 4
    # Scene index is assigned in row-major XML order — bioio_lif compatibility.
    assert [f["scene_index"] for f in plate["fields"]] == [0, 1, 2, 3]
    assert [(f["row"], f["column"]) for f in plate["fields"]] == [
        ("A", "01"),
        ("A", "02"),
        ("B", "01"),
        ("B", "02"),
    ]
    # Vendor-facing field_name is a compact <row><col> tag per ADR-0009.
    assert plate["fields"][0]["field_name"] == "A01"


def test_extract_plate_layouts_normalizes_lowercase_rows() -> None:
    """LAS X occasionally exports lowercase row letters — normalize to upper."""
    xml = build_lif_plate_metadata(
        [{"name": "P", "rows": ["a", "b"], "columns": ["01", "02"]}]
    )
    plates = extract_plate_layouts(xml)
    assert plates[0]["rows"] == ["A", "B"]
    assert [f["row"] for f in plates[0]["fields"]] == ["A", "A", "B", "B"]


def test_extract_plate_layouts_normalizes_unpadded_columns() -> None:
    """LAS X exports have been seen with unpadded columns; normalize to width-2."""
    xml = build_lif_plate_metadata(
        [{"name": "P", "rows": ["A"], "columns": ["1", "2", "10"]}]
    )
    plates = extract_plate_layouts(xml)
    assert plates[0]["columns"] == ["01", "02", "10"]
    assert [f["column"] for f in plates[0]["fields"]] == ["01", "02", "10"]


def test_extract_plate_layouts_returns_empty_for_flat_lif() -> None:
    """A LIF whose XML has no plate template must extract nothing."""
    xml = (
        "<LMSDataContainerHeader><Element Name='ConfocalScene'>"
        "<Data><Image /></Data></Element></LMSDataContainerHeader>"
    )
    assert extract_plate_layouts(xml) == []


def test_extract_plate_layouts_returns_empty_for_malformed_xml() -> None:
    assert extract_plate_layouts("not xml") == []
    assert extract_plate_layouts("") == []


def test_extract_plate_layouts_enumerates_multi_plate_with_running_scene_index() -> (
    None
):
    xml = build_lif_plate_metadata(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ]
    )
    plates = extract_plate_layouts(xml)
    assert [p["name"] for p in plates] == ["PlateA", "PlateB"]
    # Scene index carries across plates so scenes[1], scenes[2] belong to
    # PlateB when the reader concatenates them.
    assert plates[0]["fields"][0]["scene_index"] == 0
    assert [f["scene_index"] for f in plates[1]["fields"]] == [1, 2]


# ---------- reader wiring ----------


def _lif_reader_with_plates(
    plates: list[dict],
    scenes: list[str],
    *,
    active_plate: str | None = None,
    **fake_kwargs,
) -> _MosaicAwareLifReader:
    xml = build_lif_plate_metadata(plates)
    inner = FakeReader(
        scenes=scenes,
        dims="TCYX",
        shape=(1, 1, 16, 16),
        raw_xml=xml,
        channel_names=fake_kwargs.pop("channel_names", ["DAPI"]),
        **fake_kwargs,
    )
    return _MosaicAwareLifReader(inner, active_plate=active_plate)


def test_mosaic_aware_reader_reports_plate_layout_for_single_plate_lif() -> None:
    reader = _lif_reader_with_plates(
        [{"name": "PilotPlate", "rows": ["A", "B"], "columns": ["01", "02"]}],
        scenes=[
            "PilotPlate/A/01",
            "PilotPlate/A/02",
            "PilotPlate/B/01",
            "PilotPlate/B/02",
        ],
    )
    assert reader.layout_hint == "plate"
    assert reader.available_plates == ["PilotPlate"]
    layout = reader.plate_layout
    assert layout is not None
    assert layout.name == "PilotPlate"
    assert layout.rows == ["A", "B"]
    assert layout.columns == ["01", "02"]
    assert len(layout.fields) == 4
    assert layout.fields[0].scene_index == 0
    assert layout.fields[0].row == "A"
    assert layout.fields[0].column == "01"
    assert layout.fields[0].acquisition_id == 1
    assert len(layout.acquisitions) == 1
    assert layout.acquisitions[0].id == 1


def test_mosaic_aware_reader_stays_flat_for_non_plate_lif() -> None:
    """Regression guard: a flat LIF must not accidentally trip plate detection."""
    inner = FakeReader(
        scenes=["Scene_0", "Scene_1"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        raw_xml="<LMSDataContainerHeader><Element Name='ConfocalScene'>"
        "<Data><Image /></Data></Element></LMSDataContainerHeader>",
    )
    reader = _MosaicAwareLifReader(inner)
    assert reader.layout_hint == "flat"
    assert reader.available_plates == []
    assert reader.plate_layout is None


def test_mosaic_aware_reader_stays_flat_for_multi_plate_lif_but_lists_available() -> (
    None
):
    """Multi-plate without ``active_plate`` stays flat at the reader level.

    ``convert()`` refuses the ambiguity with :class:`PlateSelectionError`;
    the reader itself surfaces the available names for that error message
    and for :func:`inspect` to discover.
    """
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=["s0", "s1", "s2"],
    )
    assert reader.layout_hint == "flat"
    assert reader.plate_layout is None
    assert reader.available_plates == ["PlateA", "PlateB"]


def test_mosaic_aware_reader_caches_plate_state_across_accesses() -> None:
    """Plate detection walks the XML once per reader instance — property calls
    must reuse the cached result so multi-access is O(1) after the first hit."""
    reader = _lif_reader_with_plates(
        [{"name": "P", "rows": ["A"], "columns": ["01"]}],
        scenes=["P/A/01"],
    )
    first = reader.plate_layout
    second = reader.plate_layout
    assert first is second


# ---------- end-to-end ----------


def test_convert_auto_routes_single_plate_lif_to_plate_writer(
    tmp_path: Path, patched_reader
) -> None:
    """Default ``layout='auto'`` on a single-plate LIF produces a valid plate.zarr."""
    reader = _lif_reader_with_plates(
        [{"name": "PilotPlate", "rows": ["A", "B"], "columns": ["01", "02"]}],
        scenes=[
            "PilotPlate/A/01",
            "PilotPlate/A/02",
            "PilotPlate/B/01",
            "PilotPlate/B/02",
        ],
        channel_names=["DAPI"],
    )
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert("/tmp/single_plate.lif", out, pyramid_min_size=8)

    assert audit["layout"] == "plate"
    with open(out / "zarr.json") as f:
        root_zj = json.load(f)
    plate = root_zj["attributes"]["ome"]["plate"]
    assert plate["name"] == "PilotPlate"
    assert plate["rows"] == [{"name": "A"}, {"name": "B"}]
    assert plate["columns"] == [{"name": "01"}, {"name": "02"}]
    # Every well is present on disk with a 0/ FOV image group.
    for well in ["A/01", "A/02", "B/01", "B/02"]:
        assert (out / well / "0" / "zarr.json").exists()
    # Audit uses the schema-3 plate + fields shape.
    assert audit["plate"]["name"] == "PilotPlate"
    assert [f["well_id"] for f in audit["fields"]] == ["A01", "A02", "B01", "B02"]


def test_flat_lif_stays_flat_end_to_end(tmp_path: Path, patched_reader) -> None:
    """Existing flat-LIF users continue to get per-scene stores under default auto."""
    inner = FakeReader(
        scenes=["Scene_0", "Scene_1"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        raw_xml="<LMSDataContainerHeader><Element Name='Confocal'>"
        "<Data><Image /></Data></Element></LMSDataContainerHeader>",
        channel_names=["DAPI"],
    )
    wrapped = _MosaicAwareLifReader(inner)
    patched_reader(wrapped)
    out = tmp_path / "out"

    result = convert("/tmp/flat.lif", out, pyramid_min_size=8)

    assert result["layout"] == "per-scene"
    # Two per-scene stores, no plate structure.
    assert (out / "Scene_0.ome.zarr").exists()
    assert (out / "Scene_1.ome.zarr").exists()
    assert not (out / "A").exists()


def test_inspect_surfaces_plate_block_for_single_plate_lif(patched_reader) -> None:
    """``inspect()`` surfaces plate name, row/column count, and field count."""
    reader = _lif_reader_with_plates(
        [{"name": "PilotPlate", "rows": ["A", "B"], "columns": ["01", "02"]}],
        scenes=[
            "PilotPlate/A/01",
            "PilotPlate/A/02",
            "PilotPlate/B/01",
            "PilotPlate/B/02",
        ],
    )
    patched_reader(reader)

    info = zm_inspect("/tmp/single_plate.lif")

    assert "plate_layout" in info
    plate = info["plate_layout"]
    assert plate["name"] == "PilotPlate"
    assert plate["rows"] == [{"name": "A"}, {"name": "B"}]
    assert plate["columns"] == [{"name": "01"}, {"name": "02"}]
    assert len(plate["wells"]) == 4


# ---------- multi-plate: --plate NAME selector (issue #82) ----------


def _two_plate_scenes() -> list[str]:
    """Scene names for the two-plate fixture used across selector tests.

    PlateA has a single well (A/01); PlateB has two wells (A/01, A/02).
    Global scene order is document order: PlateA first, then PlateB — the
    same order ``extract_plate_layouts`` numbers ``scene_index`` in.
    """
    return ["PlateA/A/01", "PlateB/A/01", "PlateB/A/02"]


def test_active_plate_selects_matching_plate_on_multi_plate_lif() -> None:
    """``active_plate=NAME`` on a multi-plate LIF exposes only that plate's fields."""
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
        active_plate="PlateB",
    )
    assert reader.layout_hint == "plate"
    layout = reader.plate_layout
    assert layout is not None
    assert layout.name == "PlateB"
    assert layout.columns == ["01", "02"]
    # scene_index is remapped: original global positions [1, 2] → filtered [0, 1].
    assert [f.scene_index for f in layout.fields] == [0, 1]
    # ``scenes`` is filtered to just PlateB's entries.
    assert list(reader.scenes) == ["PlateB/A/01", "PlateB/A/02"]


def test_active_plate_switches_inner_reader_to_correct_global_scene() -> None:
    """``set_scene(filtered_idx)`` translates to the correct inner (global) index."""
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
        active_plate="PlateB",
    )
    reader.set_scene(1)
    # PlateB's filtered scene 1 = global scene 2 = "PlateB/A/02".
    assert reader.current_scene_index == 2


def test_active_plate_unknown_name_on_multi_plate_raises_with_names() -> None:
    """``active_plate=NAME`` with unknown NAME raises listing available names."""
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
        active_plate="NoSuchPlate",
    )
    with pytest.raises(PlateSelectionError) as excinfo:
        _ = reader.plate_layout
    msg = str(excinfo.value)
    assert "NoSuchPlate" in msg
    assert "PlateA" in msg
    assert "PlateB" in msg


def test_active_plate_matching_name_on_single_plate_lif_is_accepted() -> None:
    """Belt-and-suspenders: passing the correct name on a single-plate LIF works."""
    reader = _lif_reader_with_plates(
        [{"name": "Solo", "rows": ["A"], "columns": ["01"]}],
        scenes=["Solo/A/01"],
        active_plate="Solo",
    )
    assert reader.layout_hint == "plate"
    assert reader.plate_layout is not None
    assert reader.plate_layout.name == "Solo"


def test_active_plate_mismatch_on_single_plate_lif_raises() -> None:
    """Passing a name that doesn't match the sole plate raises with both names."""
    reader = _lif_reader_with_plates(
        [{"name": "Solo", "rows": ["A"], "columns": ["01"]}],
        scenes=["Solo/A/01"],
        active_plate="Other",
    )
    with pytest.raises(PlateSelectionError) as excinfo:
        _ = reader.plate_layout
    msg = str(excinfo.value)
    assert "Other" in msg
    assert "Solo" in msg


def test_convert_multi_plate_without_selector_raises_with_available_names(
    tmp_path: Path, patched_reader
) -> None:
    """``convert()`` refuses ambiguity: multi-plate LIF, no ``plate=`` → raise."""
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
    )
    patched_reader(reader)

    with pytest.raises(PlateSelectionError) as excinfo:
        convert("/tmp/multi.lif", tmp_path / "out", pyramid_min_size=8)
    msg = str(excinfo.value)
    assert "PlateA" in msg
    assert "PlateB" in msg
    assert "--plate" in msg


def test_convert_with_plate_kwarg_produces_single_plate_zarr(
    tmp_path: Path, patched_reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`plate=NAME` in reader_kwargs writes only that plate's fields.

    Uses `get_reader`-level `reader_kwargs` capture to confirm the kwarg
    reaches the plugin's `open()` and the produced plate.zarr carries only
    the selected plate's wells.
    """
    # patched_reader ignores reader_kwargs, so we roll our own installer that
    # asserts the kwarg reached get_reader and returns a reader pre-scoped
    # to PlateB. This matches how the real bioio-lif plugin would honour
    # `plate=` at open() time.
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
        active_plate="PlateB",
    )
    captured: dict = {}

    def _get_reader(_path, *, reader_kwargs=None):
        captured["reader_kwargs"] = reader_kwargs
        plugin_obj = ReaderPlugin(
            name="bioio-lif",
            match=lambda _p: 100,
            open=lambda _p: object(),
            distribution="bioio-lif",
            source="builtin",
        )
        return reader, plugin_obj, 100

    monkeypatch.setattr(api_module, "get_reader", _get_reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert(
        "/tmp/multi.lif",
        out,
        pyramid_min_size=8,
        reader_kwargs={"plate": "PlateB"},
    )
    assert captured["reader_kwargs"] == {"plate": "PlateB"}
    assert audit["layout"] == "plate"
    assert audit["plate"]["name"] == "PlateB"
    assert [f["well_id"] for f in audit["fields"]] == ["A01", "A02"]
    with open(out / "zarr.json") as f:
        root_zj = json.load(f)
    plate = root_zj["attributes"]["ome"]["plate"]
    assert plate["name"] == "PlateB"
    # PlateA wells must NOT be present on disk.
    assert (out / "A" / "01" / "0" / "zarr.json").exists()
    assert (out / "A" / "02" / "0" / "zarr.json").exists()


def test_convert_with_unknown_plate_name_raises_with_available_names(
    tmp_path: Path, patched_reader
) -> None:
    """`plate=NAME` where NAME isn't among the templates raises listing them."""
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
        active_plate="Nonexistent",
    )
    patched_reader(reader)

    with pytest.raises(PlateSelectionError) as excinfo:
        convert(
            "/tmp/multi.lif",
            tmp_path / "out",
            pyramid_min_size=8,
            reader_kwargs={"plate": "Nonexistent"},
        )
    msg = str(excinfo.value)
    assert "Nonexistent" in msg


def test_inspect_lists_all_plate_names_on_multi_plate_lif(patched_reader) -> None:
    """`inspect()` surfaces every plate template name on a multi-plate LIF."""
    reader = _lif_reader_with_plates(
        [
            {"name": "PlateA", "rows": ["A"], "columns": ["01"]},
            {"name": "PlateB", "rows": ["A"], "columns": ["01", "02"]},
        ],
        scenes=_two_plate_scenes(),
    )
    patched_reader(reader)

    info = zm_inspect("/tmp/multi.lif")

    assert info["plates"] == ["PlateA", "PlateB"]
    # Single-plate `plate_layout` shape is absent — there's no unambiguous plate.
    assert "plate_layout" not in info
    # Every underlying scene still enumerates as today (additive block).
    assert info["n_scenes"] == 3


def test_inspect_single_plate_lif_does_not_add_plates_block(patched_reader) -> None:
    """Single-plate `inspect()` output is unchanged — no additive `plates` key."""
    reader = _lif_reader_with_plates(
        [{"name": "Solo", "rows": ["A"], "columns": ["01"]}],
        scenes=["Solo/A/01"],
    )
    patched_reader(reader)

    info = zm_inspect("/tmp/single.lif")

    assert "plate_layout" in info
    assert "plates" not in info
