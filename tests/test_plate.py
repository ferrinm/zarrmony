"""End-to-end tests for ``layout='plate'`` (the OME-NGFF HCS writer).

Covers the slice-1 tracer bullet from issue #8:

- A synthetic 2x2 plate flows through ``convert(layout='plate')`` and lands as
  a spec-conformant OME-NGFF 0.5 plate store on disk.
- ``PlateLayoutError`` rejects internally-inconsistent ``PlateLayout`` inputs
  before any pixels are written.
- The audit record uses the new schema-3 ``fields`` + ``plate`` shape (no
  ``per_scene``) and is reachable both as the ``convert()`` return value and
  via ``<plate>/zarr.json#attrs.zarrmony``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zarr
from ome_types import from_xml

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.audit import AUDIT_SCHEMA_VERSION
from zarrmony.errors import LayoutMismatchError, PlateLayoutError
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin
from zarrmony.writers.plate import (
    parse_well_key,
    validate_plate_layout,
    write_plate,
)


def _fake_plugin(name: str = "bioio-fake-plate") -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=name,
        source="builtin",
    )


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    def installer(reader: FakeReader, plugin: str = "bioio-fake-plate"):
        plugin_obj = _fake_plugin(plugin)
        monkeypatch.setattr(
            api_module, "get_reader", lambda _path: (reader, plugin_obj, 100)
        )

    return installer


def _synthetic_2x2_layout() -> PlateLayout:
    """A 2x2 plate (rows A,B; columns 01,02), one FOV per well, scenes 0..3."""
    return PlateLayout(
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
            PlateField(
                scene_index=3,
                row="B",
                column="02",
                field_name="B02-f0",
                acquisition_id=1,
            ),
        ],
    )


def _synthetic_plate_reader() -> FakeReader:
    return FakeReader(
        scenes=["s0", "s1", "s2", "s3"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=_synthetic_2x2_layout(),
        channel_names=["DAPI"],
    )


# ---------- end-to-end ----------


def test_plate_end_to_end_writes_spec_conformant_store(
    tmp_path: Path, patched_reader
) -> None:
    reader = _synthetic_plate_reader()
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert(
        "/tmp/fake.czi",
        out,
        layout="plate",
        pyramid_min_size=8,
    )

    # ---- plate-root attrs.ome.plate matches expected dict ----
    with open(out / "zarr.json") as f:
        root_zj = json.load(f)
    ome_block = root_zj["attributes"]["ome"]
    assert ome_block["version"] == "0.5"
    # No bf2raw marker on plate stores (ADR-0004 rejected option).
    assert "bioformats2raw.layout" not in ome_block

    plate = ome_block["plate"]
    assert plate["name"] == "synthetic-2x2"
    assert plate["rows"] == [{"name": "A"}, {"name": "B"}]
    assert plate["columns"] == [{"name": "01"}, {"name": "02"}]
    assert plate["acquisitions"] == [{"id": 1, "name": "acq", "maximumfieldcount": 1}]
    assert plate["field_count"] == 1
    assert sorted(plate["wells"], key=lambda w: w["path"]) == [
        {"path": "A/01", "rowIndex": 0, "columnIndex": 0},
        {"path": "A/02", "rowIndex": 0, "columnIndex": 1},
        {"path": "B/01", "rowIndex": 1, "columnIndex": 0},
        {"path": "B/02", "rowIndex": 1, "columnIndex": 1},
    ]

    # ---- well-group attrs match the spec shape ----
    for well in ["A/01", "A/02", "B/01", "B/02"]:
        with open(out / well / "zarr.json") as f:
            well_zj = json.load(f)
        well_attr = well_zj["attributes"]["ome"]["well"]
        assert well_attr["version"] == "0.5"
        assert well_attr["images"] == [{"path": "0", "acquisition": 1}]

    # ---- every FOV is a valid multiscales image ----
    for well in ["A/01", "A/02", "B/01", "B/02"]:
        fov = out / well / "0"
        g = zarr.open_group(str(fov), mode="r")
        assert g["0"].shape == (1, 1, 16, 16)
        # Each FOV's image_name comes from PlateField.field_name.
        with open(fov / "zarr.json") as f:
            fov_zj = json.load(f)
        ms = fov_zj["attributes"]["ome"]["multiscales"][0]
        # The vendor field name lands in multiscales[0].name (not the path).
        assert ms["name"].startswith(well.replace("/", ""))

    # ---- combined OME/METADATA.ome.xml at plate root only ----
    xml = (out / "OME" / "METADATA.ome.xml").read_text()
    parsed = from_xml(xml)
    assert len(parsed.images) == 4
    # ---- vendor source XML once at plate root ----
    assert (out / "OME" / "source" / "raw.czi.xml").exists()

    # ---- audit (return value == on-disk attrs.zarrmony) ----
    assert audit["audit_schema_version"] == AUDIT_SCHEMA_VERSION
    assert audit["layout"] == "plate"
    assert audit["output"] == {"ome_ngff_version": "0.5"}
    assert "per_scene" not in audit
    assert len(audit["fields"]) == 4
    assert audit["plate"]["name"] == "synthetic-2x2"

    # First field record carries the per-FOV plate context.
    f0 = audit["fields"][0]
    assert f0["row"] == "A"
    assert f0["column"] == "01"
    assert f0["well_id"] == "A01"
    assert f0["field_path"] == "A/01/0"
    assert f0["field_name"] == "A01-f0"
    assert f0["acquisition_id"] == 1
    # Every field carries well_id in <row-letter><col-number> format (#66).
    assert [f["well_id"] for f in audit["fields"]] == ["A01", "A02", "B01", "B02"]
    # ADR-0008 / #61: per-field channels block. FakeReader exposes one DAPI
    # channel per scene; every field carries the same shape via the OME
    # projection.
    for field in audit["fields"]:
        assert "channels" in field
        assert len(field["channels"]) == 1
        assert field["channels"][0]["index"] == 0
        assert field["channels"][0]["name"] == "DAPI"
    # The synthetic layout has no plate_id — key must be absent, not None.
    assert "plate_id" not in audit["plate"]
    # ...and the NGFF on-disk plate attr never carries plate_id.
    assert "plate_id" not in plate

    # Audit is also persisted at the plate root.
    assert root_zj["attributes"]["zarrmony"] == audit


def test_plate_id_surfaces_in_audit_and_inspect_when_reader_supplies_it(
    tmp_path: Path, patched_reader
) -> None:
    """When PlateLayout carries plate_id, it lands in audit.plate.plate_id and
    inspect().plate_layout.plate_id. The on-disk NGFF plate attr never
    carries it (not a spec key). Covers ADR-0008 / #66."""
    layout = PlateLayout(
        name="synthetic-2x2",
        rows=["A", "B"],
        columns=["01", "02"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(scene_index=0, row="A", column="01", acquisition_id=1),
            PlateField(scene_index=1, row="A", column="02", acquisition_id=1),
            PlateField(scene_index=2, row="B", column="01", acquisition_id=1),
            PlateField(scene_index=3, row="B", column="02", acquisition_id=1),
        ],
        plate_id="Plate-BARCODE-123",
    )
    reader = FakeReader(
        scenes=["s0", "s1", "s2", "s3"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=layout,
        channel_names=["DAPI"],
    )
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert("/tmp/fake.czi", out, layout="plate", pyramid_min_size=8)

    assert audit["plate"]["plate_id"] == "Plate-BARCODE-123"

    with open(out / "zarr.json") as f:
        root_zj = json.load(f)
    # Not stamped into the OME-NGFF plate attr — audit-only surface.
    assert "plate_id" not in root_zj["attributes"]["ome"]["plate"]

    from zarrmony import inspect as zm_inspect

    info = zm_inspect("/tmp/fake.czi")
    assert info["plate_layout"]["plate_id"] == "Plate-BARCODE-123"


def test_plate_against_flat_reader_raises_layout_mismatch(
    tmp_path: Path, patched_reader
) -> None:
    """Forcing layout='plate' against a flat reader is a dispatch-matrix error."""
    flat_reader = FakeReader(scenes=["s0"], dims="TCYX", shape=(1, 1, 16, 16))
    patched_reader(flat_reader)
    with pytest.raises(LayoutMismatchError, match="layout_hint='flat'"):
        convert(
            "/tmp/fake.czi",
            tmp_path / "out.ome.zarr",
            layout="plate",
        )


def test_plate_reader_missing_plate_layout_raises(
    tmp_path: Path, patched_reader
) -> None:
    """Defensive: a reader claiming layout_hint='plate' but with plate_layout=None."""
    misconfigured = FakeReader(
        scenes=["s0"], dims="TCYX", shape=(1, 1, 16, 16), layout_hint="plate"
    )
    patched_reader(misconfigured)
    with pytest.raises(Exception, match="reader.plate_layout"):
        convert(
            "/tmp/fake.czi",
            tmp_path / "out.ome.zarr",
            layout="plate",
        )


# ---------- validation (PlateLayoutError) ----------


def test_validate_rejects_unknown_row() -> None:
    layout = PlateLayout(
        name="bad",
        rows=["A"],
        columns=["01"],
        fields=[PlateField(scene_index=0, row="Z", column="01")],
    )
    with pytest.raises(PlateLayoutError, match="unknown row"):
        validate_plate_layout(layout, n_scenes=1)


def test_validate_rejects_unknown_column() -> None:
    layout = PlateLayout(
        name="bad",
        rows=["A"],
        columns=["01"],
        fields=[PlateField(scene_index=0, row="A", column="99")],
    )
    with pytest.raises(PlateLayoutError, match="unknown column"):
        validate_plate_layout(layout, n_scenes=1)


def test_validate_rejects_duplicate_well_path() -> None:
    """Two fields claiming the same scene_index would silently double-write."""
    layout = PlateLayout(
        name="bad",
        rows=["A"],
        columns=["01", "02"],
        fields=[
            PlateField(scene_index=0, row="A", column="01"),
            PlateField(scene_index=0, row="A", column="02"),
        ],
    )
    with pytest.raises(PlateLayoutError, match="duplicate well path"):
        validate_plate_layout(layout, n_scenes=1)


def test_validate_rejects_multi_acquisition() -> None:
    layout = PlateLayout(
        name="bad",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1), Acquisition(id=2)],
        fields=[],
    )
    with pytest.raises(PlateLayoutError, match="at most 1 acquisition"):
        validate_plate_layout(layout, n_scenes=0)


def test_validate_rejects_scene_index_out_of_range() -> None:
    layout = PlateLayout(
        name="bad",
        rows=["A"],
        columns=["01"],
        fields=[PlateField(scene_index=5, row="A", column="01")],
    )
    with pytest.raises(PlateLayoutError, match="out of range"):
        validate_plate_layout(layout, n_scenes=2)


def test_writer_validates_before_any_pixel_write(tmp_path: Path) -> None:
    """A bad layout MUST NOT leave a partial store on disk."""
    out = tmp_path / "should-not-exist.ome.zarr"
    bad_layout = PlateLayout(
        name="bad",
        rows=["A"],
        columns=["01"],
        fields=[PlateField(scene_index=0, row="Z", column="01")],
    )
    reader = FakeReader(scenes=["s0"], dims="TCYX", shape=(1, 1, 16, 16))
    with pytest.raises(PlateLayoutError):
        write_plate(
            reader,
            store_path=str(out),
            plate_layout=bad_layout,
            pyramid_min_size=8,
        )
    assert not out.exists()


# ---------- well key parsing ----------


def test_parse_well_key_single_letter_row() -> None:
    assert parse_well_key("B04") == ("B", "04")


def test_parse_well_key_double_letter_row() -> None:
    """1536-well plates extend rows past Z (AA, AB, ...)."""
    assert parse_well_key("AA01") == ("AA", "01")
    assert parse_well_key("AF24") == ("AF", "24")


def test_parse_well_key_rejects_non_alpha_digit_shape() -> None:
    with pytest.raises(ValueError, match="not a valid alpha\\+digit coordinate"):
        parse_well_key("B-04")
    with pytest.raises(ValueError, match="not a valid alpha\\+digit coordinate"):
        parse_well_key("04")
    with pytest.raises(ValueError, match="not a valid alpha\\+digit coordinate"):
        parse_well_key("B")
    with pytest.raises(ValueError, match="not a valid alpha\\+digit coordinate"):
        parse_well_key("")
