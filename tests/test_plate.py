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
    resolve_per_well_metadata,
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
        monkeypatch.setattr(api_module, "get_reader", lambda _path: (reader, plugin_obj, 100))

    return installer


def _synthetic_2x2_layout() -> PlateLayout:
    """A 2x2 plate (rows A,B; columns 01,02), one FOV per well, scenes 0..3."""
    return PlateLayout(
        name="synthetic-2x2",
        rows=["A", "B"],
        columns=["01", "02"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(scene_index=0, row="A", column="01", field_name="A01-f0", acquisition_id=1),
            PlateField(scene_index=1, row="A", column="02", field_name="A02-f0", acquisition_id=1),
            PlateField(scene_index=2, row="B", column="01", field_name="B01-f0", acquisition_id=1),
            PlateField(scene_index=3, row="B", column="02", field_name="B02-f0", acquisition_id=1),
        ],
    )


def _synthetic_plate_reader() -> FakeReader:
    return FakeReader(
        scenes=["s0", "s1", "s2", "s3"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=_synthetic_2x2_layout(),
    )


# ---------- end-to-end ----------


def test_plate_end_to_end_writes_spec_conformant_store(tmp_path: Path, patched_reader) -> None:
    reader = _synthetic_plate_reader()
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert(
        "/tmp/fake.czi",
        out,
        layout="plate",
        metadata={"microscope": "Axioscan", "modality": "fluorescence"},
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
    assert "per_scene" not in audit
    assert len(audit["fields"]) == 4
    assert audit["plate"]["name"] == "synthetic-2x2"
    assert audit["user_metadata"]["microscope"] == "Axioscan"

    # First field record carries the per-FOV plate context.
    f0 = audit["fields"][0]
    assert f0["row"] == "A"
    assert f0["column"] == "01"
    assert f0["field_path"] == "A/01/0"
    assert f0["field_name"] == "A01-f0"
    assert f0["acquisition_id"] == 1

    # Audit is also persisted at the plate root.
    assert root_zj["attributes"]["zarrmony"] == audit


def test_plate_rejects_per_scene_metadata(tmp_path: Path, patched_reader) -> None:
    reader = _synthetic_plate_reader()
    patched_reader(reader)
    with pytest.raises(ValueError, match="per_well_metadata"):
        convert(
            "/tmp/fake.czi",
            tmp_path / "out.ome.zarr",
            layout="plate",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            per_scene_metadata={"s0": {"microscope": "x", "modality": "y"}},
        )


def test_plate_against_flat_reader_raises_layout_mismatch(tmp_path: Path, patched_reader) -> None:
    """Forcing layout='plate' against a flat reader is a dispatch-matrix error."""
    flat_reader = FakeReader(scenes=["s0"], dims="TCYX", shape=(1, 1, 16, 16))
    patched_reader(flat_reader)
    with pytest.raises(LayoutMismatchError, match="layout_hint='flat'"):
        convert(
            "/tmp/fake.czi",
            tmp_path / "out.ome.zarr",
            layout="plate",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        )


def test_plate_reader_missing_plate_layout_raises(tmp_path: Path, patched_reader) -> None:
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
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
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


# ---------- per_well_metadata: key parsing ----------


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


def test_resolve_per_well_metadata_rejects_lowercase() -> None:
    """Casing must match the plate's canonical row spelling."""
    layout = _synthetic_2x2_layout()
    with pytest.raises(ValueError, match="'b01'"):
        resolve_per_well_metadata({"b01": {}}, layout)


def test_resolve_per_well_metadata_rejects_padding_mismatch() -> None:
    """Zero-padding from the plate is the source of truth — 'B1' != 'B01'."""
    layout = _synthetic_2x2_layout()  # columns=["01", "02"]
    with pytest.raises(ValueError, match="'A1'"):
        resolve_per_well_metadata({"A1": {}}, layout)


def test_resolve_per_well_metadata_rejects_unknown_well() -> None:
    layout = _synthetic_2x2_layout()  # rows=["A","B"], columns=["01","02"]
    with pytest.raises(ValueError, match="'C03'"):
        resolve_per_well_metadata({"C03": {}}, layout)


def test_resolve_per_well_metadata_returns_row_col_tuples() -> None:
    layout = _synthetic_2x2_layout()
    resolved = resolve_per_well_metadata({"B02": {"k": "v"}}, layout)
    assert resolved == {("B", "02"): {"k": "v"}}


# ---------- per_well_metadata: end-to-end persistence ----------


def test_per_well_metadata_round_trip(tmp_path: Path, patched_reader) -> None:
    """Override on B02 lands on disk + audit; A01 has no zarrmony attrs."""
    reader = _synthetic_plate_reader()
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"

    audit = convert(
        "/tmp/fake.czi",
        out,
        layout="plate",
        metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        per_well_metadata={
            "B02": {"microscope": "B02-scope", "modality": "B02-mode", "study": "treated"}
        },
        pyramid_min_size=8,
    )

    # On disk: B02's well group carries attrs.zarrmony.user_metadata.
    with open(out / "B" / "02" / "zarr.json") as f:
        b02_zj = json.load(f)
    assert b02_zj["attributes"]["zarrmony"]["user_metadata"]["microscope"] == "B02-scope"
    assert b02_zj["attributes"]["zarrmony"]["user_metadata"]["study"] == "treated"
    # OME well block is unchanged (spec-clean).
    assert b02_zj["attributes"]["ome"]["well"]["images"] == [{"path": "0", "acquisition": 1}]

    # Wells without an override do not get a zarrmony attrs block.
    with open(out / "A" / "01" / "zarr.json") as f:
        a01_zj = json.load(f)
    assert "zarrmony" not in a01_zj["attributes"]

    # Audit's plate.wells[i] carries user_metadata only for the overridden well.
    audit_wells = {w["path"]: w for w in audit["plate"]["wells"]}
    assert audit_wells["B/02"]["user_metadata"]["microscope"] == "B02-scope"
    assert "user_metadata" not in audit_wells["A/01"]

    # On-disk attrs.ome.plate stays spec-clean (no user_metadata leak).
    with open(out / "zarr.json") as f:
        root_zj = json.load(f)
    on_disk_wells = {w["path"]: w for w in root_zj["attributes"]["ome"]["plate"]["wells"]}
    assert "user_metadata" not in on_disk_wells["B/02"]


def test_per_well_metadata_unknown_well_raises_before_writing(
    tmp_path: Path, patched_reader
) -> None:
    reader = _synthetic_plate_reader()
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"
    with pytest.raises(ValueError, match="'Z99'"):
        convert(
            "/tmp/fake.czi",
            out,
            layout="plate",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            per_well_metadata={"Z99": {"microscope": "x", "modality": "y"}},
        )
    assert not out.exists()


def test_per_well_metadata_rejected_outside_plate_mode(tmp_path: Path, patched_reader) -> None:
    flat_reader = FakeReader(scenes=["s0"], dims="TCYX", shape=(1, 1, 16, 16))
    patched_reader(flat_reader)
    with pytest.raises(ValueError, match="per_well_metadata is only supported in plate mode"):
        convert(
            "/tmp/fake.czi",
            tmp_path / "out.ome.zarr",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            per_well_metadata={"A01": {"microscope": "x", "modality": "y"}},
        )
