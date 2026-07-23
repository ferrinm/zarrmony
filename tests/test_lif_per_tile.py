"""End-to-end tests for the LIF per-tile mosaic write path (ADR-0005 / #36).

Drives ``convert(..., lif_mosaic="per-tile")`` against FakeReader configured to
look like a per-tile-eligible LIF mosaic scene (M-intact xarray + a
LIF-shaped metadata blob with ``<Tile>`` entries the extractor picks up).
Covers:

- on-disk shape: ``<output>/<scene>/tile_X{f:02d}Y{f:02d}.ome.zarr/`` sub-stores
- scene-level directory is a PLAIN directory (no ``zarr.json``)
- per-tile pixel data lands at the right tile (FakeReader fills tile m with m+1)
- per-tile OME-XML ``<Plane>`` carries ``PositionX/Y/Z`` in micrometers
- audit ``mosaic.per_tile=true`` + ``mosaic.tile_stores=[...]`` + schema-5
- plate + per-tile → ``LayoutMismatchError``
- auto-stitch (default) is unaffected; ``_Merged``-sibling skip still works
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import zarr
from ome_types import from_xml

from tests.conftest import FakeReader, TileScene
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.audit import AUDIT_SCHEMA_VERSION
from zarrmony.errors import (
    LayoutMismatchError,
    MosaicMergedSiblingWarning,
    MosaicStitchingWarning,
)
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
    def installer(reader: FakeReader, plugin: str = "bioio-lif"):
        plugin_obj = _fake_plugin(plugin)
        monkeypatch.setattr(
            api_module, "get_reader", lambda _path: (reader, plugin_obj, 100)
        )

    return installer


def _three_by_one_tiles() -> list[dict]:
    """A simple 3x1 mosaic with 10% overlap stage positions in meters."""
    return [
        {
            "field_x": 0,
            "field_y": 0,
            "pos_x_m": 0.04000,
            "pos_y_m": 0.01700,
            "pos_z_m": 0.01170,
        },
        {
            "field_x": 1,
            "field_y": 0,
            "pos_x_m": 0.04050,
            "pos_y_m": 0.01700,
            "pos_z_m": 0.01170,
        },
        {
            "field_x": 2,
            "field_y": 0,
            "pos_x_m": 0.04100,
            "pos_y_m": 0.01700,
            "pos_z_m": 0.01170,
        },
    ]


def _make_mosaic_reader(scene_name: str = "Position 1") -> FakeReader:
    scene = TileScene(tiles=_three_by_one_tiles(), tile_yx=(32, 32))
    return FakeReader(
        scenes=[scene_name],
        dims="TCZYX",
        shape=(1, 1, 1, 32, 32),
        channel_names=["DAPI"],
        per_tile_scenes={0: scene},
    )


# ---------- convert() signature + validation ----------


def test_invalid_lif_mosaic_value_rejected(tmp_path: Path, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    with pytest.raises(ValueError, match="lif_mosaic"):
        convert("/tmp/x.lif", tmp_path / "out", lif_mosaic="bogus")


# ---------- per-tile on-disk shape ----------


def test_per_tile_writes_one_substore_per_tile_under_scene_dir(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    scene_dir = out / "Position_1"
    assert scene_dir.is_dir(), "scene-level group should be a plain directory"
    # NOT a zarr group: no zarr.json at the scene-named directory itself
    assert not (scene_dir / "zarr.json").exists()

    # Three tile sub-stores at the expected (zero-padded) coords
    for fx in range(3):
        tile_store = scene_dir / f"tile_X{fx:02d}Y00.ome.zarr"
        assert tile_store.is_dir(), tile_store
        assert (
            tile_store / "zarr.json"
        ).exists(), f"tile sub-store {tile_store} is not a zarr group"
        assert (tile_store / "OME" / "METADATA.ome.xml").exists()

    # result.stores has one audit per tile
    assert result["layout"] == "per-scene"
    assert len(result["stores"]) == 3


def test_per_tile_pixel_data_lands_at_the_right_tile(
    tmp_path: Path, patched_reader
) -> None:
    """FakeReader fills tile m with ``m + 1``; assert each tile's array reads
    back as that constant so the M-slice → store mapping is correct."""
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    for m, fx in enumerate(range(3)):
        g = zarr.open_group(
            str(out / "Position_1" / f"tile_X{fx:02d}Y00.ome.zarr"), mode="r"
        )
        arr = g["0"][:]
        assert (
            arr == m + 1
        ).all(), f"tile {m} should be filled with {m + 1}, got {arr.min()}..{arr.max()}"


def test_per_tile_ome_xml_carries_plane_position(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    # Tile X01Y00 corresponds to scene tiles[1] with pos_x_m=0.04050,
    # pos_y_m=0.01700, pos_z_m=0.01170; meters → µm = 40500, 17000, 11700.
    xml = (
        out / "Position_1" / "tile_X01Y00.ome.zarr" / "OME" / "METADATA.ome.xml"
    ).read_text()
    parsed = from_xml(xml)
    image = parsed.images[0]
    planes = image.pixels.planes
    assert len(planes) == 1
    plane = planes[0]
    assert plane.position_x == pytest.approx(40500.0)
    assert plane.position_y == pytest.approx(17000.0)
    assert plane.position_z == pytest.approx(11700.0)
    assert plane.position_x_unit.value == "µm"


def test_adjacent_tiles_have_stage_offset_matching_overlap(
    tmp_path: Path, patched_reader
) -> None:
    """Two adjacent tiles' OME-XML PositionX must differ by the stage step
    between them (50 µm = 50e-6 m here). This is the user-facing sanity check
    from the acceptance criteria."""
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    def _pos_x_um(p: Path) -> float:
        parsed = from_xml(p.read_text())
        return parsed.images[0].pixels.planes[0].position_x

    p0 = _pos_x_um(
        out / "Position_1" / "tile_X00Y00.ome.zarr" / "OME" / "METADATA.ome.xml"
    )
    p1 = _pos_x_um(
        out / "Position_1" / "tile_X01Y00.ome.zarr" / "OME" / "METADATA.ome.xml"
    )
    assert p1 - p0 == pytest.approx(500.0)  # 0.00050 m = 500 µm


# ---------- audit shape (schema-5: mosaic.per_tile, mosaic.tile_stores) ----------


def test_per_tile_audit_records_per_tile_and_tile_stores(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    # All three tile audits carry per_tile=true + the full tile_stores list.
    for tile_audit in result["stores"]:
        assert tile_audit["audit_schema_version"] == 7
        m = tile_audit["mosaic"]
        assert m["per_tile"] is True
        assert m["tile_count"] == 3
        assert len(m["tile_stores"]) == 3
        # tile_stores entries have the documented keys
        for entry in m["tile_stores"]:
            assert set(entry.keys()) == {
                "field_x",
                "field_y",
                "store_path",
                "pos_x_m",
                "pos_y_m",
                "pos_z_m",
            }
        # The intended overlap percent surfaces alongside.
        assert m["intended_overlap_x_pct"] == 10.0
        assert m["intended_overlap_y_pct"] == 10.0
        assert tile_audit["config"]["lif_mosaic"] == "per-tile"


def test_per_tile_audit_round_trips_to_on_disk_attrs(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    with open(out / "Position_1" / "tile_X00Y00.ome.zarr" / "zarr.json") as f:
        root = json.load(f)
    audit = root["attributes"]["zarrmony"]
    assert audit["audit_schema_version"] == AUDIT_SCHEMA_VERSION == 7
    assert audit["mosaic"]["per_tile"] is True
    assert audit["mosaic"]["tile_index"] == 0
    assert len(audit["mosaic"]["tile_stores"]) == 3


# ---------- plate + per-tile incompatibility ----------


def test_plate_plus_per_tile_raises_layout_mismatch(
    tmp_path: Path, patched_reader
) -> None:
    """``layout='plate'`` + ``lif_mosaic='per-tile'`` is rejected before any
    pixels are written; the error mentions the per-tile-vs-plate conflict."""
    from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

    plate_layout = PlateLayout(
        name="x",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(
                scene_index=0,
                row="A",
                column="01",
                field_name="A01-f0",
                acquisition_id=1,
            ),
        ],
    )
    reader = FakeReader(
        scenes=["s"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=plate_layout,
    )
    patched_reader(reader)

    with pytest.raises(LayoutMismatchError, match="per-tile"):
        convert("/tmp/x.lif", tmp_path / "out", layout="plate", lif_mosaic="per-tile")


# ---------- default mode unchanged + _Merged sibling unchanged ----------


def test_auto_stitch_default_does_not_use_per_tile_path(
    tmp_path: Path, patched_reader
) -> None:
    """The default lif_mosaic value writes the scene as a single store with
    no scene-named subdirectory of tile sub-stores. (Regression guard for the
    no-flag user.)"""
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    # Default lif_mosaic — FakeReader is per-tile-eligible but the dispatch
    # only fires when lif_mosaic=="per-tile". Under the v0.7.0 cascade default
    # this fixture has stage metadata, so it lands on stage-stitch — the
    # single-store on-disk shape (asserted below) is unchanged either way.
    # Suppress MosaicPlacementWarning because this fixture's stage positions
    # aren't calibrated for a realistic overlap.
    from zarrmony.errors import MosaicPlacementWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MosaicStitchingWarning)
        warnings.simplefilter("ignore", MosaicPlacementWarning)
        result = convert("/tmp/x.lif", out, pyramid_min_size=8)

    assert (out / "Position_1.ome.zarr" / "zarr.json").exists()
    assert not (out / "Position_1").exists()
    assert len(result["stores"]) == 1
    assert "mosaic" not in result["stores"][0].get("config", {}).get("layout", "")


def test_per_tile_still_honors_merged_sibling_skip(
    tmp_path: Path, patched_reader
) -> None:
    """A mosaic scene with a ``_Merged`` sibling is skipped under BOTH
    ``lif_mosaic`` values — the merged sibling is the source of truth for
    that scene; per-tile output would be redundant."""
    reader = FakeReader(
        scenes=["Position 1", "Position 1_Merged"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        skip_reasons={0: "vendor-merged sibling 'Position 1_Merged' is present"},
    )
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.warns(MosaicMergedSiblingWarning):
        result = convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")

    # Only the merged sibling was written (one flat per-scene store).
    assert (out / "Position_1_Merged.ome.zarr").is_dir()
    assert not (out / "Position_1").exists()
    assert len(result["stores"]) == 1
    assert result["stores"][0]["scene_name"] == "Position 1_Merged"


# ---------- per-tile refuse-overwrite ----------


def test_per_tile_refuses_existing_tile_store_without_force(
    tmp_path: Path, patched_reader
) -> None:
    from zarrmony.errors import OutputExistsError

    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")
    with pytest.raises(OutputExistsError):
        convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")


def test_per_tile_force_overwrites_existing_tile_stores(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")
    result = convert(
        "/tmp/x.lif",
        out,
        pyramid_min_size=8,
        lif_mosaic="per-tile",
        force=True,
    )
    assert len(result["stores"]) == 3


# ---------- per-tile path does NOT trigger MosaicStitchingWarning ----------


def test_per_tile_does_not_emit_stitching_warning(
    tmp_path: Path, patched_reader
) -> None:
    """The per-tile path bypasses bioio-lif's auto-stitcher entirely, so the
    1-pixel-overlap warning shouldn't fire (it would lie about behavior)."""
    reader = _make_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicStitchingWarning)
        convert("/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="per-tile")
