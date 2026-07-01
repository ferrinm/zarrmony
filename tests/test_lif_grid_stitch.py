"""End-to-end tests for the LIF grid-stitch mosaic write path (#39).

Drives ``convert(..., lif_mosaic="grid-stitch")`` against FakeReader configured
as a reassembly-eligible LIF mosaic scene (M-intact xarray + a LIF-shaped
metadata blob with ``<Tile>`` entries the extractor picks up). Covers:

- on-disk shape: a single ``<scene>.ome.zarr`` per scene (one-store invariant
  preserved, unlike per-tile which produces N sub-stores per scene)
- pixel placement: tile M=i lands at ``(field_y[i]*tile_H, field_x[i]*tile_W)``
  on the canvas, NOT at M-scan-order slots
- audit: ``mosaic.stitcher="zarrmony-grid"``, ``mosaic.placement_shape``,
  ``mosaic.overlap_assumption_px=0``
- no ``MosaicStitchingWarning`` (arrangement is correct)
- composes with ``layout="plate"`` (grid-stitch produces one canvas per FOV,
  which the plate spec accepts — unlike per-tile which is rejected)
- ``_Merged`` sibling still skips (predicate shared with per-tile)
- strict metadata: convert raises ``ValueError`` naming ``per-tile`` as escape
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr

from tests.conftest import FakeReader, TileScene
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.errors import MosaicMergedSiblingWarning, MosaicStitchingWarning
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


def _shuffled_3x3_tiles() -> list[dict]:
    """3x3 mosaic where M-order deliberately differs from grid order.

    M=0 declares (fx=2, fy=1) — bioio-lif's stitcher would put its pixels at
    grid slot (0,0). Grid-stitch must put them at (row=1, col=2) instead.
    """
    m_order = [
        (2, 1),  # M=0 — the acceptance-criterion tile
        (0, 0),  # M=1
        (1, 0),  # M=2
        (2, 0),  # M=3
        (0, 1),  # M=4
        (1, 1),  # M=5
        (0, 2),  # M=6
        (1, 2),  # M=7
        (2, 2),  # M=8
    ]
    return [
        {
            "field_x": fx,
            "field_y": fy,
            "pos_x_m": 0.04 + fx * 0.0005,
            "pos_y_m": 0.017 + fy * 0.0005,
            "pos_z_m": 0.01170,
        }
        for fx, fy in m_order
    ]


def _make_3x3_mosaic_reader(scene_name: str = "Position 1") -> FakeReader:
    scene = TileScene(tiles=_shuffled_3x3_tiles(), tile_yx=(8, 8))
    return FakeReader(
        scenes=[scene_name],
        dims="TCZYX",
        shape=(1, 1, 1, 8, 8),  # scene-level shape unused in grid-stitch path
        channel_names=["DAPI"],
        per_tile_scenes={0: scene},
    )


# ---------- on-disk shape: single canvas per scene ----------


def test_grid_stitch_writes_one_store_per_scene(tmp_path: Path, patched_reader) -> None:
    """Grid-stitch preserves the one-store-per-scene invariant (unlike per-tile
    which produces a scene-named directory of tile sub-stores)."""
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")

    store = out / "Position_1.ome.zarr"
    assert store.is_dir()
    assert (store / "zarr.json").exists()
    # No scene-named subdirectory of tile sub-stores.
    assert not (out / "Position_1").exists()
    assert len(result["stores"]) == 1


def test_grid_stitch_canvas_matches_grid_dims(tmp_path: Path, patched_reader) -> None:
    """3x3 grid of 8x8 tiles → 24x24 butt-jointed canvas."""
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")

    g = zarr.open_group(str(out / "Position_1.ome.zarr"), mode="r")
    # Dims are TCZYX; last two are the reassembled Y/X.
    assert g["0"].shape[-2:] == (24, 24)


# ---------- pixel placement: acceptance criterion ----------


def test_grid_stitch_places_tile_at_declared_slot_not_m_scan_order(
    tmp_path: Path, patched_reader
) -> None:
    """The acceptance case: tile M=0 declared at (fx=2, fy=1) must land at
    canvas region [8:16, 16:24] (row 1, col 2), NOT at [0:8, 0:8] (row 0, col
    0 — where bioio-lif's M-scan-order stitcher would put it)."""
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")

    g = zarr.open_group(str(out / "Position_1.ome.zarr"), mode="r")
    canvas = g["0"][:]  # (T, C, Z, Y, X)
    # FakeReader fills tile M with value M+1 (see conftest.TileScene).
    # M=0 → value 1, declared slot (fx=2, fy=1) → canvas[..., 8:16, 16:24].
    tile_h, tile_w = 8, 8
    block_m0 = canvas[..., 1 * tile_h : 2 * tile_h, 2 * tile_w : 3 * tile_w]
    assert np.all(block_m0 == 1), "tile M=0 did not land at declared (fx=2, fy=1) slot"
    # And the (0,0) slot should hold M=1's pixels (value 2), not M=0's.
    block_at_origin = canvas[..., 0:tile_h, 0:tile_w]
    assert np.all(block_at_origin == 2), "grid slot (0,0) should hold M=1's pixels"


# ---------- audit surface ----------


def test_grid_stitch_audit_records_stitcher_and_placement_shape(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")

    scene_record = result["stores"][0]["per_scene"][0]
    mosaic = scene_record["mosaic"]
    assert mosaic["stitcher"] == "zarrmony-grid"
    assert mosaic["placement_shape"] == {"rows": 3, "cols": 3}
    assert mosaic["overlap_assumption_px"] == 0
    assert mosaic["tile_count"] == 9
    # Tile positions carried through from extract_tile_layout.
    assert len(mosaic["tiles"]) == 9
    assert mosaic["intended_overlap_x_pct"] == pytest.approx(10.0)


def test_grid_stitch_audit_round_trips_to_on_disk_attrs(
    tmp_path: Path, patched_reader
) -> None:
    """The mosaic block in the returned audit dict must also land on the store's
    root attrs.zarrmony (regression guard against the caller enriching only the
    Python dict and forgetting the on-disk write)."""
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")

    with open(out / "Position_1.ome.zarr" / "zarr.json") as f:
        root = json.load(f)
    mosaic = root["attributes"]["zarrmony"]["per_scene"][0]["mosaic"]
    assert mosaic["stitcher"] == "zarrmony-grid"
    assert mosaic["placement_shape"] == {"rows": 3, "cols": 3}


def test_grid_stitch_config_records_lif_mosaic_value(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")
    assert result["stores"][0]["config"]["lif_mosaic"] == "grid-stitch"


# ---------- no MosaicStitchingWarning ----------


def test_grid_stitch_does_not_emit_stitching_warning(
    tmp_path: Path, patched_reader
) -> None:
    """Grid-stitch fixes the arrangement bug and bypasses bioio-lif's stitcher —
    the auto-stitch warning would lie about behavior here."""
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicStitchingWarning)
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")


# ---------- _Merged sibling skip still works ----------


def test_grid_stitch_still_honors_merged_sibling_skip(
    tmp_path: Path, patched_reader
) -> None:
    """A mosaic scene with a ``_Merged`` sibling is skipped under all three
    lif_mosaic values — the merged sibling is the source of truth."""
    reader = FakeReader(
        scenes=["Position 1", "Position 1_Merged"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        skip_reasons={0: "vendor-merged sibling 'Position 1_Merged' is present"},
    )
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.warns(MosaicMergedSiblingWarning):
        result = convert(
            "/tmp/x.lif", out, pyramid_min_size=8, lif_mosaic="grid-stitch"
        )

    assert (out / "Position_1_Merged.ome.zarr").is_dir()
    assert not (out / "Position_1.ome.zarr").exists()
    assert len(result["stores"]) == 1


# ---------- strict metadata: raise on incomplete tile info ----------


def test_grid_stitch_raises_when_tile_layout_incomplete(
    tmp_path: Path, patched_reader
) -> None:
    """Grid-stitch is strict — an incomplete grid (only 3 tiles for a 2x2 layout)
    raises ValueError naming lif_mosaic='per-tile' as the graceful-degrade escape."""
    incomplete = [
        {
            "field_x": 0,
            "field_y": 0,
            "pos_x_m": 0.04,
            "pos_y_m": 0.017,
            "pos_z_m": 0.01,
        },
        {
            "field_x": 1,
            "field_y": 0,
            "pos_x_m": 0.041,
            "pos_y_m": 0.017,
            "pos_z_m": 0.01,
        },
        {
            "field_x": 0,
            "field_y": 1,
            "pos_x_m": 0.04,
            "pos_y_m": 0.018,
            "pos_z_m": 0.01,
        },
    ]
    scene = TileScene(tiles=incomplete, tile_yx=(4, 4))
    reader = FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 1, 4, 4),
        per_tile_scenes={0: scene},
    )
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"complete rectangular grid.*per-tile"):
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="grid-stitch")


# ---------- plate composability (grid-stitch WORKS with layout="plate") ----------


def test_grid_stitch_composes_with_plate_layout(tmp_path: Path, patched_reader) -> None:
    """Grid-stitch produces one canvas per FOV — plate-compatible. Unlike
    per-tile (rejected against plate), grid-stitch honors the "one FOV = one
    image" plate spec while still fixing tile arrangement."""
    from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

    plate_layout = PlateLayout(
        name="mosaic-plate",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(
                scene_index=0,
                row="A",
                column="01",
                field_name="A01-mosaic",
                acquisition_id=1,
            ),
        ],
    )
    scene = TileScene(tiles=_shuffled_3x3_tiles(), tile_yx=(8, 8))
    reader = FakeReader(
        scenes=["Position 1"],
        dims="TCZYX",
        shape=(1, 1, 1, 8, 8),
        channel_names=["DAPI"],
        layout_hint="plate",
        plate_layout=plate_layout,
        per_tile_scenes={0: scene},
    )
    patched_reader(reader)
    out = tmp_path / "out"

    audit = convert(
        "/tmp/x.lif",
        out,
        pyramid_min_size=4,
        layout="plate",
        lif_mosaic="grid-stitch",
    )

    # Plate structure exists AND the single FOV was written.
    assert (out / "A" / "01" / "0" / "zarr.json").exists()
    # FOV-scoped audit records the grid-stitch mosaic block.
    field_record = audit["fields"][0]
    assert field_record["mosaic"]["stitcher"] == "zarrmony-grid"
    assert field_record["mosaic"]["placement_shape"] == {"rows": 3, "cols": 3}

    # And the pixels are reassembled correctly: canvas is 24x24, tile M=0
    # (fill value 1) at declared (fx=2, fy=1) → canvas[..., 8:16, 16:24].
    g = zarr.open_group(str(out / "A" / "01" / "0"), mode="r")
    canvas = g["0"][:]
    assert canvas.shape[-2:] == (24, 24)
    block_m0 = canvas[..., 8:16, 16:24]
    assert np.all(block_m0 == 1)


# ---------- auto-stitch (default) is unaffected ----------


def test_default_mode_does_not_use_grid_stitch_path(
    tmp_path: Path, patched_reader
) -> None:
    """The default lif_mosaic value does NOT invoke grid-stitch, even when the
    scene would be eligible. Regression guard for the no-flag user."""
    reader = _make_3x3_mosaic_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MosaicStitchingWarning)
        result = convert("/tmp/x.lif", out, pyramid_min_size=4)

    # Default writes via the auto-stitch path (bioio-lif mosaic_xarray_dask_data),
    # whose mosaic_summary carries stitcher="bioio-lif" — not zarrmony-grid.
    # (FakeReader's mosaic_summary is None by default, so this is asserted
    # via the ABSENCE of grid-stitch fields.)
    scene_record = result["stores"][0]["per_scene"][0]
    assert scene_record.get("mosaic", {}).get("stitcher") != "zarrmony-grid"
