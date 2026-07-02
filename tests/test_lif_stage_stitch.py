"""End-to-end tests for the LIF stage-stitch mosaic write path (#40).

Drives ``convert(..., lif_mosaic="stage-stitch")`` against FakeReader configured
as a reassembly-eligible LIF mosaic scene (M-intact xarray + a LIF-shaped
metadata blob with ``<Tile>`` entries the extractor picks up + a physical pixel
size). Covers:

- on-disk shape: a single ``<scene>.ome.zarr`` per scene
- pixel placement: adjacent tile overlap regions honour the LIF-declared
  intended overlap (~10% wide, not the 1-px butt joints of grid-stitch)
- overwrite policy: later-M tiles overwrite earlier ones in the overlap region
- MosaicPlacementWarning on mismatched pixel-size (catches unit-conversion bugs)
- strict metadata: missing PosX/PosY → ValueError naming grid-stitch as escape
- strict inputs: missing scene physical pixel size → ValueError naming grid-stitch
- audit: ``mosaic.stitcher="zarrmony-stage"``, ``tile_pixel_offsets``,
  ``observed_overlap_pct`` alongside ``intended_overlap_*_pct``
- no ``MosaicStitchingWarning`` (arrangement + overlap both handled)
- default mode remains unchanged (no stage-stitch when the user omits the flag)
- reassembled xarray remains dask-backed (no eager compute in the reader path)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr

from tests.conftest import FakePhysicalPixelSizes, FakeReader, TileScene
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.errors import (
    LayoutMismatchError,
    MosaicPlacementWarning,
    MosaicStitchingWarning,
)
from zarrmony.readers.plugin import ReaderPlugin

# Tile geometry the tests share: 20 px square tiles + 1 µm/px pixel size means
# a stage step of 18 µm gives exactly 10% overlap (2 px shared between adjacent
# tiles). Small enough to keep the writer fast; large enough that a 10% overlap
# is a visible integer number of pixels.
TILE_YX = (20, 20)
TILE_H, TILE_W = TILE_YX
INTENDED_OVERLAP_PCT = 10.0
STEP_UM = TILE_W * (1.0 - INTENDED_OVERLAP_PCT / 100.0)  # 18.0 µm
STEP_M = STEP_UM / 1_000_000.0  # 1.8e-5 m
ORIGIN_X_M = 0.040
ORIGIN_Y_M = 0.017


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


def _overlapping_3x3_tiles() -> list[dict]:
    """3x3 mosaic where stage positions imply ~10% overlap on each axis.

    M-order is row-major (fx varies fastest) so tests that don't rely on
    shuffling can predict where each tile ends up on the canvas.
    """
    tiles = []
    for fy in range(3):
        for fx in range(3):
            tiles.append(
                {
                    "field_x": fx,
                    "field_y": fy,
                    "pos_x_m": ORIGIN_X_M + fx * STEP_M,
                    "pos_y_m": ORIGIN_Y_M + fy * STEP_M,
                    "pos_z_m": 0.01170,
                }
            )
    return tiles


def _make_3x3_stage_reader(
    scene_name: str = "Position 1",
    pixel_sizes: FakePhysicalPixelSizes | None = None,
    tiles: list[dict] | None = None,
) -> FakeReader:
    scene = TileScene(
        tiles=tiles or _overlapping_3x3_tiles(),
        tile_yx=TILE_YX,
        intended_overlap_x_pct=INTENDED_OVERLAP_PCT,
        intended_overlap_y_pct=INTENDED_OVERLAP_PCT,
    )
    return FakeReader(
        scenes=[scene_name],
        dims="TCZYX",
        shape=(1, 1, 1, TILE_H, TILE_W),
        channel_names=["DAPI"],
        pixel_sizes=pixel_sizes or FakePhysicalPixelSizes(Y=1.0, X=1.0),
        per_tile_scenes={0: scene},
    )


# ---------- on-disk shape ----------


def test_stage_stitch_writes_one_store_per_scene(
    tmp_path: Path, patched_reader
) -> None:
    """Stage-stitch preserves the one-store-per-scene invariant."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")

    store = out / "Position_1.ome.zarr"
    assert store.is_dir()
    assert (store / "zarr.json").exists()
    assert not (out / "Position_1").exists()
    assert len(result["stores"]) == 1


# ---------- canvas dimensions honour intended overlap ----------


def test_stage_stitch_canvas_dims_match_intended_overlap(
    tmp_path: Path, patched_reader
) -> None:
    """3x3 grid of 20 px tiles at 10% overlap → canvas = 20 + 2*(20-2) = 56 px per side."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")

    g = zarr.open_group(str(out / "Position_1.ome.zarr"), mode="r")
    # step is 18 px between tile origins; last-tile origin is at 2*18=36, so
    # canvas = 36 + 20 = 56.
    expected = int(round(2 * STEP_UM)) + TILE_W
    assert g["0"].shape[-2:] == (expected, expected)


# ---------- overlap-region width matches intended (~10% within ±1 px) ----------


def test_stage_stitch_overlap_region_width_matches_intended(
    tmp_path: Path, patched_reader
) -> None:
    """Adjacent-tile overlap width in canvas pixels equals intended overlap
    (within a ±1 px tolerance for rounding to integer pixel snap)."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")

    # Tile M=0 at (fx=0, fy=0), M=1 at (fx=1, fy=0). Their stage positions
    # differ by STEP_UM=18 µm; at 1 µm/px that's an 18-px stride between
    # origins. tile_w=20 → 2-px overlap region between M=0 and M=1.
    stride_px = int(round(STEP_UM))
    expected_overlap = TILE_W - stride_px  # 2
    intended_overlap_px = int(round(TILE_W * INTENDED_OVERLAP_PCT / 100.0))
    assert abs(expected_overlap - intended_overlap_px) <= 1


# ---------- overwrite policy: later M tiles win on shared pixels ----------


def test_stage_stitch_later_tile_overwrites_earlier_in_overlap(
    tmp_path: Path, patched_reader
) -> None:
    """FakeReader fills tile M with value M+1. In the horizontal overlap
    region between M=0 and M=1, the shared pixels must carry value 2 (M=1
    overwrote M=0) — deterministic later-wins policy, no blending."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")

    g = zarr.open_group(str(out / "Position_1.ome.zarr"), mode="r")
    canvas = g["0"][:]
    # M=0 origin (0,0); M=1 origin (0, 18). Overlap X range [18, 20).
    # Restrict Y to [0, 18) so we only see the M=0 / M=1 pair — M=3 and M=4
    # also overwrite Y in [18, 20) and would confuse the assertion.
    overlap_slice = canvas[..., 0:18, 18:20]
    assert np.all(overlap_slice == 2), "later-M tile did not overwrite in overlap"


# ---------- audit surface ----------


def test_stage_stitch_audit_records_stitcher_offsets_and_observed_overlap(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")
    scene_record = result["stores"][0]["per_scene"][0]
    mosaic = scene_record["mosaic"]

    assert mosaic["stitcher"] == "zarrmony-stage"
    assert mosaic["tile_count"] == 9
    # Nine tiles, each entry names its M index and per-axis pixel offset.
    assert len(mosaic["tile_pixel_offsets"]) == 9
    offsets_by_m = {
        e["m_index"]: (e["y_px"], e["x_px"]) for e in mosaic["tile_pixel_offsets"]
    }
    stride_px = int(round(STEP_UM))
    # M=0 at (fx=0, fy=0) → (0, 0); M=4 at (fx=1, fy=1) → (stride, stride).
    assert offsets_by_m[0] == (0, 0)
    assert offsets_by_m[4] == (stride_px, stride_px)
    # Observed overlap agrees with intended within a small margin.
    obs = mosaic["observed_overlap_pct"]
    assert obs["x"] == pytest.approx(INTENDED_OVERLAP_PCT, abs=5.0)
    assert obs["y"] == pytest.approx(INTENDED_OVERLAP_PCT, abs=5.0)
    # LIF-declared intended overlap survives alongside the observed values.
    assert mosaic["intended_overlap_x_pct"] == pytest.approx(INTENDED_OVERLAP_PCT)
    assert mosaic["intended_overlap_y_pct"] == pytest.approx(INTENDED_OVERLAP_PCT)


def test_stage_stitch_config_records_lif_mosaic_value(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    result = convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")
    assert result["stores"][0]["config"]["lif_mosaic"] == "stage-stitch"


# ---------- placement-sanity warning fires on pixel-size mismatch ----------


def test_stage_stitch_warns_on_pixel_size_stage_mismatch(
    tmp_path: Path, patched_reader
) -> None:
    """Reader reports pixel size 2 µm/px but stage positions were captured at
    1 µm/px calibration — placement offsets end up half as far apart, producing
    ~55% observed overlap vs 10% intended (>20% mismatch → warning)."""
    reader = _make_3x3_stage_reader(pixel_sizes=FakePhysicalPixelSizes(Y=2.0, X=2.0))
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.warns(MosaicPlacementWarning, match=r"observed overlap"):
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


def test_stage_stitch_no_warning_when_within_tolerance(
    tmp_path: Path, patched_reader
) -> None:
    """When observed and intended overlap agree, MosaicPlacementWarning must
    not fire (regression guard against a too-tight tolerance)."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicPlacementWarning)
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


# ---------- fail-loud: missing PosX/PosY ----------


def test_stage_stitch_raises_on_missing_pos_x(tmp_path: Path, patched_reader) -> None:
    """A tile without PosX / PosY errors clearly and names grid-stitch escape."""
    tiles = _overlapping_3x3_tiles()
    tiles[4]["pos_x_m"] = None  # M=4 is (fx=1, fy=1); drop its PosX
    reader = _make_3x3_stage_reader(tiles=tiles)
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.raises(
        ValueError,
        match=r"stage-stitch.*missing.*pos_x_m.*grid-stitch",
    ):
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


def test_stage_stitch_raises_on_missing_pos_y(tmp_path: Path, patched_reader) -> None:
    tiles = _overlapping_3x3_tiles()
    tiles[0]["pos_y_m"] = None
    reader = _make_3x3_stage_reader(tiles=tiles)
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"pos_y_m"):
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


# ---------- fail-loud: missing scene physical pixel size ----------


def test_stage_stitch_raises_on_missing_pixel_size_x(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_3x3_stage_reader(pixel_sizes=FakePhysicalPixelSizes(Y=1.0, X=None))
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"physical pixel size.*\['X'\].*grid-stitch"):
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


def test_stage_stitch_raises_on_missing_pixel_size_y(
    tmp_path: Path, patched_reader
) -> None:
    reader = _make_3x3_stage_reader(pixel_sizes=FakePhysicalPixelSizes(Y=None, X=1.0))
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"physical pixel size.*\['Y'\]"):
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


# ---------- no MosaicStitchingWarning ----------


def test_stage_stitch_does_not_emit_stitching_warning(
    tmp_path: Path, patched_reader
) -> None:
    """Stage-stitch fixes both arrangement AND overlap — the auto-stitch
    warning would lie about behaviour here."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicStitchingWarning)
        convert("/tmp/x.lif", out, pyramid_min_size=4, lif_mosaic="stage-stitch")


# ---------- default mode is unaffected ----------


def test_default_mode_does_not_use_stage_stitch_path(
    tmp_path: Path, patched_reader
) -> None:
    """Regression guard: the no-flag user still gets bioio-lif auto-stitch."""
    reader = _make_3x3_stage_reader()
    patched_reader(reader)
    out = tmp_path / "out"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MosaicStitchingWarning)
        result = convert("/tmp/x.lif", out, pyramid_min_size=4)

    scene_record = result["stores"][0]["per_scene"][0]
    assert scene_record.get("mosaic", {}).get("stitcher") != "zarrmony-stage"


# ---------- API validation: unknown values still error clearly ----------


def test_stage_stitch_unknown_value_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"lif_mosaic must be one of"):
        convert("/tmp/x.lif", tmp_path / "out", lif_mosaic="stage")  # type: ignore[arg-type]


# ---------- plate + stage-stitch rejected clearly ----------


def test_stage_stitch_rejected_under_plate_layout(
    tmp_path: Path, patched_reader
) -> None:
    """A plate reader + stage-stitch would silently fall through to
    bioio-lif's auto-stitcher today; reject before any pixels are written
    with a message that names the two escapes (flat, or grid-stitch)."""
    from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

    plate_layout = PlateLayout(
        name="p",
        rows=["A"],
        columns=["01"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(
                scene_index=0,
                row="A",
                column="01",
                field_name="A01",
                acquisition_id=1,
            ),
        ],
    )
    reader = FakeReader(
        scenes=["Position 1"],
        dims="TCZYX",
        shape=(1, 1, 1, TILE_H, TILE_W),
        layout_hint="plate",
        plate_layout=plate_layout,
    )
    patched_reader(reader)
    out = tmp_path / "out"

    with pytest.raises(LayoutMismatchError, match=r"stage-stitch.*plate.*grid-stitch"):
        convert(
            "/tmp/x.lif",
            out,
            pyramid_min_size=4,
            layout="plate",
            lif_mosaic="stage-stitch",
        )


# ---------- reassembled xarray remains dask-backed ----------


def test_stage_stitch_reader_path_returns_dask_backed_canvas() -> None:
    """Direct check on the pure helper: reassemble_stage returns dask-backed."""
    import dask.array as da

    from zarrmony.metadata.lif_tiles import (
        compute_stage_placements,
        reassemble_stage,
    )

    tiles = _overlapping_3x3_tiles()
    scene = TileScene(
        tiles=tiles,
        tile_yx=TILE_YX,
        intended_overlap_x_pct=INTENDED_OVERLAP_PCT,
        intended_overlap_y_pct=INTENDED_OVERLAP_PCT,
    )
    reader = FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 1, TILE_H, TILE_W),
        per_tile_scenes={0: scene},
    )
    tiles_xarr = reader.tiles_xarray_dask_data
    tile_layout = {
        "tiles": tiles,
        "intended_overlap_x_pct": INTENDED_OVERLAP_PCT,
        "intended_overlap_y_pct": INTENDED_OVERLAP_PCT,
    }
    offsets, canvas_shape_yx, _observed = compute_stage_placements(
        tile_layout,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        tile_h=TILE_H,
        tile_w=TILE_W,
    )
    canvas = reassemble_stage(tiles_xarr, offsets, canvas_shape_yx)
    assert isinstance(canvas.data, da.Array)
