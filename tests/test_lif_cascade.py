"""End-to-end tests for the LIF ``lif_mosaic="auto-stitch"`` cascade (v0.7.0, #41).

The default value picks a concrete stitcher per scene:

1. ``stage-stitch`` when every tile has ``PosX``/``PosY`` and the scene has
   both physical pixel sizes.
2. ``grid-stitch`` when ``FieldX``/``FieldY`` form a complete rectangular grid.
3. ``bioio-lif`` (M-scan-order fallback) otherwise.

These tests exercise each decision arm plus the two escape hatches
(``bioio-lif`` opt-in and ``stage-stitch`` explicit override) so a future
refactor can't silently change which branch fires for which metadata shape.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.conftest import FakePhysicalPixelSizes, FakeReader, TileScene
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.errors import MosaicPlacementWarning
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
    def installer(reader: FakeReader):
        plugin_obj = _fake_plugin()
        monkeypatch.setattr(
            api_module, "get_reader", lambda _path: (reader, plugin_obj, 100)
        )

    return installer


TILE_YX = (20, 20)
_STEP_M = TILE_YX[1] * 0.9 / 1_000_000.0  # 10% overlap in µm → m


def _tiles_with(*, stage: bool, grid: bool) -> list[dict]:
    """3x3 tile dicts with per-axis toggles for stage-vs-grid completeness."""
    tiles = []
    for fy in range(3):
        for fx in range(3):
            tiles.append(
                {
                    "field_x": fx if grid else None,
                    "field_y": fy if grid else None,
                    "pos_x_m": (0.04 + fx * _STEP_M) if stage else None,
                    "pos_y_m": (0.017 + fy * _STEP_M) if stage else None,
                    "pos_z_m": 0.01170,
                }
            )
    return tiles


def _cascade_reader(
    *,
    stage: bool = True,
    grid: bool = True,
    pixel_sizes: FakePhysicalPixelSizes | None = None,
    include_tile_metadata: bool = True,
) -> FakeReader:
    """Reassembly-eligible reader whose tile metadata surface is controllable.

    ``include_tile_metadata=False`` simulates a mosaic scene with no ``<Tile>``
    entries in the scene XML — the "cascade lands on bioio-lif" case.
    """
    if include_tile_metadata:
        scene = TileScene(tiles=_tiles_with(stage=stage, grid=grid), tile_yx=TILE_YX)
        per_tile_scenes = {0: scene}
    else:
        scene = TileScene(tiles=[], tile_yx=TILE_YX)
        per_tile_scenes = {0: scene}
    return FakeReader(
        scenes=["Position 1"],
        dims="TCZYX",
        shape=(1, 1, 1, TILE_YX[0], TILE_YX[1]),
        channel_names=["DAPI"],
        pixel_sizes=pixel_sizes or FakePhysicalPixelSizes(Y=1.0, X=1.0),
        per_tile_scenes=per_tile_scenes,
    )


def _get_mosaic(result: dict) -> dict:
    return result["stores"][0]["per_scene"][0]["mosaic"]


# ---------- 1. Full metadata → stage ----------


def test_cascade_lands_on_stage_when_all_metadata_present(
    tmp_path: Path, patched_reader
) -> None:
    reader = _cascade_reader(stage=True, grid=True)
    patched_reader(reader)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MosaicPlacementWarning)
        result = convert("/tmp/x.lif", tmp_path / "out", pyramid_min_size=4)

    mosaic = _get_mosaic(result)
    assert mosaic["stitcher"] == "zarrmony-stage"
    assert mosaic["cascade_selected"] is True


# ---------- 2. Missing pixel size + complete grid → grid ----------


def test_cascade_falls_back_to_grid_when_pixel_size_missing(
    tmp_path: Path, patched_reader
) -> None:
    reader = _cascade_reader(
        stage=True, grid=True, pixel_sizes=FakePhysicalPixelSizes(Y=None, X=None)
    )
    patched_reader(reader)

    result = convert("/tmp/x.lif", tmp_path / "out", pyramid_min_size=4)

    mosaic = _get_mosaic(result)
    assert mosaic["stitcher"] == "zarrmony-grid"
    assert mosaic["cascade_selected"] is True


# ---------- 3. Missing per-tile positions + complete grid → grid ----------


def test_cascade_falls_back_to_grid_when_tile_positions_missing(
    tmp_path: Path, patched_reader
) -> None:
    reader = _cascade_reader(stage=False, grid=True)
    patched_reader(reader)

    result = convert("/tmp/x.lif", tmp_path / "out", pyramid_min_size=4)

    mosaic = _get_mosaic(result)
    assert mosaic["stitcher"] == "zarrmony-grid"
    assert mosaic["cascade_selected"] is True


# ---------- 4. No <Tile> metadata at all → bioio-lif ----------


def test_cascade_falls_all_the_way_to_bioio_lif_without_tile_metadata(
    tmp_path: Path, patched_reader
) -> None:
    # FakeReader doesn't wrap through _MosaicAwareLifReader so it can't emit
    # MosaicStitchingWarning — that path is exercised in test_readers.py.
    # Here we assert the cascade DECISION: neither zarrmony reassembler ran,
    # which is the observable difference between "cascade landed on
    # bioio-lif" and "cascade landed on grid/stage".
    reader = _cascade_reader(include_tile_metadata=False)
    patched_reader(reader)

    result = convert("/tmp/x.lif", tmp_path / "out", pyramid_min_size=4)

    scene_record = result["stores"][0]["per_scene"][0]
    mosaic = scene_record.get("mosaic")
    if mosaic is not None:
        assert mosaic.get("stitcher") not in ("zarrmony-stage", "zarrmony-grid")


# ---------- 5. Plate mode → grid (stage plate-rejected under cascade) ----------


def test_cascade_under_plate_layout_skips_stage_and_lands_on_grid(
    tmp_path: Path, patched_reader
) -> None:
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
    scene = TileScene(tiles=_tiles_with(stage=True, grid=True), tile_yx=TILE_YX)
    reader = FakeReader(
        scenes=["Position 1"],
        dims="TCZYX",
        shape=(1, 1, 1, TILE_YX[0], TILE_YX[1]),
        channel_names=["DAPI"],
        pixel_sizes=FakePhysicalPixelSizes(Y=1.0, X=1.0),
        layout_hint="plate",
        plate_layout=plate_layout,
        per_tile_scenes={0: scene},
    )
    patched_reader(reader)

    audit = convert("/tmp/x.lif", tmp_path / "out", pyramid_min_size=4, layout="plate")

    field = audit["fields"][0]
    assert field["mosaic"]["stitcher"] == "zarrmony-grid"
    assert field["mosaic"]["cascade_selected"] is True


# ---------- 6. Explicit bioio-lif → routes there regardless ----------


def test_explicit_bioio_lif_bypasses_cascade_even_with_full_metadata(
    tmp_path: Path, patched_reader
) -> None:
    """Passing lif_mosaic='bioio-lif' opts back into the pre-v0.7.0 default
    (M-scan-order stitcher + 1-px overlap). The cascade does NOT run — the
    audit must not carry cascade_selected=True for this branch, even though
    the fixture has enough metadata to have picked stage under the default.
    """
    reader = _cascade_reader(stage=True, grid=True)
    patched_reader(reader)

    result = convert(
        "/tmp/x.lif",
        tmp_path / "out",
        pyramid_min_size=4,
        lif_mosaic="bioio-lif",
    )

    scene_record = result["stores"][0]["per_scene"][0]
    mosaic = scene_record.get("mosaic")
    if mosaic is not None:
        assert mosaic.get("cascade_selected") is not True
        assert mosaic.get("stitcher") not in ("zarrmony-stage", "zarrmony-grid")


# ---------- 7. Explicit stage-stitch → still stage; cascade flag absent ----------


def test_explicit_stage_stitch_bypasses_cascade_flag(
    tmp_path: Path, patched_reader
) -> None:
    """When the user explicitly requests stage-stitch, the concrete stitcher
    is the same as what the cascade would pick — but cascade_selected must
    stay False so the audit accurately records intent.
    """
    reader = _cascade_reader(stage=True, grid=True)
    patched_reader(reader)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MosaicPlacementWarning)
        result = convert(
            "/tmp/x.lif",
            tmp_path / "out",
            pyramid_min_size=4,
            lif_mosaic="stage-stitch",
        )

    mosaic = _get_mosaic(result)
    assert mosaic["stitcher"] == "zarrmony-stage"
    assert mosaic.get("cascade_selected") is not True
