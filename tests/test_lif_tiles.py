"""Tests for the stdlib-only Leica LIF tile-layout extractor + grid reassembly.

Exercises a 3x3 mosaic with positions + overlap settings, a scene without
``<Tile>`` elements, malformed/oversized input, the missing-overlap
graceful-degradation path, and the ``reassemble_grid`` / ``validate_grid_layout``
/ ``grid_shape`` helpers used by the ``lif_mosaic="grid-stitch"`` writer path.
"""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from zarrmony.metadata.lif_tiles import (
    extract_tile_layout,
    grid_shape,
    reassemble_grid,
    validate_grid_layout,
)


def _scene_xml(tiles_xml: str = "", stitching_xml: str = "") -> str:
    """Wrap tile/stitching snippets in the smallest realistic scene scaffold."""
    return (
        f"<Element><Data><Image>"
        f'<Attachment Name="TileScanInfo" Application="LAS AF">{tiles_xml}</Attachment>'
        f'<Attachment Name="HardwareSetting">{stitching_xml}</Attachment>'
        f"</Image></Data></Element>"
    )


def _grid_3x3_tiles() -> str:
    out = []
    for fy in range(3):
        for fx in range(3):
            pos_x = 0.0400 + fx * 0.0005
            pos_y = 0.0170 + fy * 0.0005
            out.append(
                f'<Tile FieldX="{fx}" FieldY="{fy}" '
                f'PosX="{pos_x:.10f}" PosY="{pos_y:.10f}" PosZ="0.0117032516" />'
            )
    return "".join(out)


# --- 3x3 mosaic with positions and overlap settings ------------------------


def test_3x3_mosaic_extracts_nine_tiles_in_document_order() -> None:
    stitching = (
        '<StitchingSettings IsAutoStitching="1" OverlapXmanual="0" '
        'OverlapYmanual="0" OverlapPercentageX="0.10" OverlapPercentageY="0.10" />'
    )
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles(), stitching))
    assert out is not None
    assert len(out["tiles"]) == 9
    assert [(t["field_x"], t["field_y"]) for t in out["tiles"]] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    ]


def test_3x3_mosaic_preserves_positions_in_meters_verbatim() -> None:
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles()))
    assert out is not None
    first = out["tiles"][0]
    assert first == {
        "field_x": 0,
        "field_y": 0,
        "pos_x_m": 0.04,
        "pos_y_m": 0.017,
        "pos_z_m": 0.0117032516,
    }


def test_overlap_percent_is_converted_from_decimal_fraction() -> None:
    # LIF stores 0.10 to mean 10%; the surface key represents actual percent.
    stitching = (
        '<StitchingSettings OverlapPercentageX="0.10" OverlapPercentageY="0.15" />'
    )
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles(), stitching))
    assert out is not None
    assert out["intended_overlap_x_pct"] == pytest.approx(10.0)
    assert out["intended_overlap_y_pct"] == pytest.approx(15.0)


def test_extractor_keys_are_exactly_the_contract() -> None:
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles()))
    assert out is not None
    assert set(out) == {"tiles", "intended_overlap_x_pct", "intended_overlap_y_pct"}
    for tile in out["tiles"]:
        assert set(tile) == {"field_x", "field_y", "pos_x_m", "pos_y_m", "pos_z_m"}


# --- absent / partial overlap settings -------------------------------------


def test_missing_stitching_settings_yields_none_overlaps() -> None:
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles()))
    assert out is not None
    assert out["intended_overlap_x_pct"] is None
    assert out["intended_overlap_y_pct"] is None


def test_missing_overlap_attributes_degrade_independently() -> None:
    # OverlapPercentageY present but X missing — surface degrades only the missing axis.
    stitching = '<StitchingSettings OverlapPercentageY="0.07" />'
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles(), stitching))
    assert out is not None
    assert out["intended_overlap_x_pct"] is None
    assert out["intended_overlap_y_pct"] == pytest.approx(7.0)


def test_unparseable_overlap_attribute_degrades_to_none() -> None:
    stitching = '<StitchingSettings OverlapPercentageX="not-a-number" OverlapPercentageY="0.05" />'
    out = extract_tile_layout(_scene_xml(_grid_3x3_tiles(), stitching))
    assert out is not None
    assert out["intended_overlap_x_pct"] is None
    assert out["intended_overlap_y_pct"] == pytest.approx(5.0)


# --- absence / fail-closed paths -------------------------------------------


def test_scene_with_no_tile_elements_returns_none() -> None:
    assert extract_tile_layout(_scene_xml("", "")) is None
    assert extract_tile_layout("<Element/>") is None


def test_malformed_xml_returns_none() -> None:
    assert extract_tile_layout("<Element><Data><Image") is None
    assert extract_tile_layout("not xml at all") is None


def test_empty_and_nonstring_return_none() -> None:
    assert extract_tile_layout("") is None
    assert extract_tile_layout(None) is None  # type: ignore[arg-type]


def test_oversized_input_returns_none() -> None:
    huge = "<root>" + ("x" * (33 * 1024 * 1024)) + "</root>"
    assert extract_tile_layout(huge) is None


def test_doctype_is_rejected() -> None:
    assert extract_tile_layout('<!DOCTYPE r SYSTEM "x.dtd"><r/>') is None


def test_external_entity_is_rejected() -> None:
    xxe = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        "<root>&x;</root>"
    )
    assert extract_tile_layout(xxe) is None


# --- per-tile graceful degradation -----------------------------------------


def test_partially_stamped_tile_keeps_other_fields() -> None:
    # A tile missing PosZ still names its grid slot — degrade per field, not per tile.
    tile = '<Tile FieldX="0" FieldY="0" PosX="0.04" PosY="0.017" />'
    out = extract_tile_layout(_scene_xml(tile))
    assert out is not None
    assert out["tiles"] == [
        {
            "field_x": 0,
            "field_y": 0,
            "pos_x_m": 0.04,
            "pos_y_m": 0.017,
            "pos_z_m": None,
        }
    ]


def test_non_finite_positions_degrade_to_none() -> None:
    tile = '<Tile FieldX="0" FieldY="0" PosX="inf" PosY="nan" PosZ="0.01" />'
    out = extract_tile_layout(_scene_xml(tile))
    assert out is not None
    assert out["tiles"][0]["pos_x_m"] is None
    assert out["tiles"][0]["pos_y_m"] is None
    assert out["tiles"][0]["pos_z_m"] == 0.01


# --- grid_shape ------------------------------------------------------------


def test_grid_shape_returns_rows_cols_from_max_field_indices() -> None:
    tiles = [
        {"field_x": 0, "field_y": 0},
        {"field_x": 1, "field_y": 0},
        {"field_x": 0, "field_y": 1},
        {"field_x": 1, "field_y": 1},
    ]
    assert grid_shape(tiles) == (2, 2)


def test_grid_shape_returns_none_none_when_any_field_missing() -> None:
    tiles = [
        {"field_x": 0, "field_y": 0},
        {"field_x": None, "field_y": 0},
    ]
    assert grid_shape(tiles) == (None, None)


# --- validate_grid_layout --------------------------------------------------


def _tiles_from_grid(pairs: list[tuple[int, int]]) -> list[dict]:
    """Build minimal tile dicts from a list of (field_x, field_y) pairs."""
    return [
        {"field_x": x, "field_y": y, "pos_x_m": 0.0, "pos_y_m": 0.0, "pos_z_m": 0.0}
        for x, y in pairs
    ]


def test_validate_grid_layout_returns_rows_cols_on_complete_grid() -> None:
    tiles = _tiles_from_grid([(x, y) for y in range(2) for x in range(3)])
    assert validate_grid_layout(tiles, m_size=6) == (2, 3)


def test_validate_grid_layout_rejects_empty_tiles_and_names_escape() -> None:
    with pytest.raises(ValueError, match=r"no <Tile> entries.*per-tile"):
        validate_grid_layout([], m_size=4)


def test_validate_grid_layout_rejects_count_mismatch_and_names_escape() -> None:
    tiles = _tiles_from_grid([(0, 0), (1, 0), (0, 1), (1, 1)])
    with pytest.raises(
        ValueError, match=r"tile-metadata count \(4\).*M dim \(3\).*per-tile"
    ):
        validate_grid_layout(tiles, m_size=3)


def test_validate_grid_layout_rejects_missing_field_indices() -> None:
    tiles = [
        {"field_x": 0, "field_y": 0, "pos_x_m": 0, "pos_y_m": 0, "pos_z_m": 0},
        {"field_x": None, "field_y": 0, "pos_x_m": 0, "pos_y_m": 0, "pos_z_m": 0},
    ]
    with pytest.raises(ValueError, match=r"missing FieldX and/or FieldY.*per-tile"):
        validate_grid_layout(tiles, m_size=2)


def test_validate_grid_layout_rejects_duplicate_slots() -> None:
    tiles = _tiles_from_grid([(0, 0), (0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValueError, match=r"duplicate \(field_x, field_y\)"):
        validate_grid_layout(tiles, m_size=4)


def test_validate_grid_layout_rejects_incomplete_rectangle() -> None:
    # Gap at (1,1) — 3 tiles, max_x=1, max_y=1 → expects 2x2 grid but only 3 filled.
    tiles = _tiles_from_grid([(0, 0), (1, 0), (0, 1)])
    with pytest.raises(ValueError, match=r"complete rectangular grid.*2x2"):
        validate_grid_layout(tiles, m_size=3)


# --- reassemble_grid -------------------------------------------------------


def _tiles_xarr(
    m_order: list[tuple[int, int]], tile_h: int = 4, tile_w: int = 4
) -> xr.DataArray:
    """Build a raw M-intact xarray where M=i is filled with value i+1.

    ``m_order`` is deliberately shuffled from grid order so tests can prove
    reassemble_grid places tiles by declared (fx, fy), not by M index.
    """
    m = len(m_order)
    arr = np.zeros((m, 1, 1, tile_h, tile_w), dtype=np.uint16)
    for i in range(m):
        arr[i, ...] = i + 1
    return xr.DataArray(da.from_array(arr), dims=["M", "C", "Z", "Y", "X"])


def test_reassemble_grid_places_tile_at_declared_field_slot() -> None:
    # 3x3 grid; M-order is NOT grid-order — M=0 declared at (fx=2, fy=1).
    # A correct reassembler must place M=0's pixels at (row=1, col=2), not (0,0).
    m_order = [
        (2, 1),  # M=0 — the acceptance case: bioio-lif would put this at (0,0)
        (0, 0),  # M=1
        (1, 0),  # M=2
        (2, 0),  # M=3
        (0, 1),  # M=4
        (1, 1),  # M=5
        (0, 2),  # M=6
        (1, 2),  # M=7
        (2, 2),  # M=8
    ]
    tile_h, tile_w = 4, 4
    tiles_xarr = _tiles_xarr(m_order, tile_h=tile_h, tile_w=tile_w)
    tile_layout = {
        "tiles": _tiles_from_grid(m_order),
        "intended_overlap_x_pct": None,
        "intended_overlap_y_pct": None,
    }

    canvas = reassemble_grid(tiles_xarr, tile_layout)

    assert "M" not in canvas.dims
    assert canvas.sizes["Y"] == 3 * tile_h
    assert canvas.sizes["X"] == 3 * tile_w

    data = canvas.data.compute()
    for m, (fx, fy) in enumerate(m_order):
        y0, y1 = fy * tile_h, (fy + 1) * tile_h
        x0, x1 = fx * tile_w, (fx + 1) * tile_w
        block = data[..., y0:y1, x0:x1]
        assert (
            block == m + 1
        ).all(), f"tile M={m} declared (fx={fx}, fy={fy}) did not land at that slot"


def test_reassemble_grid_returns_dask_backed_xarray() -> None:
    tiles_xarr = _tiles_xarr([(0, 0), (1, 0), (0, 1), (1, 1)])
    tile_layout = {
        "tiles": _tiles_from_grid([(0, 0), (1, 0), (0, 1), (1, 1)]),
        "intended_overlap_x_pct": None,
        "intended_overlap_y_pct": None,
    }
    canvas = reassemble_grid(tiles_xarr, tile_layout)
    # No eager compute — the underlying array is still lazy dask.
    assert isinstance(canvas.data, da.Array)


def test_reassemble_grid_raises_on_missing_layout() -> None:
    tiles_xarr = _tiles_xarr([(0, 0), (1, 0), (0, 1), (1, 1)])
    with pytest.raises(ValueError, match=r"per-tile grid metadata.*per-tile"):
        reassemble_grid(tiles_xarr, None)


def test_reassemble_grid_raises_on_empty_tiles_list() -> None:
    tiles_xarr = _tiles_xarr([(0, 0), (1, 0), (0, 1), (1, 1)])
    with pytest.raises(ValueError, match=r"no <Tile> entries"):
        reassemble_grid(
            tiles_xarr,
            {
                "tiles": [],
                "intended_overlap_x_pct": None,
                "intended_overlap_y_pct": None,
            },
        )


def test_reassemble_grid_preserves_non_m_coords() -> None:
    # Attach a C coord — must survive the reassembly (grid_stitch preserves
    # channel identity, T/C/Z axes untouched).
    m_order = [(0, 0), (1, 0), (0, 1), (1, 1)]
    tiles_xarr = _tiles_xarr(m_order)
    tiles_xarr = tiles_xarr.assign_coords(C=["DAPI"])
    tile_layout = {
        "tiles": _tiles_from_grid(m_order),
        "intended_overlap_x_pct": None,
        "intended_overlap_y_pct": None,
    }
    canvas = reassemble_grid(tiles_xarr, tile_layout)
    assert "C" in canvas.coords
    assert list(canvas.coords["C"].values) == ["DAPI"]
