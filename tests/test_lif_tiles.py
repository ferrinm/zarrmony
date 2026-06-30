"""Tests for the stdlib-only Leica LIF tile-layout extractor.

Exercises a 3x3 mosaic with positions + overlap settings, a scene without
``<Tile>`` elements, malformed/oversized input, and the missing-overlap
graceful-degradation path.
"""

import pytest

from zarrmony.metadata.lif_tiles import extract_tile_layout


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
