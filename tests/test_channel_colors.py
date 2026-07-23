"""Tests for the emission-band colorblind palette (ADR-0007).

Covers:

- band-boundary edges (500 nm, 545 nm, 660 nm) for both emission and excitation,
- the emission → excitation → dye-name → palette resolution ladder,
- the dye-name fallback (used by CZI/ND2/OME-TIFF paths without wavelength),
- collision handling with ``ChannelColorCollisionWarning``,
- source-file color parsing (LIF LUTName, OME-XML ARGB int, hex),
- overrides (keyed on channel name / dye).

Uses the module-level color constants ``CYAN``/``GREEN``/``YELLOW``/``MAGENTA``/
``WHITE`` throughout so a test's intent reads at a glance without hex-matching.
"""

from __future__ import annotations

import warnings

import pytest

from zarrmony.errors import ChannelColorCollisionWarning
from zarrmony.metadata.channel_colors import (
    CYAN,
    DYE_TO_BAND,
    EMISSION_BANDS,
    EXCITATION_BANDS,
    GREEN,
    MAGENTA,
    UNKNOWN_PALETTE,
    WHITE,
    YELLOW,
    assign_colors,
    color_for_dye_name,
    color_for_emission_nm,
    color_for_excitation_nm,
    colors_for_channels,
    parse_source_color,
)

# --- band lookups: table contents + boundary edges --------------------------


def test_emission_band_table_covers_the_expected_bands() -> None:
    # Sanity-check that the table matches the ADR-0007 palette: exactly five
    # distinct colors across eight bands.
    band_colors = {c for (_lo, _hi, c) in EMISSION_BANDS}
    assert band_colors == {CYAN, GREEN, YELLOW, MAGENTA, WHITE}
    assert len(EMISSION_BANDS) == 8


def test_emission_dapi_is_cyan() -> None:
    # DAPI midpoint ~460 nm; under the deep-blue band.
    assert color_for_emission_nm(460) == CYAN


def test_emission_gfp_is_green() -> None:
    # EGFP midpoint ~510 nm.
    assert color_for_emission_nm(510) == GREEN


def test_emission_mcherry_is_magenta() -> None:
    # mCherry midpoint ~610 nm (red band).
    assert color_for_emission_nm(620) == MAGENTA


def test_emission_af647_is_white() -> None:
    assert color_for_emission_nm(670) == WHITE


def test_emission_boundary_500_lands_in_green() -> None:
    # Half-open [lo, hi): 500 nm exactly falls into the green [500, 545) band,
    # matching the "500–545 → green" reading of the ADR-0007 table.
    assert color_for_emission_nm(500) == GREEN


def test_emission_boundary_545_lands_in_yellow() -> None:
    assert color_for_emission_nm(545) == YELLOW


def test_emission_boundary_580_lands_in_yellow() -> None:
    # 580 nm sits at the yellow → orange edge; both bands map to yellow, but
    # 580 must not slip into the red band.
    assert color_for_emission_nm(580) == YELLOW


def test_emission_boundary_610_lands_in_magenta() -> None:
    assert color_for_emission_nm(610) == MAGENTA


def test_emission_boundary_660_lands_in_white() -> None:
    assert color_for_emission_nm(660) == WHITE


def test_emission_boundary_740_lands_in_white_nir() -> None:
    # 740 nm exactly enters the NIR band — still white by the palette.
    assert color_for_emission_nm(740) == WHITE


def test_emission_none_input_returns_none() -> None:
    assert color_for_emission_nm(None) is None


# --- excitation bands: laser-line buckets ----------------------------------


def test_excitation_dapi_405_is_cyan() -> None:
    assert color_for_excitation_nm(405) == CYAN


def test_excitation_af488_499_is_green() -> None:
    # AF488 excitation ~499 nm — must land in green, not the < 450 cyan band.
    assert color_for_excitation_nm(499) == GREEN


def test_excitation_af546_555_is_magenta() -> None:
    # Load-bearing for the ADR-0007 real-file example: AF546 at 555 nm
    # excitation is the yellow-orange laser bucket → magenta.
    assert color_for_excitation_nm(555) == MAGENTA


def test_excitation_af594_590_is_magenta() -> None:
    assert color_for_excitation_nm(590) == MAGENTA


def test_excitation_af647_651_is_white() -> None:
    assert color_for_excitation_nm(651) == WHITE


def test_excitation_boundary_620_lands_in_white() -> None:
    # [620, 720) → white; 620 exactly is red-laser territory.
    assert color_for_excitation_nm(620) == WHITE


def test_excitation_nir_is_white() -> None:
    assert color_for_excitation_nm(780) == WHITE


# --- dye name substring fallback (non-LIF readers) --------------------------


def test_dye_name_dapi_is_cyan() -> None:
    assert color_for_dye_name("DAPI") == CYAN


def test_dye_name_gfp_is_green() -> None:
    assert color_for_dye_name("GFP") == GREEN


def test_dye_name_mcherry_is_magenta() -> None:
    assert color_for_dye_name("mCherry") == MAGENTA


def test_dye_name_alexa_647_is_white() -> None:
    assert color_for_dye_name("AF647") == WHITE
    assert color_for_dye_name("Alexa 647") == WHITE


def test_dye_name_is_case_insensitive() -> None:
    assert color_for_dye_name("dapi") == CYAN
    assert color_for_dye_name("MCHERRY") == MAGENTA


def test_dye_name_substring_match() -> None:
    # Common real-world channel names include wavelength / conjugate suffixes.
    assert color_for_dye_name("DAPI_405") == CYAN
    assert color_for_dye_name("GFP-EM") == GREEN
    assert color_for_dye_name("Alexa 647 conjugate") == WHITE


def test_dye_name_unknown_returns_none() -> None:
    assert color_for_dye_name("WEIRD_CHANNEL") is None
    assert color_for_dye_name("") is None
    assert color_for_dye_name(None) is None


def test_dye_name_brightfield_is_white() -> None:
    assert color_for_dye_name("Brightfield") == WHITE
    assert color_for_dye_name("DIC") == WHITE


# --- source-file color parsing (LIF LUTName, OME-XML ARGB) ------------------


def test_parse_source_color_hex_string() -> None:
    assert parse_source_color("ff8800") == "ff8800"
    assert parse_source_color("#FF8800") == "ff8800"
    assert parse_source_color("00FFFF") == "00ffff"


def test_parse_source_color_lut_names() -> None:
    assert parse_source_color("Green") == GREEN
    assert parse_source_color("Blue") == "0000ff"
    assert parse_source_color("Magenta") == MAGENTA
    assert parse_source_color("Gray") == WHITE
    # Case-insensitive.
    assert parse_source_color("green") == GREEN


def test_parse_source_color_leica_gradient_lut() -> None:
    # "Gradient (R,G,B)" — the R,G,B triple is the display color.
    assert parse_source_color("Gradient (233,141,52)") == "e98d34"


def test_parse_source_color_argb_int() -> None:
    # OME-XML <Channel Color> is a signed 32-bit ARGB; the alpha byte drops.
    assert parse_source_color(0xFFFF8800) == "ff8800"
    assert parse_source_color(0x00FF8800) == "ff8800"


def test_parse_source_color_none_and_unknown_return_none() -> None:
    assert parse_source_color(None) is None
    assert parse_source_color("nonsense-name") is None
    assert parse_source_color("") is None
    assert parse_source_color(True) is None  # bool is not a color


# --- assign_colors: end-to-end resolution + collision handling --------------


def _info(**kw) -> dict:
    """Build a channel-info dict with defaults so tests read compactly."""
    base = {
        "name": None,
        "dye": None,
        "fluor": None,
        "excitation_nm": None,
        "emission_low_nm": None,
        "emission_high_nm": None,
    }
    base.update(kw)
    return base


def test_assign_colors_prefers_emission_over_excitation() -> None:
    # Emission midpoint 620 (magenta) beats excitation 405 (would be cyan).
    infos = [_info(emission_low_nm=610, emission_high_nm=630, excitation_nm=405)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assign_colors(infos) == [MAGENTA]


def test_assign_colors_falls_back_to_excitation_when_no_emission() -> None:
    infos = [_info(excitation_nm=555)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assign_colors(infos) == [MAGENTA]


def test_assign_colors_falls_back_to_dye_name_when_no_wavelength() -> None:
    infos = [_info(name="DAPI_405", dye="DAPI_405")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assign_colors(infos) == [CYAN]


def test_assign_colors_falls_back_to_palette_when_nothing_resolves() -> None:
    # Two mystery channels: distinct UNKNOWN_PALETTE slots, no collision.
    infos = [_info(name="MYSTERY_A"), _info(name="MYSTERY_B")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        colors = assign_colors(infos)
    assert colors[0] == UNKNOWN_PALETTE[0]
    assert colors[1] == UNKNOWN_PALETTE[1]


def test_assign_colors_reallocates_collisions_and_warns() -> None:
    # Two AF647 channels → both would land on white. Second reallocates through
    # UNKNOWN_PALETTE and the warning fires.
    infos = [
        _info(name="AF647-1", dye="AF647", emission_low_nm=670, emission_high_nm=680),
        _info(name="AF647-2", dye="AF647", emission_low_nm=670, emission_high_nm=680),
    ]
    with pytest.warns(ChannelColorCollisionWarning) as record:
        colors = assign_colors(infos)
    assert colors[0] == WHITE
    assert colors[1] != WHITE
    assert colors[1] in UNKNOWN_PALETTE
    assert len(record) == 1
    body = str(record[0].message)
    assert "AF647-2" in body


def test_assign_colors_source_file_beats_band_scheme() -> None:
    # A channel with emission that would be magenta gets overridden by its
    # source-file color hint (a hex).
    infos = [_info(emission_low_nm=610, emission_high_nm=630)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assign_colors(infos, source_file_colors=["ff8800"]) == ["ff8800"]


def test_assign_colors_overrides_beat_everything() -> None:
    # Override on channel `name` wins over even the source-file color.
    infos = [_info(name="DAPI", emission_low_nm=440, emission_high_nm=480)]
    result = assign_colors(
        infos,
        source_file_colors=[GREEN],
        overrides={"DAPI": "112233"},
    )
    assert result == ["112233"]


def test_assign_colors_overrides_case_insensitive() -> None:
    infos = [_info(name="DAPI", dye="DAPI (dsDNA bound)")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = assign_colors(infos, overrides={"dapi": "abcdef"})
    assert result == ["abcdef"]


def test_assign_colors_source_file_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        assign_colors([_info(name="X")], source_file_colors=["ff8800", "112233"])


def test_assign_colors_empty_input_is_empty() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assign_colors([]) == []


# --- colors_for_channels: the name-only adapter used by non-LIF paths ------


def test_colors_for_channels_uses_dye_name_fallback() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = colors_for_channels(["DAPI", "GFP", "mCherry", "AF647"])
    assert out == [CYAN, GREEN, MAGENTA, WHITE]


def test_colors_for_channels_with_overrides() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = colors_for_channels(
            ["DAPI", "GFP", "Custom_X"], overrides={"Custom_X": "abcdef"}
        )
    assert out == [CYAN, GREEN, "abcdef"]


def test_colors_for_channels_unknown_name_uses_palette() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = colors_for_channels(["WEIRD_A", "WEIRD_B"])
    assert out[0] == UNKNOWN_PALETTE[0]
    assert out[1] == UNKNOWN_PALETTE[1]


# --- the ADR-0007 acceptance-criterion real-file case ----------------------


def test_dapi_af546_af647_produces_cyan_magenta_white() -> None:
    """The reporter's real file — DAPI 405 nm exc / AF546 555 nm exc / AF647 651 nm exc.

    The three channels are shaped as excitation-only (no emission band) to
    match a reader path where filter cubes weren't surfaced. The default
    emission-band scheme must still land them on the CMY palette:
    cyan (deep-blue) / magenta (AF546 red-emitter) / white (far-red).
    """
    infos = [
        _info(name="DAPI", dye="DAPI", excitation_nm=405),
        _info(name="Alexa 546", dye="ALEXA 546", excitation_nm=555),
        _info(name="Alexa 647", dye="ALEXA 647", excitation_nm=651),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        colors = assign_colors(infos)
    assert colors == [CYAN, MAGENTA, WHITE]


# --- DYE_TO_BAND table smoke tests -----------------------------------------


def test_dye_to_band_table_uses_palette_colors_only() -> None:
    # The DYE_TO_BAND lookup is a name → band-color shim; every value must be
    # a palette color, not some legacy true-color hex.
    palette = {CYAN, GREEN, YELLOW, MAGENTA, WHITE}
    for name, color in DYE_TO_BAND.items():
        assert color in palette, f"{name} maps to non-palette color {color}"


def test_dye_to_band_includes_all_four_red_family_slots() -> None:
    # ADR-0007 subdivides the red side into four visually-distinct slots.
    assert DYE_TO_BAND["mOrange"] == YELLOW
    assert DYE_TO_BAND["mCherry"] == MAGENTA
    assert DYE_TO_BAND["AF647"] == WHITE
    assert DYE_TO_BAND["Cy7"] == WHITE


def test_dye_to_band_and_emission_bands_agree_on_palette() -> None:
    # Sanity: every band color that appears in EMISSION_BANDS also appears in
    # DYE_TO_BAND (so a dye-name fallback can reach every band).
    emission_colors = {c for (_lo, _hi, c) in EMISSION_BANDS}
    dye_band_colors = set(DYE_TO_BAND.values())
    assert emission_colors.issubset(dye_band_colors)


def test_excitation_bands_use_the_same_palette() -> None:
    palette = {CYAN, GREEN, YELLOW, MAGENTA, WHITE}
    for _lo, _hi, color in EXCITATION_BANDS:
        assert color in palette
