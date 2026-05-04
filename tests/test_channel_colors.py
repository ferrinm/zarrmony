from zarrmony.metadata.channel_colors import (
    NAME_COLORS,
    PALETTE,
    color_for_channel,
    colors_for_channels,
)


def test_dapi_is_blue() -> None:
    assert color_for_channel("DAPI") == NAME_COLORS["DAPI"]


def test_gfp_is_green() -> None:
    assert color_for_channel("GFP") == "00ff00"


def test_mcherry_is_red() -> None:
    assert color_for_channel("mCherry") == "ff0000"


def test_cy5_is_magenta() -> None:
    assert color_for_channel("Cy5") == "ff00ff"


def test_brightfield_is_white() -> None:
    assert color_for_channel("Brightfield") == "ffffff"
    assert color_for_channel("DIC") == "ffffff"


def test_lookup_is_case_insensitive() -> None:
    assert color_for_channel("dapi") == color_for_channel("DAPI")
    assert color_for_channel("MCHERRY") == color_for_channel("mCherry")


def test_substring_match() -> None:
    # Common in real channel names: dye + wavelength suffix
    assert color_for_channel("DAPI_405") == color_for_channel("DAPI")
    assert color_for_channel("GFP-EM") == color_for_channel("GFP")


def test_unknown_falls_back_to_palette() -> None:
    assert color_for_channel("WEIRD_CHANNEL", fallback_index=0) == PALETTE[0]
    assert color_for_channel("WEIRD_CHANNEL", fallback_index=1) == PALETTE[1]


def test_palette_wraps_around() -> None:
    big_idx = len(PALETTE) + 2
    assert color_for_channel("UNKNOWN", fallback_index=big_idx) == PALETTE[2]


def test_colors_for_channels_with_overrides() -> None:
    names = ["DAPI", "GFP", "Custom_X"]
    overrides = {"Custom_X": "abcdef"}
    out = colors_for_channels(names, overrides=overrides)
    assert out[0] == NAME_COLORS["DAPI"]
    assert out[1] == "00ff00"
    assert out[2] == "abcdef"


def test_colors_for_channels_no_overrides() -> None:
    out = colors_for_channels(["DAPI", "GFP"])
    assert out == [NAME_COLORS["DAPI"], "00ff00"]
