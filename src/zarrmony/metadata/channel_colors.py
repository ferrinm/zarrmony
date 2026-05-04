"""Heuristic channel-name → hex-color mapping with palette fallback.

Used by the per-scene writer to assign default OMERO channel colors when the
caller doesn't supply explicit colors. Coverage targets the dyes most common in
fluorescence microscopy at Calico (DAPI, GFP, mCherry, Cy5 family) plus a
brightfield/DIC fallback. Any unmatched channel falls back to a small
round-robin palette.
"""

NAME_COLORS: dict[str, str] = {
    # Blue (UV / 405)
    "DAPI": "0099ff",
    "Hoechst": "0099ff",
    "BFP": "0040ff",
    # Green (488)
    "GFP": "00ff00",
    "EGFP": "00ff00",
    "AF488": "00ff00",
    "Alexa488": "00ff00",
    "FITC": "00ff00",
    # Yellow (515 / 561 short)
    "YFP": "ccff00",
    "Citrine": "ccff00",
    # Orange / Red (561 / 568)
    "RFP": "ff0000",
    "mCherry": "ff0000",
    "AF555": "ff8000",
    "Alexa555": "ff8000",
    "Cy3": "ff8000",
    "TexasRed": "ff4000",
    "TRITC": "ff4000",
    # Far red (640+)
    "AF647": "ff00ff",
    "Alexa647": "ff00ff",
    "Cy5": "ff00ff",
    "AF680": "ff00bb",
    "AF750": "808080",
    # Brightfield / transmission — display as white
    "Brightfield": "ffffff",
    "BF": "ffffff",
    "DIC": "ffffff",
    "Phase": "ffffff",
    "PhaseContrast": "ffffff",
}

PALETTE: tuple[str, ...] = (
    "00ff00",
    "ff0000",
    "0099ff",
    "ffff00",
    "00ffff",
    "ff00ff",
    "ff8000",
    "ffffff",
)


def color_for_channel(name: str, fallback_index: int = 0) -> str:
    """Return a hex color for ``name``.

    Lookup is case-insensitive. If no exact match, tries substring match (e.g.
    "DAPI_405" matches "DAPI"). Falls back to PALETTE[fallback_index % len].
    """
    name_lower = name.lower()

    for key, color in NAME_COLORS.items():
        if key.lower() == name_lower:
            return color

    for key, color in NAME_COLORS.items():
        if key.lower() in name_lower:
            return color

    return PALETTE[fallback_index % len(PALETTE)]


def colors_for_channels(
    channel_names: list[str],
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Resolve a hex color for each channel name.

    Per-channel overrides win over the heuristic. Unmatched channels round-robin
    through PALETTE in their position order.
    """
    overrides = overrides or {}
    return [
        overrides.get(name, color_for_channel(name, fallback_index=i))
        for i, name in enumerate(channel_names)
    ]
