"""Emission-band → colorblind-friendly display color mapping (ADR-0007).

Assigns per-channel hex display colors from **what the eye sees** — the
fluorophore's emission midpoint — rather than the dye's true color. Uses the
Nature Methods "Points of View" CMY / green palette so overlays stay
distinguishable under red-green colorblindness (~8% of Northern European males)
and stack cleanly for 3–4 channel merged images.

Resolution ladder (per :func:`assign_colors`):

1. Per-channel override from ``convert(..., channel_colors={<name>: <hex6>})``.
2. Source file's stored channel color (opt-in via ``channel_colors="source-file"``,
   parsed by :func:`parse_source_color` from LIF ``LUTName`` / OME-XML
   ``Channel@Color``).
3. Emission midpoint → :data:`EMISSION_BANDS`.
4. Excitation → :data:`EXCITATION_BANDS` (blueshifted, laser-line oriented).
5. Dye-name substring → :data:`DYE_TO_BAND` (for readers without wavelengths).
6. Round-robin through :data:`UNKNOWN_PALETTE`.

After per-channel resolution, :func:`assign_colors` walks the list left-to-right
and — when a later channel collides with an earlier one — reallocates it out of
``UNKNOWN_PALETTE`` (skipping already-taken colors) and fires
:class:`ChannelColorCollisionWarning`. First-in-order channels never move.

Band boundaries are half-open ``[lo, hi)`` intervals so an exact-boundary
emission (500 nm, 545 nm, 580 nm, 610 nm, 660 nm, 740 nm) lands in the *upper*
band — matches the "500–545" reading of the table where 500 is green not cyan.
"""

from __future__ import annotations

import re
import warnings

from zarrmony.errors import ChannelColorCollisionWarning

# The five colors of the palette — kept as module constants so tests and
# downstream code can reference them symbolically instead of matching hex.
CYAN = "00ffff"
GREEN = "00ff00"
YELLOW = "ffff00"
MAGENTA = "ff00ff"
WHITE = "ffffff"

# The primaries, used *only* for a folded samples axis (see
# ``transforms.fold_samples_axis``) and deliberately outside the palette above.
# Those five hues encode fluorescence emission bands and are chosen to stay
# distinguishable for colorblind viewers. An RGB image's samples are not
# emissions — they are the literal red, green and blue components of a colour
# photograph, and compositing them in any other hue reproduces the wrong
# picture. Alpha gets white because it modulates rather than carries colour.
RED = "ff0000"
BLUE = "0000ff"
RGB_SAMPLE_LABELS: tuple[str, ...] = ("Red", "Green", "Blue", "Alpha")
RGB_SAMPLE_COLORS: tuple[str, ...] = (RED, GREEN, BLUE, WHITE)


def sample_axis_channels(n_samples: int) -> tuple[list[str], list[str]]:
    """``(labels, colors)`` for a samples axis of size ``n_samples``.

    Covers the interleaved layouts Bio-Formats reports as ``S``: 3 (RGB) and 4
    (RGBA) are the ones that occur in practice, and 2 shows up in the odd
    two-sample scan. Anything wider is not a colour model we can name, so it
    degrades to positional labels in white rather than guessing a mapping.
    """
    if n_samples <= len(RGB_SAMPLE_LABELS):
        return (
            list(RGB_SAMPLE_LABELS[:n_samples]),
            list(RGB_SAMPLE_COLORS[:n_samples]),
        )
    return ([f"S:{i}" for i in range(n_samples)], [WHITE] * n_samples)


# Emission-midpoint → color, per the ADR-0007 table. Half-open intervals so
# exact-boundary emissions (500, 545, 580, 610, 660, 740 nm) land in the upper
# band. NIR + brightfield/DIC/phase both surface as white (the "no principled
# color" bucket).
EMISSION_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 450.0, CYAN),  # deep-blue / UV (DAPI, Hoechst)
    (450.0, 500.0, CYAN),  # cyan (CFP, BFP, mTurquoise)
    (500.0, 545.0, GREEN),  # green (GFP, AF488, FITC)
    (545.0, 580.0, YELLOW),  # yellow (YFP, Citrine)
    (580.0, 610.0, YELLOW),  # orange (mOrange, AF555, Cy3)
    (610.0, 660.0, MAGENTA),  # red (mCherry, AF546/568/594)
    (660.0, 740.0, WHITE),  # far-red (AF647, Cy5, AF680)
    (740.0, float("inf"), WHITE),  # NIR (Cy7, AF750)
)

# Excitation-only fallback bands. Blueshifted from the emission table and
# clustered around common Leica laser lines so a channel that reports only its
# excitation wavelength still lands in the right colorblind slot:
#
#     laser line → typical dye class → band color
#     405        → DAPI/CFP/BFP     → cyan
#     488        → GFP/AF488        → green
#     514/532    → YFP/Citrine      → yellow
#     552/561/594→ AF546/AF555/     → magenta
#                   mCherry/AF594
#     633/640/647→ AF647/Cy5        → white
#     730+       → NIR              → white
#
# The user-file example that drove ADR-0007 (DAPI 405 nm exc / AF546 555 nm exc
# / AF647 651 nm exc → cyan / magenta / white) lands correctly here because
# the yellow-orange laser bucket [545, 620) is a single magenta band.
EXCITATION_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 450.0, CYAN),  # UV / 405 laser
    (450.0, 500.0, GREEN),  # 488 laser
    (500.0, 545.0, YELLOW),  # 514/532 lasers
    (545.0, 620.0, MAGENTA),  # 552/561/594 lasers
    (620.0, 720.0, WHITE),  # 633/640/647 red lasers
    (720.0, float("inf"), WHITE),  # NIR
)

# Dye-name substring → band color, for readers that don't surface wavelength
# (CZI / ND2 / OME-TIFF paths that never populate emission or excitation).
# Keys are case-insensitive substrings; longer keys win to avoid "Cy5" also
# matching "Cy55" or "Cy2" matching "Cy20". Colors come from the same palette
# constants above so the band table stays the single source of truth.
DYE_TO_BAND: dict[str, str] = {
    # Deep-blue / UV
    "DAPI": CYAN,
    "Hoechst": CYAN,
    # Cyan
    "CFP": CYAN,
    "Cerulean": CYAN,
    "mTurquoise": CYAN,
    "BFP": CYAN,
    # Green
    "GFP": GREEN,
    "EGFP": GREEN,
    "Alexa Fluor 488": GREEN,
    "Alexa 488": GREEN,
    "Alexa488": GREEN,
    "AF488": GREEN,
    "FITC": GREEN,
    "FAM": GREEN,
    "Cy2": GREEN,
    "mNeonGreen": GREEN,
    "Emerald": GREEN,
    # Yellow (yellow-green fluorescent proteins)
    "YFP": YELLOW,
    "Citrine": YELLOW,
    "mCitrine": YELLOW,
    "Venus": YELLOW,
    "Alexa 514": YELLOW,
    "AF514": YELLOW,
    "Alexa 532": YELLOW,
    "AF532": YELLOW,
    # Orange (still yellow on the CMY palette)
    "mOrange": YELLOW,
    "dTomato": YELLOW,
    "Alexa 555": YELLOW,
    "AF555": YELLOW,
    "Cy3": YELLOW,
    "TRITC": YELLOW,
    # Red (magenta on the CMY palette)
    "mCherry": MAGENTA,
    "mRFP": MAGENTA,
    "RFP": MAGENTA,
    "Alexa 546": MAGENTA,
    "AF546": MAGENTA,
    "Alexa 568": MAGENTA,
    "AF568": MAGENTA,
    "Alexa 594": MAGENTA,
    "AF594": MAGENTA,
    "Texas Red": MAGENTA,
    "TexasRed": MAGENTA,
    # Far-red (white)
    "Alexa 647": WHITE,
    "AF647": WHITE,
    "Cy5": WHITE,
    "Alexa 680": WHITE,
    "AF680": WHITE,
    "iRFP670": WHITE,
    # NIR (white)
    "Cy7": WHITE,
    "Alexa 700": WHITE,
    "AF700": WHITE,
    "Alexa 750": WHITE,
    "AF750": WHITE,
    "Alexa 790": WHITE,
    "AF790": WHITE,
    # Brightfield / transmission (white)
    "Brightfield": WHITE,
    "BF": WHITE,
    "DIC": WHITE,
    "Phase": WHITE,
    "PhaseContrast": WHITE,
}

# Round-robin palette for channels that don't resolve any other way, and for
# collision reallocation. Ordering keeps the CMY primary triple first so
# unresolved channels stay visually coherent alongside band-assigned ones.
UNKNOWN_PALETTE: tuple[str, ...] = (CYAN, GREEN, MAGENTA, YELLOW, WHITE)

# Named LUTs a source file might store as its per-channel display color. Leica
# LAS X writes ``<ChannelDescription LUTName="Green">``, ``LUTName="Blue"``,
# ``LUTName="Gradient (R,G,B)"``, etc.; OME-XML uses a signed 32-bit ARGB int.
# Only names that unambiguously resolve to a display color live here.
_LUT_NAME_TO_HEX: dict[str, str] = {
    "blue": "0000ff",
    "cyan": CYAN,
    "green": GREEN,
    "yellow": YELLOW,
    "orange": "ff8000",
    "red": "ff0000",
    "magenta": MAGENTA,
    "white": WHITE,
    "gray": WHITE,
    "grey": WHITE,
}
_GRADIENT_LUT_RE = re.compile(
    r"^\s*Gradient\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$",
    re.IGNORECASE,
)
_HEX6_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _band_lookup(bands: tuple[tuple[float, float, str], ...], nm: float) -> str | None:
    """Return the hex color whose ``[lo, hi)`` interval contains ``nm``."""
    for lo, hi, hex_ in bands:
        if lo <= nm < hi:
            return hex_
    return None


def color_for_emission_nm(nm: float | int | None) -> str | None:
    """Look up ``nm`` in :data:`EMISSION_BANDS`; ``None`` if out of range/None."""
    if nm is None:
        return None
    return _band_lookup(EMISSION_BANDS, float(nm))


def color_for_excitation_nm(nm: float | int | None) -> str | None:
    """Look up ``nm`` in :data:`EXCITATION_BANDS`; ``None`` if out of range/None."""
    if nm is None:
        return None
    return _band_lookup(EXCITATION_BANDS, float(nm))


def color_for_dye_name(name: str | None) -> str | None:
    """Substring-match ``name`` against :data:`DYE_TO_BAND` (case-insensitive).

    Longest key wins so ``"AF488"`` isn't shadowed by a shorter substring.
    Returns ``None`` when nothing matches.
    """
    if not name:
        return None
    lower = name.lower()
    matches = sorted(
        (k for k in DYE_TO_BAND if k.lower() in lower),
        key=len,
        reverse=True,
    )
    if matches:
        return DYE_TO_BAND[matches[0]]
    return None


def parse_source_color(raw: str | int | None) -> str | None:
    """Interpret a source file's stored per-channel color hint as a hex6 string.

    Accepts:

    - a plain hex6 string ``"RRGGBB"`` (case-insensitive, optional leading ``#``);
    - a named LUT (``"Green"``, ``"Blue"``, ``"Magenta"``, …) — the set
      Leica LAS X writes as ``<ChannelDescription LUTName="...">``;
    - a Leica gradient LUT (``"Gradient (R,G,B)"``) — the ``R,G,B`` triple is
      taken as the display color;
    - a signed 32-bit ARGB integer (from OME-XML ``<Channel Color>``) — the
      alpha byte is dropped.

    Returns ``None`` when the input can't be resolved so the caller falls
    through to the band scheme.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        rgb = raw & 0x00FFFFFF
        return f"{rgb:06x}"
    if not isinstance(raw, str):
        return None
    text = raw.strip().lstrip("#")
    if _HEX6_RE.match(text):
        return text.lower()
    hit = _LUT_NAME_TO_HEX.get(text.lower())
    if hit is not None:
        return hit
    m = _GRADIENT_LUT_RE.match(text)
    if m:
        r, g, b = (int(x) & 0xFF for x in m.groups())
        return f"{r:02x}{g:02x}{b:02x}"
    return None


def _emission_midpoint(
    low: float | int | None, high: float | int | None
) -> float | None:
    if low is None or high is None:
        return None
    return (float(low) + float(high)) / 2.0


def _normalize_hex(color: str) -> str:
    """Lowercase a hex color and strip an optional leading ``#``."""
    return color.lstrip("#").lower()


def _resolve_one(info: dict, source_color: str | None) -> str | None:
    """Resolve a single channel down the wavelength → dye ladder; ``None`` if unresolved.

    Source-file color, when provided, wins over every wavelength/name lookup —
    that's the whole point of ``channel_colors="source-file"``. Palette-index
    fallback is left to the caller.
    """
    if source_color is not None:
        return _normalize_hex(source_color)
    midpoint = _emission_midpoint(
        info.get("emission_low_nm"), info.get("emission_high_nm")
    )
    color = color_for_emission_nm(midpoint)
    if color is not None:
        return color
    color = color_for_excitation_nm(info.get("excitation_nm"))
    if color is not None:
        return color
    color = color_for_dye_name(info.get("dye") or info.get("name"))
    if color is not None:
        return color
    return None


def _override_for(info: dict, overrides: dict[str, str]) -> str | None:
    """Look up ``overrides`` by channel name / dye (case-insensitive fallback).

    The user's override key can match any of the identifiers we know for the
    channel — the omero label (``"DAPI (405 nm)"``), the raw dye
    (``"DAPI (dsDNA bound)"``), or the cleaned dye / channel name
    (``"DAPI"``). First hit wins.
    """
    if not overrides:
        return None
    lower_overrides = {k.lower(): v for k, v in overrides.items()}
    for key in (
        info.get("name"),
        info.get("dye"),
        info.get("fluor"),
    ):
        if key is None:
            continue
        if key in overrides:
            return overrides[key]
        if key.lower() in lower_overrides:
            return lower_overrides[key.lower()]
    return None


def _channel_label(info: dict, index: int) -> str:
    """Best-effort human name for a channel, for warning bodies."""
    return str(
        info.get("name") or info.get("dye") or info.get("fluor") or f"channel {index}"
    )


def assign_colors(
    channel_infos: list[dict],
    *,
    source_file_colors: list[str | None] | None = None,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Resolve display colors for a batch of channels, with collision handling.

    ``channel_infos[i]`` is a dict that may include ``name``, ``dye``,
    ``fluor``, ``excitation_nm``, ``emission_low_nm``, ``emission_high_nm``.
    Missing fields degrade — a dict with only ``name`` still lands via the
    dye-name substring path.

    Per-channel resolution ladder (top wins):

    1. ``overrides`` — keyed by channel name / dye / fluor (case-insensitive).
    2. ``source_file_colors[i]`` — the parsed source-file color, or ``None``.
    3. Emission midpoint → :func:`color_for_emission_nm`.
    4. Excitation → :func:`color_for_excitation_nm`.
    5. Dye name substring → :func:`color_for_dye_name`.
    6. ``UNKNOWN_PALETTE[i % len]`` — position-based fallback.

    After per-channel resolution, walk left-to-right and reallocate later
    duplicates out of ``UNKNOWN_PALETTE`` (skipping already-taken colors);
    the first-in-order channel keeps its natural color. Fires
    :class:`ChannelColorCollisionWarning` once per batch when any reallocation
    happens, naming the affected channels and the colors they would have taken.
    """
    n = len(channel_infos)
    if source_file_colors is None:
        source_file_colors = [None] * n
    if len(source_file_colors) != n:
        raise ValueError(
            f"source_file_colors length {len(source_file_colors)} != "
            f"channel_infos length {n}"
        )
    overrides = overrides or {}

    # Pass 1: per-channel resolution (no cross-channel awareness).
    resolved: list[str] = []
    for i, info in enumerate(channel_infos):
        override = _override_for(info, overrides)
        if override is not None:
            resolved.append(_normalize_hex(override))
            continue
        color = _resolve_one(info, source_file_colors[i])
        if color is None:
            color = UNKNOWN_PALETTE[i % len(UNKNOWN_PALETTE)]
        resolved.append(color)

    # Pass 2: collision reallocation. First-in-order keeps its color; later
    # duplicates round-robin through UNKNOWN_PALETTE skipping taken slots.
    taken: set[str] = set()
    final: list[str] = []
    collisions: list[tuple[str, str]] = []  # (channel label, original color)
    for i, color in enumerate(resolved):
        if color not in taken:
            taken.add(color)
            final.append(color)
            continue
        new_color = color
        for candidate in UNKNOWN_PALETTE:
            if candidate not in taken:
                new_color = candidate
                break
        # If every palette slot is taken, we're out of distinct colors — keep
        # the original color for this channel; the warning still fires so
        # users know two channels look identical.
        taken.add(new_color)
        final.append(new_color)
        collisions.append((_channel_label(channel_infos[i], i), color))

    if collisions:
        parts = ", ".join(
            f"{name!r} (would collide on #{color})" for name, color in collisions
        )
        warnings.warn(
            f"channel color collision: reassigned {parts}; pass "
            f"channel_colors={{<channel>: <hex6>}} to override",
            ChannelColorCollisionWarning,
            stacklevel=2,
        )
    return final


def colors_for_channels(
    channel_names: list[str],
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Resolve one hex color per channel name, for reader paths without wavelength.

    Thin adapter around :func:`assign_colors` for the CZI/ND2/OME-TIFF branch
    where only the channel's display name is known. Reaches the same band
    mapping via the dye-name fallback, then collision-reallocates.
    """
    infos: list[dict] = [{"name": n, "dye": n} for n in channel_names]
    return assign_colors(infos, overrides=overrides)
