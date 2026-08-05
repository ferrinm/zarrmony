"""Pure, stdlib-only per-scene acquisition/instrument extractor for LIF scene XML.

A Leica LIF scene records the acquisition date + instrument identity in the
same XML blob that :mod:`zarrmony.metadata.lif_channels` and
:mod:`zarrmony.metadata.objective` walk. This extractor projects those fields
into the ADR-0008 / #62 ``acquisition`` audit block:

    {
      "date":              str | absent,     # ISO 8601, taken from TimeStamp
      "microscope":        str | absent,     # brand + model (e.g. "STELLARIS 8")
      "microscope_serial": str | absent,     # instrument serial number
      "imaging_method":    list[str] | absent, # normalised OME tokens
    }

Every key is optional. A scene with no extractable fields returns ``None`` (not
an empty dict) so the audit either records the ``acquisition`` key with real
content or omits it entirely — mirrors :func:`extract_objective`.

``imaging_method`` is always a ``list[str]`` when present (matches the BQ
``imaging_method`` ``REPEATED STRING`` shape); a single-modality confocal scene
emits ``["confocal"]``. Multi-modal scenes (widefield preview + confocal detail
scan concatenated in one file, rare) emit each modality in the order
encountered.

Distinction: ``microscope`` is the brand/model (``"Stellaris"``, ``"SP8"``) —
what the LIF ``HardwareSetting`` records. ``microscope_name`` (Calico's
internal instance name like ``"Snouty"``) has no source-file surface and is
NOT extracted here; Aperture ingest fills that column from the submission form.

Hardening: fail-closed like the sibling extractors. Oversized input, DTDs,
entity declarations (billion-laughs / XXE), external entities, and malformed
XML all yield ``None`` — never an exception, never entity expansion, never
a hang. Metadata never crashes a conversion.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_MAX_BYTES = 32 * 1024 * 1024

_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

# LIF ``DataSourceTypeName`` / `LightPathMode` strings → OME-conventional
# imaging-method tokens. The token set is deliberately loose (single lowercase
# snake-case identifier per modality) so CZI / ND2 / OME-TIFF extractors can
# emit the same tokens from their own surfaces. Add new tokens as new
# reader-format branches call for them.
_LIF_MODALITY_TOKEN: dict[str, str] = {
    "CONFOCAL": "confocal",
    "WIDEFIELD": "widefield_fluorescence",
    "WIDEFIELDFLUORESCENCE": "widefield_fluorescence",
    "SPINNINGDISK": "spinning_disk_confocal",
    "SPINNING DISK": "spinning_disk_confocal",
    "SPINNINGDISKCONFOCAL": "spinning_disk_confocal",
    "LIGHTSHEET": "light_sheet",
    "STED": "sted",
    "TIRF": "tirf",
    "MULTIPHOTON": "multiphoton",
    "TWOPHOTON": "multiphoton",
}

# LIF ``WideFieldChannelInfo.ContrastingMethodName`` → imaging-method token.
# Populated per-channel on widefield systems (Thunder, DMi8) where
# ``HardwareSetting.DataSourceTypeName`` reports the vendor-neutral ``"Camera"``
# rather than a modality string. Keys are normalised (uppercased, underscores
# and hyphens stripped) so ``TL-BF`` and ``TL_BF`` both match ``TLBF``.
_LIF_CONTRASTING_METHOD_TOKEN: dict[str, str] = {
    "FLUO": "widefield_fluorescence",
    "FLUORESCENCE": "widefield_fluorescence",
    "BF": "bright_field",
    "TLBF": "bright_field",
    "BRIGHTFIELD": "bright_field",
    "DIC": "dic",
    "TLDIC": "dic",
    "PH": "phase_contrast",
    "TLPH": "phase_contrast",
    "PHASECONTRAST": "phase_contrast",
    "POL": "polarised_light",
    "TLPOL": "polarised_light",
    "DARKFIELD": "dark_field",
    "DF": "dark_field",
}


class _EntityRejectingTarget:
    """ExpatBuilder target that refuses DTDs and entity definitions."""

    def __init__(self) -> None:
        self._builder = ET.TreeBuilder()

    def entity_decl(self, *_args, **_kwargs):  # pragma: no cover - defensive
        raise ValueError("entity declarations are not permitted")

    def unparsed_entity_decl(self, *_args, **_kwargs):  # pragma: no cover
        raise ValueError("entity declarations are not permitted")

    def start_doctype_decl(self, *_args, **_kwargs):  # pragma: no cover
        raise ValueError("DOCTYPE is not permitted")

    def start(self, tag, attrib):
        return self._builder.start(tag, attrib)

    def end(self, tag):
        return self._builder.end(tag)

    def data(self, text):
        return self._builder.data(text)

    def close(self):
        return self._builder.close()


def _safe_parse(scene_xml: str) -> ET.Element | None:
    """Parse ``scene_xml`` into a root element, fail-closed (never raises)."""
    if not isinstance(scene_xml, str) or not scene_xml:
        return None
    if len(scene_xml.encode("utf-8", "ignore")) > _MAX_BYTES:
        return None
    if _DOCTYPE_OR_ENTITY.search(scene_xml):
        return None
    try:
        parser = ET.XMLParser(target=_EntityRejectingTarget())
        parser.feed(scene_xml)
        return parser.close()
    except Exception:
        return None


def _first_nonempty(*values: str | None) -> str | None:
    """Return the first non-empty (stripped) string, or ``None``."""
    for v in values:
        if v is None:
            continue
        stripped = v.strip()
        if stripped:
            return stripped
    return None


# Windows FILETIME epoch = 1601-01-01 UTC. LIF ``TimeStamp`` elements typically
# store 64-bit (HighInteger, LowInteger) pairs of 100-nanosecond ticks since
# that epoch; converting to a POSIX timestamp is one subtraction. We convert
# to an ISO 8601 UTC string so consumers don't have to know the LIF encoding.
_FILETIME_EPOCH_DIFF_SECONDS = 11644473600  # (1970 - 1601) in seconds


def _ticks_to_iso(ticks: int) -> str | None:
    """Convert a 64-bit FILETIME tick count → ISO 8601 UTC. ``None`` on failure."""
    if ticks < 0:
        return None
    from datetime import UTC, datetime

    try:
        posix = ticks / 10_000_000 - _FILETIME_EPOCH_DIFF_SECONDS
        return datetime.fromtimestamp(posix, tz=UTC).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _filetime_to_iso(high: str | None, low: str | None) -> str | None:
    """Convert a LIF ``(HighInteger, LowInteger)`` FILETIME tick pair → ISO 8601 UTC.

    Returns ``None`` on any parse failure. Never raises.
    """
    try:
        h = int(high) if high is not None else None
        low_int = int(low) if low is not None else None
    except (TypeError, ValueError):
        return None
    if h is None or low_int is None:
        return None
    if h < 0 or low_int < 0:
        return None
    return _ticks_to_iso((h << 32) | low_int)


def _hex_filetime_to_iso(hex_ticks: str | None) -> str | None:
    """Convert a hex-encoded 64-bit FILETIME tick count → ISO 8601 UTC.

    The LIF ``<TimeStampList>`` element carries per-frame timestamps as
    space-separated hex FILETIME values (e.g. ``"1dc28bfd6199e60"``). Returns
    ``None`` on any parse failure.
    """
    if not hex_ticks:
        return None
    try:
        ticks = int(hex_ticks, 16)
    except (TypeError, ValueError):
        return None
    return _ticks_to_iso(ticks)


def _extract_date(root: ET.Element) -> str | None:
    """First recognisable ``<TimeStamp>``/``<TimeStampList>`` in the tree.

    Handles three encodings seen in the wild:

    1. ``<TimeStamp HighInteger="..." LowInteger="..."/>`` — the older LIF
       per-scene shape.
    2. ``<TimeStampList>`` with space-separated hex FILETIME values in the
       element text — the newer LIF shape (Thunder / LAS X 3.x). The first
       hex value in the list is the acquisition start time.
    3. Plain-text ISO date attribute or text content — fallback for exports
       that inline the timestamp.

    Returns ``None`` when no timestamp is present or parseable.
    """
    for ts in root.iter("TimeStamp"):
        # LIF FILETIME ticks — the older shape.
        iso = _filetime_to_iso(
            ts.attrib.get("HighInteger"), ts.attrib.get("LowInteger")
        )
        if iso is not None:
            return iso
        # Fallback: ISO-shaped attribute (some LIF exports).
        for key in ("Date", "DateTime", "date"):
            candidate = ts.attrib.get(key)
            if candidate:
                return candidate.strip()
        # Fallback: text content.
        text = (ts.text or "").strip()
        if text:
            return text
    # Newer LIF shape: TimeStampList with space-separated hex FILETIME values.
    for tsl in root.iter("TimeStampList"):
        text = (tsl.text or "").strip()
        if not text:
            continue
        first = text.split()[0]
        iso = _hex_filetime_to_iso(first)
        if iso is not None:
            return iso
    return None


def _iter_hardware_settings(root: ET.Element):
    """Yield every LIF ``HardwareSetting`` element in ``root``.

    LIF wraps hardware settings in an ``<Attachment Name="HardwareSetting">``
    element (an attribute-tagged marker on the generic ``Attachment`` element).
    A rare export may instead expose a bare ``<HardwareSetting>`` element;
    both shapes are yielded so the extractor doesn't have to know which one
    a given file uses.
    """
    for element in root.iter():
        if element.tag == "HardwareSetting":
            yield element
        elif (
            element.tag == "Attachment"
            and element.attrib.get("Name") == "HardwareSetting"
        ):
            yield element


def _microscope_from_attrs(
    attrib: dict, microscope: str | None, serial: str | None
) -> tuple[str | None, str | None]:
    """Fold one element's attributes into a ``(microscope, serial)`` pair.

    First-writer-wins per field — matches :mod:`zarrmony.metadata.objective`.
    """
    if microscope is None:
        candidate = _first_nonempty(
            attrib.get("SystemTypeName"),
            attrib.get("MicroscopeModel"),
        )
        # "0" is Leica's placeholder for "not populated" on MicroscopeModel —
        # never a real microscope. Reject it explicitly.
        if candidate and candidate != "0":
            microscope = candidate
    if serial is None:
        candidate = _first_nonempty(
            attrib.get("SystemSerialNumber"),
            attrib.get("MicroscopeSerialNumber"),
            attrib.get("SerialNumber"),
        )
        if candidate:
            serial = candidate
    return microscope, serial


def _extract_microscope(root: ET.Element) -> tuple[str | None, str | None]:
    """Return ``(microscope, microscope_serial)`` from the LIF scene metadata.

    Walks ``HardwareSetting`` elements first (the Attachment-wrapped root
    hardware header, e.g. ``SystemTypeName="STELLARIS 8"``,
    ``SystemSerialNumber="8300000404"``), then falls back to
    ``ATLConfocalSettingDefinition`` blocks — the per-sequence acquisition
    settings, which some LIF exports use as the authoritative store for the
    same fields (the header can be truncated on older exports).

    ``microscope`` prefers ``SystemTypeName`` (Leica's brand + model string,
    e.g. ``"STELLARIS 8"``, ``"SP8"``) over ``MicroscopeModel`` (the microscope
    body's model number, e.g. ``"DM6B-Z-CFS"``) — SystemTypeName is what a user
    would say the microscope is.
    """
    microscope: str | None = None
    serial: str | None = None
    for hw in _iter_hardware_settings(root):
        microscope, serial = _microscope_from_attrs(hw.attrib, microscope, serial)
        if microscope is not None and serial is not None:
            return microscope, serial
    for atl in root.iter("ATLConfocalSettingDefinition"):
        microscope, serial = _microscope_from_attrs(atl.attrib, microscope, serial)
        if microscope is not None and serial is not None:
            return microscope, serial
    return microscope, serial


def _normalise_modality(raw: str | None) -> str | None:
    """LIF modality name → the OME-conventional token, or ``None``."""
    if not raw:
        return None
    key = raw.strip().upper().replace("_", "").replace("-", "")
    return _LIF_MODALITY_TOKEN.get(key)


def _normalise_contrasting_method(raw: str | None) -> str | None:
    """LIF ``ContrastingMethodName`` → the OME-conventional token, or ``None``."""
    if not raw:
        return None
    key = raw.strip().upper().replace("_", "").replace("-", "")
    return _LIF_CONTRASTING_METHOD_TOKEN.get(key)


def _extract_imaging_method(root: ET.Element) -> list[str] | None:
    """Extract the imaging modalities as a list of OME-conventional tokens.

    Tiered detection — later tiers only fire when earlier tiers found nothing
    for that channel/scene:

    1. ``HardwareSetting.DataSourceTypeName`` — the classic LIF surface for
       confocal systems (SP8, STELLARIS) that report ``"Confocal"`` here.
       Widefield-family systems (Thunder, DMi8) report the generic ``"Camera"``,
       which we deliberately do NOT map (Camera is a hardware category, not a
       modality — CSU spinning disk cameras and TIRF cameras report Camera too).
    2. ``WideFieldChannelInfo.ContrastingMethodName`` — per-channel contrasting
       method on Leica widefield systems (``FLUO``, ``BF``, ``DIC``, etc.).
       Multi-channel scenes with mixed contrasting methods surface every one.
    3. ``ATLConfocalSettingDefinition`` presence — a scene with an ATL confocal
       block is confocal by construction, even when the header string is
       missing.
    4. ``ATLCameraSettingDefinition`` presence — final fallback for camera-based
       Leica scenes where neither ``ContrastingMethodName`` nor any of the above
       fired. Camera-only LIFs from Leica default to widefield fluorescence.

    Deduplicates while preserving first-seen order. Returns a ``list[str]``
    when at least one modality was extracted, ``None`` when none was.
    """
    seen: list[str] = []
    # Tier 1: HardwareSetting.DataSourceTypeName (confocal-family surface).
    for hw in _iter_hardware_settings(root):
        token = _normalise_modality(hw.attrib.get("DataSourceTypeName"))
        if token and token not in seen:
            seen.append(token)
    # Tier 2: WideFieldChannelInfo.ContrastingMethodName (widefield-family
    # per-channel surface — mixed-method scenes emit every distinct token).
    for wf in root.iter("WideFieldChannelInfo"):
        token = _normalise_contrasting_method(wf.attrib.get("ContrastingMethodName"))
        if token and token not in seen:
            seen.append(token)
    # Tier 3: ATLConfocalSettingDefinition presence → confocal.
    if not seen:
        for _ in root.iter("ATLConfocalSettingDefinition"):
            seen.append("confocal")
            break
    # Tier 4: ATLCameraSettingDefinition presence (widefield-family fallback).
    if not seen:
        for _ in root.iter("ATLCameraSettingDefinition"):
            seen.append("widefield_fluorescence")
            break
    return seen or None


def extract_acquisition(scene_xml: str) -> dict | None:
    """Extract the ADR-0008 ``acquisition`` audit block from one LIF scene XML.

    Returns a dict with any subset of the keys ``date``, ``microscope``,
    ``microscope_serial``, ``imaging_method`` — only the keys the LIF actually
    surfaced. Missing individual fields are *omitted* (never ``None`` /
    empty); a scene with no acquisition info at all yields ``None`` rather
    than an empty dict, so the audit either records the ``acquisition`` key
    with real content or omits it entirely.

    ``imaging_method`` is always a ``list[str]`` when present, even when
    single-valued — matches the BQ REPEATED shape.

    Fail-closed: any parse failure, unsafe input, or missing structure yields
    ``None``. Metadata never crashes a conversion.
    """
    root = _safe_parse(scene_xml)
    if root is None:
        return None
    try:
        result: dict = {}
        date = _extract_date(root)
        if date is not None:
            result["date"] = date
        microscope, serial = _extract_microscope(root)
        if microscope is not None:
            result["microscope"] = microscope
        if serial is not None:
            result["microscope_serial"] = serial
        method = _extract_imaging_method(root)
        if method:
            result["imaging_method"] = method
        return result or None
    except Exception:
        return None


__all__ = ["extract_acquisition"]
