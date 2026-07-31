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
    from datetime import UTC, datetime

    try:
        ticks = (h << 32) | low_int
        posix = ticks / 10_000_000 - _FILETIME_EPOCH_DIFF_SECONDS
        return datetime.fromtimestamp(posix, tz=UTC).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _extract_date(root: ET.Element) -> str | None:
    """First recognisable ``<TimeStamp>`` in the tree, as an ISO 8601 UTC string.

    Handles two encodings seen in the wild: ``(HighInteger, LowInteger)``
    FILETIME ticks (the common LIF encoding) and a plain-text ISO date attribute
    (fallback). Returns ``None`` when no timestamp is present or parseable.
    """
    for ts in root.iter("TimeStamp"):
        # LIF FILETIME ticks — the common shape.
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


def _extract_imaging_method(root: ET.Element) -> list[str] | None:
    """Extract the imaging modalities as a list of OME-conventional tokens.

    Walks the ``HardwareSetting`` ``DataSourceTypeName`` attributes across every
    ``<Image>`` in the tree; falls back to the ``ATLConfocalSettingDefinition``
    presence check when no ``DataSourceTypeName`` is populated (a scene with an
    ATL block is confocal by construction, even when the header string is
    missing). Deduplicates while preserving first-seen order.

    Always returns a ``list[str]`` when at least one modality was extracted;
    ``None`` when none was. Never emits a scalar — matches the BQ REPEATED shape.
    """
    seen: list[str] = []
    for hw in _iter_hardware_settings(root):
        token = _normalise_modality(hw.attrib.get("DataSourceTypeName"))
        if token and token not in seen:
            seen.append(token)
    # Fallback: any ATLConfocalSettingDefinition in this scene → confocal.
    if not seen:
        for _ in root.iter("ATLConfocalSettingDefinition"):
            if "confocal" not in seen:
                seen.append("confocal")
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
