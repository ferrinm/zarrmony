"""Pure, stdlib-only per-tile layout extractor for Leica LIF scene XML.

A mosaic LIF scene carries two pieces of acquisition metadata that bioio-lif's
stitcher ignores:

* ``<Tile FieldX FieldY PosX PosY PosZ />`` elements inside the scene's
  ``<Attachment Name="TileScanInfo">`` — the per-tile (row, column) grid index
  and the stage position (meters) of each tile's origin.
* ``<StitchingSettings OverlapPercentageX OverlapPercentageY .../>`` — the
  user-configured intended overlap fraction (LIF stores 0.10 to mean 10%).

:func:`extract_tile_layout` returns a structured dict surfacing both, suitable
for the audit's ``mosaic`` block. It is fail-closed (mirrors
:mod:`zarrmony.metadata.lif_channels`): oversized input, DTDs / entity
definitions, external entities, and any malformed XML yield ``None``. A scene
with no ``<Tile>`` elements also yields ``None`` — the audit/warning fall back
to today's generic shape rather than carrying a nonsense empty list.

The extractor is dependency-free and intended to lift into ``bioio-lif`` later.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

# A confocal scene blob is a few hundred KB; 32 MiB is generous headroom while
# still bounding parse work. Mirrors :mod:`lif_channels`.
_MAX_BYTES = 32 * 1024 * 1024

# A LIF scene blob is plain element/attribute XML. Any DOCTYPE or entity
# declaration is malformed or hostile (billion-laughs / XXE). Reject textually
# before parsing — stdlib ``ElementTree`` *does* expand internal entities.
_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


class _EntityRejectingTarget:
    """ExpatBuilder target that refuses DTDs and entity definitions.

    Belt-and-suspenders alongside the textual pre-scan: if a declaration slipped
    past the regex, expat's ``entity_decl`` / ``unparsed_entity_decl``
    callbacks fire here and abort the parse instead of expanding anything.
    """

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


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return int(value) if math.isfinite(value) and value.is_integer() else None


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tile_entries(root: ET.Element) -> list[dict]:
    """One dict per ``<Tile>`` element, in document order.

    Each tile carries the grid index (``field_x``, ``field_y``) and the stage
    position (``pos_x_m``, ``pos_y_m``, ``pos_z_m``) verbatim from LIF — meters,
    floats. Any attribute that fails to parse degrades to ``None`` rather than
    dropping the tile, so a partially-stamped tile still names its grid slot.
    """
    tiles: list[dict] = []
    for tile in root.iter("Tile"):
        attrib = tile.attrib
        tiles.append(
            {
                "field_x": _to_int(attrib.get("FieldX")),
                "field_y": _to_int(attrib.get("FieldY")),
                "pos_x_m": _to_float(attrib.get("PosX")),
                "pos_y_m": _to_float(attrib.get("PosY")),
                "pos_z_m": _to_float(attrib.get("PosZ")),
            }
        )
    return tiles


def _overlap_pct(root: ET.Element, attr: str) -> float | None:
    """The first ``StitchingSettings``'s ``attr`` as a percent (0.10 → 10.0).

    LIF stores ``OverlapPercentageX/Y`` as a *fraction* (0.10 == 10%) despite
    the misleading name; the surface key ``intended_overlap_x_pct`` represents
    actual percent, so we multiply by 100. Returns ``None`` when no
    ``StitchingSettings`` element exists or the attribute is absent/unparseable.
    """
    for settings in root.iter("StitchingSettings"):
        fraction = _to_float(settings.attrib.get(attr))
        if fraction is None:
            return None
        return fraction * 100.0
    return None


def extract_tile_layout(scene_xml: str) -> dict | None:
    """Extract tile positions and intended overlap from a LIF scene XML string.

    Returns ``{"tiles": [...], "intended_overlap_x_pct": float|None,
    "intended_overlap_y_pct": float|None}`` when the scene has at least one
    ``<Tile>`` element; otherwise ``None`` (also for missing/malformed/oversized
    input). Each tile dict has the keys ``field_x``, ``field_y``, ``pos_x_m``,
    ``pos_y_m``, ``pos_z_m``; positions are LIF's stage coordinates in meters.

    Fail-closed: any unexpected structural surprise yields ``None`` so the audit
    falls back cleanly. Metadata never crashes a conversion.
    """
    root = _safe_parse(scene_xml)
    if root is None:
        return None
    try:
        tiles = _tile_entries(root)
        if not tiles:
            return None
        return {
            "tiles": tiles,
            "intended_overlap_x_pct": _overlap_pct(root, "OverlapPercentageX"),
            "intended_overlap_y_pct": _overlap_pct(root, "OverlapPercentageY"),
        }
    except Exception:
        return None
