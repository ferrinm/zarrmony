"""Pure LMSDataContainer XML walker for LIF plate templates.

Per ADR-0009, LIF plate detection is driven by walking the ``LMSDataContainer``
XML tree — not by parsing the derived ``{plate}/{row}/{col}`` scene-name
pattern. This module owns that walk in one place so both the reader (which sets
``layout_hint`` / ``plate_layout``) and the ``inspect()`` surface can consume
identical structured output.

The walker collects every plate template it finds in first-encountered order.
A "plate template" is an Element whose ``Children/Element`` children carry
alpha-only ``Name`` attributes (rows) that themselves carry
``Children/Element`` children with numeric ``Name`` attributes (columns).
Each column element records a scene index that matches ``bioio_lif``'s
document-order scene enumeration (``_scene_to_well_map``), so
``PlateField.scene_index`` values plug back into ``reader.scenes`` without
re-parsing scene names.

Row and column strings are normalized at extraction time (single uppercase
letters, width-2 zero-padded numeric columns — ``zarrmony-phenix``
convention). This matches ADR-0009 §Consequences and keeps the on-disk
``<plate>/<row>/<col>/`` shape identical for the same physical plate regardless
of how LAS X happened to serialize the labels.

The extractor is fail-closed (mirrors :mod:`lif_tiles`): oversized input,
DTDs / entity definitions, external entities, and any malformed XML yield an
empty list. Non-plate LIF files (no plate templates in the XML) also yield an
empty list — the reader treats that as "flat" and leaves ``layout_hint``
alone.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# LIF plate XML blobs are typically a few hundred KB even for 96-well plates;
# 32 MiB is generous headroom while still bounding parse work. Mirrors
# :mod:`lif_tiles`.
_MAX_BYTES = 32 * 1024 * 1024

# Reject DOCTYPE / entity declarations before parsing — stdlib ``ElementTree``
# *does* expand internal entities. Same billion-laughs / XXE guard as
# :mod:`lif_tiles`.
_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

# Row Name is one or more letters (single uppercase for 96/384-well plates,
# double-letter for 1536-well plates past Z). Case-insensitive match at the
# XML boundary; canonicalized to upper case in the output.
_ROW_NAME_RE = re.compile(r"^[A-Za-z]+$")

# Column Name is one or more digits; canonicalized to width-2 zero-padded.
_COL_NAME_RE = re.compile(r"^\d+$")


class _EntityRejectingTarget:
    """ExpatBuilder target that refuses DTDs and entity definitions.

    Belt-and-suspenders alongside the textual pre-scan (see :mod:`lif_tiles`
    for the same guard).
    """

    def __init__(self) -> None:
        self._builder = ET.TreeBuilder()

    def entity_decl(self, *_args, **_kwargs):  # pragma: no cover — defensive
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


def _safe_parse(source_xml: str) -> ET.Element | None:
    """Parse ``source_xml`` into a root element, fail-closed (never raises)."""
    if not isinstance(source_xml, str) or not source_xml:
        return None
    if len(source_xml.encode("utf-8", "ignore")) > _MAX_BYTES:
        return None
    if _DOCTYPE_OR_ENTITY.search(source_xml):
        return None
    try:
        parser = ET.XMLParser(target=_EntityRejectingTarget())
        parser.feed(source_xml)
        return parser.close()
    except Exception:  # noqa: BLE001 — fail-closed
        return None


def _normalize_row(name: str) -> str:
    return name.upper()


def _normalize_column(name: str) -> str:
    return f"{int(name):02d}"


def _is_plate_template(element: ET.Element) -> bool:
    """True when ``element`` looks like a plate: alpha rows with numeric columns.

    The predicate needs at least one row + column pair to fire so that regular
    confocal ``<Element>`` nodes (which may nest arbitrarily) never trigger a
    false-positive plate detection.
    """
    for row in element.findall("./Children/Element"):
        row_name = (row.attrib.get("Name") or "").strip()
        if not row_name or not _ROW_NAME_RE.match(row_name):
            continue
        for col in row.findall("./Children/Element"):
            col_name = (col.attrib.get("Name") or "").strip()
            if col_name and _COL_NAME_RE.match(col_name):
                return True
    return False


def _walk_plate(element: ET.Element, scene_offset: int) -> dict:
    """Extract one plate template's shape + fields, starting at ``scene_offset``.

    Iteration order matches ``bioio_lif``'s ``_scene_to_well_map`` (rows in
    XML document order, columns in XML document order under each row), so
    ``PlateField.scene_index`` values line up with ``reader.scenes`` positions
    without a second pass. Returns the plate dict plus the total field count
    consumed so multi-plate walkers can advance ``scene_offset``.
    """
    plate_name = element.attrib.get("Name", "") or ""
    rows: list[str] = []
    columns: list[str] = []
    seen_rows: set[str] = set()
    seen_columns: set[str] = set()
    fields: list[dict] = []
    scene_index = scene_offset

    for row_elem in element.findall("./Children/Element"):
        row_name = (row_elem.attrib.get("Name") or "").strip()
        if not row_name or not _ROW_NAME_RE.match(row_name):
            continue
        row_norm = _normalize_row(row_name)
        if row_norm not in seen_rows:
            seen_rows.add(row_norm)
            rows.append(row_norm)
        for col_elem in row_elem.findall("./Children/Element"):
            col_name = (col_elem.attrib.get("Name") or "").strip()
            if not col_name or not _COL_NAME_RE.match(col_name):
                continue
            col_norm = _normalize_column(col_name)
            if col_norm not in seen_columns:
                seen_columns.add(col_norm)
                columns.append(col_norm)
            fields.append(
                {
                    "scene_index": scene_index,
                    "row": row_norm,
                    "column": col_norm,
                    "field_name": f"{row_norm}{col_norm}",
                }
            )
            scene_index += 1

    return {
        "name": plate_name,
        "rows": sorted(rows),
        "columns": sorted(columns),
        "fields": fields,
    }


def _find_plate_templates(root: ET.Element) -> list[ET.Element]:
    """Every plate-template Element in the XML tree, in document order.

    A plate template can sit at any depth (bioio_lif walks ``.//Element``
    without depth constraint) — we mirror that but only emit elements whose
    structural shape actually matches a plate (:func:`_is_plate_template`).
    """
    return [elem for elem in root.iter("Element") if _is_plate_template(elem)]


def extract_plate_layouts(source_xml: str) -> list[dict]:
    """Return every plate template found in a LIF ``LMSDataContainer`` blob.

    Each plate is a dict::

        {
            "name": str,            # plate template name from XML
            "rows": [str, ...],     # canonical uppercase letters (sorted)
            "columns": [str, ...],  # width-2 zero-padded numeric (sorted)
            "fields": [
                {"scene_index": int, "row": str, "column": str, "field_name": str},
                ...
            ],
        }

    Scene indexes are assigned in ``bioio_lif`` document-order enumeration —
    across plates, rows within a plate, then columns within a row — so a
    reader consumer can drop the field dicts straight into
    :class:`~zarrmony.readers.plate.PlateField` without re-numbering.

    Returns ``[]`` for non-plate LIF files or any parse failure. Never raises:
    plate detection is metadata, and metadata never crashes a conversion.
    """
    root = _safe_parse(source_xml)
    if root is None:
        return []
    try:
        plates: list[dict] = []
        scene_offset = 0
        for plate_elem in _find_plate_templates(root):
            plate = _walk_plate(plate_elem, scene_offset)
            if not plate["fields"]:
                continue
            scene_offset += len(plate["fields"])
            plates.append(plate)
        return plates
    except Exception:  # noqa: BLE001 — never break a conversion over metadata
        return []


__all__ = ["extract_plate_layouts"]
