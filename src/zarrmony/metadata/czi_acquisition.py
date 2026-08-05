"""Fill the CZI microscope-model gap that bioio-czi's OME projection leaves.

bioio-czi's XSLT projects ``Microscope.manufacturer = "Zeiss"`` from the CZI
vendor XML but leaves ``Microscope.model`` empty, so
``per_scene[i].acquisition.microscope`` from
:func:`~zarrmony.metadata.ome_extractors.extract_acquisition_from_ome` lands as
just ``"Zeiss"`` — indistinguishable across Axio Scan, LSM 900/980, Elyra,
Celldiscoverer, and Lattice Lightsheet. The raw CZI XML the file actually
carries has the model at ``Metadata/Information/Instrument/Microscopes/
Microscope[@Name]``; this extractor reads it and returns the ADR-0008
``acquisition`` fields the OME projection missed.

Used as a vendor-specific tier layered between the LIF extractor and the OME
projection in :func:`zarrmony.api._audit_acquisition_for_scene` — sits above
the OME projection so its ``Zeiss <Model>`` string wins the ``setdefault``
race against OME's ``"Zeiss"`` alone. LIF still takes precedence (LIF is a
different reader entirely).

Fail-closed like the LIF extractor: any parse failure, unsafe input, or
missing element yields ``None`` — never raises. Metadata never crashes a
conversion.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

_MAX_BYTES = 32 * 1024 * 1024

_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


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


def _safe_parse(raw: str) -> ET.Element | None:
    """Parse ``raw`` XML into a root element, fail-closed (never raises)."""
    if not isinstance(raw, str) or not raw:
        return None
    if len(raw.encode("utf-8", "ignore")) > _MAX_BYTES:
        return None
    if _DOCTYPE_OR_ENTITY.search(raw):
        return None
    try:
        parser = ET.XMLParser(target=_EntityRejectingTarget())
        parser.feed(raw)
        return parser.close()
    except Exception:
        return None


def _coerce_root(raw_metadata: Any) -> ET.Element | None:
    """Coerce ``reader.metadata`` (string or ``Element``) into a root ``Element``.

    bioio-czi exposes ``reader.metadata`` as an
    :class:`xml.etree.ElementTree.Element`; other reader paths that serialise
    the vendor XML as a string are also accepted so this extractor can be
    used against captured ``raw.czi.xml`` fixtures directly.
    """
    if raw_metadata is None:
        return None
    if isinstance(raw_metadata, ET.Element):
        return raw_metadata
    if isinstance(raw_metadata, str):
        return _safe_parse(raw_metadata)
    return None


def _first_nonempty(*values: str | None) -> str | None:
    for v in values:
        if v is None:
            continue
        stripped = v.strip()
        if stripped:
            return stripped
    return None


def _extract_microscope_model(root: ET.Element) -> str | None:
    """Return the first ``<Microscope Name="...">`` model string in the tree.

    CZI puts the microscope model at
    ``Metadata/Information/Instrument/Microscopes/Microscope[@Name]``, but the
    surface is heterogeneous across Zeiss product lines (Axio Scan, LSM 900,
    Elyra, Celldiscoverer, Lattice Lightsheet). We walk the whole tree for
    a ``Microscope`` element and prefer ``@Name`` (the human-friendly product
    name — e.g. ``"Axioscan 7"``), falling back to a ``<System>`` child text
    on the rare export that uses that shape.
    """
    for scope in root.iter("Microscope"):
        name = _first_nonempty(scope.attrib.get("Name"))
        if name is not None:
            return name
        system = scope.find("System")
        if system is not None:
            text = _first_nonempty(system.text)
            if text is not None:
                return text
    return None


def _with_zeiss_prefix(model: str) -> str:
    """Prepend ``"Zeiss "`` if ``model`` doesn't already start with it.

    ``"Axioscan 7"`` → ``"Zeiss Axioscan 7"``; ``"Zeiss LSM 980"`` stays as-is.
    Case-insensitive check so ``"ZEISS Elyra 7"`` isn't double-prefixed.
    """
    stripped = model.strip()
    if stripped.lower().startswith("zeiss"):
        return stripped
    return f"Zeiss {stripped}"


def extract_czi_acquisition(raw_metadata: Any) -> dict | None:
    """Extract the ADR-0008 ``acquisition`` fields the CZI vendor XML carries.

    Currently populates only ``microscope`` — the OME projection already
    surfaces ``date`` (from ``acquisition_date``), ``imaging_method`` (from
    per-channel ``AcquisitionMode``), and the manufacturer half of
    ``microscope``. This extractor's job is to fill the model gap
    (``"Zeiss Axioscan 7"`` instead of just ``"Zeiss"``).

    Returns ``None`` when the CZI tree has no extractable microscope model.
    Fail-safe: any exception yields ``None``.
    """
    root = _coerce_root(raw_metadata)
    if root is None:
        return None
    try:
        model = _extract_microscope_model(root)
        if model is None:
            return None
        return {"microscope": _with_zeiss_prefix(model)}
    except Exception:  # noqa: BLE001 — never crash audit
        return None


__all__ = ["extract_czi_acquisition"]
