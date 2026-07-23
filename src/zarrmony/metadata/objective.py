"""Pure, stdlib-only per-scene objective-lens extractor for Leica LIF scene XML.

A Leica LIF scene records the acquisition objective in two overlapping shapes:

* every ``<ATLConfocalSettingDefinition>`` (the sequential-settings blocks that
  :mod:`zarrmony.metadata.lif_channels` already walks for channel identity) carries
  the objective as *attributes* — ``ObjectiveName``, ``NumericalAperture``,
  ``Immersion``, and either ``ObjectiveMag`` / ``NominalMagnification`` when
  present, or an ``NNx`` embedded in ``ObjectiveName`` (``"HC PL APO CS2 20x/0.75 DRY"``).
* some LIF exports also include an ``<Objective>`` element carrying the same
  fields; walk it and merge so either shape yields the same audit dict.

:func:`extract_objective` returns one dict per scene with only the keys the LIF
actually surfaced — missing fields are omitted (never ``None`` / ``0``), and a
scene with no objective info at all yields ``None`` (rather than an empty dict).
This is the shape the audit persists under ``per_scene[i].objective`` (issue #52)
and the shape :mod:`zarrmony.writers.ome_xml` projects into a top-level
``<Instrument><Objective/></Instrument>`` + per-image ``<ObjectiveSettings/>``.

The extractor is *dependency-free* (stdlib only) for the same reason
:mod:`lif_channels` is: it is meant to lift into ``bioio-lif`` unchanged.

Hardening: fail-closed like the sibling extractors. Oversized input, DTDs /
entity declarations (billion-laughs / XXE), external entities, and any
malformed XML all yield ``None`` — never an exception, never entity expansion,
never a hang. Metadata never crashes a conversion.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

# Mirrors the caps used by lif_channels / lif_tiles. A confocal scene blob is
# a few hundred KB; 32 MiB is generous headroom while still bounding parse work.
_MAX_BYTES = 32 * 1024 * 1024

# Any DOCTYPE / entity declaration in a LIF scene blob is malformed or hostile.
# Reject textually before parsing — stdlib ``ElementTree`` *does* expand
# internal entities.
_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

# Parses ``"HC PL APO CS2    20x/0.75 DRY"`` → ``20`` (integral) or
# ``"63x/1.4"`` → ``63``. Anchored so a decimal like ``1.4x`` inside a longer
# fragment (impossible in practice, defensive here) is still matched; the
# negative lookbehind prevents matching the trailing digit of ``"CS2 20x"``.
_OBJECTIVE_NAME_MAG = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)x", re.IGNORECASE)

# LIF ``Immersion`` attribute strings → OME ``Objective_Immersion`` enum values.
# ``DRY`` is Leica shorthand for an air (no-medium) objective. Unknown strings
# fall through to ``"Other"`` per the OME enum's own catch-all — never omit
# the key when a raw value was present, since "we saw an immersion but don't
# recognise it" is different from "no immersion metadata".
_LIF_IMMERSION_TO_OME: dict[str, str] = {
    "OIL": "Oil",
    "WATER": "Water",
    "AIR": "Air",
    "DRY": "Air",
    "GLYC": "Glycerol",
    "GLYCEROL": "Glycerol",
    "MULTI": "Multi",
    "OTHER": "Other",
}


class _EntityRejectingTarget:
    """ExpatBuilder target that refuses DTDs and entity definitions.

    Belt-and-suspenders alongside the textual pre-scan; mirrors the pattern in
    :mod:`zarrmony.metadata.lif_channels`. If a declaration slipped past the
    regex, expat's ``entity_decl`` / ``unparsed_entity_decl`` callbacks fire
    here and abort the parse instead of expanding anything.
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


def _to_number(text: str | None) -> int | float | None:
    """Parse a numeric string, returning int when integral, else float, else None."""
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _clean_model(raw: str | None) -> str | None:
    """Collapse LIF's whitespace-padded objective name into a canonical string.

    ``"HC PL APO CS2    20x/0.75 DRY "`` → ``"HC PL APO CS2 20x/0.75 DRY"``.
    Empty / whitespace-only inputs degrade to ``None`` so the caller omits the
    ``model`` key rather than persisting a bogus placeholder.
    """
    if raw is None:
        return None
    collapsed = " ".join(raw.split()).strip()
    return collapsed or None


def _map_immersion(raw: str | None) -> str | None:
    """LIF ``Immersion`` attribute → OME ``Objective_Immersion`` enum value.

    Unknown-but-present strings map to ``"Other"`` (never dropped: "we saw an
    immersion we don't recognise" is different from "no immersion metadata").
    A missing / empty attribute yields ``None`` so the caller omits the key.
    """
    if raw is None:
        return None
    key = str(raw).strip().upper()
    if not key:
        return None
    return _LIF_IMMERSION_TO_OME.get(key, "Other")


def _parse_name_magnification(name: str | None) -> int | float | None:
    """Extract the ``NNx`` magnification embedded in an objective name, or ``None``.

    ``"HC PL APO CS2 20x/0.75 DRY"`` → ``20``; ``"63x/1.40 OIL"`` → ``63``.
    Returns ``None`` when the pattern is absent so the caller keeps searching
    for a magnification via other attributes.
    """
    if not name:
        return None
    match = _OBJECTIVE_NAME_MAG.search(name)
    if match is None:
        return None
    return _to_number(match.group(1))


def _fill_from_attrs(dest: dict, attrib: dict) -> None:
    """Merge one element's attributes into ``dest``, first-writer-wins per key.

    ``nominal_magnification`` prefers an explicit ``ObjectiveMag`` /
    ``NominalMagnification`` attribute; if neither is present it falls back to
    parsing an ``NNx`` from ``ObjectiveName``. The bare ``Magnification``
    attribute on ``ATLConfocalSettingDefinition`` is *total* zoom (objective
    magnification × zoom factor) and is deliberately NOT read — recording ``21``
    instead of ``20`` for a ``20x`` objective would silently corrupt the audit.

    ``working_distance_um`` is taken verbatim from the ``WorkingDistance``
    attribute. LIF's stored unit for this field is not consistently documented
    (some exports millimeters, others micrometers); recording the raw value
    preserves fidelity — consumers that know their instrument can convert.
    """
    if "nominal_magnification" not in dest:
        for key in ("ObjectiveMag", "NominalMagnification"):
            value = _to_number(attrib.get(key))
            if value is not None:
                dest["nominal_magnification"] = value
                break
    if "nominal_magnification" not in dest:
        value = _parse_name_magnification(attrib.get("ObjectiveName"))
        if value is not None:
            dest["nominal_magnification"] = value

    if "numerical_aperture" not in dest:
        value = _to_number(attrib.get("NumericalAperture"))
        if value is not None:
            dest["numerical_aperture"] = value

    if "immersion" not in dest:
        mapped = _map_immersion(attrib.get("Immersion"))
        if mapped is not None:
            dest["immersion"] = mapped

    if "model" not in dest:
        model = _clean_model(attrib.get("ObjectiveName") or attrib.get("Model"))
        if model is not None:
            dest["model"] = model

    if "working_distance_um" not in dest:
        value = _to_number(attrib.get("WorkingDistance"))
        if value is not None:
            dest["working_distance_um"] = value


def extract_objective(scene_xml: str) -> dict | None:
    """Extract the acquisition objective from one Leica LIF scene XML string.

    Returns a dict with any subset of the keys ``nominal_magnification``,
    ``numerical_aperture``, ``immersion``, ``model``, ``working_distance_um`` —
    only the keys the LIF actually surfaced. Missing individual fields are
    *omitted* (never ``None`` / ``0``); a scene with no objective info at all
    yields ``None`` rather than an empty dict, so the audit either records the
    ``objective`` key with real content or omits it entirely.

    Both ``<ATLConfocalSettingDefinition>`` and ``<Objective>`` elements are
    walked and merged (first-writer-wins per key): different LIF exports carry
    the same fields on either element, and either shape yields the same audit
    dict.

    Fail-closed: any parse failure, unsafe input, or missing structure yields
    ``None``. Metadata never crashes a conversion.
    """
    root = _safe_parse(scene_xml)
    if root is None:
        return None
    try:
        result: dict = {}
        for atl in root.iter("ATLConfocalSettingDefinition"):
            _fill_from_attrs(result, atl.attrib)
        for obj in root.iter("Objective"):
            _fill_from_attrs(result, obj.attrib)
        return result or None
    except Exception:
        # Any unexpected structural surprise stays fail-closed.
        return None
