"""Pure, stdlib-only per-channel identity extractor for Leica LIF scene XML.

A single Leica ``.lif`` scene preserves its full acquisition settings as an XML
blob (the same text zarrmony copies verbatim to ``OME/source/raw.lif.xml``).
:func:`extract_channels` turns that blob into one identity dict per image
channel — dye, detector, excitation line, and emission band — without touching
``bioio``, ``ome_types``, or anything outside the standard library. Keeping it
dependency-free is deliberate: this is the load-bearing core later slices build
on, and it is meant to lift into ``bioio-lif``'s ``ome_metadata`` unchanged.

The hard part is the JOIN, not the parsing. A Leica scene stores several copies
of its instrument settings, and only one set is the *real* acquisition:

* ``LDM_Block_Sequential / LDM_Block_Sequential_List`` holds the genuine
  sequential settings, in document order. A channel's ``SequentialSettingIndex``
  indexes directly into this list. This is the source of truth.
* ``LDM_Block_Sequential_Master`` and the top-level
  ``Attachment[HardwareSetting]`` copies are reference/duplicate snapshots, NOT
  real sequences. The Master copy in particular can carry a phantom laser line
  that excites nothing in the acquired data; folding it in yields wrong answers.
  We never read from them.

Within a real sequence, excitation is recovered by spectral pairing: the active
laser lines (``LaserLineSetting`` with ``IntensityDev > 0``) are matched, in
ascending wavelength order, to the active detectors (``Detector`` with
``IsActive == "1"``) in ascending channel order. A channel's excitation is the
laser paired to its own physical detector channel. Emission comes from the
``MultiBand`` whose ``Channel`` matches that same physical channel.

Hardening: the parser is fail-closed. Oversized input, DTDs / entity
definitions (the "billion laughs" expansion vector), external entities, and any
malformed or structurally-missing XML all yield ``[]`` — never an exception,
never an entity expansion, never a hang.

Projections (below :func:`extract_channels`) — :func:`channels_to_omero` and
:func:`channels_to_ome_channels` — map those identity dicts into the two shapes
consumers actually read: the OME-XML ``<Channel>`` element (canonical) and the
omero display ``label``/``color`` (derived). They are deliberately *zarrmony-side*
and so are allowed the ``ome_types`` + ``zarrmony.metadata.channel_colors``
dependencies the core forbids itself; those imports are kept local to the
projections so the bioio-portable core can still be imported and called with the
standard library alone.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ome_types.model import Channel

# Refuse anything larger than this up front. A confocal scene blob is a few
# hundred KB; 32 MiB is generous headroom while still bounding parse work.
_MAX_BYTES = 32 * 1024 * 1024

# A LIF scene blob is plain element/attribute XML with no document type. Any
# DOCTYPE or entity declaration is therefore either malformed or hostile (the
# billion-laughs / external-entity vectors). stdlib ``ElementTree`` *does*
# expand internal entities, so we reject these textually before we ever parse.
_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

# Vendor prefix stripped from dye names: "Leica/ALEXA 594" -> "ALEXA 594".
_DYE_VENDOR_PREFIX = "Leica/"


class _EntityRejectingTarget:
    """ExpatBuilder target that refuses DTDs and entity definitions.

    Belt-and-suspenders alongside the textual pre-scan: even if a declaration
    slipped past the regex, expat's ``entity_decl`` / ``unparsed_entity_decl``
    callbacks fire here and abort the parse instead of expanding anything.
    """

    def __init__(self) -> None:
        self._builder = ET.TreeBuilder()

    # --- declarations we forbid -------------------------------------------
    def entity_decl(self, *_args, **_kwargs):  # pragma: no cover - defensive
        raise ValueError("entity declarations are not permitted")

    def unparsed_entity_decl(self, *_args, **_kwargs):  # pragma: no cover
        raise ValueError("entity declarations are not permitted")

    def start_doctype_decl(self, *_args, **_kwargs):  # pragma: no cover
        raise ValueError("DOCTYPE is not permitted")

    # --- normal tree construction -----------------------------------------
    def start(self, tag, attrib):
        return self._builder.start(tag, attrib)

    def end(self, tag):
        return self._builder.end(tag)

    def data(self, text):
        return self._builder.data(text)

    def close(self):
        return self._builder.close()


def _safe_parse(scene_xml: str) -> ET.Element | None:
    """Parse ``scene_xml`` into a root element, fail-closed.

    Returns ``None`` (never raises) on oversized input, DTD/entity content,
    external entities, or any malformed XML.
    """
    if not isinstance(scene_xml, str) or not scene_xml:
        return None
    # Size cap on the encoded bytes — char count alone understates memory.
    if len(scene_xml.encode("utf-8", "ignore")) > _MAX_BYTES:
        return None
    if _DOCTYPE_OR_ENTITY.search(scene_xml):
        return None
    try:
        parser = ET.XMLParser(target=_EntityRejectingTarget())
        parser.feed(scene_xml)
        return parser.close()
    except Exception:
        # ParseError, ValueError from the target, recursion limits, etc.
        return None


def _cp_map(channel_desc: ET.Element) -> dict[str, str]:
    """Flatten a ``ChannelDescription``'s ``ChannelProperty`` Key/Value pairs."""
    props: dict[str, str] = {}
    for prop in channel_desc.iter("ChannelProperty"):
        key = prop.find("Key")
        val = prop.find("Value")
        if key is not None and key.text:
            props[key.text.strip()] = (
                (val.text or "").strip() if val is not None else ""
            )
    return props


def _to_number(text: str | None) -> int | float | None:
    """Parse a numeric string, returning int when integral, else float, else None.

    Non-finite values (``inf`` / ``nan``) are rejected as ``None`` — they are
    never valid wavelengths and would otherwise break ``round``.
    """
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _sequence_blocks(root: ET.Element) -> list[ET.Element]:
    """The real acquisition sequences, in document order.

    These are the ``ATLConfocalSettingDefinition`` children of
    ``LDM_Block_Sequential / LDM_Block_Sequential_List`` — and ONLY those. The
    ``LDM_Block_Sequential_Master`` block and the top-level
    ``Attachment[HardwareSetting]`` copies are reference/duplicate snapshots and
    are deliberately excluded; ``SequentialSettingIndex`` indexes into this list.
    """
    blocks: list[ET.Element] = []
    # ``iter`` walks in document order, so each list's settings stay ordered and
    # multiple sequential lists (rare) concatenate in the order they appear.
    for seq_list in root.iter("LDM_Block_Sequential_List"):
        for atl in seq_list.iter("ATLConfocalSettingDefinition"):
            blocks.append(atl)
    return blocks


def _detector_channel_map(root: ET.Element) -> dict[str, int]:
    """Map every ``Detector`` ``Name`` to its physical ``Channel`` number.

    The mapping (e.g. ``HyD S 1`` -> 1, ``HyD X 2`` -> 2) is consistent across
    every block, so we scan the whole document to be robust to partial blocks.
    """
    mapping: dict[str, int] = {}
    for det in root.iter("Detector"):
        name = det.attrib.get("Name")
        channel = det.attrib.get("Channel")
        if not name or channel is None:
            continue
        try:
            mapping[name] = int(channel)
        except ValueError:
            continue
    return mapping


def _active_lasers(block: ET.Element) -> list[int | float]:
    """Excited laser lines (``IntensityDev > 0``) for a sequence, ascending."""
    lasers: list[int | float] = []
    for laser in block.iter("LaserLineSetting"):
        intensity = _to_number(laser.attrib.get("IntensityDev"))
        line = _to_number(laser.attrib.get("LaserLine"))
        if intensity is not None and intensity > 0 and line is not None:
            lasers.append(line)
    return sorted(lasers)


def _active_detector_channels(block: ET.Element) -> list[int]:
    """Physical channels of the active detectors (``IsActive == "1"``), ascending."""
    channels: list[int] = []
    for det in block.iter("Detector"):
        if det.attrib.get("IsActive") != "1":
            continue
        channel = det.attrib.get("Channel")
        if channel is None:
            continue
        try:
            channels.append(int(channel))
        except ValueError:
            continue
    return sorted(channels)


def _excitation_for(
    block: ET.Element, physical_channel: int | None
) -> int | float | None:
    """Excitation line for ``physical_channel`` within one sequence.

    Spectral pairing: the i-th lowest active laser drives the i-th lowest active
    detector. Find this channel's position among the active detectors, then read
    the laser at the same position.
    """
    if physical_channel is None:
        return None
    lasers = _active_lasers(block)
    detectors = _active_detector_channels(block)
    if physical_channel not in detectors:
        return None
    position = detectors.index(physical_channel)
    if position >= len(lasers):
        return None
    return lasers[position]


def _emission_for(
    block: ET.Element, physical_channel: int | None
) -> tuple[int | None, int | None]:
    """``(low, high)`` emission band (rounded nm) for ``physical_channel``."""
    if physical_channel is None:
        return None, None
    for band in block.iter("MultiBand"):
        channel = _to_number(band.attrib.get("Channel"))
        if channel is None or int(channel) != physical_channel:
            continue
        left = _to_number(band.attrib.get("LeftWorld"))
        right = _to_number(band.attrib.get("RightWorld"))
        low = round(left) if left is not None else None
        high = round(right) if right is not None else None
        return low, high
    return None, None


def _strip_dye_prefix(dye: str | None) -> str | None:
    """Drop the leading ``"Leica/"`` vendor prefix from a dye name."""
    if dye is None:
        return None
    return dye.removeprefix(_DYE_VENDOR_PREFIX)


def extract_channels(scene_xml: str) -> list[dict]:
    """Extract per-channel identity from one Leica LIF scene XML string.

    Returns one dict per image channel, in acquisition (document) order, each
    with exactly the keys ``index``, ``dye``, ``fluor``, ``detector``,
    ``excitation_nm``, ``emission_low_nm``, ``emission_high_nm``.

    Fail-closed: any parse failure, unsafe input, or missing structure yields
    ``[]``. Individual undeterminable fields degrade to ``None`` rather than
    dropping the channel.
    """
    root = _safe_parse(scene_xml)
    if root is None:
        return []

    try:
        sequences = _sequence_blocks(root)
        detector_to_channel = _detector_channel_map(root)

        channels: list[dict] = []
        for index, channel_desc in enumerate(root.iter("ChannelDescription")):
            props = _cp_map(channel_desc)

            dye = _strip_dye_prefix(props.get("DyeName") or None)
            detector_name = props.get("DetectorName") or None
            physical_channel = (
                detector_to_channel.get(detector_name) if detector_name else None
            )

            # Resolve which real sequence acquired this channel.
            block = None
            seq_index = _to_number(props.get("SequentialSettingIndex"))
            if seq_index is not None and 0 <= int(seq_index) < len(sequences):
                block = sequences[int(seq_index)]

            if block is not None:
                excitation = _excitation_for(block, physical_channel)
                emission_low, emission_high = _emission_for(block, physical_channel)
            else:
                excitation = None
                emission_low = emission_high = None

            channels.append(
                {
                    "index": index,
                    "dye": dye,
                    "fluor": dye,
                    "detector": detector_name,
                    "excitation_nm": excitation,
                    "emission_low_nm": emission_low,
                    "emission_high_nm": emission_high,
                }
            )
        return channels
    except Exception:
        # Any unexpected structural surprise stays fail-closed.
        return []


# ---------------------------------------------------------------------------
# Projections: identity dicts -> the shapes consumers read.
#
# These are zarrmony-side, NOT the bioio-portable core, so they are allowed
# the ``ome_types`` + ``channel_colors`` dependencies ``extract_channels``
# forbids itself. Those imports stay local to keep the core stdlib-only on
# import. Each is a small pure function over the dicts ``extract_channels``
# returns; they degrade per-field (omit/None) rather than inventing values, so
# garbage or partial input never raises.
# ---------------------------------------------------------------------------

# A trailing " (...)" parenthetical qualifier on a dye name, e.g. the
# "(dsDNA bound)" in "DAPI (dsDNA bound)". Anchored to the end so an interior
# parenthesis in an (unusual) dye name is left intact.
_TRAILING_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")


def _clean_dye(dye: str | None) -> str | None:
    """Strip a trailing ``" (...)"`` qualifier from a dye name.

    ``"DAPI (dsDNA bound)"`` -> ``"DAPI"``; ``"ALEXA 594"`` -> ``"ALEXA 594"``;
    ``None`` -> ``None``. Leaves a name with no trailing parenthetical untouched.
    """
    if dye is None:
        return None
    return _TRAILING_PARENTHETICAL.sub("", dye).strip() or None


def _color_for(cleaned_dye: str | None, index: int) -> str:
    """Heuristic hex color for a (cleaned) dye, palette-fallback by index.

    Imported locally so the bioio-portable core stays stdlib-only on import:
    ``zarrmony.metadata.channel_colors`` won't exist once ``extract_channels``
    is lifted into ``bioio-lif``, but these projections stay zarrmony-side.
    """
    from zarrmony.metadata.channel_colors import color_for_channel

    return color_for_channel(cleaned_dye or "", index)


def _omero_label(
    cleaned_dye: str | None, excitation_nm: int | float | None
) -> str | None:
    """The omero label per the ladder: dye + excitation, then either, then None."""
    if cleaned_dye and excitation_nm is not None:
        return f"{cleaned_dye} ({excitation_nm} nm)"
    if cleaned_dye:
        return cleaned_dye
    if excitation_nm is not None:
        return f"{excitation_nm} nm"
    return None


def channels_to_omero(channels: list[dict]) -> list[dict]:
    """Project identity dicts into omero display ``{"label", "color"}`` dicts.

    One dict per input channel, in order. ``label`` follows the dye/excitation
    ladder (``None`` only when neither is known); ``color`` is the cleaned dye's
    heuristic hex (falling back to the palette by index), so it is always a valid
    six-hex string even for an unnamed channel.
    """
    out: list[dict] = []
    for index, ch in enumerate(channels):
        cleaned_dye = _clean_dye(ch.get("dye"))
        out.append(
            {
                "label": _omero_label(cleaned_dye, ch.get("excitation_nm")),
                "color": _color_for(cleaned_dye, index),
            }
        )
    return out


def channels_to_ome_channels(channels: list[dict]) -> list[Channel]:
    """Project identity dicts into canonical ome_types ``Channel`` elements.

    One ``Channel`` per input channel: ``id=Channel:0:<index>``, ``name`` the
    cleaned dye, ``fluor`` the full (un-cleaned) dye, excitation/emission with
    ``nm`` units, and the same hex ``color`` as :func:`channels_to_omero`. Any
    field whose source is ``None`` is omitted entirely rather than set to a
    bogus value, so the element never claims wavelengths it doesn't have.
    """
    # Local import: the bioio-portable core stays stdlib-only on module import.
    from ome_types.model import Channel

    out: list[Channel] = []
    for index, ch in enumerate(channels):
        cleaned_dye = _clean_dye(ch.get("dye"))
        fields: dict = {
            "id": f"Channel:0:{index}",
            "color": _color_for(cleaned_dye, index),
        }
        if cleaned_dye is not None:
            fields["name"] = cleaned_dye
        fluor = ch.get("fluor")
        if fluor is not None:
            fields["fluor"] = fluor

        excitation = ch.get("excitation_nm")
        if excitation is not None:
            fields["excitation_wavelength"] = excitation
            fields["excitation_wavelength_unit"] = "nm"

        low = ch.get("emission_low_nm")
        high = ch.get("emission_high_nm")
        if low is not None and high is not None:
            fields["emission_wavelength"] = round((low + high) / 2)
            fields["emission_wavelength_unit"] = "nm"

        out.append(Channel(**fields))
    return out
