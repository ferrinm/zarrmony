"""Tests for the CZI vendor-XML microscope extractor (issue #77).

Covers :mod:`zarrmony.metadata.czi_acquisition` — reads
``Metadata/Information/Instrument/Microscopes/Microscope[@Name]`` from the
raw CZI XML and returns ``{"microscope": "Zeiss <Model>"}`` so
``per_scene[i].acquisition.microscope`` lands as ``"Zeiss Axioscan 7"``
instead of the OME-projection's ``"Zeiss"`` alone.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from zarrmony.metadata.czi_acquisition import extract_czi_acquisition

# Minimal CZI XML skeleton — mirrors the shape of a real Axio Scan 7 export.
_AXIOSCAN_XML = """\
<ImageDocument>
 <Metadata>
  <Information>
   <Instrument>
    <Microscopes>
     <Microscope Id="Microscope:1" Name="Axioscan 7">
      <Type>Upright</Type>
      <UserDefinedName>4661000515</UserDefinedName>
     </Microscope>
    </Microscopes>
   </Instrument>
  </Information>
 </Metadata>
</ImageDocument>
"""


def test_extracts_zeiss_model_from_axioscan_xml() -> None:
    """The canonical case from #77: real Axioscan XML → 'Zeiss Axioscan 7'."""
    result = extract_czi_acquisition(_AXIOSCAN_XML)
    assert result == {"microscope": "Zeiss Axioscan 7"}


def test_accepts_elementtree_root_directly() -> None:
    """bioio-czi's ``reader.metadata`` returns an ``Element`` — no re-parse."""
    root = ET.fromstring(_AXIOSCAN_XML)
    assert extract_czi_acquisition(root) == {"microscope": "Zeiss Axioscan 7"}


def test_does_not_double_prefix_when_name_already_has_zeiss() -> None:
    """Some CZI exports carry the brand in ``Name`` (e.g. ``"Zeiss LSM 980"``)."""
    xml = _AXIOSCAN_XML.replace('Name="Axioscan 7"', 'Name="Zeiss LSM 980"')
    assert extract_czi_acquisition(xml) == {"microscope": "Zeiss LSM 980"}


def test_case_insensitive_zeiss_prefix_check() -> None:
    """``"ZEISS Elyra 7"`` isn't double-prefixed to ``"Zeiss ZEISS Elyra 7"``."""
    xml = _AXIOSCAN_XML.replace('Name="Axioscan 7"', 'Name="ZEISS Elyra 7"')
    assert extract_czi_acquisition(xml) == {"microscope": "ZEISS Elyra 7"}


def test_none_when_microscope_element_missing() -> None:
    """A CZI tree without a ``<Microscope>`` element yields None (fill-nothing)."""
    xml = "<ImageDocument><Metadata><Information></Information></Metadata></ImageDocument>"
    assert extract_czi_acquisition(xml) is None


def test_none_when_microscope_has_no_name_or_system() -> None:
    """A ``<Microscope>`` element with no model info yields None."""
    xml = """\
<ImageDocument>
 <Metadata>
  <Information>
   <Instrument>
    <Microscopes>
     <Microscope Id="Microscope:1">
      <Type>Upright</Type>
     </Microscope>
    </Microscopes>
   </Instrument>
  </Information>
 </Metadata>
</ImageDocument>
"""
    assert extract_czi_acquisition(xml) is None


def test_falls_back_to_system_child_text() -> None:
    """Rare export shape: model in ``<System>`` child instead of ``@Name``."""
    xml = """\
<ImageDocument>
 <Metadata>
  <Information>
   <Instrument>
    <Microscopes>
     <Microscope Id="Microscope:1">
      <System>LSM 980</System>
     </Microscope>
    </Microscopes>
   </Instrument>
  </Information>
 </Metadata>
</ImageDocument>
"""
    assert extract_czi_acquisition(xml) == {"microscope": "Zeiss LSM 980"}


def test_none_for_none_input() -> None:
    assert extract_czi_acquisition(None) is None


def test_none_for_unrecognised_input_type() -> None:
    """Non-str, non-Element input (int, dict, list) yields None — never crashes."""
    for bad in (42, ["<xml/>"], {"root": "x"}):
        assert extract_czi_acquisition(bad) is None


def test_malformed_xml_yields_none() -> None:
    """A parse error must not crash the audit."""
    assert extract_czi_acquisition("<not valid xml") is None


def test_rejects_doctype() -> None:
    """XXE-hardening: any DOCTYPE declaration in the input is refused."""
    xml = """<!DOCTYPE root>\n""" + _AXIOSCAN_XML
    assert extract_czi_acquisition(xml) is None


def test_rejects_entity_declaration() -> None:
    """XXE-hardening: any ENTITY declaration is refused."""
    xml = """<!ENTITY x "y">\n""" + _AXIOSCAN_XML
    assert extract_czi_acquisition(xml) is None
