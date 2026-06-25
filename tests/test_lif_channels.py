"""Tests for the stdlib-only Leica LIF channel extractor.

Exercises the confocal fixture (the JOIN + phantom-block disambiguation), the
fail-closed hardening (entity bombs, XXE, malformed/oversized input), and
graceful degradation on partial metadata.
"""

from pathlib import Path

from zarrmony.metadata.lif_channels import extract_channels

FIXTURE = Path(__file__).parent / "fixtures" / "lif_confocal_7ch.xml"


def _channels() -> list[dict]:
    return extract_channels(FIXTURE.read_text(encoding="utf-8"))


# --- the confocal fixture: identity + the JOIN -----------------------------


def test_extracts_exactly_seven_channels_in_order() -> None:
    channels = _channels()
    assert len(channels) == 7
    assert [c["index"] for c in channels] == [0, 1, 2, 3, 4, 5, 6]


def test_each_channel_has_exactly_the_contract_keys() -> None:
    expected = {
        "index",
        "dye",
        "fluor",
        "detector",
        "excitation_nm",
        "emission_low_nm",
        "emission_high_nm",
    }
    for channel in _channels():
        assert set(channel) == expected


def test_dyes_have_vendor_prefix_stripped() -> None:
    assert [c["dye"] for c in _channels()] == [
        "DAPI (dsDNA bound)",
        "ALEXA 594",
        "ALEXA 750",
        "ALEXA 488",
        "ALEXA 647-R-PE",
        "ALEXA 555",
        "ALEXA 700",
    ]


def test_fluor_mirrors_dye() -> None:
    for channel in _channels():
        assert channel["fluor"] == channel["dye"]


def test_detectors() -> None:
    assert [c["detector"] for c in _channels()] == [
        "HyD S 1",
        "HyD S 3",
        "HyD R 5",
        "HyD X 2",
        "HyD X 4",
        "HyD X 2",
        "HyD X 4",
    ]


def test_excitation_uses_real_sequences_not_phantom() -> None:
    # The trap: the Master block's phantom 620 nm line and the duplicate
    # top-level Attachment copy must not corrupt the spectral pairing.
    excitations = [c["excitation_nm"] for c in _channels()]
    assert excitations == [405, 590, 753, 499, 653, 553, 696]
    assert 620 not in excitations
    assert all(e is not None for e in excitations)


def test_excitation_values_are_ints_when_integral() -> None:
    for channel in _channels():
        assert isinstance(channel["excitation_nm"], int)


def test_emission_bands_rounded() -> None:
    bands = [(c["emission_low_nm"], c["emission_high_nm"]) for c in _channels()]
    assert bands == [
        (430, 499),
        (601, 640),
        (768, 829),
        (506, 548),
        (663, 688),
        (562, 600),
        (706, 749),
    ]


# --- fail-closed hardening --------------------------------------------------


def test_billion_laughs_returns_empty_fast() -> None:
    bomb = '<?xml version="1.0"?>\n<!DOCTYPE lolz [\n <!ENTITY a "aaaaaaaaaa">\n'
    prev = "a"
    for i in range(2, 12):
        bomb += f' <!ENTITY a{i} "{"&" + prev + ";" * 10}">\n'  # nested expansion
        prev = f"a{i}"
    bomb += "]>\n<root>&a11;</root>"
    assert extract_channels(bomb) == []


def test_external_entity_returns_empty() -> None:
    xxe = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        "<root>&x;</root>"
    )
    assert extract_channels(xxe) == []


def test_doctype_is_rejected_even_without_entities() -> None:
    assert extract_channels('<!DOCTYPE r SYSTEM "x.dtd"><r/>') == []


def test_malformed_xml_returns_empty() -> None:
    assert extract_channels("<Element><Data><Image") == []
    assert extract_channels("not xml at all") == []


def test_empty_and_nonstring_return_empty() -> None:
    assert extract_channels("") == []
    assert extract_channels(None) == []  # type: ignore[arg-type]


def test_oversized_input_returns_empty() -> None:
    huge = "<root>" + ("x" * (33 * 1024 * 1024)) + "</root>"
    assert extract_channels(huge) == []


def test_valid_xml_without_channels_returns_empty() -> None:
    assert extract_channels("<root><a/><b/></root>") == []


# --- graceful degradation on partial metadata ------------------------------


def test_non_finite_band_values_degrade_without_dropping_channel() -> None:
    # A pathological inf/nan emission must not crash round() or drop the
    # channel — the band degrades to None while excitation is still kept.
    xml = """<Element><Data><Image><ImageDescription><Channels>
      <ChannelDescription>
        <ChannelProperty><Key>DetectorName</Key><Value>HyD S 1</Value></ChannelProperty>
        <ChannelProperty><Key>SequentialSettingIndex</Key><Value>0</Value></ChannelProperty>
      </ChannelDescription>
    </Channels></ImageDescription>
    <Attachment Name="HardwareSetting"><LDM_Block_Sequential><LDM_Block_Sequential_List>
      <ATLConfocalSettingDefinition>
        <Detector Channel="1" Name="HyD S 1" IsActive="1"/>
        <LaserLineSetting LaserLine="488" IntensityDev="5"/>
        <MultiBand Channel="1" LeftWorld="inf" RightWorld="nan"/>
      </ATLConfocalSettingDefinition>
    </LDM_Block_Sequential_List></LDM_Block_Sequential></Attachment>
    </Image></Data></Element>"""
    out = extract_channels(xml)
    assert len(out) == 1
    assert out[0]["excitation_nm"] == 488
    assert out[0]["emission_low_nm"] is None
    assert out[0]["emission_high_nm"] is None


def test_missing_fields_degrade_to_none_without_dropping_channel() -> None:
    xml = """<Element><Data><Image><ImageDescription><Channels>
      <ChannelDescription>
        <ChannelProperty><Key>DetectorName</Key><Value>HyD X 2</Value></ChannelProperty>
        <ChannelProperty><Key>SequentialSettingIndex</Key><Value>0</Value></ChannelProperty>
      </ChannelDescription>
    </Channels></ImageDescription>
    <Attachment Name="HardwareSetting"><LDM_Block_Sequential><LDM_Block_Sequential_List>
      <ATLConfocalSettingDefinition>
        <Detector Channel="2" Name="HyD X 2" IsActive="0"/>
      </ATLConfocalSettingDefinition>
    </LDM_Block_Sequential_List></LDM_Block_Sequential></Attachment>
    </Image></Data></Element>"""
    out = extract_channels(xml)
    assert len(out) == 1
    assert out[0]["detector"] == "HyD X 2"
    assert out[0]["dye"] is None
    assert out[0]["fluor"] is None
    assert out[0]["excitation_nm"] is None
    assert out[0]["emission_low_nm"] is None
    assert out[0]["emission_high_nm"] is None
