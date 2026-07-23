"""Tests for the stdlib-only Leica LIF objective-lens extractor.

Exercises the real captured confocal fixture (attributes on
``ATLConfocalSettingDefinition``), the sibling ``<Objective>`` shape some
LIF exports use, the "complete"/"partial"/"absent" acceptance triple from
issue #52, and the fail-closed hardening shared with
:mod:`zarrmony.metadata.lif_channels`.
"""

from pathlib import Path

from zarrmony.metadata.objective import extract_objective

FIXTURE = Path(__file__).parent / "fixtures" / "lif_confocal_7ch.xml"


# --- real captured confocal fixture ----------------------------------------


def test_real_fixture_yields_objective_fields() -> None:
    obj = extract_objective(FIXTURE.read_text(encoding="utf-8"))
    assert obj is not None
    # ObjectiveName in the fixture is "HC PL APO CS2    20x/0.75 DRY ";
    # the extractor collapses whitespace and strips the trailing space.
    assert obj["model"] == "HC PL APO CS2 20x/0.75 DRY"
    # The bare ATLConfocalSettingDefinition Magnification="21" is *total*
    # magnification (obj_mag * zoom) and MUST NOT be used; the true objective
    # magnification is recovered from the "20x" embedded in the name.
    assert obj["nominal_magnification"] == 20
    assert obj["numerical_aperture"] == 0.75
    # LIF "DRY" is Leica shorthand for an air (no-medium) objective.
    assert obj["immersion"] == "Air"


def test_real_fixture_returns_only_known_keys() -> None:
    obj = extract_objective(FIXTURE.read_text(encoding="utf-8"))
    # The fixture does not carry WorkingDistance — the key must be omitted
    # (never null / 0), per the "missing individual fields" rule in #52.
    assert "working_distance_um" not in obj
    assert set(obj).issubset(
        {
            "nominal_magnification",
            "numerical_aperture",
            "immersion",
            "model",
            "working_distance_um",
        }
    )


# --- acceptance triple: complete / partial / absent ------------------------
#
# Per #52 the writer must handle three cases distinctly: a fully populated
# objective block, a partial one (magnification + NA only), and a scene with
# no objective info at all (``objective`` key OMITTED from the audit, not
# a dict of Nones). Hand-crafted minimal XML fixtures pin each case.


def _wrap_atl(attrs: str) -> str:
    """Wrap a bare attribute string into a minimal valid LIF-shaped XML doc.

    The extractor walks by tag name (``ATLConfocalSettingDefinition``) so
    ancestry beyond a container root is irrelevant to the assertion under test.
    """
    return f"<Element><Data><Image>{attrs}</Image></Data></Element>"


def test_complete_objective_populates_every_field() -> None:
    xml = _wrap_atl(
        '<ATLConfocalSettingDefinition ObjectiveName="HC PL APO CS2 63x/1.40 OIL" '
        'ObjectiveMag="63" NumericalAperture="1.4" Immersion="OIL" '
        'WorkingDistance="140"/>'
    )
    obj = extract_objective(xml)
    assert obj == {
        "nominal_magnification": 63,
        "numerical_aperture": 1.4,
        "immersion": "Oil",
        "model": "HC PL APO CS2 63x/1.40 OIL",
        "working_distance_um": 140,
    }


def test_partial_objective_omits_missing_fields() -> None:
    # Magnification + NA only — per #52's "partial" acceptance case. The
    # missing model / immersion / working_distance_um keys MUST be absent
    # from the dict entirely, not set to None or 0.
    xml = _wrap_atl(
        '<ATLConfocalSettingDefinition ObjectiveMag="20" NumericalAperture="0.8"/>'
    )
    obj = extract_objective(xml)
    assert obj == {"nominal_magnification": 20, "numerical_aperture": 0.8}
    assert "immersion" not in obj
    assert "model" not in obj
    assert "working_distance_um" not in obj


def test_scene_with_no_objective_info_returns_none() -> None:
    # A scene XML with no ATLConfocalSettingDefinition and no <Objective>
    # element must yield None (NOT an empty dict) — that's the discriminator
    # the audit uses to omit the ``objective`` key from ``per_scene[i]``.
    assert extract_objective("<Element><Data/></Element>") is None


# --- alternative source: <Objective> sibling element -----------------------


def test_objective_element_supplies_fields() -> None:
    # Some LIF exports put the same fields on an <Objective> element beside
    # (or instead of) the ATL block. Either shape must yield the same dict.
    xml = (
        "<Element><Data><Image>"
        '<Objective ObjectiveName="Plan-Apo 40x/1.10 W" ObjectiveMag="40" '
        'NumericalAperture="1.1" Immersion="WATER"/>'
        "</Image></Data></Element>"
    )
    obj = extract_objective(xml)
    assert obj == {
        "nominal_magnification": 40,
        "numerical_aperture": 1.1,
        "immersion": "Water",
        "model": "Plan-Apo 40x/1.10 W",
    }


def test_atl_and_objective_merge_first_writer_wins() -> None:
    # When both surfaces carry fields, first-writer-wins per key — the
    # ATL block is iterated first, so its ObjectiveName wins over the
    # Objective element's Model. Fields only present on the Objective
    # element still contribute (the merge is *union*, not exclusion).
    xml = (
        "<Element><Data><Image>"
        '<ATLConfocalSettingDefinition ObjectiveName="HC PL APO 63x/1.4 OIL" '
        'NumericalAperture="1.4"/>'
        '<Objective Model="Ignored" ObjectiveMag="63" Immersion="OIL"/>'
        "</Image></Data></Element>"
    )
    obj = extract_objective(xml)
    assert obj["model"] == "HC PL APO 63x/1.4 OIL"
    assert obj["numerical_aperture"] == 1.4
    assert obj["nominal_magnification"] == 63  # from the Objective element
    assert obj["immersion"] == "Oil"


# --- immersion mapping (LIF strings → OME enum) ----------------------------


def test_immersion_maps_dry_to_air() -> None:
    # LIF "DRY" is not the OME enum value; it's Leica shorthand for air.
    xml = _wrap_atl('<ATLConfocalSettingDefinition Immersion="DRY"/>')
    assert extract_objective(xml) == {"immersion": "Air"}


def test_immersion_maps_glyc_to_glycerol() -> None:
    xml = _wrap_atl('<ATLConfocalSettingDefinition Immersion="GLYC"/>')
    assert extract_objective(xml) == {"immersion": "Glycerol"}


def test_immersion_unknown_falls_through_to_other() -> None:
    # An unrecognised-but-present immersion string is meaningful ("we saw
    # an immersion medium we don't recognise") and must map to "Other"
    # rather than being dropped — that's a different signal from "no
    # immersion metadata at all".
    xml = _wrap_atl('<ATLConfocalSettingDefinition Immersion="SILICONE"/>')
    assert extract_objective(xml) == {"immersion": "Other"}


def test_immersion_empty_attribute_is_omitted() -> None:
    xml = _wrap_atl('<ATLConfocalSettingDefinition Immersion=""/>')
    assert extract_objective(xml) is None


# --- graceful degradation --------------------------------------------------


def test_magnification_from_name_when_attribute_missing() -> None:
    # No ObjectiveMag / NominalMagnification, but the name embeds "20x".
    xml = _wrap_atl(
        '<ATLConfocalSettingDefinition ObjectiveName="HC PL APO 20x/0.75 DRY"/>'
    )
    obj = extract_objective(xml)
    assert obj["nominal_magnification"] == 20
    assert obj["model"] == "HC PL APO 20x/0.75 DRY"


def test_bare_magnification_attr_is_ignored() -> None:
    # ATL's "Magnification" attribute is total zoom (obj_mag * zoom), NOT the
    # objective's nominal magnification. Recording it would silently corrupt
    # the audit (e.g. 21 for a 20x objective at 1.05x zoom).
    xml = _wrap_atl('<ATLConfocalSettingDefinition Magnification="21"/>')
    assert extract_objective(xml) is None


def test_non_finite_numeric_values_degrade_to_omission() -> None:
    xml = _wrap_atl(
        '<ATLConfocalSettingDefinition NumericalAperture="inf" '
        'ObjectiveMag="nan" WorkingDistance="not-a-number"/>'
    )
    assert extract_objective(xml) is None


# --- fail-closed hardening (mirrors lif_channels) --------------------------


def test_billion_laughs_returns_none_fast() -> None:
    bomb = '<?xml version="1.0"?>\n<!DOCTYPE lolz [\n <!ENTITY a "aaaaaaaaaa">\n'
    prev = "a"
    for i in range(2, 12):
        bomb += f' <!ENTITY a{i} "{"&" + prev + ";" * 10}">\n'
        prev = f"a{i}"
    bomb += "]>\n<root>&a11;</root>"
    assert extract_objective(bomb) is None


def test_external_entity_returns_none() -> None:
    xxe = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        "<root>&x;</root>"
    )
    assert extract_objective(xxe) is None


def test_doctype_is_rejected_even_without_entities() -> None:
    assert extract_objective('<!DOCTYPE r SYSTEM "x.dtd"><r/>') is None


def test_malformed_xml_returns_none() -> None:
    assert extract_objective("<Element><Data><Image") is None
    assert extract_objective("not xml at all") is None


def test_empty_and_nonstring_return_none() -> None:
    assert extract_objective("") is None
    assert extract_objective(None) is None  # type: ignore[arg-type]


def test_oversized_input_returns_none() -> None:
    huge = "<root>" + ("x" * (33 * 1024 * 1024)) + "</root>"
    assert extract_objective(huge) is None
