"""Tests for the ADR-0008 / #61 audit-channels projections.

Covers ``zarrmony.metadata.audit_channels.from_lif_extracted`` (LIF path) and
``from_ome_channels`` (CZI/ND2/OME-TIFF path) — the two projections into the
shared per-scene ``channels`` audit shape.
"""

from ome_types.model import Channel

from zarrmony.metadata.audit_channels import (
    from_lif_extracted,
    from_ome_channels,
)

# --- LIF projection ---------------------------------------------------------


def test_lif_projection_carries_all_9_keys_when_populated() -> None:
    extracted = [
        {
            "index": 0,
            "dye": "DAPI (dsDNA bound)",
            "fluor": "DAPI (dsDNA bound)",
            "detector": "HyD S 1",
            "excitation_nm": 405,
            "emission_low_nm": 420,
            "emission_high_nm": 480,
            "lut_name": "Blue",
        }
    ]
    records = from_lif_extracted(extracted, colors=["0099ff"])
    assert records == [
        {
            "index": 0,
            "name": "DAPI",
            "dye": "DAPI (dsDNA bound)",
            "fluor": "DAPI (dsDNA bound)",
            "excitation_nm": 405,
            "emission_low_nm": 420,
            "emission_high_nm": 480,
            "color": "0099ff",
            "lut_name": "Blue",
        }
    ]


def test_lif_projection_omits_missing_keys_instead_of_null() -> None:
    # Detector-only channel: no dye/fluor/wavelengths/lut_name/color.
    extracted = [
        {
            "index": 0,
            "dye": None,
            "fluor": None,
            "detector": "HyD X 2",
            "excitation_nm": None,
            "emission_low_nm": None,
            "emission_high_nm": None,
            "lut_name": None,
        }
    ]
    records = from_lif_extracted(extracted, colors=None)
    assert records == [{"index": 0}]  # only `index` survives; every other key omitted


def test_lif_projection_omits_color_when_colors_length_mismatches() -> None:
    extracted = [{"index": 0, "dye": "DAPI"}, {"index": 1, "dye": "GFP"}]
    # Wrong length → treat as absent, per fail-safe behaviour.
    records = from_lif_extracted(extracted, colors=["0099ff"])
    assert "color" not in records[0]
    assert "color" not in records[1]


def test_lif_projection_strips_internal_detector_field() -> None:
    extracted = [
        {
            "index": 0,
            "dye": "DAPI",
            "fluor": "DAPI",
            "detector": "HyD S 1",
            "excitation_nm": 405,
            "emission_low_nm": 420,
            "emission_high_nm": 480,
            "lut_name": "Blue",
        }
    ]
    r = from_lif_extracted(extracted, colors=["0099ff"])[0]
    # `detector` is LIF-internal — never surfaces in the ADR-0008 shape.
    assert "detector" not in r


# --- OME projection ---------------------------------------------------------


def test_ome_projection_from_band_center_wavelength() -> None:
    ome_ch = Channel(
        id="Channel:0:0",
        name="DAPI",
        fluor="DAPI",
        excitation_wavelength=405.0,
        excitation_wavelength_unit="nm",
        emission_wavelength=450.0,
        emission_wavelength_unit="nm",
    )
    records = from_ome_channels([ome_ch], colors=["0099ff"])
    # OME gives a single emission point; ADR-0008 uses low == high for that.
    assert records == [
        {
            "index": 0,
            "name": "DAPI",
            "fluor": "DAPI",
            "excitation_nm": 405,
            "emission_low_nm": 450,
            "emission_high_nm": 450,
            "color": "0099ff",
        }
    ]


def test_ome_projection_omits_missing_fields() -> None:
    ome_ch = Channel(id="Channel:0:0", name="Ch0")
    records = from_ome_channels([ome_ch], colors=None)
    assert records == [{"index": 0, "name": "Ch0"}]


def test_ome_projection_empty_input_yields_empty_list() -> None:
    assert from_ome_channels([]) == []
    assert from_lif_extracted([]) == []
