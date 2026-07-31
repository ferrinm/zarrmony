"""Tests for the zarrmony-side LIF channel projections + api wiring.

Covers:
- ``channels_to_omero`` — the label ladder, the trailing-parenthetical strip on
  the dye, and the always-valid hex color.
- ``channels_to_ome_channels`` — cleaned name vs full fluor, excitation/emission
  units, band-center rounding, and field-omission when a source is None.
- the fail-safe contract: empty / garbage input degrades, never raises.
- the api wiring: a LIF-shaped reader (exposes ``scene_root``) gets real
  ``<Channel>`` elements + omero label/color in its store; a non-LIF reader is
  untouched and a LIF whose channel count disagrees with the data falls back.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import zarr
from ome_types import from_xml

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.errors import ChannelColorCollisionWarning
from zarrmony.metadata.lif_channels import (
    channels_to_ome_channels,
    channels_to_omero,
    extract_channels,
)
from zarrmony.readers.plugin import ReaderPlugin

FIXTURE = Path(__file__).parent / "fixtures" / "lif_confocal_7ch.xml"


def _channels() -> list[dict]:
    return extract_channels(FIXTURE.read_text(encoding="utf-8"))


# --- channels_to_omero ------------------------------------------------------


def test_omero_labels_match_the_confocal_fixture() -> None:
    labels = [o["label"] for o in channels_to_omero(_channels())]
    assert labels == [
        "DAPI (405 nm)",
        "ALEXA 594 (590 nm)",
        "ALEXA 750 (753 nm)",
        "ALEXA 488 (499 nm)",
        "ALEXA 647-R-PE (653 nm)",
        "ALEXA 555 (553 nm)",
        "ALEXA 700 (696 nm)",
    ]


def test_omero_colors_are_always_valid_hex6() -> None:
    for o in channels_to_omero(_channels()):
        assert isinstance(o["color"], str)
        assert len(o["color"]) == 6
        int(o["color"], 16)  # parses as hex


def test_omero_label_ladder() -> None:
    rows = [
        # dye + excitation -> "dye (exc nm)" with the parenthetical stripped
        ({"dye": "DAPI (dsDNA bound)", "excitation_nm": 405}, "DAPI (405 nm)"),
        # dye only -> cleaned dye
        ({"dye": "ALEXA 594", "excitation_nm": None}, "ALEXA 594"),
        # excitation only -> "exc nm"
        ({"dye": None, "excitation_nm": 488}, "488 nm"),
        # neither -> None
        ({"dye": None, "excitation_nm": None}, None),
        # a dye that is *only* a parenthetical collapses to None, not "" ->
        # excitation (if any) takes over per the ladder
        ({"dye": "(unmixed)", "excitation_nm": 561}, "561 nm"),
    ]
    for ch, expected in rows:
        assert channels_to_omero([ch])[0]["label"] == expected


def test_omero_color_uses_cleaned_dye_not_full_name() -> None:
    # "DAPI (dsDNA bound)" must map to DAPI's blue via the cleaned name, the
    # same as a bare "DAPI" would — not the palette fallback.
    cleaned = channels_to_omero([{"dye": "DAPI (dsDNA bound)"}])[0]["color"]
    bare = channels_to_omero([{"dye": "DAPI"}])[0]["color"]
    assert cleaned == bare


# --- channels_to_ome_channels -----------------------------------------------


def test_ome_channels_shape_and_identity() -> None:
    ome = channels_to_ome_channels(_channels())
    assert len(ome) == 7
    assert [c.id for c in ome] == [f"Channel:0:{i}" for i in range(7)]
    # name is the *cleaned* dye; fluor is the *full* dye.
    assert ome[0].name == "DAPI"
    assert ome[0].fluor == "DAPI (dsDNA bound)"
    # excitation carried with nm units.
    assert float(ome[0].excitation_wavelength) == 405.0
    assert str(ome[0].excitation_wavelength_unit) in ("nm", "UnitsLength.NANOMETER")
    # emission at band center (430..499 -> 464.5 -> 464), with nm units.
    assert float(ome[0].emission_wavelength) == 464.0
    assert str(ome[0].emission_wavelength_unit) in ("nm", "UnitsLength.NANOMETER")


def test_ome_channels_omit_fields_when_source_is_none() -> None:
    # detector-only channel: no dye/fluor, no excitation, no emission band.
    ch = {
        "index": 0,
        "dye": None,
        "fluor": None,
        "detector": "HyD X 2",
        "excitation_nm": None,
        "emission_low_nm": None,
        "emission_high_nm": None,
    }
    c = channels_to_ome_channels([ch])[0]
    assert c.id == "Channel:0:0"
    assert c.name is None
    assert c.fluor is None
    assert c.excitation_wavelength is None
    assert c.emission_wavelength is None
    # color is still set (palette fallback), never None.
    assert c.color is not None


def test_ome_channels_omit_emission_when_band_half_missing() -> None:
    ch = {"emission_low_nm": 500, "emission_high_nm": None}
    assert channels_to_ome_channels([ch])[0].emission_wavelength is None


# --- fail-safe: empty / garbage never raises --------------------------------


def test_empty_input_yields_empty_projections() -> None:
    assert channels_to_omero([]) == []
    assert channels_to_ome_channels([]) == []


def test_garbage_dicts_degrade_without_raising() -> None:
    garbage = [{}, {"dye": ""}, {"excitation_nm": None, "dye": None}]
    omero = channels_to_omero(garbage)
    assert [o["label"] for o in omero] == [None, None, None]
    assert all(isinstance(o["color"], str) and len(o["color"]) == 6 for o in omero)
    # ome_types projection also survives empty dicts.
    assert len(channels_to_ome_channels(garbage)) == 3


# --- api wiring -------------------------------------------------------------


class FakeLifReader(FakeReader):
    """A LIF-shaped FakeReader: adds the ``scene_root`` the api keys off."""

    def __init__(self, *, scene_xml: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._scene_xml = scene_xml

    @property
    def scene_root(self) -> ET.Element:
        return ET.fromstring(self._scene_xml)


def _install(monkeypatch, reader: FakeReader) -> None:
    plugin = ReaderPlugin(
        name="bioio-lif",
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution="bioio-lif",
        source="builtin",
    )
    monkeypatch.setattr(api_module, "get_reader", lambda _path: (reader, plugin, 100))


def test_api_lif_path_writes_real_channels(tmp_path: Path, monkeypatch) -> None:
    xml = FIXTURE.read_text(encoding="utf-8")
    reader = FakeLifReader(
        scene_xml=xml,
        scenes=["scene0"],
        dims="CYX",
        shape=(7, 16, 16),  # SizeC must equal the 7 extracted channels
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    store = out / "scene0.ome.zarr"

    # OME-XML carries real <Channel> elements with the cleaned names + waves.
    parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    chans = parsed.images[0].pixels.channels
    assert [c.name for c in chans] == [
        "DAPI",
        "ALEXA 594",
        "ALEXA 750",
        "ALEXA 488",
        "ALEXA 647-R-PE",
        "ALEXA 555",
        "ALEXA 700",
    ]
    assert float(chans[0].excitation_wavelength) == 405.0
    assert float(chans[0].emission_wavelength) == 464.0
    assert chans[0].fluor == "DAPI (dsDNA bound)"

    # omero (display) label/color landed in the zarr group attrs.
    g = zarr.open_group(str(store), mode="r")
    omero = g.attrs["ome"]["omero"]
    labels = [c["label"] for c in omero["channels"]]
    assert labels[0] == "DAPI (405 nm)"
    assert labels[1] == "ALEXA 594 (590 nm)"
    for c in omero["channels"]:
        assert isinstance(c["color"], str) and len(c["color"]) == 6

    # ADR-0008 / #61: the audit block also carries the extracted channel
    # identity, projected into the shared 9-key shape. Same colors as omero.
    audit_channels = g.attrs["zarrmony"]["per_scene"][0]["channels"]
    assert len(audit_channels) == 7
    assert audit_channels[0]["index"] == 0
    assert audit_channels[0]["name"] == "DAPI"
    assert audit_channels[0]["dye"] == "DAPI (dsDNA bound)"
    assert audit_channels[0]["fluor"] == "DAPI (dsDNA bound)"
    assert audit_channels[0]["excitation_nm"] == 405
    assert audit_channels[0]["emission_low_nm"] == 430
    assert audit_channels[0]["emission_high_nm"] == 499
    assert audit_channels[0]["color"] == omero["channels"][0]["color"]
    # LIF-internal `detector` never surfaces in the audit shape.
    assert all("detector" not in c for c in audit_channels)


def test_api_non_lif_reader_is_untouched(tmp_path: Path, monkeypatch) -> None:
    # No scene_root -> the LIF decision is a no-op and the name-based path runs.
    reader = FakeReader(
        scenes=["s"],
        dims="CYX",
        shape=(2, 16, 16),
        channel_names=["DAPI", "GFP"],
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.czi", out, pyramid_min_size=8)

    store = out / "s.ome.zarr"
    g = zarr.open_group(str(store), mode="r")
    labels = [c["label"] for c in g.attrs["ome"]["omero"]["channels"]]
    # name-based omero, exactly as before this slice.
    assert labels == ["DAPI", "GFP"]
    # OME-XML uses the reader's ome_metadata stub (no LIF <Channel> identity).
    parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    assert parsed.images[0].name == "s"


def test_api_lif_channel_count_mismatch_falls_back(tmp_path: Path, monkeypatch) -> None:
    # 7 channels extracted but the data only has SizeC=2: don't write an
    # OME-XML that contradicts the array. Fall back to the existing path.
    xml = FIXTURE.read_text(encoding="utf-8")
    reader = FakeLifReader(
        scene_xml=xml,
        scenes=["scene0"],
        dims="CYX",
        shape=(2, 16, 16),
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    # Must not raise; conversion completes.
    convert("/tmp/x.lif", out, pyramid_min_size=8)

    store = out / "scene0.ome.zarr"
    parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    # SizeC honors the data (2), and the contradictory 7-channel identity was
    # NOT forced in (channel count, if any, never exceeds SizeC).
    assert parsed.images[0].pixels.size_c == 2
    assert len(parsed.images[0].pixels.channels) <= 2
    # The omero block also stays consistent: never 7 labels on a 2-channel image.
    g = zarr.open_group(str(store), mode="r")
    assert len(g.attrs["ome"]["omero"]["channels"]) == 2


@pytest.mark.parametrize(
    "dtype, expected_window",
    [
        (np.uint8, {"min": 0, "max": 255, "start": 0, "end": 255}),
        (np.uint16, {"min": 0, "max": 65535, "start": 0, "end": 65535}),
        (np.float32, {"min": 0.0, "max": 1.0, "start": 0.0, "end": 1.0}),
    ],
)
def test_api_lif_omero_window_matches_reader_dtype(
    tmp_path: Path, monkeypatch, dtype, expected_window
) -> None:
    """The LIF ``_lif_scene_channels`` path must also honor the reader's dtype
    when constructing the OMERO display window. Same regression as #50 but for
    the LIF-identity branch that would otherwise ship 0–255 next to real
    fluorophore labels.

    Pinned to ``contrast_percentile=None`` so this test isolates the dtype-range
    behavior; issue-#53 percentile contrast is exercised separately.
    """
    xml = FIXTURE.read_text(encoding="utf-8")
    reader = FakeLifReader(
        scene_xml=xml,
        scenes=["scene0"],
        dims="CYX",
        shape=(7, 16, 16),
        dtype=dtype,
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, contrast_percentile=None)

    g = zarr.open_group(str(out / "scene0.ome.zarr"), mode="r")
    channels = g.attrs["ome"]["omero"]["channels"]
    assert len(channels) == 7
    for c in channels:
        assert c["window"] == expected_window


def test_api_lif_with_garbage_scene_root_falls_back(
    tmp_path: Path, monkeypatch
) -> None:
    # scene_root present but yields no channels -> graceful fallback, no crash.
    reader = FakeLifReader(
        scene_xml="<root><nothing/></root>",
        scenes=["s"],
        dims="CYX",
        shape=(1, 16, 16),
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    store = out / "s.ome.zarr"
    assert (store / "OME" / "METADATA.ome.xml").exists()


# --- ADR-0007: channel_colors="source-file" and dict overrides -------------


def _one_channel_lif_xml(*, lut_name: str, dye: str = "DAPI (dsDNA bound)") -> str:
    """Minimal LIF-shaped XML for a single-channel scene with a LUTName hint.

    The parser needs a ChannelDescription with a LUTName attribute plus a
    valid sequence block so excitation/emission can attach; only the LUTName
    is load-bearing for source-file-mode tests but the rest matches a real
    Leica scene closely enough to route through the normal extraction path.
    """
    return f"""<Element><Data><Image><ImageDescription><Channels>
      <ChannelDescription LUTName="{lut_name}">
        <ChannelProperty><Key>DyeName</Key><Value>Leica/{dye}</Value></ChannelProperty>
        <ChannelProperty><Key>DetectorName</Key><Value>HyD S 1</Value></ChannelProperty>
        <ChannelProperty><Key>SequentialSettingIndex</Key><Value>0</Value></ChannelProperty>
      </ChannelDescription>
    </Channels></ImageDescription>
    <Attachment Name="HardwareSetting"><LDM_Block_Sequential><LDM_Block_Sequential_List>
      <ATLConfocalSettingDefinition>
        <Detector Channel="1" Name="HyD S 1" IsActive="1"/>
        <LaserLineSetting LaserLine="405" IntensityDev="5"/>
        <MultiBand Channel="1" LeftWorld="430" RightWorld="480"/>
      </ATLConfocalSettingDefinition>
    </LDM_Block_Sequential_List></LDM_Block_Sequential></Attachment>
    </Image></Data></Element>"""


def test_api_source_file_mode_uses_stored_lut_color(
    tmp_path: Path, monkeypatch
) -> None:
    """``channel_colors="source-file"`` swaps in the LIF LUTName color.

    Emission (430–480 midpoint 455) would be cyan under the default; the
    file's ``LUTName="Red"`` overrides that to plain red — the whole point of
    source-file mode is to trust the reader's stored per-channel hint over
    our band scheme.
    """
    reader = FakeLifReader(
        scene_xml=_one_channel_lif_xml(lut_name="Red"),
        scenes=["scene0"],
        dims="CYX",
        shape=(1, 16, 16),
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, channel_colors="source-file")

    store = out / "scene0.ome.zarr"
    g = zarr.open_group(str(store), mode="r")
    colors = [c["color"] for c in g.attrs["ome"]["omero"]["channels"]]
    assert colors == ["ff0000"]  # LUTName="Red" → plain red, not band cyan


def test_api_source_file_mode_falls_through_when_lut_missing(
    tmp_path: Path, monkeypatch
) -> None:
    # No LUTName attribute → source-file mode has nothing to consume, so the
    # emission-band scheme takes over (405 nm exc / 430–480 nm emission → cyan).
    xml = _one_channel_lif_xml(lut_name="")  # empty LUTName is treated as absent
    xml = xml.replace('LUTName=""', "")
    reader = FakeLifReader(
        scene_xml=xml,
        scenes=["scene0"],
        dims="CYX",
        shape=(1, 16, 16),
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8, channel_colors="source-file")

    store = out / "scene0.ome.zarr"
    g = zarr.open_group(str(store), mode="r")
    colors = [c["color"] for c in g.attrs["ome"]["omero"]["channels"]]
    # Emission midpoint 455 → deep-blue band → cyan.
    assert colors == ["00ffff"]


def test_api_dict_override_wins_over_band_scheme(tmp_path: Path, monkeypatch) -> None:
    # A dict override keyed on the cleaned dye ("DAPI") beats the emission band.
    reader = FakeLifReader(
        scene_xml=_one_channel_lif_xml(lut_name="Blue"),
        scenes=["scene0"],
        dims="CYX",
        shape=(1, 16, 16),
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert(
        "/tmp/x.lif",
        out,
        pyramid_min_size=8,
        channel_colors={"DAPI": "112233"},
    )

    store = out / "scene0.ome.zarr"
    g = zarr.open_group(str(store), mode="r")
    colors = [c["color"] for c in g.attrs["ome"]["omero"]["channels"]]
    assert colors == ["112233"]


def test_convert_rejects_unknown_channel_colors_string(tmp_path: Path) -> None:
    # Any string other than "source-file" is an error at the API surface —
    # matches the ChannelColorSpec type and prevents typo-silent fallthrough.
    import pytest

    with pytest.raises(ValueError, match="source-file"):
        convert(
            "/tmp/x.lif",
            tmp_path / "out",
            pyramid_min_size=8,
            channel_colors="soure-file",  # typo
        )


def test_convert_preserves_channel_colors_verbatim_in_audit(
    tmp_path: Path, monkeypatch
) -> None:
    """The audit's ``config.channel_colors`` records exactly what the user passed.

    ADR-0007: downstream consumers must be able to tell "user passed None"
    apart from "user passed a dict" apart from "user passed 'source-file'";
    the audit is the only durable record.
    """
    reader = FakeLifReader(
        scene_xml=_one_channel_lif_xml(lut_name="Green"),
        scenes=["scene0"],
        dims="CYX",
        shape=(1, 16, 16),
    )
    _install(monkeypatch, reader)

    result = convert(
        "/tmp/x.lif",
        tmp_path / "out",
        pyramid_min_size=8,
        channel_colors="source-file",
    )
    audit_config = result["stores"][0]["config"]
    assert audit_config["channel_colors"] == "source-file"


# --- ADR-0007: collision handling on a real projection ---------------------


def test_channels_to_omero_reallocates_collisions() -> None:
    """Two far-red channels both landing on white → second-in-order moves.

    Exercises the projection surface end-to-end (not just ``assign_colors``)
    so any regression in the LIF-side wiring is caught.
    """
    import warnings

    channels = [
        {"dye": "AF647", "emission_low_nm": 663, "emission_high_nm": 688},
        {"dye": "AF680", "emission_low_nm": 680, "emission_high_nm": 720},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        omero = channels_to_omero(channels)
    assert omero[0]["color"] == "ffffff"
    assert omero[1]["color"] != "ffffff"
    assert any(
        isinstance(w.message, ChannelColorCollisionWarning) for w in caught
    ), "collision warning must fire"


# --- ADR-0007: default writer path (non-LIF) uses band scheme via dye name -


def test_non_lif_reader_default_channels_use_band_scheme(
    tmp_path: Path, monkeypatch
) -> None:
    """``writers/scene.py::_default_channels`` reaches the emission bands.

    A non-LIF reader with recognizable dye names ('DAPI', 'GFP', 'mCherry',
    'AF647') should land those channels on the colorblind palette via the
    dye-name substring fallback — no all-white fallback like the old default.
    """
    reader = FakeReader(
        scenes=["s"],
        dims="CYX",
        shape=(4, 16, 16),
        channel_names=["DAPI", "GFP", "mCherry", "AF647"],
    )
    _install(monkeypatch, reader)
    out = tmp_path / "out"

    convert("/tmp/x.czi", out, pyramid_min_size=8)

    store = out / "s.ome.zarr"
    g = zarr.open_group(str(store), mode="r")
    colors = [c["color"] for c in g.attrs["ome"]["omero"]["channels"]]
    assert colors == ["00ffff", "00ff00", "ff00ff", "ffffff"]
