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

import zarr
from ome_types import from_xml

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert
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


def _md() -> dict:
    return {"microscope": "Stellaris", "modality": "fluorescence"}


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

    convert("/tmp/x.lif", out, metadata=_md(), pyramid_min_size=8)

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

    convert("/tmp/x.czi", out, metadata=_md(), pyramid_min_size=8)

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
    convert("/tmp/x.lif", out, metadata=_md(), pyramid_min_size=8)

    store = out / "scene0.ome.zarr"
    parsed = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    # SizeC honors the data (2), and the contradictory 7-channel identity was
    # NOT forced in (channel count, if any, never exceeds SizeC).
    assert parsed.images[0].pixels.size_c == 2
    assert len(parsed.images[0].pixels.channels) <= 2
    # The omero block also stays consistent: never 7 labels on a 2-channel image.
    g = zarr.open_group(str(store), mode="r")
    assert len(g.attrs["ome"]["omero"]["channels"]) == 2


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

    convert("/tmp/x.lif", out, metadata=_md(), pyramid_min_size=8)

    store = out / "s.ome.zarr"
    assert (store / "OME" / "METADATA.ome.xml").exists()
