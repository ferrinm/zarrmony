"""End-to-end wiring tests for LIF objective-lens metadata (issue #52).

Drives ``convert()`` against a FakeReader whose ``metadata`` is a LIF-shaped
XML blob carrying (or deliberately not carrying) an
``<ATLConfocalSettingDefinition>`` with objective attributes. Asserts, for each
of the three acceptance cases in #52 — complete / partial / absent — both:

* the on-disk audit at ``attrs.zarrmony.per_scene[0].objective``, and
* the on-disk OME-XML's ``<Instrument><Objective/></Instrument>`` +
  ``<ObjectiveSettings/>`` reference on the ``<Image>``.

Uses the same ``FakeReader`` + ``patched_reader`` doubles as the rest of the
LIF wiring tests so no real ``bioio_lif`` import is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ome_types import from_xml

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.readers.plugin import ReaderPlugin


def _fake_plugin() -> ReaderPlugin:
    return ReaderPlugin(
        name="bioio-lif",
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution="bioio-lif",
        source="builtin",
    )


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    def installer(reader: FakeReader):
        monkeypatch.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, _fake_plugin(), 100),
        )

    return installer


def _lif_metadata_with_atl(atl_attrs: str) -> str:
    """A LIF-shaped metadata blob with one ``<Image>`` carrying a single ATL
    settings block containing ``atl_attrs`` (which may be empty for the
    "no objective info" case).

    The extractor doesn't require the surrounding ChannelDescription /
    Sequential blocks — objective attrs live on ATLConfocalSettingDefinition
    directly — so we keep the fixture minimal.
    """
    atl = f"<ATLConfocalSettingDefinition {atl_attrs}/>" if atl_attrs else ""
    return (
        "<LMSDataContainerHeader><Element><Children><Element>"
        "<Data><Image>"
        f'<Attachment Name="HardwareSetting">{atl}</Attachment>'
        "</Image></Data></Element></Children></Element></LMSDataContainerHeader>"
    )


def _reader_with_metadata(metadata: str) -> FakeReader:
    return FakeReader(
        scenes=["scene0"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        channel_names=["DAPI"],
        raw_xml=metadata,
    )


# --- complete objective block ----------------------------------------------


def test_complete_objective_lands_in_audit_and_ome_xml(
    tmp_path: Path, patched_reader
) -> None:
    reader = _reader_with_metadata(
        _lif_metadata_with_atl(
            'ObjectiveName="HC PL APO CS2 63x/1.40 OIL" '
            'ObjectiveMag="63" NumericalAperture="1.4" Immersion="OIL" '
            'WorkingDistance="140"'
        )
    )
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    store = out / "scene0.ome.zarr"

    # (1) Audit — full dict with every field populated
    audit = json.loads((store / "zarr.json").read_text())["attributes"]["zarrmony"]
    objective = audit["per_scene"][0]["objective"]
    assert objective == {
        "nominal_magnification": 63,
        "numerical_aperture": 1.4,
        "immersion": "Oil",
        "model": "HC PL APO CS2 63x/1.40 OIL",
        "working_distance_um": 140,
    }

    # (2) OME-XML — top-level <Instrument> with <Objective>, and per-image
    # <ObjectiveSettings> + <InstrumentRef>.
    ome = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    assert len(ome.instruments) == 1
    instrument = ome.instruments[0]
    assert len(instrument.objectives) == 1
    obj = instrument.objectives[0]
    assert obj.nominal_magnification == 63.0
    assert obj.lens_na == 1.4
    assert obj.model == "HC PL APO CS2 63x/1.40 OIL"
    assert obj.immersion.value == "Oil"
    assert obj.working_distance == 140.0
    assert obj.working_distance_unit.value == "µm"

    image = ome.images[0]
    assert image.instrument_ref is not None
    assert image.instrument_ref.id == instrument.id
    assert image.objective_settings is not None
    assert image.objective_settings.id == obj.id


# --- partial objective block (magnification + NA only) ---------------------


def test_partial_objective_omits_missing_keys_from_audit(
    tmp_path: Path, patched_reader
) -> None:
    reader = _reader_with_metadata(
        _lif_metadata_with_atl('ObjectiveMag="20" NumericalAperture="0.8"')
    )
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    store = out / "scene0.ome.zarr"
    audit = json.loads((store / "zarr.json").read_text())["attributes"]["zarrmony"]

    # Audit dict has ONLY the two extracted keys — no None-valued placeholders
    # for model / immersion / working_distance_um.
    objective = audit["per_scene"][0]["objective"]
    assert objective == {"nominal_magnification": 20, "numerical_aperture": 0.8}
    for missing in ("model", "immersion", "working_distance_um"):
        assert missing not in objective

    # OME-XML: the partial <Objective> element is still emitted with an ID and
    # the two populated fields; the omitted OME attributes stay absent.
    ome = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    obj = ome.instruments[0].objectives[0]
    assert obj.nominal_magnification == 20.0
    assert obj.lens_na == 0.8
    assert obj.model is None
    assert obj.immersion is None
    assert obj.working_distance is None


# --- no objective info at all ----------------------------------------------


def test_absent_objective_omits_key_from_audit_and_no_instrument(
    tmp_path: Path, patched_reader
) -> None:
    # LIF-shaped metadata with a scene <Image> but NO ATLConfocalSettingDefinition
    # (nor <Objective> element) — the extractor returns None, so the audit
    # must OMIT the ``objective`` key entirely (not set it to null / {}), and
    # the OME-XML must carry no <Instrument> at all.
    reader = _reader_with_metadata(_lif_metadata_with_atl(""))
    patched_reader(reader)
    out = tmp_path / "out"

    convert("/tmp/x.lif", out, pyramid_min_size=8)

    store = out / "scene0.ome.zarr"
    audit = json.loads((store / "zarr.json").read_text())["attributes"]["zarrmony"]

    assert "objective" not in audit["per_scene"][0]

    ome = from_xml((store / "OME" / "METADATA.ome.xml").read_text())
    assert ome.instruments == []
    assert ome.images[0].instrument_ref is None
    assert ome.images[0].objective_settings is None
