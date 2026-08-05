"""Vendor-extractor tier wiring for `_audit_acquisition_for_scene` (issues #77, #78).

Pins the seam that dispatches to the CZI / ND2 vendor extractors when the
reader's class lives in the ``bioio_czi`` / ``bioio_nd2`` module namespace.
Precedence: LIF → vendor → OME → hook, uniform ``setdefault`` — so a vendor
extractor's ``"Zeiss Axioscan 7"`` overrides OME's ``"Zeiss"``, and the LIF
extractor still wins over both.
"""

from __future__ import annotations

from typing import Any

import pytest
from ome_types import OME
from ome_types.model import (
    Channel,
    Image,
    Instrument,
    Microscope,
    Pixels,
    PixelType,
)

from zarrmony.api import _audit_acquisition_for_scene, _vendor_acquisition_extras

_AXIOSCAN_XML = """\
<ImageDocument>
 <Metadata>
  <Information>
   <Instrument>
    <Microscopes>
     <Microscope Id="Microscope:1" Name="Axioscan 7">
      <Type>Upright</Type>
     </Microscope>
    </Microscopes>
   </Instrument>
  </Information>
 </Metadata>
</ImageDocument>
"""


def _ome_manufacturer_only() -> OME:
    """The bioio-czi projection: manufacturer set, model empty → ``"Zeiss"``."""
    return OME(
        images=[
            Image(
                id="Image:0",
                pixels=Pixels(
                    id="Pixels:0",
                    size_x=8,
                    size_y=8,
                    size_z=1,
                    size_c=1,
                    size_t=1,
                    dimension_order="XYZCT",
                    type=PixelType.UINT16,
                    channels=[
                        Channel(id="Channel:0:0", acquisition_mode="WideField"),
                    ],
                ),
            )
        ],
        instruments=[
            Instrument(
                id="Instrument:0",
                microscope=Microscope(manufacturer="Zeiss"),
            )
        ],
    )


def _make_reader(module: str, **attrs: Any) -> Any:
    """Build a stub reader whose class ``__module__`` is ``module``.

    Vendor dispatch keys on the class's module — creating a class with
    ``__module__="bioio_czi.stub"`` is enough to make the dispatch fire.
    """

    def _init(self: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)

    cls = type("_StubReader", (), {"__init__": _init, "__module__": module})
    return cls()


# --- _vendor_acquisition_extras: dispatch ---------------------------------


def test_vendor_dispatch_ignores_non_vendor_reader() -> None:
    """A reader whose class isn't from bioio-czi/nd2 → no vendor extras."""
    r = _make_reader("some.other.module", metadata=_AXIOSCAN_XML)
    assert _vendor_acquisition_extras(r) is None


def test_vendor_dispatch_reads_czi_metadata() -> None:
    """bioio-czi-shaped reader → CZI extractor runs on ``reader.metadata``."""
    r = _make_reader("bioio_czi.reader", metadata=_AXIOSCAN_XML)
    assert _vendor_acquisition_extras(r) == {"microscope": "Zeiss Axioscan 7"}


def test_vendor_dispatch_nd2_uses_text_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bioio-nd2-shaped reader → ND2 extractor opens the file via nd2 SDK."""
    import sys
    import types

    class _FakeND2File:
        def text_info(self) -> dict:
            return {"capturing": "Ti2"}

        def close(self) -> None:
            pass

    module = types.ModuleType("nd2")
    module.ND2File = lambda _path: _FakeND2File()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nd2", module)

    r = _make_reader("bioio_nd2.reader", _path="/tmp/fake.nd2")
    assert _vendor_acquisition_extras(r) == {"microscope": "Nikon Ti2"}


# --- _audit_acquisition_for_scene: layered precedence -------------------


def test_vendor_beats_ome_manufacturer_only_string() -> None:
    """The canonical #77 case: CZI OME → 'Zeiss'; vendor → 'Zeiss Axioscan 7'.

    Vendor tier ``setdefault``-fills first, so its full string blocks OME's
    manufacturer-only fallback for the ``microscope`` key. OME still fills
    other keys the vendor tier didn't emit (``imaging_method`` here).
    """
    r = _make_reader(
        "bioio_czi.reader",
        metadata=_AXIOSCAN_XML,
        ome_metadata=_ome_manufacturer_only(),
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {
        "microscope": "Zeiss Axioscan 7",
        "imaging_method": ["widefield_fluorescence"],
    }


def test_ome_fills_gaps_vendor_extractor_left() -> None:
    """A CZI whose raw XML lacks a Microscope element still gets OME's
    manufacturer-only ``"Zeiss"`` — vendor tier omitted rather than blocked."""
    r = _make_reader(
        "bioio_czi.reader",
        metadata="<ImageDocument><Metadata/></ImageDocument>",
        ome_metadata=_ome_manufacturer_only(),
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {
        "microscope": "Zeiss",
        "imaging_method": ["widefield_fluorescence"],
    }


def test_lif_wins_over_vendor_and_ome() -> None:
    """LIF-extracted microscope holds even when vendor tier would also fire."""
    lif = {"microscope": "STELLARIS 8", "imaging_method": ["confocal"]}
    r = _make_reader(
        "bioio_czi.reader",  # deliberately misdirected — LIF still wins
        metadata=_AXIOSCAN_XML,
        ome_metadata=_ome_manufacturer_only(),
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=lif, scene_index=0)
    assert result["microscope"] == "STELLARIS 8"
    assert result["imaging_method"] == ["confocal"]


def test_vendor_extractor_failure_falls_through_to_ome() -> None:
    """A CZI vendor extractor that returns None must not block OME from firing."""
    r = _make_reader(
        "bioio_czi.reader",
        metadata="",  # empty → CZI extractor returns None
        ome_metadata=_ome_manufacturer_only(),
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {
        "microscope": "Zeiss",
        "imaging_method": ["widefield_fluorescence"],
    }
