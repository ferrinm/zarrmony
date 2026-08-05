"""Reader ``acquisition_audit`` hook wiring (issue #76).

The hook lets any reader inject fields into the per-scene acquisition audit
block regardless of source format — the escape hatch for readers whose
modality is known by construction (SmartSPIM = light-sheet, Blaze =
light-sheet) but neither the LIF scene-XML extractor nor OME's per-channel
``AcquisitionMode`` surface produces it.

These tests pin the seam directly (``_audit_acquisition_for_scene``) so no
real bioio reader is needed. Precedence: LIF (LIF scenes) → OME projection
→ reader hook, uniform ``setdefault`` — later layers fill only gaps left by
earlier ones. Fail-safe: any exception at any tier degrades to that tier
contributing nothing.
"""

from __future__ import annotations

from typing import Any

from ome_types import OME
from ome_types.model import Channel, Image, Pixels, PixelType

from zarrmony.api import _audit_acquisition_for_scene, _reader_acquisition_extras


def _ome_with_confocal_channel() -> OME:
    """One-image OME with a single confocal channel — populates imaging_method."""
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
                        Channel(
                            id="Channel:0:0",
                            acquisition_mode="LaserScanningConfocalMicroscopy",
                        )
                    ],
                ),
            )
        ],
    )


class _StubReader:
    """Minimal reader with a settable ``acquisition_audit`` + ``ome_metadata``."""

    def __init__(
        self,
        *,
        acquisition_audit: Any = None,
        ome_metadata: Any = None,
    ) -> None:
        # Only set the attribute when supplied so absence is observable via
        # getattr — a reader without the hook must not accidentally look
        # like a reader whose hook returned None.
        if acquisition_audit is not None:
            self.acquisition_audit = acquisition_audit
        if ome_metadata is not None:
            self.ome_metadata = ome_metadata


# --- _reader_acquisition_extras direct probes ------------------------------


def test_extras_absent_when_no_hook() -> None:
    """Reader with no ``acquisition_audit`` attribute → no extras."""
    assert _reader_acquisition_extras(_StubReader()) is None


def test_extras_reads_dict_hook() -> None:
    r = _StubReader(acquisition_audit={"imaging_method": ["light_sheet"]})
    assert _reader_acquisition_extras(r) == {"imaging_method": ["light_sheet"]}


def test_extras_reads_callable_hook() -> None:
    """The hook may also be a ``@property``-style callable that returns a dict."""
    r = _StubReader(acquisition_audit=lambda: {"imaging_method": ["light_sheet"]})
    assert _reader_acquisition_extras(r) == {"imaging_method": ["light_sheet"]}


def test_extras_raising_hook_degrades_silently() -> None:
    """A hook that raises must yield no extras — conversion never crashes."""

    def _boom() -> dict:
        raise RuntimeError("simulated reader hook failure")

    r = _StubReader(acquisition_audit=_boom)
    assert _reader_acquisition_extras(r) is None


def test_extras_non_dict_return_ignored() -> None:
    """Non-dict return (int, list, str) is treated as no extras — the shape
    contract is a dict."""
    for bad in (42, ["light_sheet"], "confocal"):
        r = _StubReader(acquisition_audit=bad)
        assert _reader_acquisition_extras(r) is None


def test_extras_empty_dict_ignored() -> None:
    r = _StubReader(acquisition_audit={})
    assert _reader_acquisition_extras(r) is None


# --- _audit_acquisition_for_scene: layered precedence ---------------------


def test_hook_alone_lands_when_lif_and_ome_empty() -> None:
    """SmartSPIM shape: no LIF, no OME AcquisitionMode → hook fills the gap."""
    r = _StubReader(acquisition_audit={"imaging_method": ["light_sheet"]})
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {"imaging_method": ["light_sheet"]}


def test_hook_does_not_override_ome_imaging_method() -> None:
    """OME says confocal; hook claims light_sheet. OME wins under setdefault."""
    r = _StubReader(
        acquisition_audit={"imaging_method": ["light_sheet"]},
        ome_metadata=_ome_with_confocal_channel(),
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {"imaging_method": ["confocal"]}


def test_hook_fills_key_ome_omitted() -> None:
    """OME populates microscope only; hook adds imaging_method that OME didn't."""

    scope_only = OME(
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
                ),
            )
        ],
    )
    r = _StubReader(
        acquisition_audit={
            "imaging_method": ["light_sheet"],
            "microscope": "OverrideMe",
        },
        ome_metadata=scope_only,
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {"imaging_method": ["light_sheet"], "microscope": "OverrideMe"}


def test_lif_wins_over_ome_and_hook_for_populated_keys() -> None:
    """LIF-extracted key holds; OME and hook only fill LIF's gaps."""
    lif = {"microscope": "STELLARIS 8", "imaging_method": ["confocal"]}
    r = _StubReader(
        acquisition_audit={
            "imaging_method": ["light_sheet"],
            "microscope_serial": "SN-42",  # neither LIF nor OME has this
        },
        ome_metadata=_ome_with_confocal_channel(),
    )
    result = _audit_acquisition_for_scene(r, lif_acquisition=lif, scene_index=0)
    assert result == {
        "microscope": "STELLARIS 8",
        "imaging_method": ["confocal"],
        "microscope_serial": "SN-42",
    }


def test_reader_without_hook_matches_prior_behavior() -> None:
    """Regression guard: a reader with no acquisition_audit attribute yields
    the same dict as before the hook existed (LIF or OME only)."""
    r = _StubReader(ome_metadata=_ome_with_confocal_channel())
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {"imaging_method": ["confocal"]}


def test_all_three_sources_empty_yields_none() -> None:
    """No LIF, no OME, no hook → block omitted entirely."""
    r = _StubReader()
    assert _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0) is None


def test_raising_ome_metadata_still_lets_hook_contribute() -> None:
    """Reader whose ome_metadata raises must not block the hook from filling in."""

    class _RaisingOme(_StubReader):
        @property
        def ome_metadata(self) -> Any:
            raise RuntimeError("bioio surface failure")

    r = _RaisingOme(acquisition_audit={"imaging_method": ["light_sheet"]})
    result = _audit_acquisition_for_scene(r, lif_acquisition=None, scene_index=0)
    assert result == {"imaging_method": ["light_sheet"]}
