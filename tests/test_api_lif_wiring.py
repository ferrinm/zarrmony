"""LIF api-wiring fail-safe regression.

The LIF channel-metadata branch must never crash a conversion and must decline
cleanly on: a non-LIF reader, a ``scene_root`` property that *raises* (the common
non-plate confocal case in bioio_lif), or a channel-count vs SizeC mismatch.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from zarrmony.api import _lif_scene_channels

FIX = Path(__file__).parent / "fixtures" / "lif_confocal_7ch.xml"


class _Xarr:
    def __init__(self, c):
        self.dims = ("C",) if c is not None else ()
        self.sizes = {"C": c} if c is not None else {}


class _Raising:
    """bioio_lif's scene_root raises ValueError for ordinary (non-plate) scenes."""

    xarray_dask_data = _Xarr(7)

    @property
    def scene_root(self):
        raise ValueError("scene is not a plate well")


class _NonLif:
    xarray_dask_data = _Xarr(7)


class _Good:
    xarray_dask_data = _Xarr(7)

    @property
    def scene_root(self):
        return ET.parse(FIX).getroot()


class _CountMismatch:
    xarray_dask_data = _Xarr(2)

    @property
    def scene_root(self):
        return ET.parse(FIX).getroot()


def test_raising_scene_root_falls_back():
    assert _lif_scene_channels(_Raising()) == (None, None)


def test_non_lif_falls_back():
    assert _lif_scene_channels(_NonLif()) == (None, None)


def test_count_mismatch_declines():
    assert _lif_scene_channels(_CountMismatch()) == (None, None)


def test_happy_path_wiring():
    extracted, omero = _lif_scene_channels(_Good())
    assert extracted is not None and len(extracted) == 7
    assert [c.label for c in omero][1] == "ALEXA 594 (590 nm)"
