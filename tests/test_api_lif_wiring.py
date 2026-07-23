"""LIF api-wiring regression — the ``_lif_scene_channels`` scene→identity seam.

Channel identity is located two-tier (see ``api._lif_scene_root_fast`` /
``_lif_scene_image``): bioio-lif's ``scene_root`` plate-well locator is tried
first, then — because ``scene_root`` *raises* for ordinary non-plate confocal
scenes — the document-order ``.//Image[current_scene_index]`` locator (bioio-lif
PR #52) is the fallback that makes confocal scenes work. These tests pin both that
the confocal path yields real fluorophore identity AND that the seam never crashes
a conversion: it declines cleanly (``(None, None)``) for a non-LIF reader, a reader
exposing neither a usable ``scene_root`` nor ``metadata``, a count-vs-SizeC
mismatch, or any unexpected reader-surface error.

Driven by faithful reader doubles over the real captured ``lif_confocal_7ch.xml``
fixture — no ``bioio_lif`` import and no large data file, so it runs in CI.
"""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from zarrmony.api import _lif_scene_channels

FIX = Path(__file__).parent / "fixtures" / "lif_confocal_7ch.xml"

# Every double below mirrors bioio's ``Reader.dtype`` surface (uint16 is the
# LIF acquisition default) — ``_lif_scene_channels`` reads it to size the
# OMERO display window per array-dtype range (issue #50).
_DTYPE = np.dtype(np.uint16)


class _Xarr:
    def __init__(self, c):
        self.dims = ("C",) if c is not None else ()
        self.sizes = {"C": c} if c is not None else {}


# --- plate / degenerate / non-LIF doubles (the scene_root fast path) -------


class _Raising:
    """scene_root raises AND no ``metadata`` is exposed — a degenerate reader that
    can't be located either way, so it must decline (not crash)."""

    xarray_dask_data = _Xarr(7)
    dtype = _DTYPE

    @property
    def scene_root(self):
        raise ValueError("scene is not a plate well")


class _NonLif:
    """Neither ``scene_root`` nor ``metadata`` — a non-LIF reader."""

    xarray_dask_data = _Xarr(7)
    dtype = _DTYPE


class _Good:
    """Plate-shaped reader: ``scene_root`` returns the scene ``<Element>`` directly."""

    xarray_dask_data = _Xarr(7)
    dtype = _DTYPE

    @property
    def scene_root(self):
        return ET.parse(FIX).getroot()


class _CountMismatch:
    xarray_dask_data = _Xarr(2)
    dtype = _DTYPE

    @property
    def scene_root(self):
        return ET.parse(FIX).getroot()


# --- confocal double (scene_root raises; scenes live in ``metadata``) ------


def _two_scene_metadata() -> ET.Element:
    """A LIF metadata tree with two confocal scenes (the second dye-renamed).

    Scene 0 is the real captured 7-channel ``<Element>``; scene 1 is the same with
    its ``DyeName`` values prefixed so correct *positional* indexing is observable.
    """
    scene0 = copy.deepcopy(ET.parse(FIX).getroot())  # <Element ..><Data><Image>..
    scene1 = copy.deepcopy(scene0)
    for cp in scene1.iter("ChannelProperty"):
        key, val = cp.find("Key"), cp.find("Value")
        if (
            key is not None
            and (key.text or "").strip() == "DyeName"
            and val is not None
            and val.text
        ):
            val.text = "B_" + val.text
    root = ET.Element("LMSDataContainerHeader")
    children = ET.SubElement(ET.SubElement(root, "Element"), "Children")
    children.append(scene0)
    children.append(scene1)
    return root


class _ConfocalReader:
    """Confocal LIF reader: ``scene_root`` raises; scenes live in ``metadata`` as
    the document-order ``<Image>`` elements (the common confocal case)."""

    def __init__(self, c=7, have_metadata=True):
        self._md = _two_scene_metadata() if have_metadata else None
        self._c = c
        self.current_scene_index = 0
        self.dtype = _DTYPE

    @property
    def metadata(self):
        return self._md

    @property
    def scene_root(self):
        raise ValueError(
            "Row or column value is missing; cannot locate the scene node."
        )

    @property
    def xarray_dask_data(self):
        return _Xarr(self._c)


# --- fail-safe / fall-back paths -------------------------------------------


def test_raising_scene_root_without_metadata_declines():
    assert _lif_scene_channels(_Raising()) == (None, None, None)


def test_non_lif_falls_back():
    assert _lif_scene_channels(_NonLif()) == (None, None, None)


def test_count_mismatch_declines_channels_but_still_surfaces_objective():
    # Objective extraction is decoupled from the SizeC check — a garbled
    # channel list must not drop the (still-valid) objective dict.
    extracted, omero, objective = _lif_scene_channels(_CountMismatch())
    assert extracted is None and omero is None
    assert objective is not None and objective.get("nominal_magnification") == 20


# --- plate fast path -------------------------------------------------------


def test_happy_path_wiring():
    extracted, omero, objective = _lif_scene_channels(_Good())
    assert extracted is not None and len(extracted) == 7
    assert [c.label for c in omero][1] == "ALEXA 594 (590 nm)"
    # Objective co-extraction: same scene XML, same call — the fixture's
    # ATLConfocalSettingDefinition carries a 20x/0.75 DRY objective.
    assert objective is not None
    assert objective["nominal_magnification"] == 20
    assert objective["numerical_aperture"] == 0.75
    assert objective["immersion"] == "Air"  # LIF "DRY" -> OME Air


# --- confocal (scene_root raises -> metadata .//Image locator) -------------


def test_confocal_scene_yields_fluorophore_channels():
    r = _ConfocalReader()
    r.current_scene_index = 0
    extracted, omero, _ = _lif_scene_channels(r)
    assert omero is not None, "confocal channel metadata is inert"
    assert [c.label for c in omero][1] == "ALEXA 594 (590 nm)"
    assert extracted[1]["excitation_nm"] == 590


def test_confocal_scene_is_located_positionally():
    r = _ConfocalReader()
    r.current_scene_index = 1
    _, omero, _ = _lif_scene_channels(r)
    assert omero is not None
    assert [c.label for c in omero][1] == "B_Leica/ALEXA 594 (590 nm)"


def test_confocal_count_mismatch_declines():
    r = _ConfocalReader(c=2)  # XML has 7 channels, array claims 2
    extracted, omero, objective = _lif_scene_channels(r)
    assert (extracted, omero) == (None, None)
    # Objective is still surfaced — see the plate-path test for the reason.
    assert objective is not None


def test_confocal_no_metadata_falls_back():
    r = _ConfocalReader(have_metadata=False)
    assert _lif_scene_channels(r) == (None, None, None)
