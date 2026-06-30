"""Locate the current scene's settings XML on a (possibly LIF) bioio reader.

Both the channel-identity extractor (``api._lif_scene_channels``) and the tile-
layout extractor (``readers.lif._MosaicAwareLifReader.mosaic_summary``) need
the same per-scene XML blob. Two reader surfaces can produce it:

1. ``reader.scene_root`` — bioio-lif's *plate* row/column locator. It returns
   the well's ``<Element>`` node directly for plate scenes but RAISES
   ``ValueError`` ("Row or column value is missing…") for ordinary non-plate
   confocal scenes, and is absent entirely on non-LIF readers.
2. ``reader.metadata`` ``.//Image[current_scene_index]`` — the document-order
   locator from bioio-lif PR #52. Works for the confocal case where (1) raises.

Every reader-surface access is guarded so a partial reader cannot raise out of
here: a raising/absent ``scene_root``, missing/``None`` metadata, a metadata
object without ``findall``, an empty ``<Image>`` list, and an out-of-range
``current_scene_index`` all yield ``None``. Metadata never crashes a conversion.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


def _scene_root_fast(reader: Any) -> ET.Element | None:
    """``reader.scene_root`` when it works, else ``None`` (never raises)."""
    try:
        return getattr(reader, "scene_root", None)
    except Exception:  # noqa: BLE001 — a raise here just means "not this path"
        return None


def _scene_image(reader: Any) -> ET.Element | None:
    """The current scene's ``<Image>`` from ``reader.metadata``, or ``None``."""
    metadata = getattr(reader, "metadata", None)
    if metadata is None or not hasattr(metadata, "findall"):
        return None
    images = metadata.findall(".//Image")
    if not images:
        return None
    index = getattr(reader, "current_scene_index", None)
    if not isinstance(index, int) or not (0 <= index < len(images)):
        return None
    return images[index]


def find_scene_xml(reader: Any) -> str | None:
    """Serialized per-scene settings XML for ``reader``'s current scene, or ``None``.

    Tries ``scene_root`` (plate fast path) then the ``.//Image`` fallback (the
    confocal case). Test ``is None`` explicitly rather than truthiness — an
    ``Element`` with no children is falsy, so ``or`` would wrongly skip a valid
    childless ``scene_root``.
    """
    element = _scene_root_fast(reader)
    if element is None:
        element = _scene_image(reader)
    if element is None:
        return None
    try:
        return ET.tostring(element, encoding="unicode")
    except Exception:  # noqa: BLE001 — never break a conversion over metadata
        return None
