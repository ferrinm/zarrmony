"""Default reader path: ``bioio.BioImage`` with plugin auto-discovery.

Used for any input format that does not have a registered override. Works for
OME-TIFF, ND2, and any other format with an installed bioio plugin.
"""

from pathlib import Path
from typing import Any

from bioio import BioImage


def open_default_reader(path: str | Path) -> tuple[Any, str]:
    """Open ``path`` via ``BioImage`` and return ``(reader, plugin_name)``.

    Plugin name is derived from the underlying reader's module (e.g.
    ``"bioio-ome-tiff"``) for the audit trail.
    """
    img = BioImage(str(path))
    plugin = "bioio"
    try:
        underlying = img.reader
        module = type(underlying).__module__
        plugin = module.split(".")[0].replace("_", "-")
    except (AttributeError, IndexError):
        pass
    return img, plugin
