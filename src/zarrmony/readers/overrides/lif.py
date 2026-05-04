"""LIF reader override.

``BioImage(path)`` for a LIF file only exposes the first scene by default; the
format-specific ``bioio_lif.Reader`` exposes ``.scenes`` for full iteration.
"""

from pathlib import Path
from typing import Any

from bioio_lif import Reader


def open_lif_reader(path: str | Path) -> tuple[Any, str]:
    return Reader(str(path)), "bioio-lif"
