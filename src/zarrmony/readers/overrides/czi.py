"""CZI reader override.

Uses the default ``pylibczirw`` backend, which auto-stitches mosaic scenes into
single images. The alternative ``aicspylibczi`` backend exposes the M (tile)
dimension and timelapse interval, but adds a 4th spatial axis (M) that is not
permitted by the OME-Zarr 0.5 axes spec. If you need per-tile positions or
timelapse intervals for analysis, they are preserved verbatim in
``OME/source/raw.czi.xml`` (the raw vendor metadata).
"""

from pathlib import Path
from typing import Any

from bioio_czi import Reader


def open_czi_reader(path: str | Path) -> tuple[Any, str]:
    return Reader(str(path)), "bioio-czi"
