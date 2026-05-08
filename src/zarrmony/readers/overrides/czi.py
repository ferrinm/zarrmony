"""CZI reader override.

Uses the default ``pylibczirw`` backend, which auto-stitches mosaic scenes into
single images. The alternative ``aicspylibczi`` backend exposes the M (tile)
dimension and timelapse interval, but adds a 4th spatial axis (M) that is not
permitted by the OME-Zarr 0.5 axes spec. If you need per-tile positions or
timelapse intervals for analysis, they are preserved verbatim in
``OME/source/raw.czi.xml`` (the raw vendor metadata).

Exposed as ``czi_plugin`` (a :class:`ReaderPlugin`) and registered in
``readers/__init__.py`` at zarrmony import time.
"""

from pathlib import Path
from typing import Any

from bioio_czi import Reader

from zarrmony.readers.plugin import ReaderPlugin


def _match_czi(path: Path) -> int | None:
    return 100 if path.suffix.lower() == ".czi" else None


def _open_czi(path: Path) -> Any:
    return Reader(str(path))


czi_plugin = ReaderPlugin(
    name="bioio-czi",
    match=_match_czi,
    open=_open_czi,
    distribution="bioio-czi",
    source="builtin",
)


def open_czi_reader(path: str | Path) -> tuple[Any, str]:
    """Legacy 2-tuple factory kept for the deprecated ``_OVERRIDES`` path."""
    return Reader(str(path)), "bioio-czi"
