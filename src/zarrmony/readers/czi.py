"""CZI reader plugin.

Uses the default ``pylibczirw`` backend, which auto-stitches mosaic scenes into
single images. The alternative ``aicspylibczi`` backend exposes the M (tile)
dimension and timelapse interval, but adds a 4th spatial axis (M) that is not
permitted by the OME-Zarr 0.5 axes spec. If you need per-tile positions or
timelapse intervals for analysis, they are preserved verbatim in
``OME/source/raw.czi.xml`` (the raw vendor metadata).

``bioio_czi`` is imported lazily, inside ``_open_czi``, because it is not in
the default dependency set on Linux aarch64 (issue #142). It hard-depends on
``aicspylibczi`` — which zarrmony never calls, but which ``bioio_czi`` imports
at module scope — and ``aicspylibczi`` publishes no linux-aarch64 wheel, so
installing it there means a C++ source build. The ``czi`` extra opts back in.
A module-scope import would make ``import zarrmony`` fail outright on that
platform, so the cost of the absent backend is deferred to the point where a
user actually opens a ``.czi`` file.

Exposed as ``czi_plugin`` and registered in ``readers/__init__.py`` at zarrmony
import time. It registers on every platform, with its usual match score, so a
``.czi`` input never falls through to the ``bioio`` catch-all — that plugin's
hint names the ``bioformats`` extra, which is the wrong advice for CZI.
"""

import importlib.util
from pathlib import Path
from typing import Any

from zarrmony.errors import UnsupportedFormatError
from zarrmony.readers.plugin import ReaderPlugin


def _match_czi(path: Path) -> int | None:
    return 100 if path.suffix.lower() == ".czi" else None


def _czi_backend_installed() -> bool:
    """Is ``bioio-czi`` importable? Gates the ``czi`` extra install hint."""
    try:
        return importlib.util.find_spec("bioio_czi") is not None
    except (ImportError, ValueError):  # pragma: no cover — broken install
        return False


def _missing_backend_error(path: Path, exc: ImportError) -> UnsupportedFormatError:
    """Turn a failed backend import into advice the user can act on."""
    if _czi_backend_installed():
        # Naming the extra would be noise — it is already installed. Report
        # what the import actually said instead. A compiled backend that is
        # present but unloadable lands here: an ABI mismatch against the
        # runtime's libstdc++, a half-finished source build, an API shift.
        return UnsupportedFormatError(
            f"zarrmony cannot read {path}. The CZI backend is installed but "
            f"cannot be imported: {exc}."
        )
    return UnsupportedFormatError(
        f"zarrmony cannot read {path}. The CZI backend is not installed. "
        f'Install it with `pip install "zarrmony[czi]"`. On Linux arm64 that '
        f"backend has no prebuilt wheel, so the install compiles it from "
        f"source and needs cmake and a C++ compiler."
    )


def _open_czi(path: Path) -> Any:
    try:
        from bioio_czi import Reader
    except ImportError as exc:
        # ImportError, not just ModuleNotFoundError: a compiled backend can be
        # installed and still fail to load. The CLI reports a ZarrmonyError and
        # lets anything else through as a traceback, so an uncaught ImportError
        # here costs the user the message that names the fix.
        raise _missing_backend_error(path, exc) from exc
    return Reader(str(path))


czi_plugin = ReaderPlugin(
    name="bioio-czi",
    match=_match_czi,
    open=_open_czi,
    distribution="bioio-czi",
    source="builtin",
)
