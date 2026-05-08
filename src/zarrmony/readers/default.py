"""Default reader: ``bioio.BioImage`` with bioio's plugin auto-discovery.

Used for any input that no extension-specific override (CZI, LIF, ND2) claims.
Exposed as ``default_plugin`` (a :class:`ReaderPlugin`) and registered in
``readers/__init__.py`` at zarrmony import time. Its ``match()`` returns the
lowest possible score (0) so any extension-specific plugin outranks it.

``derive_bioio_distribution()`` recovers the actual underlying bioio
sub-package (``bioio-ome-tiff``, etc.) from an opened ``BioImage`` so the
audit record's ``distribution`` field stays specific even when the catch-all
plugin won.
"""

from pathlib import Path
from typing import Any

from bioio import BioImage

from zarrmony.readers.plugin import ReaderPlugin


def _match_default(_path: Path) -> int | None:
    return 0


def _open_default(path: Path) -> Any:
    return BioImage(str(path))


default_plugin = ReaderPlugin(
    name="bioio",
    match=_match_default,
    open=_open_default,
    distribution=None,
    source="builtin",
)


def derive_bioio_distribution(reader: Any) -> str | None:
    """Recover the bioio sub-package name (e.g. ``bioio-ome-tiff``) from an
    opened ``BioImage``. Returns ``None`` if introspection fails.
    """
    try:
        underlying = reader.reader
        module = type(underlying).__module__
        return module.split(".")[0].replace("_", "-")
    except (AttributeError, IndexError):
        return None


def open_default_reader(path: str | Path) -> tuple[Any, str]:
    """Legacy 2-tuple factory kept for the deprecated ``_OVERRIDES`` path."""
    img = BioImage(str(path))
    return img, derive_bioio_distribution(img) or "bioio"
