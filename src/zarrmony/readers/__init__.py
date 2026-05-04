"""Reader registry: dispatches an input path to the right bioio reader.

Default path is ``bioio.BioImage`` with plugin auto-discovery. Per-format
overrides (CZI, LIF) bypass that default for known-broken cases. Other formats
can register overrides at runtime via ``register_override``.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .default import open_default_reader
from .overrides.czi import open_czi_reader
from .overrides.lif import open_lif_reader

ReaderFactory = Callable[[str | Path], tuple[Any, str]]

_OVERRIDES: dict[str, ReaderFactory] = {
    ".czi": open_czi_reader,
    ".lif": open_lif_reader,
}


def get_reader(path: str | Path) -> tuple[Any, str]:
    """Open ``path`` with the registered override for its extension, or fall
    back to ``bioio.BioImage``. Returns ``(reader, plugin_name)``.
    """
    ext = Path(str(path)).suffix.lower()
    factory = _OVERRIDES.get(ext, open_default_reader)
    return factory(path)


def register_override(extension: str, factory: ReaderFactory) -> None:
    """Add or replace a reader override for ``extension``. Extension is
    normalized to lowercase with a leading dot.
    """
    if not extension.startswith("."):
        extension = "." + extension
    _OVERRIDES[extension.lower()] = factory


__all__ = ["get_reader", "register_override", "ReaderFactory"]
