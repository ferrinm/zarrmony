"""ND2 reader override.

Pins the format-specific ``bioio_nd2.Reader`` so ND2 input fails fast at import
time if the plugin is missing, rather than silently falling through bioio's
plugin discovery and surfacing as a generic "no reader found" error at convert
time. The plugin name is also recorded explicitly for the audit trail.

Exposed as ``nd2_plugin`` (a :class:`ReaderPlugin`) and registered in
``readers/__init__.py`` at zarrmony import time.
"""

from pathlib import Path
from typing import Any

from bioio_nd2 import Reader

from zarrmony.readers.plugin import ReaderPlugin


def _match_nd2(path: Path) -> int | None:
    return 100 if path.suffix.lower() == ".nd2" else None


def _open_nd2(path: Path) -> Any:
    return Reader(str(path))


nd2_plugin = ReaderPlugin(
    name="bioio-nd2",
    match=_match_nd2,
    open=_open_nd2,
    distribution="bioio-nd2",
    source="builtin",
)


def open_nd2_reader(path: str | Path) -> tuple[Any, str]:
    """Legacy 2-tuple factory kept for the deprecated ``_OVERRIDES`` path."""
    return Reader(str(path)), "bioio-nd2"
