"""ND2 reader override.

Pins the format-specific ``bioio_nd2.Reader`` so ND2 input fails fast at import
time if the plugin is missing, rather than silently falling through bioio's
plugin discovery and surfacing as a generic "no reader found" error at convert
time. The plugin name is also recorded explicitly for the audit trail.
"""

from pathlib import Path
from typing import Any

from bioio_nd2 import Reader


def open_nd2_reader(path: str | Path) -> tuple[Any, str]:
    return Reader(str(path)), "bioio-nd2"
