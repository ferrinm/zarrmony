"""Reader registry: register built-in :class:`ReaderPlugin` instances and keep
the legacy ``_OVERRIDES`` API alive for back-compat.

Active dispatch lives in ``readers/plugin.py``: ``convert()`` and ``inspect()``
call ``plugin.get_reader(path)`` which walks the registered plugins (default
+ CZI/LIF/ND2 below + any external entry-point readers). The legacy
``_OVERRIDES`` dict and ``register_override`` API still exist for users who
wired into them directly, but zarrmony's own dispatch no longer consults them;
they will be removed in a follow-up slice.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .default import default_plugin, open_default_reader
from .overrides.czi import czi_plugin, open_czi_reader
from .overrides.lif import lif_plugin, open_lif_reader
from .overrides.nd2 import nd2_plugin, open_nd2_reader
from .plugin import register_plugin

ReaderFactory = Callable[[str | Path], tuple[Any, str]]

# Register built-in plugins at import time. Order matters: built-ins go in
# before any entry-point plugins are walked, so equal-priority ties resolve
# to built-ins (per ADR-0001).
register_plugin(default_plugin)
register_plugin(czi_plugin)
register_plugin(lif_plugin)
register_plugin(nd2_plugin)


# ---- Legacy back-compat API (deprecated; removed in a follow-up slice) ----

_OVERRIDES: dict[str, ReaderFactory] = {
    ".czi": open_czi_reader,
    ".lif": open_lif_reader,
    ".nd2": open_nd2_reader,
}


def get_reader(path: str | Path) -> tuple[Any, str]:
    """Legacy 2-tuple dispatch. New code should call
    :func:`zarrmony.readers.plugin.get_reader` instead.
    """
    ext = Path(str(path)).suffix.lower()
    factory = _OVERRIDES.get(ext, open_default_reader)
    return factory(path)


def register_override(extension: str, factory: ReaderFactory) -> None:
    """Add or replace a reader override for ``extension``. Legacy API; new
    code should expose a :class:`ReaderPlugin` and call ``register_plugin``.
    """
    if not extension.startswith("."):
        extension = "." + extension
    _OVERRIDES[extension.lower()] = factory


__all__ = ["get_reader", "register_override", "ReaderFactory"]
