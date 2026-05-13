"""Reader plugin system: Protocol, ReaderPlugin, registry, dispatch.

A zarrmony reader is anything that satisfies ``ReaderProtocol``. Plugins are
registered as ``ReaderPlugin`` instances — at zarrmony import time (built-ins),
via Python entry points (group ``zarrmony.readers``), or via runtime
``register_plugin()``. ``get_reader(path)`` walks all registered plugins,
calls each ``match()`` cheaply, and ``open()``s the highest-scoring one.

This module is the new plugin infrastructure (see ADR-0001). The legacy
extension-keyed dispatch in ``readers/__init__.py`` will be migrated onto this
registry as a follow-up; the two coexist for now.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "zarrmony.readers"

PluginSource = Literal["builtin", "entry_point", "runtime"]
LayoutHint = Literal["flat", "plate"]


@runtime_checkable
class ReaderProtocol(Protocol):
    """Structural interface ``convert()`` consumes from any reader.

    Required: ``scenes``, ``set_scene``, ``xarray_dask_data``,
    ``physical_pixel_sizes``. Soft-optional (accessed via getattr or
    try/except): ``channel_names``, ``ome_metadata``, ``metadata``, ``close``.
    ``layout_hint`` selects the writer (``"flat"`` → per-scene/bf2raw,
    ``"plate"`` → plate). Plate-shaped readers also populate
    ``plate_layout`` with the structured plate shape (see ADR-0004).
    """

    scenes: list[str]
    layout_hint: LayoutHint
    plate_layout: PlateLayout | None

    def set_scene(self, index: int) -> None: ...

    @property
    def xarray_dask_data(self) -> Any: ...

    @property
    def physical_pixel_sizes(self) -> Any: ...


@dataclass(frozen=True)
class ReaderPlugin:
    """A registered reader plugin.

    ``match`` must be cheap and side-effect-free; it returns a priority score
    (higher wins) or ``None`` for no match. ``open`` may be expensive and is
    only called on the winning plugin.
    """

    name: str
    match: Callable[[Path], int | None]
    open: Callable[[Path], ReaderProtocol]
    distribution: str | None = None
    source: PluginSource = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)


class NoMatchingPluginError(Exception):
    """Raised by ``get_reader`` when no registered plugin matches the input."""


_PLUGINS: dict[str, ReaderPlugin] = {}
_ENTRY_POINTS_LOADED = False


def register_plugin(plugin: ReaderPlugin, *, replace: bool = False) -> None:
    """Register ``plugin``. Rejects duplicate names unless ``replace=True``."""
    if plugin.name in _PLUGINS and not replace:
        raise ValueError(
            f"plugin name {plugin.name!r} is already registered; pass replace=True to override"
        )
    _PLUGINS[plugin.name] = plugin


def unregister_plugin(name: str) -> None:
    """Remove ``name`` from the registry. No-op if absent."""
    _PLUGINS.pop(name, None)


def list_plugins() -> list[ReaderPlugin]:
    """Return all registered plugins in registration order."""
    _ensure_entry_points_loaded()
    return list(_PLUGINS.values())


def _ensure_entry_points_loaded() -> None:
    """Walk ``zarrmony.readers`` entry points once and register what they yield.

    Built-ins must register *before* the first call (typically at zarrmony
    import time) so that equal-score ties resolve to built-ins via stable sort.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            plugin = ep.load()
        except (
            Exception
        ) as exc:  # noqa: BLE001 — third-party loader errors are heterogeneous
            warnings.warn(
                f"failed to load zarrmony.readers entry point {ep.name!r}: {exc}",
                stacklevel=2,
            )
            continue
        if not isinstance(plugin, ReaderPlugin):
            warnings.warn(
                f"entry point {ep.name!r} did not yield a ReaderPlugin "
                f"(got {type(plugin).__name__})",
                stacklevel=2,
            )
            continue
        try:
            register_plugin(plugin)
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)


def get_reader(path: str | Path) -> tuple[ReaderProtocol, ReaderPlugin, int]:
    """Open ``path`` with the highest-scoring registered plugin.

    Returns ``(reader, plugin, match_score)``. Matchers that raise are logged
    and treated as no-match. The winning ``open()`` raising is fatal.
    """
    _ensure_entry_points_loaded()
    p = Path(str(path))
    candidates: list[tuple[int, ReaderPlugin]] = []
    for plugin in _PLUGINS.values():
        try:
            score = plugin.match(p)
        except Exception as exc:  # noqa: BLE001 — see ADR-0001 trust model
            log.warning(
                "plugin %r raised in match(%s): %s; treating as no-match",
                plugin.name,
                p,
                exc,
            )
            continue
        if score is not None:
            candidates.append((score, plugin))
    if not candidates:
        raise NoMatchingPluginError(f"no registered plugin matched {path!s}")
    # Stable sort preserves registration order on ties → built-ins win equal
    # scores against entry-point plugins (built-ins register first).
    candidates.sort(key=lambda c: -c[0])
    score, plugin = candidates[0]
    reader = plugin.open(p)
    return reader, plugin, score


__all__ = [
    "ENTRY_POINT_GROUP",
    "Acquisition",
    "LayoutHint",
    "NoMatchingPluginError",
    "PlateField",
    "PlateLayout",
    "PluginSource",
    "ReaderPlugin",
    "ReaderProtocol",
    "get_reader",
    "list_plugins",
    "register_plugin",
    "unregister_plugin",
]
