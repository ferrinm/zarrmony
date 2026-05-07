"""Filesystem-safe scene-name handling for per-scene output directories.

In per-scene mode, each scene becomes its own ``<sanitized_scene_name>.ome.zarr``
directory under the user-supplied output. Source scene names can contain any
characters the original vendor allowed (slashes, colons, whitespace, control
chars), so we sanitize them to a portable subset before using them as directory
names. When two scene names sanitize to the same string, we disambiguate by
suffixing the scene index: ``<sanitized>__<scene_index>``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Replace anything outside [A-Za-z0-9._-] (and the empty string) with underscore.
_UNSAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9._-]+")
_LEADING_DOTS_RE = re.compile(r"^\.+")
_FALLBACK_NAME = "scene"


def sanitize_scene_name(name: str) -> str:
    """Return a filesystem-safe version of ``name``.

    - Collapses runs of unsafe chars (slashes, colons, whitespace, etc.) into
      a single ``_``.
    - Strips leading dots so the result is never a hidden file.
    - Trims leading/trailing ``_``/``-``/``.`` from the result.
    - Falls back to ``"scene"`` if the result is empty.
    """
    if name is None:
        return _FALLBACK_NAME
    s = str(name).strip()
    s = _UNSAFE_CHAR_RE.sub("_", s)
    s = _LEADING_DOTS_RE.sub("", s)
    s = s.strip("_-.")
    return s or _FALLBACK_NAME


def resolve_scene_dirnames(scene_names: Sequence[str]) -> list[str]:
    """Sanitize ``scene_names`` and disambiguate sanitization collisions.

    Two scenes that sanitize to the same string both get the ``__<index>``
    suffix (not just the second one) so neither store wins by accident of
    iteration order. Scenes whose sanitized name is unique are returned
    unsuffixed.
    """
    sanitized = [sanitize_scene_name(n) for n in scene_names]
    counts: dict[str, int] = {}
    for s in sanitized:
        counts[s] = counts.get(s, 0) + 1
    return [f"{s}__{i}" if counts[s] > 1 else s for i, s in enumerate(sanitized)]
