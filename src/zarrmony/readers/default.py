"""Default reader: ``bioio.BioImage`` with bioio's plugin auto-discovery.

Used for any input that no extension-specific plugin (CZI, LIF, ND2) claims.
Exposed as ``default_plugin`` and registered in ``readers/__init__.py`` at
zarrmony import time. Its ``match()`` returns the lowest possible score (0) so
any extension-specific plugin outranks it.

``derive_bioio_distribution()`` recovers the actual underlying bioio
sub-package (``bioio-ome-tiff``, etc.) from an opened ``BioImage`` so the
audit record's ``distribution`` field stays specific even when the catch-all
plugin won.

``reader_kwargs`` reach the backend through here: ``_open_default`` forwards
``**kwargs`` to ``BioImage``, which forwards them to whichever bioio backend
won discovery. Two keys are coerced from their CLI string form first
(``dask_tiles``, ``tile_size``) because the backends that accept them are
third-party and will not coerce for us; everything else passes through
verbatim.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from zarrmony.errors import ReaderKwargError, UnsupportedFormatError
from zarrmony.readers.plugin import ReaderPlugin

_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})

# Named in the hint below because they are the formats users most often arrive
# with and search for; the real list is every Bio-Formats-only format.
_BIOFORMATS_HINT = (
    "zarrmony has no reader for {path}. If this is a vendor format covered by "
    "Bio-Formats (Olympus cellSens VSI, Zeiss ZVI, Hamamatsu NDPI, ...), try "
    '`pip install "zarrmony[bioformats]"` — see the README for the GPL-3.0 '
    "note on that extra."
)


def _match_default(_path: Path) -> int | None:
    return 0


def _coerce_bool(key: str, value: Any) -> Any:
    """``"true"``/``"0"``/``"yes"`` → ``bool``. Non-strings pass through."""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise ReaderKwargError(
        f"reader kwarg {key}={value!r} must be a boolean "
        f"(true/false, 1/0, yes/no, on/off)"
    )


def _coerce_int_pair(key: str, value: Any) -> Any:
    """``"1024,1024"`` → ``(1024, 1024)``. Non-strings pass through.

    A single int (``"1024"``) means a square, which is how every backend that
    takes a tile size is used in practice.
    """
    if not isinstance(value, str):
        return value
    parts = [part.strip() for part in value.split(",")]
    try:
        ints = [int(part) for part in parts]
    except ValueError as exc:
        raise ReaderKwargError(
            f"reader kwarg {key}={value!r} must be one or two comma-separated "
            f"ints (e.g. '1024,1024' or '1024')"
        ) from exc
    if len(ints) == 1:
        return (ints[0], ints[0])
    if len(ints) == 2:
        return (ints[0], ints[1])
    raise ReaderKwargError(
        f"reader kwarg {key}={value!r} must be one or two comma-separated "
        f"ints (e.g. '1024,1024' or '1024'); got {len(ints)} values"
    )


# Keys whose string form the CLI cannot leave to the reader, because the
# reader is a third-party bioio backend zarrmony does not control. Everything
# absent from this table keeps the documented string-passthrough contract.
_DEFAULT_KWARG_COERCIONS = {
    "dask_tiles": _coerce_bool,
    "tile_size": _coerce_int_pair,
}


def _coerce_default_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            _DEFAULT_KWARG_COERCIONS[key](key, value)
            if key in _DEFAULT_KWARG_COERCIONS
            else value
        )
        for key, value in kwargs.items()
    }


def _bioformats_installed() -> bool:
    """Is ``bioio-bioformats`` importable? Gates the ADR-0011 install hint."""
    try:
        return importlib.util.find_spec("bioio_bioformats") is not None
    except (ImportError, ValueError):  # pragma: no cover — broken install
        return False


def _open_default(path: Path, **kwargs: Any) -> Any:
    """Open ``path`` with ``BioImage``, forwarding ``kwargs`` to the backend.

    Unknown kwargs surface as the backend constructor's native ``TypeError``,
    matching the contract every other plugin honours (see ``plugin.py``).
    """
    try:
        return BioImage(str(path), **_coerce_default_kwargs(kwargs))
    except UnsupportedFileFormatError as exc:
        if _bioformats_installed():
            # Bio-Formats is already in the environment and still cannot read
            # this; naming the extra would be noise.
            raise
        raise UnsupportedFormatError(_BIOFORMATS_HINT.format(path=path)) from exc


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
