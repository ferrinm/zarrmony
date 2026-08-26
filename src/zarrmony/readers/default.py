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

``BioImage`` reports *every* dispatch failure as ``UnsupportedFileFormatError``
recommending you install an extra, including failures that have nothing to do
with which backends are installed — the real cause goes to a log line and is
dropped. ``_open_default`` therefore runs a post-mortem before re-raising: it
checks whether the path is readable at all, and folds bioio's discarded
per-backend errors into the message. See ``_explain_unsupported``.
"""

from __future__ import annotations

import errno
import importlib.util
import logging
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from zarrmony.errors import (
    InputAccessError,
    ReaderKwargError,
    UnsupportedFormatError,
    ZarrmonyError,
)
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


# bioio logs each backend's failure here before discarding it. Module-scoped
# logger name, so it moves only if bioio renames the module.
_BIOIO_DISPATCH_LOGGER = "bioio.bio_image"

_MAX_REPORTED_ATTEMPTS = 5

# ``get_reader`` wraps every input in ``Path``, which collapses "s3://bucket"
# to "s3:/bucket" — so match one *or* two slashes. The 2+ character scheme
# keeps Windows drive letters ("C:/data") from reading as remote.
_REMOTE_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+://?")

_MACOS_PRIVACY_HINT = (
    "On macOS an EPERM here is usually the privacy layer rather than the file "
    "mode: grant Full Disk Access to the application hosting this process "
    "(Terminal, iTerm, Emacs, VS Code) under System Settings > Privacy & "
    "Security, then restart that application — restarting only the Python "
    "process does not pick up a new grant. Network and removable volumes "
    "under /Volumes need this even when `ls` on the file appears to work."
)


@contextmanager
def _captured_dispatch_attempts(path: Path) -> Iterator[list[str]]:
    """Collect the per-backend failures bioio logs and then throws away.

    ``BioImage.determine_plugin`` swallows every backend exception into a
    ``log.warning`` and raises a generic ``UnsupportedFileFormatError``, so
    those records are the only evidence of what actually went wrong. Records
    are filtered to those naming ``path`` — a concurrent open of a different
    file shares this logger and must not bleed into our diagnosis.
    """
    captured: list[str] = []
    needle = str(path)

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if needle in message:
                captured.append(message)

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger(_BIOIO_DISPATCH_LOGGER)
    logger.addHandler(handler)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)


def _describe_access_problem(path: Path) -> str | None:
    """Return why the OS refuses to read ``path``, or ``None`` if it reads fine.

    ``None`` also covers inputs this probe cannot speak to: directories
    (bioformats2raw bundles, Zarr stores) and remote URLs, which ``get_reader``
    has already wrapped in a ``Path`` by the time we see them.
    """
    if _REMOTE_URI.match(str(path)):
        return None
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except IsADirectoryError:
        return None
    except FileNotFoundError:
        return f"{path} does not exist"
    except PermissionError as exc:
        detail = f"{path} exists but cannot be opened for reading ({exc.strerror})."
        if sys.platform == "darwin" and exc.errno == errno.EPERM:
            detail = f"{detail} {_MACOS_PRIVACY_HINT}"
        return detail
    except OSError as exc:  # pragma: no cover — EIO, ELOOP, stale NFS handle, ...
        return f"{path} could not be opened for reading ({exc.strerror})"
    return None


def _parent_is_unlistable(path: Path) -> bool:
    """Can we read ``path`` but not list the directory holding it?

    Multi-file formats need both. Olympus VSI keeps its pyramid in a sibling
    ``_<name>_/stackNNNNN/`` tree, so a readable ``.vsi`` with an unlistable
    parent fails in a way that looks exactly like an unsupported format.
    """
    try:
        with os.scandir(path.parent) as entries:
            next(entries, None)
    except OSError:
        return True
    return False


def _access_error(path: Path) -> InputAccessError | None:
    """``InputAccessError`` if the input itself is unreachable, else ``None``."""
    access = _describe_access_problem(path)
    if access is None:
        return None
    return InputAccessError(f"zarrmony cannot read {path}. {access}")


def _explain_unsupported(path: Path, attempts: list[str]) -> ZarrmonyError:
    """Turn bioio's one-size-fits-all dispatch failure into a specific one."""
    inaccessible = _access_error(path)
    if inaccessible is not None:
        return inaccessible

    parts = []
    if _parent_is_unlistable(path):
        parts.append(
            f"zarrmony can read {path} but cannot list {path.parent}, which "
            f"multi-file formats need (Olympus VSI keeps its pyramid in a "
            f"sibling directory)."
        )
        if sys.platform == "darwin":
            parts.append(_MACOS_PRIVACY_HINT)
    elif _bioformats_installed():
        # Naming the extra would be noise — it is already installed. Say what
        # was tried instead, since bioio's own message will not.
        parts.append(
            f"zarrmony has no reader for {path}. bioio-bioformats is installed, "
            f"so this format is not one Bio-Formats covers, or the file is not "
            f"the format its extension claims."
        )
    else:
        parts.append(_BIOFORMATS_HINT.format(path=path))

    if attempts:
        reported = attempts[:_MAX_REPORTED_ATTEMPTS]
        dropped = len(attempts) - len(reported)
        lines = "\n  ".join(reported)
        suffix = f"\n  (+{dropped} more)" if dropped else ""
        parts.append(f"bioio tried:\n  {lines}{suffix}")

    return UnsupportedFormatError(" ".join(parts))


def _open_default(path: Path, **kwargs: Any) -> Any:
    """Open ``path`` with ``BioImage``, forwarding ``kwargs`` to the backend.

    Unknown kwargs surface as the backend constructor's native ``TypeError``,
    matching the contract every other plugin honours (see ``plugin.py``).
    """
    coerced = _coerce_default_kwargs(kwargs)
    with _captured_dispatch_attempts(path) as attempts:
        try:
            return BioImage(str(path), **coerced)
        except FileNotFoundError as exc:
            # bioio reports this as a stringified fsspec tuple
            # ("('file', 'local'):///path"), and the CLI takes INPUT as a raw
            # string so nothing has checked the path before now. Only claim it
            # when the probe agrees the *input* is the missing file — a backend
            # can also raise this for a sibling we know nothing about.
            inaccessible = _access_error(path)
            if inaccessible is None:
                raise
            raise inaccessible from exc
        except UnsupportedFileFormatError as exc:
            raise _explain_unsupported(path, attempts) from exc


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
