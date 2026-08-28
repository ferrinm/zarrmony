"""What an input actually consists of, when it is more than the named path.

``convert()`` and ``inspect()`` are handed one path, but for several vendor
formats that path is an index and the pixels live beside it: a ``.vsi`` is a
few megabytes of metadata next to tens of gigabytes of ``.ets`` tiles in a
sibling ``_<name>_/`` directory. Sizing and hashing the named path alone then
describes the index and says nothing about the data that was converted, so
``input.size_bytes`` is wrong by four orders of magnitude and ``--checksum``
does not cover a single converted voxel (#116).

Bio-Formats knows the answer — ``IFormatReader.getUsedFiles()``, surfaced by
``bffile`` as ``BioFile.used_files()``. Nothing else in the reader stack does.
So this module asks whoever can answer and stays quiet when nobody can: a
reader that cannot report its file set leaves ``input.files`` absent, which is
different from reporting one file, and the audit says which case it is rather
than emitting a number that looks total.

The digest over a multi-file set is a *manifest* digest — SHA256 over
``<relpath>\\0<sha256>\\n`` lines, sorted, with paths taken relative to the
set's common ancestor. Relative paths keep it stable when the dataset is
moved; including the names means adding or renaming a sidecar changes it, not
just editing one.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePath
from typing import Any

from zarrmony._storage import format_bytes

#: How many member paths the audit lists before it stops and sets
#: ``listing_truncated``. A VSI names six; a TIFF series can name thousands,
#: and store attrs are not an inventory system. ``count`` is always exact.
MAX_LISTED_FILES = 64

_HASH_BLOCK = 2**20


def reader_used_files(reader: Any) -> list[str] | None:
    """Every file ``reader`` read, resolved and sorted, or ``None`` if it can't say.

    Two sources, in order: a ``used_files`` attribute on the plugin's reader
    (the opt-in hook for zarrmony's own plugins — a sequence, or a callable
    returning one), then ``BioFile.used_files()`` reached through bioio's
    ``BioImage.reader._bf``. Everything is guarded: a reader that raises, or
    that names files that no longer exist, is treated as unable to answer
    rather than allowed to break a conversion over an audit field.
    """
    for candidate in _used_files_candidates(reader):
        try:
            raw = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: BLE001 — heterogeneous reader/JVM errors
            continue
        if not raw:
            continue
        resolved = _resolve_existing(raw)
        if resolved:
            return resolved
    return None


def _used_files_candidates(reader: Any) -> Iterable[Any]:
    hook = getattr(reader, "used_files", None)
    if hook is not None:
        yield hook
    # bioio's BioImage delegates to a backend Reader, which holds the BioFile.
    for owner in (reader, getattr(reader, "reader", None)):
        bf = getattr(owner, "_bf", None)
        bf_hook = getattr(bf, "used_files", None)
        if bf_hook is not None:
            yield bf_hook


def _resolve_existing(paths: Iterable[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for entry in paths:
        p = Path(str(entry))
        try:
            if not p.is_file():
                continue
            seen.setdefault(str(p.resolve()), None)
        except OSError:
            continue
    return sorted(seen)


def summarize_used_files(
    used_files: Sequence[str],
    *,
    checksum: bool = False,
) -> dict[str, Any]:
    """The ``input.files`` block: exact count and total, a capped path listing.

    ``sha256`` is the manifest digest over the whole set and is only present
    for a set of more than one file — for a single file it would restate
    ``input.sha256``, which already covers it.
    """
    total = 0
    for path in used_files:
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    block: dict[str, Any] = {
        "count": len(used_files),
        "size_bytes": total,
        "size_human": format_bytes(total),
        "paths": [str(p) for p in used_files[:MAX_LISTED_FILES]],
        "listing_truncated": len(used_files) > MAX_LISTED_FILES,
    }
    if checksum and len(used_files) > 1:
        block["sha256"] = manifest_digest(used_files)
    return block


def manifest_digest(paths: Sequence[str]) -> str:
    """SHA256 over ``<relpath>\\0<file sha256>\\n`` for every path, sorted.

    Relative to the set's common ancestor, so relocating the dataset does not
    change the digest but renaming or adding a member does.
    """
    ordered = sorted(paths)
    root = os.path.commonpath(ordered) if len(ordered) > 1 else ""
    h = hashlib.sha256()
    for path in ordered:
        rel = PurePath(path).relative_to(root).as_posix() if root else Path(path).name
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(file_digest(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def file_digest(path: str | Path) -> str:
    """SHA256 of one file, memoized on (path, size, mtime).

    Per-scene conversions build one audit record per store and each one used to
    re-hash the input; against a 37 GB VSI file set that is minutes of pointless
    I/O per extra scene.
    """
    p = Path(path)
    st = p.stat()
    return _cached_digest(str(p), st.st_size, st.st_mtime_ns)


_digest_cache: dict[tuple[str, int, int], str] = {}


def _cached_digest(path: str, size: int, mtime_ns: int) -> str:
    key = (path, size, mtime_ns)
    cached = _digest_cache.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_BLOCK), b""):
            h.update(block)
    digest = h.hexdigest()
    _digest_cache[key] = digest
    return digest
