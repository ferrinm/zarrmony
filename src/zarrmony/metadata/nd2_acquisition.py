"""Fill the ND2 microscope-model gap that bioio-nd2's OME projection leaves.

bioio-nd2 doesn't populate ``<Instrument><Microscope/>`` in its OME projection
at all, so ``per_scene[i].acquisition.microscope`` is missing entirely for
ND2 files. The Nikon SDK does record the microscope in the file — via
``nd2.ND2File(...).text_info().get("capturing")``, a free-text field the
NIS-Elements acquisition software writes with the microscope stand and
software version (typically ``"Nikon Instruments Inc.\nTi2\n..."``). This
extractor reopens the file via the ``nd2`` package to read it.

Layered between the LIF extractor and the OME projection in
:func:`zarrmony.api._audit_acquisition_for_scene` — sits above the OME
projection so ``"Nikon Ti2"`` wins the ``setdefault`` race against OME's
(currently empty) ``microscope`` slot.

Fail-closed throughout: a missing ``nd2`` package, an unreadable path, or a
``text_info`` chunk that lacks the ``capturing`` field all yield ``None``.
"""

from __future__ import annotations

from typing import Any

# Free-text ``capturing`` returned by NIS-Elements typically bundles software,
# vendor, and hardware lines. We drop anything matching these tokens so the
# remaining line is the microscope stand.
_CAPTURING_DROP_TOKENS: tuple[str, ...] = (
    "nis-elements",
    "nikon instruments",
    "nikon corporation",
)


def _clean_capturing_lines(capturing: str) -> list[str]:
    """Split a free-text ``capturing`` string into candidate microscope lines.

    NIS-Elements writes multi-line free text — software version, vendor,
    microscope stand, sometimes camera info — separated by newlines and/or
    semicolons. Split on either, strip, drop empties and known-non-microscope
    lines (``NIS-Elements``, ``Nikon Instruments Inc.``).
    """
    lines: list[str] = []
    for chunk in capturing.replace(";", "\n").splitlines():
        stripped = chunk.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(tok in lowered for tok in _CAPTURING_DROP_TOKENS):
            continue
        lines.append(stripped)
    return lines


def _microscope_from_text_info(text_info: dict | None) -> str | None:
    """Pick a ``microscope`` string from an ``nd2.ND2File.text_info()`` dict.

    Prefers the first non-vendor line of ``capturing`` (the microscope stand,
    e.g. ``"Ti2"``, ``"Ti-E"``, ``"A1 R"``). Prepends ``"Nikon "`` when the
    line doesn't already carry the brand — bioio-nd2 is a Nikon-only reader
    so the brand is known even when the file doesn't spell it out.

    Returns ``None`` when ``text_info`` is missing, ``capturing`` is empty,
    or every line was dropped as vendor/software noise.
    """
    if not isinstance(text_info, dict):
        return None
    capturing = text_info.get("capturing")
    if not isinstance(capturing, str) or not capturing.strip():
        return None
    lines = _clean_capturing_lines(capturing)
    if not lines:
        return None
    model = lines[0]
    if model.lower().startswith("nikon"):
        return model
    return f"Nikon {model}"


def _open_nd2(path: str) -> Any | None:
    """Open ``path`` via the ``nd2`` package, or ``None`` if unavailable.

    Import happens on-demand — an environment without ``nd2`` installed (rare
    for zarrmony users but possible if the ND2 reader plugin was disabled)
    still won't crash the audit path.
    """
    try:
        import nd2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — never crash audit over an import failure
        return None
    try:
        return nd2.ND2File(path)
    except Exception:  # noqa: BLE001 — never crash audit over vendor SDK errors
        return None


def extract_nd2_acquisition(reader: Any) -> dict | None:
    """Extract ``acquisition`` fields the ND2 vendor SDK carries.

    Currently populates only ``microscope`` — the OME projection already
    surfaces ``date`` and ``imaging_method`` for ND2 via bioio-nd2.

    Requires the reader to expose its underlying file path (bioio-nd2 stores
    it as ``reader._path``). A reader without a discoverable path or a
    missing ``nd2`` package both yield ``None``.
    """
    path = getattr(reader, "_path", None)
    if not isinstance(path, str) or not path:
        return None
    handle = _open_nd2(path)
    if handle is None:
        return None
    try:
        text_info = handle.text_info()
    except Exception:  # noqa: BLE001 — never crash audit
        text_info = None
    finally:
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — never crash audit on close failure
                pass
    try:
        model = _microscope_from_text_info(text_info)
        if model is None:
            return None
        return {"microscope": model}
    except Exception:  # noqa: BLE001 — never crash audit
        return None


__all__ = ["extract_nd2_acquisition"]
