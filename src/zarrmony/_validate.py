"""Optional OME-NGFF v0.5 validation via ``ome-zarr-models``.

After conversion finishes, ``convert()`` calls :func:`validate_store` on each
written store and records any findings under ``attrs.zarrmony.validation_warnings``
(see :class:`zarrmony.errors.ValidationWarning`).

``ome-zarr-models`` is an optional dependency installed via the
``zarrmony[validate]`` extra. When the extra is not installed, validation is
skipped silently — :func:`is_available` returns False and callers should
respect that rather than crashing.

Coverage caveats (inherited from ``ome-zarr-models``):

- The ``omero`` rendering block is not validated.
- For ``bioformats2raw.layout`` bundles, only the wrapper attrs are validated
  (the per-series ``0/``, ``1/``, ... subgroups are validated separately by
  walking them as ``Image`` groups).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import zarr

from zarrmony._storage import open_root_group
from zarrmony.errors import ValidationWarning

ResolvedLayout = Literal["per-scene", "bf2raw", "plate"]


def is_available() -> bool:
    """Return True if the optional ``ome-zarr-models`` package is importable."""
    try:
        import ome_zarr_models  # noqa: F401
    except ImportError:
        return False
    return True


def _validate_group_as(group: zarr.Group, kind: str) -> list[dict[str, Any]]:
    """Validate a Zarr group against an ``ome-zarr-models`` v0.5 class.

    ``kind`` is one of ``"image"``, ``"plate"``, ``"bf2raw"``. Returns an
    empty list on success, or a single-element list with a ``{field, error}``
    issue dict on failure.
    """
    from ome_zarr_models import v05

    classes = {
        "image": v05.Image,
        "plate": v05.HCS,
        "bf2raw": v05.BioFormats2Raw,
    }
    cls = classes[kind]
    try:
        cls.from_zarr(group)
    except Exception as e:  # noqa: BLE001 — pydantic raises subclassed ValidationError
        return [
            {
                "kind": kind,
                "path": str(group.store_path or ""),
                "error": f"{type(e).__name__}: {e}",
            }
        ]
    return []


def validate_store(
    store_path: str | Path, layout: ResolvedLayout
) -> list[dict[str, Any]]:
    """Validate a converted store and return any spec-violation findings.

    For ``per-scene`` ``store_path`` points at a single image store (the
    caller iterates over each scene's store). ``bf2raw`` validates the
    bundle root plus every numbered subgroup as a v0.5 ``Image``. ``plate``
    validates the root as a v0.5 ``HCS`` plate group.
    """
    if not is_available():
        return []
    root = open_root_group(store_path, mode="r")
    if layout == "per-scene":
        return _validate_group_as(root, "image")
    if layout == "plate":
        return _validate_group_as(root, "plate")
    if layout == "bf2raw":
        findings = _validate_group_as(root, "bf2raw")
        series = list(root.attrs.get("ome", {}).get("series", []))
        for s in series:
            try:
                sub = root[s]
            except KeyError:
                findings.append(
                    {
                        "kind": "image",
                        "path": str(s),
                        "error": f"KeyError: subgroup {s!r} listed in OME/series is missing",
                    }
                )
                continue
            findings.extend(_validate_group_as(sub, "image"))
        return findings
    raise ValueError(f"unknown layout: {layout!r}")


def run_validation(
    store_path: str | Path,
    layout: ResolvedLayout,
    validate: bool,
) -> list[dict[str, Any]]:
    """Validate ``store_path`` if requested and the validator is installed.

    Returns the list of findings (empty on success). Each finding is also
    surfaced as a :class:`~zarrmony.errors.ValidationWarning` so users see it on
    stderr; the caller threads the list into the audit record's
    ``validation_warnings``.

    Shared by ``convert`` and ``rechunk``: both produce stores of the same three
    layouts, and a store's spec compliance does not depend on which command
    wrote it.
    """
    if not validate:
        return []
    if not is_available():
        warnings.warn(
            "validate=True but the ome-zarr-models extra is not installed; "
            "skipping OME-NGFF validation of the written store. "
            "Install with `pip install zarrmony[validate]` to enable.",
            ValidationWarning,
            stacklevel=3,
        )
        return []
    findings = validate_store(store_path, layout)
    for f in findings:
        warnings.warn(
            f"OME-NGFF validation: {f['kind']} at {f.get('path', store_path)}: {f['error']}",
            ValidationWarning,
            stacklevel=3,
        )
    return findings
