"""Structured plate-layout dataclasses for plate-shaped readers.

Per ADR-0004, plate-shaped readers populate ``ReaderProtocol.plate_layout``
with a :class:`PlateLayout` describing rows, columns, acquisitions, and the
per-FOV mapping back to ``reader.scenes``. Zarrmony's plate writer consumes
this structure directly — string parsing of scene names was rejected.

The dataclasses are deliberately minimal: they carry exactly what
OME-NGFF 0.5 ``plate``/``well`` metadata and the audit record need.
``acquisition_id`` is reserved for the v2 multi-acquisition follow-up;
the v1 writer asserts ``len(acquisitions) <= 1`` and ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Acquisition:
    """One plate acquisition (a single imaging pass).

    ``id`` is the integer used in the OME-NGFF ``plate.acquisitions`` list and
    referenced by :class:`PlateField.acquisition_id`. ``name`` and
    ``maximumfieldcount`` are optional spec fields surfaced to callers.
    """

    id: int
    name: str | None = None
    maximumfieldcount: int | None = None


@dataclass(frozen=True)
class PlateField:
    """One field of view within a plate well.

    ``scene_index`` points back into ``reader.scenes`` so the writer can call
    ``reader.set_scene(scene_index)`` and reuse :func:`writers.scene.write_scene`
    per FOV. ``row`` and ``column`` are the plate coordinates as listed in
    :attr:`PlateLayout.rows` / :attr:`PlateLayout.columns`. ``field_name`` is
    the vendor's human-readable label (preserved in audit + multiscales name,
    NOT the on-disk path — see ADR-0004 locked decision #4).
    """

    scene_index: int
    row: str
    column: str
    field_name: str | None = None
    acquisition_id: int | None = None


@dataclass(frozen=True)
class PlateLayout:
    """Structured plate shape exposed by plate-shaped readers.

    ``rows`` and ``columns`` MUST list every physical row/column of the plate
    even when only some are imaged — sparse-plate semantics are the reader's
    responsibility (see ADR-0004 §Consequences).

    ``plate_id`` is the source-file's plate identifier (CZI ``Plate`` element,
    Opera Phenix plate barcode, etc.). Populated by the reader when the source
    encodes it; ``None`` otherwise. Surfaces in the audit's ``plate`` block as
    ``plate_id`` (per ADR-0008 / #66) and in ``inspect().plate_layout``; the
    on-disk OME-NGFF ``attrs.ome.plate`` block does NOT carry it because the
    NGFF 0.5 plate schema does not define such a key.
    """

    name: str
    rows: list[str]
    columns: list[str]
    acquisitions: list[Acquisition] = field(default_factory=list)
    fields: list[PlateField] = field(default_factory=list)
    plate_id: str | None = None


__all__ = ["Acquisition", "PlateField", "PlateLayout"]
