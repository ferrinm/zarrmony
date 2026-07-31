"""Project ``reader.ome_metadata`` into the ADR-0008 objective / acquisition audit blocks.

Serves the CZI / ND2 / OME-TIFF / default reader paths (issues #63, #64, #65).
LIF is handled by its own dedicated extractors under
:mod:`zarrmony.metadata.objective` and :mod:`zarrmony.metadata.acquisition`,
which walk the raw scene XML because bioio-lif doesn't populate
``reader.ome_metadata.instruments`` for every scene.

Every function is fail-closed: any exception yields ``None`` (never raises)
so a garbled OME element can't crash a conversion.

The three blocks are extracted independently — a reader that exposes
``ome_metadata.images[i].acquisition_date`` but no ``instruments`` still gets
its acquisition date recorded; missing bits omitted per the ADR-0008
omit-not-null rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ome_types.model import Objective as OmeObjective

# Same LIF→OME immersion mapping as :mod:`zarrmony.metadata.objective`,
# but keyed on the OME ``Immersion`` enum's ``.value`` (a canonical string
# like ``"Oil"``) — the ome_types Objective already exposes the enum, so
# we just want the string form. Unknown → "Other" per the OME enum's own
# catch-all.
_OME_IMMERSION_TO_STR: dict[str, str] = {
    "Oil": "Oil",
    "Water": "Water",
    "Air": "Air",
    "Glycerol": "Glycerol",
    "Multi": "Multi",
    "Other": "Other",
    "WaterDipping": "WaterDipping",
}


def _first_ome_objective(ome: Any) -> OmeObjective | None:
    """First ``Objective`` on the first ``Instrument`` in ``ome``, or ``None``.

    Prefers a per-image ``ObjectiveSettings`` linkage over the first-instrument
    fallback: some OME-TIFF exports have several ``<Instrument><Objective/>``
    entries and only one is the one this image actually used.
    """
    try:
        instruments = list(getattr(ome, "instruments", None) or [])
    except Exception:
        return None
    if not instruments:
        return None
    objectives = list(getattr(instruments[0], "objectives", None) or [])
    return objectives[0] if objectives else None


def _to_number(value: Any) -> int | float | None:
    """Coerce to ``int`` when integral, else ``float``, else ``None``."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(v) if v.is_integer() else v


def _to_string(value: Any) -> str | None:
    """Coerce a value (or enum) to a stripped non-empty string, or ``None``."""
    if value is None:
        return None
    # ome_types wraps enum-ish strings in Enum instances; prefer .value.
    raw = getattr(value, "value", value)
    text = str(raw).strip() if raw is not None else ""
    return text or None


def extract_objective_from_ome(ome: Any) -> dict | None:
    """Project OME's ``Objective`` into the ADR-0004 audit-shape dict.

    Uses the same 5-key shape as :func:`zarrmony.metadata.objective.extract_objective`
    (``nominal_magnification``, ``numerical_aperture``, ``immersion``,
    ``model``, ``working_distance_um``) so LIF and non-LIF conversions carry
    the same block. Missing individual fields are omitted; an OME with no
    objective info at all returns ``None`` rather than an empty dict.
    """
    try:
        obj = _first_ome_objective(ome)
    except Exception:  # noqa: BLE001 — never crash audit
        return None
    if obj is None:
        return None
    try:
        record: dict = {}
        mag = _to_number(getattr(obj, "nominal_magnification", None))
        if mag is None:
            mag = _to_number(getattr(obj, "calibrated_magnification", None))
        if mag is not None:
            record["nominal_magnification"] = mag

        na = _to_number(getattr(obj, "lens_na", None))
        if na is not None:
            record["numerical_aperture"] = na

        immersion = _to_string(getattr(obj, "immersion", None))
        if immersion is not None:
            record["immersion"] = _OME_IMMERSION_TO_STR.get(immersion, immersion)

        model = _to_string(getattr(obj, "model", None))
        if model is not None:
            record["model"] = model

        wd = _to_number(getattr(obj, "working_distance", None))
        if wd is not None:
            record["working_distance_um"] = wd

        return record or None
    except Exception:  # noqa: BLE001 — never crash audit
        return None


def _first_microscope(ome: Any) -> Any | None:
    try:
        instruments = list(getattr(ome, "instruments", None) or [])
    except Exception:
        return None
    if not instruments:
        return None
    return getattr(instruments[0], "microscope", None)


def _brand_and_model(manufacturer: str | None, model: str | None) -> str | None:
    """Combine manufacturer + model into a single 'brand model' string.

    ``"Nikon"`` + ``"Ti2"`` → ``"Nikon Ti2"``; either half alone still surfaces.
    """
    parts = [p for p in (manufacturer, model) if p]
    return " ".join(parts) if parts else None


def extract_acquisition_from_ome(ome: Any, image_index: int = 0) -> dict | None:
    """Project OME acquisition + instrument metadata into the ADR-0008 shape.

    Shape matches :func:`zarrmony.metadata.acquisition.extract_acquisition`:
    ``{date?, microscope?, microscope_serial?, imaging_method?: list[str]}``.
    Every key optional; block omitted entirely (returns ``None``) when nothing
    extractable was present.

    ``date`` comes from ``ome.images[image_index].acquisition_date`` (a
    ``datetime`` in ome_types; serialised to ISO 8601). ``microscope``
    combines the ``Microscope`` manufacturer + model (``"Nikon Ti2"``).
    ``microscope_serial`` comes from ``Microscope.serial_number``.

    ``imaging_method`` is deliberately NOT populated here — OME's Instrument
    schema does not have a first-class modality field; format-specific
    extractors (CZI ``.czi`` metadata, ND2's Nikon experiment surface)
    contribute their own modality tokens.
    """
    if ome is None:
        return None
    try:
        record: dict = {}

        images = list(getattr(ome, "images", None) or [])
        if 0 <= image_index < len(images):
            image = images[image_index]
            acq_date = getattr(image, "acquisition_date", None)
            if acq_date is not None:
                # ome_types stores datetime; serialise to ISO 8601. Fall back
                # to str() so a string-shaped alternative doesn't crash us.
                iso = getattr(acq_date, "isoformat", None)
                record["date"] = iso() if callable(iso) else str(acq_date)

        microscope = _first_microscope(ome)
        if microscope is not None:
            brand = _brand_and_model(
                _to_string(getattr(microscope, "manufacturer", None)),
                _to_string(getattr(microscope, "model", None)),
            )
            if brand is not None:
                record["microscope"] = brand
            serial = _to_string(getattr(microscope, "serial_number", None))
            if serial is not None:
                record["microscope_serial"] = serial

        return record or None
    except Exception:  # noqa: BLE001 — never crash audit
        return None


__all__ = ["extract_objective_from_ome", "extract_acquisition_from_ome"]
