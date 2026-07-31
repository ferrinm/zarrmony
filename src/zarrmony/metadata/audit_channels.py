"""Project per-scene channel identity into the ADR-0008 audit shape.

The audit block per ADR-0008 (#61) is a list of one dict per channel, in
acquisition order, with these 9 keys — every key except ``index`` is optional
and omitted (not ``None``) when the reader couldn't extract it:

    {
      "index": 0,
      "name": "DAPI",
      "dye": "DAPI",
      "fluor": "DAPI",
      "excitation_nm": 405,
      "emission_low_nm": 420,
      "emission_high_nm": 480,
      "color": "0099ff",
      "lut_name": "Blue"
    }

Two projections live here:

* :func:`from_lif_extracted` — feeds off the identity dicts returned by
  :func:`zarrmony.metadata.lif_channels.extract_channels` (already carries
  dye/fluor/excitation/emission/lut_name; ``name`` comes from the cleaned dye).
* :func:`from_ome_channels` — projects ``ome_types.model.Channel`` objects (the
  surface CZI / ND2 / OME-TIFF and any other bioio reader with
  ``reader.ome_metadata.images[0].pixels.channels`` exposes).

Both helpers are fail-closed: any per-channel exception omits that channel's
optional fields rather than crashing the audit. Missing per-channel fields
degrade to key-absent so downstream can distinguish "reader didn't extract
this field" from "reader tried and got nothing" (the latter reserved for
future nullable-value extensions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarrmony.metadata.lif_channels import _clean_dye

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ome_types.model import Channel


def _set_if(record: dict[str, Any], key: str, value: Any) -> None:
    """Assign ``value`` to ``record[key]`` only when it is not ``None``.

    Omit-on-absent per ADR-0008 — never emit explicit ``null`` for a missing
    per-channel field.
    """
    if value is not None:
        record[key] = value


def from_lif_extracted(
    extracted: list[dict], colors: list[str] | None = None
) -> list[dict]:
    """Project LIF-extracted identity dicts into the ADR-0008 audit shape.

    ``extracted`` is the list returned by
    :func:`zarrmony.metadata.lif_channels.extract_channels`; ``colors`` is the
    parallel list of resolved hex colors (from
    :func:`zarrmony.metadata.lif_channels.resolve_channel_colors` or the same
    batch used to build the omero block). If ``colors`` is ``None`` or a different
    length, the ``color`` key is omitted from every record.

    Drops the LIF-internal ``detector`` field — the ADR-0008 shape does not
    include it. ``name`` is the cleaned dye (matches what the omero label
    surfaces to the user).
    """
    if colors is not None and len(colors) != len(extracted):
        colors = None

    records: list[dict] = []
    for i, ch in enumerate(extracted):
        record: dict[str, Any] = {"index": i}
        cleaned_dye = _clean_dye(ch.get("dye"))
        _set_if(record, "name", cleaned_dye)
        _set_if(record, "dye", ch.get("dye"))
        _set_if(record, "fluor", ch.get("fluor"))
        _set_if(record, "excitation_nm", ch.get("excitation_nm"))
        _set_if(record, "emission_low_nm", ch.get("emission_low_nm"))
        _set_if(record, "emission_high_nm", ch.get("emission_high_nm"))
        if colors is not None:
            _set_if(record, "color", colors[i])
        _set_if(record, "lut_name", ch.get("lut_name"))
        records.append(record)
    return records


def _ome_wavelength_nm(wavelength: Any, unit: Any) -> float | int | None:
    """Return the wavelength coerced to nm, or ``None`` on any surprise.

    OME's ``excitation_wavelength`` / ``emission_wavelength`` are usually
    already in nm (``unit == "nm"``), but the schema permits micrometers etc.
    Fail-closed: unrecognised unit → ``None`` rather than misreporting.

    Handles both string units (``"nm"``) and ``ome_types.UnitsLength`` enums
    (whose ``.value`` is the string form) — the ome_types API returns the enum
    when a Channel is constructed with ``excitation_wavelength_unit="nm"``.
    """
    if wavelength is None:
        return None
    try:
        value = float(wavelength)
    except (TypeError, ValueError):
        return None
    if unit is None:
        unit_norm = ""
    else:
        # ome_types wraps unit strings in a UnitsLength enum whose .value is
        # the canonical short form (e.g. "nm", "µm"). Prefer .value; fall back
        # to str() for a raw string.
        raw = getattr(unit, "value", unit)
        unit_norm = str(raw).strip().lower()
    if unit_norm in ("", "nm", "nanometer"):
        pass
    elif unit_norm in ("micrometer", "µm", "um"):
        value *= 1000.0
    elif unit_norm in ("angstrom", "å"):
        value *= 0.1
    else:
        return None
    return int(value) if value.is_integer() else value


def from_ome_channels(
    ome_channels: list[Channel], colors: list[str] | None = None
) -> list[dict]:
    """Project a list of ``ome_types.model.Channel`` into the ADR-0008 shape.

    Consumes the surface CZI / ND2 / OME-TIFF (and other bioio readers)
    already expose as ``reader.ome_metadata.images[0].pixels.channels``. When
    the OME element records a single ``emission_wavelength`` rather than a
    band, populates ``emission_low_nm == emission_high_nm == wavelength_nm``
    so downstream code sees a uniform band shape across reader paths.

    Falls back to omitting per-channel fields on any per-channel exception —
    a garbled Channel element must not take down the audit for the rest.
    """
    if colors is not None and len(colors) != len(ome_channels):
        colors = None

    records: list[dict] = []
    for i, ch in enumerate(ome_channels):
        record: dict[str, Any] = {"index": i}
        try:
            name = getattr(ch, "name", None)
            fluor = getattr(ch, "fluor", None)
            excitation_nm = _ome_wavelength_nm(
                getattr(ch, "excitation_wavelength", None),
                getattr(ch, "excitation_wavelength_unit", None),
            )
            emission_nm = _ome_wavelength_nm(
                getattr(ch, "emission_wavelength", None),
                getattr(ch, "emission_wavelength_unit", None),
            )
            _set_if(record, "name", name)
            # OME's <Channel> has no "dye" attribute distinct from name/fluor.
            # Consumers keying off `dye` fall back to `fluor` per ADR-0008's
            # `channel_fluorophores` join rule; leave `dye` absent for the
            # OME projection.
            _set_if(record, "fluor", fluor)
            _set_if(record, "excitation_nm", excitation_nm)
            if emission_nm is not None:
                # OME gives a single point wavelength; ADR-0008 wants a band.
                # Populate low == high so cross-reader consumers see one shape.
                record["emission_low_nm"] = emission_nm
                record["emission_high_nm"] = emission_nm
            if colors is not None:
                _set_if(record, "color", colors[i])
        except Exception:  # noqa: BLE001 — never crash audit over one channel
            pass
        records.append(record)
    return records


__all__ = ["from_lif_extracted", "from_ome_channels"]
