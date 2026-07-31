"""Tests for the OME-metadata → audit-block projections (ADR-0008 / #63–#65).

Covers ``zarrmony.metadata.ome_extractors`` — the non-LIF path that reads
`reader.ome_metadata` for CZI / ND2 / OME-TIFF and any other bioio reader
whose OME surface exposes ``instruments[0].objectives`` /
``instruments[0].microscope`` / ``images[i].acquisition_date``.
"""

from datetime import UTC, datetime

from ome_types import OME
from ome_types.model import (
    Image,
    Instrument,
    Microscope,
    Objective,
    Pixels,
    PixelType,
)

from zarrmony.metadata.ome_extractors import (
    extract_acquisition_from_ome,
    extract_objective_from_ome,
)


def _synth_ome(
    *,
    objective: Objective | None = None,
    microscope: Microscope | None = None,
    acquisition_date: datetime | None = None,
) -> OME:
    """Build a minimal OME tree from the given pieces."""
    instruments = []
    if objective is not None or microscope is not None:
        instruments.append(
            Instrument(
                id="Instrument:0",
                objectives=[objective] if objective else [],
                microscope=microscope,
            )
        )
    image = Image(
        id="Image:0",
        acquisition_date=acquisition_date,
        pixels=Pixels(
            id="Pixels:0",
            size_x=16,
            size_y=16,
            size_z=1,
            size_c=1,
            size_t=1,
            dimension_order="XYZCT",
            type=PixelType.UINT16,
        ),
    )
    return OME(images=[image], instruments=instruments)


def test_extract_objective_shape_from_ome() -> None:
    obj = Objective(
        id="Objective:0",
        nominal_magnification=63.0,
        lens_na=1.4,
        immersion="Oil",
        model="Plan Apo 63x/1.4",
        working_distance=190.0,
    )
    result = extract_objective_from_ome(_synth_ome(objective=obj))
    assert result == {
        "nominal_magnification": 63,
        "numerical_aperture": 1.4,
        "immersion": "Oil",
        "model": "Plan Apo 63x/1.4",
        "working_distance_um": 190,
    }


def test_extract_objective_omits_missing_fields() -> None:
    obj = Objective(id="Objective:0", nominal_magnification=20.0)
    result = extract_objective_from_ome(_synth_ome(objective=obj))
    assert result == {"nominal_magnification": 20}


def test_extract_objective_none_when_no_instrument() -> None:
    assert extract_objective_from_ome(_synth_ome()) is None


def test_extract_acquisition_combines_manufacturer_and_model() -> None:
    scope = Microscope(manufacturer="Nikon", model="Ti2-E", serial_number="TI2-0042")
    result = extract_acquisition_from_ome(_synth_ome(microscope=scope))
    assert result == {"microscope": "Nikon Ti2-E", "microscope_serial": "TI2-0042"}


def test_extract_acquisition_date_serialises_to_iso() -> None:
    when = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    result = extract_acquisition_from_ome(_synth_ome(acquisition_date=when))
    assert result is not None
    parsed = datetime.fromisoformat(result["date"])
    assert parsed == when


def test_extract_acquisition_none_when_empty() -> None:
    assert extract_acquisition_from_ome(_synth_ome()) is None


def test_extract_acquisition_manufacturer_only_still_populates_microscope() -> None:
    """A microscope with only the manufacturer set should still surface as
    the `microscope` field — one of `manufacturer` or `model` is enough."""
    scope = Microscope(manufacturer="Zeiss")
    result = extract_acquisition_from_ome(_synth_ome(microscope=scope))
    assert result == {"microscope": "Zeiss"}


def test_extract_never_raises_on_none_input() -> None:
    assert extract_objective_from_ome(None) is None
    assert extract_acquisition_from_ome(None) is None
