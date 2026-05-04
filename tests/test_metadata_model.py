import pytest
from pydantic import ValidationError

from zarrmony.metadata.model import UserMetadata
from zarrmony.metadata.schema import export_schema, export_schema_json


def test_required_fields_enforced() -> None:
    with pytest.raises(ValidationError):
        UserMetadata()


def test_required_fields_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        UserMetadata(microscope="", modality="fluorescence")


def test_minimum_valid_metadata() -> None:
    m = UserMetadata(microscope="Axioscan", modality="multiplex")
    assert m.microscope == "Axioscan"
    assert m.modality == "multiplex"
    assert m.objective is None


def test_optional_fields_default_none() -> None:
    m = UserMetadata(microscope="Thunder", modality="fluorescence")
    assert m.objective is None
    assert m.detector_gain is None
    assert m.laser_power is None


def test_extra_fields_allowed_pre_finalization() -> None:
    m = UserMetadata(microscope="Thunder", modality="fluorescence", custom_tag="experimental")
    dumped = m.model_dump()
    assert dumped["custom_tag"] == "experimental"


def test_strip_whitespace_on_strings() -> None:
    m = UserMetadata(microscope="  Axioscan  ", modality="multiplex")
    assert m.microscope == "Axioscan"


def test_export_schema_contains_required_fields() -> None:
    schema = export_schema()
    assert "required" in schema
    assert "microscope" in schema["required"]
    assert "modality" in schema["required"]


def test_export_schema_json_round_trip() -> None:
    s = export_schema_json()
    import json

    parsed = json.loads(s)
    assert parsed == export_schema()
