"""Tests for the LIF acquisition/instrument extractor (ADR-0008 / #62)."""

from datetime import UTC, datetime
from pathlib import Path

from zarrmony.metadata.acquisition import extract_acquisition

FIXTURE = Path(__file__).parent / "fixtures" / "lif_confocal_7ch.xml"


def _fixture_xml() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_extract_from_fixture_populates_microscope_and_serial() -> None:
    """The captured Stellaris fixture carries ``SystemTypeName="STELLARIS 8"``
    and ``SystemSerialNumber="8300000404"`` on its HardwareSetting."""
    result = extract_acquisition(_fixture_xml())
    assert result is not None
    assert result["microscope"] == "STELLARIS 8"
    assert result["microscope_serial"] == "8300000404"


def test_extract_from_fixture_infers_confocal_from_atl_blocks() -> None:
    """The fixture has ATLConfocalSettingDefinition blocks even without a
    populated DataSourceTypeName — the extractor infers confocal."""
    result = extract_acquisition(_fixture_xml())
    assert result is not None
    assert result["imaging_method"] == ["confocal"]


def test_imaging_method_is_always_a_list_even_when_single_valued() -> None:
    """BQ column is REPEATED STRING — must never emit a scalar."""
    result = extract_acquisition(_fixture_xml())
    assert isinstance(result["imaging_method"], list)


def test_missing_fields_omitted_never_null() -> None:
    """A scene with no hardware/timestamp emits ``None`` (not an empty dict)."""
    minimal = (
        "<LMSDataContainerHeader><Element><Data><Image/></Data></Element>"
        "</LMSDataContainerHeader>"
    )
    assert extract_acquisition(minimal) is None


def test_datasource_type_name_widefield_normalises_to_ome_token() -> None:
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        '<Attachment Name="HardwareSetting" DataSourceTypeName="Widefield" />'
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    assert result["imaging_method"] == ["widefield_fluorescence"]


def test_multiple_hardware_settings_dedupe_imaging_method() -> None:
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        '<Attachment Name="HardwareSetting" DataSourceTypeName="Confocal" />'
        '<Attachment Name="HardwareSetting" DataSourceTypeName="Confocal" />'
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    assert result["imaging_method"] == ["confocal"]


def test_filetime_timestamp_projects_to_iso_utc() -> None:
    # 2026-05-15 12:00:00 UTC in FILETIME ticks (100 ns since 1601-01-01).
    ticks = int(
        (datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC).timestamp() + 11644473600)
        * 10_000_000
    )
    high = ticks >> 32
    low = ticks & 0xFFFFFFFF
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        f'<Attachment Name="TimeStampList"><TimeStamp HighInteger="{high}" '
        f'LowInteger="{low}" /></Attachment>'
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    # ISO string, UTC, matches the expected instant.
    parsed = datetime.fromisoformat(result["date"])
    assert parsed == datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)


def test_garbage_input_returns_none() -> None:
    assert extract_acquisition("") is None
    assert extract_acquisition("not xml") is None
    assert extract_acquisition("<!DOCTYPE foo><root/>") is None


def test_placeholder_microscope_model_zero_is_rejected() -> None:
    """LIF stores ``MicroscopeModel="0"`` as a not-populated placeholder."""
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        '<Attachment Name="HardwareSetting" MicroscopeModel="0" />'
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    # No microscope key when only the placeholder was present.
    assert result is None or "microscope" not in result
