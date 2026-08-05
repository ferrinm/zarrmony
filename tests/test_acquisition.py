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


# --- widefield detection tiers (Leica Thunder / DMi8 patterns) -------------


def test_widefield_channel_info_fluo_yields_widefield_fluorescence() -> None:
    """Thunder scenes report DataSourceTypeName='Camera' (generic) but carry
    WideFieldChannelInfo with ContrastingMethodName='FLUO' — that's what
    identifies the scene as widefield fluorescence."""
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        '<Attachment Name="HardwareSetting" DataSourceTypeName="Camera" />'
        '<WideFieldChannelInfo ContrastingMethodName="FLUO" />'
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    assert result["imaging_method"] == ["widefield_fluorescence"]


def test_mixed_contrasting_methods_surface_every_distinct_token() -> None:
    """Multi-channel widefield with FLUO + BF channels emits both tokens
    in first-seen order (matches OME AcquisitionMode multi-mode behaviour)."""
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        '<WideFieldChannelInfo ContrastingMethodName="BF" />'
        '<WideFieldChannelInfo ContrastingMethodName="FLUO" />'
        '<WideFieldChannelInfo ContrastingMethodName="FLUO" />'
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    assert result["imaging_method"] == ["bright_field", "widefield_fluorescence"]


def test_atl_camera_setting_fallback_when_no_contrasting_method() -> None:
    """A camera-based Leica scene with no ContrastingMethodName and no ATL
    confocal block falls back to widefield_fluorescence via
    ATLCameraSettingDefinition presence."""
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        '<Attachment Name="HardwareSetting" DataSourceTypeName="Camera" />'
        "<ATLCameraSettingDefinition />"
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    assert result["imaging_method"] == ["widefield_fluorescence"]


def test_contrasting_method_variants_normalise_correctly() -> None:
    """Case-insensitive; hyphens and underscores stripped (TL-BF == TL_BF == TLBF)."""
    for raw in ("bf", "BF", "Tl-Bf", "TL_BF"):
        xml = (
            "<LMSDataContainerHeader><Element><Data><Image>"
            f'<WideFieldChannelInfo ContrastingMethodName="{raw}" />'
            "</Image></Data></Element></LMSDataContainerHeader>"
        )
        result = extract_acquisition(xml)
        assert result is not None
        assert result["imaging_method"] == ["bright_field"], f"failed for {raw!r}"


def test_confocal_tier_wins_over_widefield_camera_fallback() -> None:
    """A scene with both an ATLConfocal block AND ATLCameraSetting reports
    confocal (tier 3 fires first, tier 4 is skipped)."""
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        "<ATLConfocalSettingDefinition />"
        "<ATLCameraSettingDefinition />"
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    assert result["imaging_method"] == ["confocal"]


# --- TimeStampList hex FILETIME parsing (Thunder / LAS X 3.x) --------------


def test_hex_timestamp_list_projects_first_value_to_iso() -> None:
    """LAS X 3.x LIFs carry per-frame timestamps in ``<TimeStampList>`` as
    space-separated hex FILETIME values. The first value is the scene start."""
    when = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    ticks = int((when.timestamp() + 11644473600) * 10_000_000)
    hex_ticks = f"{ticks:x}"
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        f'<TimeStampList NumberOfTimeStamps="3">{hex_ticks} deadbeef1234567 '
        f"cafef00d0000000</TimeStampList>"
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    parsed = datetime.fromisoformat(result["date"])
    assert parsed == when


def test_timestamp_element_takes_precedence_over_timestamp_list() -> None:
    """Older shape wins when both are present — same ordering as the extractor
    walks (iter TimeStamp first, then TimeStampList)."""
    when_new = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    ticks_new = int((when_new.timestamp() + 11644473600) * 10_000_000)
    when_old = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
    ticks_old = int((when_old.timestamp() + 11644473600) * 10_000_000)
    high = ticks_old >> 32
    low = ticks_old & 0xFFFFFFFF
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        f'<TimeStamp HighInteger="{high}" LowInteger="{low}" />'
        f"<TimeStampList>{ticks_new:x}</TimeStampList>"
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is not None
    parsed = datetime.fromisoformat(result["date"])
    assert parsed == when_old


def test_timestamp_list_empty_or_garbage_yields_no_date() -> None:
    xml = (
        "<LMSDataContainerHeader><Element><Data><Image>"
        "<TimeStampList></TimeStampList>"
        "<TimeStampList>not-hex</TimeStampList>"
        "</Image></Data></Element></LMSDataContainerHeader>"
    )
    result = extract_acquisition(xml)
    assert result is None or "date" not in result
