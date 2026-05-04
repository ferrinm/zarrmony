from datetime import UTC, datetime
from pathlib import Path

import zarr

from zarrmony import __version__
from zarrmony.audit import build_audit_record, write_audit_record


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_build_audit_record_minimum_keys(tmp_path: Path) -> None:
    src = tmp_path / "input.czi"
    src.write_bytes(b"\x00" * 1024)

    audit = build_audit_record(
        input_path=src,
        reader_plugin="bioio-czi",
        config={"pyramid_min_size": 256},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:01:30"),
    )

    assert audit["version"] == __version__
    assert audit["reader_plugin"] == "bioio-czi"
    assert audit["input"]["path"].endswith("input.czi")
    assert audit["input"]["size_bytes"] == 1024
    assert "mtime_iso" in audit["input"]
    assert "sha256" not in audit["input"]
    assert audit["config"] == {"pyramid_min_size": 256}
    assert audit["per_scene"] == []
    assert audit["metadata_warnings"] == []


def test_build_audit_record_with_checksum(tmp_path: Path) -> None:
    src = tmp_path / "input.lif"
    src.write_bytes(b"hello world")

    audit = build_audit_record(
        input_path=src,
        reader_plugin="bioio-lif",
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:10"),
        checksum=True,
    )

    # Pre-computed: sha256("hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert audit["input"]["sha256"] == expected


def test_build_audit_record_unknown_reader_plugin(tmp_path: Path) -> None:
    src = tmp_path / "x.czi"
    src.write_bytes(b"")
    audit = build_audit_record(
        input_path=src,
        reader_plugin="not-a-real-package",
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
    )
    assert audit["reader_plugin"] == "not-a-real-package"
    assert audit["reader_plugin_version"] is None


def test_build_audit_record_includes_per_scene_and_warnings(tmp_path: Path) -> None:
    src = tmp_path / "x.lif"
    src.write_bytes(b"")
    per_scene = [{"scene_index": 0, "scene_name": "a"}]
    warnings = [{"field": "binning", "error": "Row or column missing"}]
    audit = build_audit_record(
        input_path=src,
        reader_plugin=None,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
        per_scene=per_scene,
        metadata_warnings=warnings,
    )
    assert audit["per_scene"] == per_scene
    assert audit["metadata_warnings"] == warnings


def test_write_audit_record_to_zarr(tmp_path: Path) -> None:
    out = tmp_path / "audit.zarr"
    # Create an empty zarr v3 group at the path
    zarr.create_group(str(out), zarr_format=3)

    audit = {
        "version": __version__,
        "reader_plugin": "bioio-czi",
        "config": {"force": True},
    }
    write_audit_record(out, audit)

    g = zarr.open_group(str(out), mode="r")
    assert dict(g.attrs)["zarrmony"] == audit
