from datetime import UTC, datetime
from pathlib import Path

import zarr

from zarrmony import __version__
from zarrmony.audit import AUDIT_SCHEMA_VERSION, build_audit_record, write_audit_record
from zarrmony.readers.plugin import ReaderPlugin


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _fake_plugin(
    name: str = "bioio-czi",
    distribution: str | None = "bioio-czi",
    source: str = "builtin",
) -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=distribution,
        source=source,  # type: ignore[arg-type]
    )


def test_build_audit_record_minimum_keys(tmp_path: Path) -> None:
    src = tmp_path / "input.czi"
    src.write_bytes(b"\x00" * 1024)

    audit = build_audit_record(
        input_path=src,
        reader_plugin=_fake_plugin(),
        match_score=100,
        config={"pyramid_min_size": 256},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:01:30"),
    )

    assert audit["audit_schema_version"] == AUDIT_SCHEMA_VERSION
    assert audit["version"] == __version__
    assert audit["reader_plugin"]["name"] == "bioio-czi"
    assert audit["reader_plugin"]["distribution"] == "bioio-czi"
    assert audit["reader_plugin"]["source"] == "builtin"
    assert audit["reader_plugin"]["match_score"] == 100
    assert audit["input"]["path"].endswith("input.czi")
    assert audit["input"]["size_bytes"] == 1024
    assert audit["input"]["size_human"] == "1.0 KiB"
    assert "mtime_iso" in audit["input"]
    assert "sha256" not in audit["input"]
    assert audit["config"] == {"pyramid_min_size": 256}
    assert audit["per_scene"] == []
    assert audit["metadata_warnings"] == []
    # Flat fields removed.
    assert "reader_plugin_version" not in audit


def test_build_audit_record_directory_input_reports_recursive_size(
    tmp_path: Path,
) -> None:
    """Directory-tree inputs (e.g. .zarr stores, multi-file .czi) must report
    the full recursive byte count, not just the top-level inode size."""
    src = tmp_path / "input.zarr"
    (src / "a").mkdir(parents=True)
    (src / "a" / "leaf.bin").write_bytes(b"x" * 500)
    (src / "top.bin").write_bytes(b"y" * 1500)

    audit = build_audit_record(
        input_path=src,
        reader_plugin=_fake_plugin(),
        match_score=100,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
    )
    assert audit["input"]["size_bytes"] == 2000
    assert audit["input"]["size_human"] == "2.0 KiB"


def test_build_audit_record_with_checksum(tmp_path: Path) -> None:
    src = tmp_path / "input.lif"
    src.write_bytes(b"hello world")

    audit = build_audit_record(
        input_path=src,
        reader_plugin=_fake_plugin(name="bioio-lif", distribution="bioio-lif"),
        match_score=100,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:10"),
        checksum=True,
    )

    # Pre-computed: sha256("hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert audit["input"]["sha256"] == expected


def test_build_audit_record_unknown_distribution_yields_null_version(
    tmp_path: Path,
) -> None:
    src = tmp_path / "x.czi"
    src.write_bytes(b"")
    audit = build_audit_record(
        input_path=src,
        reader_plugin=_fake_plugin(
            name="not-a-real-package", distribution="not-a-real-package"
        ),
        match_score=100,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
    )
    assert audit["reader_plugin"]["name"] == "not-a-real-package"
    assert audit["reader_plugin"]["distribution"] == "not-a-real-package"
    assert audit["reader_plugin"]["version"] is None


def test_build_audit_record_distribution_override_wins(tmp_path: Path) -> None:
    """The catch-all default plugin has distribution=None at registration; the
    audit caller injects the dynamically-resolved distribution explicitly.
    """
    src = tmp_path / "x.tif"
    src.write_bytes(b"")
    plugin = _fake_plugin(name="bioio", distribution=None)

    audit = build_audit_record(
        input_path=src,
        reader_plugin=plugin,
        match_score=0,
        distribution="bioio-ome-tiff",
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
    )
    assert audit["reader_plugin"]["name"] == "bioio"
    assert audit["reader_plugin"]["distribution"] == "bioio-ome-tiff"
    assert audit["reader_plugin"]["match_score"] == 0


def test_build_audit_record_includes_per_scene_and_warnings(tmp_path: Path) -> None:
    src = tmp_path / "x.lif"
    src.write_bytes(b"")
    per_scene = [{"scene_index": 0, "scene_name": "a"}]
    warnings = [{"field": "binning", "error": "Row or column missing"}]
    audit = build_audit_record(
        input_path=src,
        reader_plugin=None,
        match_score=None,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
        per_scene=per_scene,
        metadata_warnings=warnings,
    )
    assert audit["reader_plugin"] is None
    assert audit["per_scene"] == per_scene
    assert audit["metadata_warnings"] == warnings


def test_build_audit_record_carries_output_ome_ngff_version(tmp_path: Path) -> None:
    """ADR-0008 / #70: writer's NGFF version is a first-class audit surface."""
    src = tmp_path / "x.czi"
    src.write_bytes(b"")
    audit = build_audit_record(
        input_path=src,
        reader_plugin=_fake_plugin(),
        match_score=100,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
    )
    assert audit["output"] == {"ome_ngff_version": "0.5"}


def test_build_audit_record_reader_plugin_has_exact_keys(tmp_path: Path) -> None:
    """Pin the new reader_plugin dict shape per ADR-0001 / Q7."""
    src = tmp_path / "x.czi"
    src.write_bytes(b"")
    audit = build_audit_record(
        input_path=src,
        reader_plugin=_fake_plugin(),
        match_score=100,
        config={},
        started_at=_ts("2026-05-02T10:00:00"),
        finished_at=_ts("2026-05-02T10:00:01"),
    )
    assert set(audit["reader_plugin"].keys()) == {
        "name",
        "version",
        "source",
        "distribution",
        "match_score",
    }


def test_write_audit_record_to_zarr(tmp_path: Path) -> None:
    out = tmp_path / "audit.zarr"
    # Create an empty zarr v3 group at the path
    zarr.create_group(str(out), zarr_format=3)

    audit = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "version": __version__,
        "reader_plugin": {
            "name": "bioio-czi",
            "version": None,
            "source": "builtin",
            "distribution": "bioio-czi",
            "match_score": 100,
        },
        "config": {"force": True},
    }
    write_audit_record(out, audit)

    g = zarr.open_group(str(out), mode="r")
    assert dict(g.attrs)["zarrmony"] == audit
