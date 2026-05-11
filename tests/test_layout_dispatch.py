"""Tests for the ``layout='auto'`` dispatch matrix (issue #9 / ADR-0004).

Covers every cell of the matrix:

- ``auto`` + flat reader → ``per-scene``.
- ``auto`` + plate reader → ``plate``.
- ``per-scene`` / ``bf2raw`` + plate reader → that flat layout, with a
  :class:`LayoutDowngradeWarning` naming the reader and the dropped metadata.
- ``plate`` + flat reader → :class:`LayoutMismatchError` (covered in
  ``tests/test_plate.py::test_plate_against_flat_reader_raises_layout_mismatch``).

Also exercises:

- The plate writer's unreferenced-scenes warning (a plate reader with scenes
  not referenced by any ``PlateField``).
- The CLI's ``--layout auto`` (and omitted) default for both reader shapes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert
from zarrmony.cli import app
from zarrmony.errors import LayoutDowngradeWarning, LayoutMismatchError
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin


def _fake_plugin(name: str = "bioio-fake") -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p: 100,
        open=lambda _p: object(),
        distribution=name,
        source="builtin",
    )


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    def installer(reader: FakeReader, plugin: str = "bioio-fake") -> None:
        plugin_obj = _fake_plugin(plugin)
        monkeypatch.setattr(api_module, "get_reader", lambda _path: (reader, plugin_obj, 100))

    return installer


def _plate_layout_2x2() -> PlateLayout:
    return PlateLayout(
        name="dispatch-2x2",
        rows=["A", "B"],
        columns=["01", "02"],
        acquisitions=[Acquisition(id=1)],
        fields=[
            PlateField(scene_index=0, row="A", column="01", acquisition_id=1),
            PlateField(scene_index=1, row="A", column="02", acquisition_id=1),
            PlateField(scene_index=2, row="B", column="01", acquisition_id=1),
            PlateField(scene_index=3, row="B", column="02", acquisition_id=1),
        ],
    )


def _plate_reader() -> FakeReader:
    return FakeReader(
        scenes=["s0", "s1", "s2", "s3"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=_plate_layout_2x2(),
    )


def _flat_reader(n_scenes: int = 2) -> FakeReader:
    return FakeReader(
        scenes=[f"s{i}" for i in range(n_scenes)],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="flat",
    )


# ---------- auto resolves to per-scene / plate by reader hint ----------


def test_auto_plus_flat_reader_resolves_to_per_scene(tmp_path: Path, patched_reader) -> None:
    patched_reader(_flat_reader(2))
    result = convert(
        "/tmp/fake.czi",
        tmp_path / "out",
        layout="auto",
        metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        pyramid_min_size=8,
    )
    assert result["layout"] == "per-scene"
    assert (tmp_path / "out" / "s0.ome.zarr").exists()
    assert (tmp_path / "out" / "s1.ome.zarr").exists()


def test_auto_plus_plate_reader_resolves_to_plate(tmp_path: Path, patched_reader) -> None:
    patched_reader(_plate_reader())
    out = tmp_path / "plate.ome.zarr"
    audit = convert(
        "/tmp/fake.czi",
        out,
        layout="auto",
        metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        pyramid_min_size=8,
    )
    assert audit["layout"] == "plate"
    assert audit["plate"]["name"] == "dispatch-2x2"
    assert (out / "A" / "01" / "0").is_dir()


def test_auto_is_the_default_layout(tmp_path: Path, patched_reader) -> None:
    """Calling convert() with no ``layout=`` arg uses auto-dispatch."""
    patched_reader(_plate_reader())
    out = tmp_path / "plate.ome.zarr"
    audit = convert(
        "/tmp/fake.czi",
        out,
        metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        pyramid_min_size=8,
    )
    assert audit["layout"] == "plate"


# ---------- explicit flat against plate reader: downgrade ----------


def test_per_scene_against_plate_reader_warns_and_writes_flat(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(_plate_reader(), plugin="bioio-fake-plate")
    out = tmp_path / "out"
    with pytest.warns(LayoutDowngradeWarning, match="bioio-fake-plate"):
        result = convert(
            "/tmp/fake.czi",
            out,
            layout="per-scene",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            pyramid_min_size=8,
        )
    assert result["layout"] == "per-scene"
    # Per-scene path actually wrote one store per scene (no plate structure).
    assert (out / "s0.ome.zarr").exists()
    assert not (out / "A").exists()


def test_bf2raw_against_plate_reader_warns_and_writes_bundle(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(_plate_reader(), plugin="bioio-fake-plate")
    out = tmp_path / "bundle.ome.zarr"
    with pytest.warns(LayoutDowngradeWarning, match="rows, columns, wells"):
        audit = convert(
            "/tmp/fake.czi",
            out,
            layout="bf2raw",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            pyramid_min_size=8,
        )
    assert audit["layout"] == "bf2raw"
    assert (out / "0").is_dir()
    assert (out / "OME" / "METADATA.ome.xml").exists()


# ---------- plate + flat reader (LayoutMismatchError) ----------


def test_plate_against_flat_reader_message_names_layout_hint(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(_flat_reader(1))
    with pytest.raises(LayoutMismatchError) as exc:
        convert(
            "/tmp/fake.czi",
            tmp_path / "out.ome.zarr",
            layout="plate",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        )
    msg = str(exc.value)
    assert "layout_hint='flat'" in msg
    assert "bioio-fake" in msg


# ---------- per-scene/bf2raw + flat reader: silent (no warning) ----------


def test_per_scene_against_flat_reader_does_not_warn(tmp_path: Path, patched_reader) -> None:
    patched_reader(_flat_reader(2))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        convert(
            "/tmp/fake.czi",
            tmp_path / "out",
            layout="per-scene",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            pyramid_min_size=8,
        )
    downgrade = [w for w in record if issubclass(w.category, LayoutDowngradeWarning)]
    assert downgrade == []


# ---------- unreferenced-scenes warning from the plate writer ----------


def test_plate_warns_on_unreferenced_scenes(tmp_path: Path, patched_reader) -> None:
    """Reader exposes 5 scenes but plate_layout only references 4 of them."""
    layout = _plate_layout_2x2()  # references scenes 0..3
    reader = FakeReader(
        scenes=["s0", "s1", "s2", "s3", "s4"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=layout,
    )
    patched_reader(reader)
    out = tmp_path / "plate.ome.zarr"
    with pytest.warns(LayoutDowngradeWarning, match="not referenced"):
        convert(
            "/tmp/fake.czi",
            out,
            layout="auto",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            pyramid_min_size=8,
        )


def test_plate_does_not_warn_when_all_scenes_referenced(tmp_path: Path, patched_reader) -> None:
    patched_reader(_plate_reader())
    out = tmp_path / "plate.ome.zarr"
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        convert(
            "/tmp/fake.czi",
            out,
            layout="auto",
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
            pyramid_min_size=8,
        )
    downgrade = [w for w in record if issubclass(w.category, LayoutDowngradeWarning)]
    assert downgrade == []


# ---------- input validation ----------


def test_unknown_layout_value_rejected(tmp_path: Path, patched_reader) -> None:
    patched_reader(_flat_reader(1))
    with pytest.raises(ValueError, match="layout must be one of"):
        convert(
            "/tmp/fake.czi",
            tmp_path / "out",
            layout="bogus",  # type: ignore[arg-type]
            metadata={"microscope": "Axioscan", "modality": "fluorescence"},
        )


# ---------- CLI: --layout auto (omitted) for both reader shapes ----------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_md(path: Path) -> Path:
    path.write_text(json.dumps({"microscope": "Axioscan", "modality": "fluorescence"}))
    return path


def test_cli_auto_default_writes_per_scene_for_flat_reader(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    patched_reader(_flat_reader(2))
    md = _write_md(tmp_path / "md.json")
    out = tmp_path / "out"

    # No --layout flag → defaults to auto.
    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 2 stores to" in result.output
    assert (out / "s0.ome.zarr").exists()
    assert (out / "s1.ome.zarr").exists()


def test_cli_auto_default_writes_plate_for_plate_reader(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    patched_reader(_plate_reader())
    md = _write_md(tmp_path / "md.json")
    out = tmp_path / "plate.ome.zarr"

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    assert result.exit_code == 0, result.output
    assert "(plate)" in result.output
    assert (out / "A" / "01" / "0").is_dir()


def test_cli_explicit_layout_auto_matches_omitted(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    patched_reader(_plate_reader())
    md = _write_md(tmp_path / "md.json")
    out = tmp_path / "plate.ome.zarr"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--layout",
            "auto",
            "-m",
            str(md),
            "--pyramid-min-size",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(plate)" in result.output
