"""Tests for the click CLI commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony.cli import app
from zarrmony.readers.plugin import ReaderPlugin


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    def installer(reader: FakeReader, plugin: str = "bioio-fake") -> None:
        plugin_obj = ReaderPlugin(
            name=plugin,
            match=lambda _p: 100,
            open=lambda _p: object(),
            distribution=plugin,
            source="builtin",
        )
        monkeypatch.setattr(api_module, "get_reader", lambda _path: (reader, plugin_obj, 100))

    return installer


def _write_metadata_file(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


# ---------- convert (per-scene default) ----------


def test_convert_per_scene_default_writes_one_store_per_scene(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["alpha", "beta"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    md = _write_metadata_file(
        tmp_path / "md.json", {"microscope": "Axioscan", "modality": "fluorescence"}
    )
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--metadata-file", str(md), "--pyramid-min-size", "8"],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote 2 stores to" in result.output
    assert (out / "alpha.ome.zarr" / "OME" / "METADATA.ome.xml").exists()
    assert (out / "beta.ome.zarr" / "OME" / "METADATA.ome.xml").exists()


def test_convert_per_scene_singular_store_phrasing(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    md = _write_metadata_file(
        tmp_path / "md.json", {"microscope": "Axioscan", "modality": "fluorescence"}
    )
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 1 store to" in result.output


def test_convert_permissive_bypasses_gate(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app, ["convert", "/tmp/x.lif", str(out), "--permissive", "--pyramid-min-size", "8"]
    )
    assert result.exit_code == 0, result.output
    assert (out / "s.ome.zarr" / "zarr.json").exists()


def test_convert_no_metadata_no_permissive_fails_friendly(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(app, ["convert", "/tmp/x.lif", str(out), "--pyramid-min-size", "8"])
    assert result.exit_code != 0
    assert "Metadata validation failed" in result.output


def test_convert_existing_store_without_force_fails_friendly(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    md = _write_metadata_file(
        tmp_path / "md.json", {"microscope": "Axioscan", "modality": "multiplex"}
    )
    out = tmp_path / "out"

    r1 = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    assert r1.exit_code == 0

    r2 = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    assert r2.exit_code != 0
    assert "already exists" in r2.output
    assert "force" in r2.output.lower()


def test_convert_force_overwrites(tmp_path: Path, runner: CliRunner, patched_reader) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    md = _write_metadata_file(
        tmp_path / "md.json", {"microscope": "Axioscan", "modality": "multiplex"}
    )
    out = tmp_path / "out"

    runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    r2 = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "-m",
            str(md),
            "--pyramid-min-size",
            "8",
            "--force",
        ],
    )
    assert r2.exit_code == 0


def test_convert_chunk_shape_invalid_format(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--permissive", "--chunk-shape", "not,a,number"],
    )
    assert result.exit_code != 0
    assert "chunk-shape" in result.output


# ---------- convert (--layout bf2raw opt-in) ----------


def test_convert_layout_bf2raw_writes_bundle(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    md = _write_metadata_file(
        tmp_path / "md.json", {"microscope": "Axioscan", "modality": "fluorescence"}
    )
    out = tmp_path / "x.ome.zarr"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--layout",
            "bf2raw",
            "-m",
            str(md),
            "--pyramid-min-size",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bf2raw bundle" in result.output
    assert (out / "OME" / "METADATA.ome.xml").exists()
    assert (out / "0").is_dir()
    assert (out / "1").is_dir()


def test_convert_layout_invalid_choice_rejected(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--layout", "bogus", "--permissive"],
    )
    assert result.exit_code != 0
    assert "layout" in result.output.lower()


# ---------- inspect ----------


def test_inspect_text_output(tmp_path: Path, runner: CliRunner, patched_reader) -> None:
    reader = FakeReader(
        scenes=["alpha", "beta"],
        dims="TCYX",
        shape=(1, 2, 256, 256),
        channel_names=["DAPI", "GFP"],
    )
    patched_reader(reader, plugin="bioio-fake")

    result = runner.invoke(app, ["inspect", "/tmp/x.lif"])
    assert result.exit_code == 0, result.output
    assert "Plugin: bioio-fake" in result.output
    assert "Scenes: 2" in result.output
    assert "alpha" in result.output and "beta" in result.output
    assert "DAPI, GFP" in result.output


def test_inspect_json_output(tmp_path: Path, runner: CliRunner, patched_reader) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)

    result = runner.invoke(app, ["inspect", "/tmp/x.lif", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["n_scenes"] == 1
    assert parsed["scenes"][0]["name"] == "only"


# ---------- schema dump ----------


def test_schema_dump_emits_valid_json_schema(runner: CliRunner) -> None:
    result = runner.invoke(app, ["schema", "dump"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "required" in parsed
    assert "microscope" in parsed["required"]
    assert "modality" in parsed["required"]
    assert "properties" in parsed
