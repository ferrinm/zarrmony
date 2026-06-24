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
        monkeypatch.setattr(
            api_module, "get_reader", lambda _path: (reader, plugin_obj, 100)
        )

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
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--metadata-file",
            str(md),
            "--pyramid-min-size",
            "8",
        ],
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
        app,
        ["convert", "/tmp/x.lif", str(out), "--permissive", "--pyramid-min-size", "8"],
    )
    assert result.exit_code == 0, result.output
    assert (out / "s.ome.zarr" / "zarr.json").exists()


def test_convert_no_metadata_no_permissive_fails_friendly(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app, ["convert", "/tmp/x.lif", str(out), "--pyramid-min-size", "8"]
    )
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


def test_convert_force_overwrites(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
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


def test_convert_prints_input_and_output_size_lines_per_scene(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    src = tmp_path / "in.lif"
    src.write_bytes(b"\x00" * 4096)
    md = _write_metadata_file(
        tmp_path / "md.json", {"microscope": "Axioscan", "modality": "fluorescence"}
    )
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", str(src), str(out), "-m", str(md), "--pyramid-min-size", "8"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    wrote_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Wrote "))
    input_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Input:"))
    output_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Output:"))
    assert wrote_idx < input_idx < output_idx
    assert "4.0 KB" in lines[input_idx]


def test_convert_chunk_shape_invalid_format(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--permissive",
            "--chunk-shape",
            "not,a,number",
        ],
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


def test_inspect_text_output_prints_size_line(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)
    src = tmp_path / "input.lif"
    src.write_bytes(b"\x00" * 2048)

    result = runner.invoke(app, ["inspect", str(src)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    input_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Input:"))
    plugin_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Plugin:"))
    size_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Size:"))
    assert input_idx < size_idx < plugin_idx
    assert "2.0 KB" in lines[size_idx]


def test_inspect_json_output_unchanged_by_size_line(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)
    src = tmp_path / "input.lif"
    src.write_bytes(b"\x00" * 16)

    result = runner.invoke(app, ["inspect", str(src), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "size_bytes" not in parsed
    assert "Size" not in result.output


def test_inspect_text_output_omits_plate_header_for_flat_reader(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)

    result = runner.invoke(app, ["inspect", "/tmp/x.lif"])
    assert result.exit_code == 0
    assert "Plate:" not in result.output


def test_inspect_text_output_prints_plate_header_for_plate_reader(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

    plate_layout = PlateLayout(
        name="synthetic-2x2",
        rows=["A", "B"],
        columns=["01", "02"],
        acquisitions=[Acquisition(id=1, name="acq", maximumfieldcount=1)],
        fields=[
            PlateField(
                scene_index=0,
                row="A",
                column="01",
                field_name="A01-f0",
                acquisition_id=1,
            ),
            PlateField(
                scene_index=1,
                row="A",
                column="02",
                field_name="A02-f0",
                acquisition_id=1,
            ),
            PlateField(
                scene_index=2,
                row="B",
                column="01",
                field_name="B01-f0",
                acquisition_id=1,
            ),
        ],
    )
    reader = FakeReader(
        scenes=["s0", "s1", "s2"],
        dims="TCYX",
        shape=(1, 1, 16, 16),
        layout_hint="plate",
        plate_layout=plate_layout,
    )
    patched_reader(reader, plugin="bioio-fake-plate")

    result = runner.invoke(app, ["inspect", "/tmp/x.czi"])
    assert result.exit_code == 0, result.output
    assert 'Plate: "synthetic-2x2"' in result.output
    assert "3/4 wells imaged" in result.output
    assert "1 field per well" in result.output
    assert "1 acquisition" in result.output


# ---------- schema dump ----------


def test_schema_dump_emits_valid_json_schema(runner: CliRunner) -> None:
    result = runner.invoke(app, ["schema", "dump"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "required" in parsed
    assert "microscope" in parsed["required"]
    assert "modality" in parsed["required"]
    assert "properties" in parsed
