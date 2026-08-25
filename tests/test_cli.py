"""Tests for the click CLI commands."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from tests.conftest import FakePhysicalPixelSizes, FakeReader
from zarrmony import api as api_module
from zarrmony.cli import app
from zarrmony.geometry import DEFAULT_GEOMETRY
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
        # Accept the reader_kwargs kwarg (issue #79) so tests that don't pass
        # it still work, and CLI parsing tests can capture it.
        monkeypatch.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, plugin_obj, 100),
        )

    return installer


# ---------- convert (per-scene default) ----------


def test_convert_per_scene_default_writes_one_store_per_scene(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["alpha", "beta"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
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
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--pyramid-min-size", "8"],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 1 store to" in result.output


def test_convert_existing_store_without_force_fails_friendly(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    r1 = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--pyramid-min-size", "8"],
    )
    assert r1.exit_code == 0

    r2 = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--pyramid-min-size", "8"],
    )
    assert r2.exit_code != 0
    assert "already exists" in r2.output
    assert "force" in r2.output.lower()


def test_convert_force_overwrites(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(out), "--pyramid-min-size", "8"],
    )
    r2 = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
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
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["convert", str(src), str(out), "--pyramid-min-size", "8"],
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
            "--chunk-shape",
            "not,a,number",
        ],
    )
    assert result.exit_code != 0
    assert "chunk-shape" in result.output


# ---------- convert (geometry flags, ADR-0010) ----------


def test_convert_geometry_flags_build_one_policy(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every geometry flag arrives as one ``Geometry``, not as loose kwargs.

    ``resolve_geometry`` refuses ``geometry=`` alongside the retained sugar, so
    the CLI cannot mix the two spellings even though it exposes both flags.
    """
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(tmp_path / "out"),
            "--chunk-target-bytes",
            str(2 * 1024 * 1024),
            "--pyramid-min-size",
            "64",
        ],
    )
    assert result.exit_code == 0, result.output
    geometry = captured["geometry"]
    assert geometry.chunk_target_bytes == 2 * 1024 * 1024
    assert geometry.pyramid_min_size == 64
    # Untouched fields stay at the ADR-0010 defaults...
    assert geometry.chunk_shape is None
    assert geometry.isotropy_tolerance == DEFAULT_GEOMETRY.isotropy_tolerance
    # ...and the sugar kwargs are not passed a second way.
    assert "pyramid_min_size" not in captured
    assert "chunk_shape" not in captured


def test_convert_without_geometry_flags_passes_no_policy(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    result = runner.invoke(app, ["convert", "/tmp/x.lif", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    # None, so convert() reaches for DEFAULT_GEOMETRY itself.
    assert captured["geometry"] is None


def test_convert_chunk_target_bytes_reaches_the_written_store(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 8, 128, 128),
        pixel_sizes=FakePhysicalPixelSizes(Z=4.0, Y=0.5, X=0.5),
    )
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--chunk-target-bytes",
            "4096",
            "--pyramid-min-size",
            "32",
            "--no-contrast",
            "--no-validate",
        ],
    )
    assert result.exit_code == 0, result.output

    array = json.loads((out / "s.ome.zarr" / "0" / "zarr.json").read_text())
    chunks = array["chunk_grid"]["configuration"]["chunk_shape"]
    assert chunks == [1, 1, 2, 32, 32]
    # 2 * 32 * 32 uint16 voxels is exactly the 4096-byte target.
    assert math.prod(chunks) * 2 == 4096


def test_convert_chunk_target_bytes_must_be_positive(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(tmp_path / "out"), "--chunk-target-bytes", "0"],
    )
    assert result.exit_code != 0
    assert "positive int" in result.output


def test_convert_chunk_shape_and_chunk_target_bytes_are_exclusive(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    # --chunk-shape skips the planner outright, so the byte target it would
    # have aimed for is never read; say so rather than silently ignore it.
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(tmp_path / "out"),
            "--chunk-shape",
            "1,1,16,16",
            "--chunk-target-bytes",
            "4096",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_convert_chunk_target_bytes_is_documented_in_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--chunk-target-bytes" in result.output


def _anisotropic_stack_shapes(store: Path) -> list[list[int]]:
    """Every level's on-disk shape, in level order."""
    group = json.loads((store / "zarr.json").read_text())
    paths = [
        d["path"] for d in group["attributes"]["ome"]["multiscales"][0]["datasets"]
    ]
    return [json.loads((store / p / "zarr.json").read_text())["shape"] for p in paths]


def _anisotropic_reader() -> FakeReader:
    """64 planes at Z 4.0 µm over a 128² field at 0.5 µm — 8:1 anisotropic."""
    return FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 64, 128, 128),
        pixel_sizes=FakePhysicalPixelSizes(Z=4.0, Y=0.5, X=0.5),
    )


def test_convert_holds_the_scarce_axis_by_default(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """Z at 8x the lateral spacing is outside the default 1.5 tolerance."""
    patched_reader(_anisotropic_reader())
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(out),
            "--pyramid-min-size", "32", "--no-contrast", "--no-validate",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output

    assert _anisotropic_stack_shapes(out / "s.ome.zarr") == [
        [1, 1, 64, 128, 128],
        [1, 1, 64, 64, 64],
        [1, 1, 64, 32, 32],
    ]


def test_convert_isotropy_tolerance_changes_the_written_levels(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """A tolerance wide enough to admit any axis halves Z alongside Y and X."""
    patched_reader(_anisotropic_reader())
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(out),
            "--isotropy-tolerance", "1e9",
            "--pyramid-min-size", "32", "--no-contrast", "--no-validate",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output

    # Z stops at 32 — the axis floor, which the tolerance does not override.
    assert _anisotropic_stack_shapes(out / "s.ome.zarr") == [
        [1, 1, 64, 128, 128],
        [1, 1, 32, 64, 64],
        [1, 1, 32, 32, 32],
    ]


def test_convert_isotropy_tolerance_must_be_at_least_one(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    # Below 1.0 no axis could ever be within tolerance of the finest one —
    # including the finest axis itself, which is within 1.0x of itself.
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32)))

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(tmp_path / "out"), "--isotropy-tolerance", "0.5"],
    )
    assert result.exit_code != 0
    assert "isotropy_tolerance" in result.output


def test_convert_isotropy_tolerance_is_documented_in_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--isotropy-tolerance" in result.output


# ---------- convert (coarse-level bounds, ADR-0010) ----------


def _coarse_reader() -> FakeReader:
    """8 planes of 256² at 0.5 µm isotropic.

    Z sits below the axis floor, so only Y/X ever halve and the written level
    shapes read as the depth rule alone.
    """
    return FakeReader(
        scenes=["s"],
        dims="TCZYX",
        shape=(1, 1, 8, 256, 256),
        pixel_sizes=FakePhysicalPixelSizes(Z=0.5, Y=0.5, X=0.5),
    )


def test_convert_coarse_bounds_build_one_policy(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(tmp_path / "out"),
            "--coarse-max-bytes", "32768",
            "--coarse-max-long-axis", "512",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    geometry = captured["geometry"]
    assert geometry.coarse_max_bytes == 32768
    assert geometry.coarse_max_long_axis == 512
    # The floor the bounds can win over is untouched by setting them.
    assert geometry.pyramid_min_size == DEFAULT_GEOMETRY.pyramid_min_size


def test_convert_coarse_max_bytes_extends_the_written_levels(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """A byte bound the Y/X rule's last level misses buys another level."""
    patched_reader(_coarse_reader())

    def _run(out: Path, *extra: str) -> list[list[int]]:
        result = runner.invoke(
            app,
            # fmt: off
            [
                "convert", "/tmp/x.lif", str(out),
                "--pyramid-min-size", "64", "--no-contrast", "--no-validate",
                *extra,
            ],
            # fmt: on
        )
        assert result.exit_code == 0, result.output
        return _anisotropic_stack_shapes(out / "s.ome.zarr")

    # The Y/X rule alone stops at 64: its last level is 64 KiB per (t, c),
    # comfortably inside the default 64 MiB bound.
    assert _run(tmp_path / "default") == [
        [1, 1, 8, 256, 256],
        [1, 1, 8, 128, 128],
        [1, 1, 8, 64, 64],
    ]
    # Halve the bound past that level's 64 KiB and the pyramid keeps going.
    assert _run(tmp_path / "tight", "--coarse-max-bytes", "32768") == [
        [1, 1, 8, 256, 256],
        [1, 1, 8, 128, 128],
        [1, 1, 8, 64, 64],
        [1, 1, 8, 32, 32],
    ]


def test_convert_coarse_max_long_axis_extends_the_written_levels(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """The second bound extends depth on its own, bytes notwithstanding."""
    patched_reader(_coarse_reader())
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(out),
            "--coarse-max-long-axis", "32",
            "--pyramid-min-size", "64", "--no-contrast", "--no-validate",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output

    # 64 KiB is inside the byte bound at every level here; it is the 64-voxel
    # lateral extent that keeps the pyramid halving to 32.
    assert _anisotropic_stack_shapes(out / "s.ome.zarr") == [
        [1, 1, 8, 256, 256],
        [1, 1, 8, 128, 128],
        [1, 1, 8, 64, 64],
        [1, 1, 8, 32, 32],
    ]


def test_convert_coarse_max_bytes_must_be_positive(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32)))

    result = runner.invoke(
        app,
        ["convert", "/tmp/x.lif", str(tmp_path / "out"), "--coarse-max-bytes", "0"],
    )
    assert result.exit_code != 0
    assert "positive int" in result.output


def test_convert_coarse_bounds_are_documented_in_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--coarse-max-bytes" in result.output
    assert "--coarse-max-long-axis" in result.output


# ---------- convert (downsample method, ADR-0010) ----------


def _punctum_reader() -> FakeReader:
    """A 64² field of uniform 100 background with one 1000-valued voxel.

    The sparse-label case ADR-0010 exposes `--downsample-method max` for: at the
    32x cumulative factor the levels below reach, one punctum in 1024 pooled
    voxels either survives at full intensity or vanishes into the background.
    """
    volume = np.full((1, 1, 64, 64), 100, dtype=np.uint16)
    volume[0, 0, 32, 32] = 1000
    return FakeReader(
        scenes=["s"],
        dims="TCYX",
        shape=volume.shape,
        channel_names=["GFP"],
        data=volume,
    )


def test_convert_downsample_method_builds_one_policy(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(tmp_path / "out"),
            "--downsample-method", "max",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    assert captured["geometry"].downsample_method == "max"
    # Naming the kernel says nothing about the shapes.
    assert captured["geometry"].pyramid_min_size == DEFAULT_GEOMETRY.pyramid_min_size


@pytest.mark.parametrize(
    ("flag", "coarsest_peak"),
    # No flag at all is the third case worth pinning: the default has to stay
    # mean, not merely be spelled "mean" in the help text.
    [
        ([], 100),
        (["--downsample-method", "mean"], 100),
        (["--downsample-method", "max"], 1000),
    ],
)
def test_convert_downsample_method_reaches_the_written_pixels(
    tmp_path: Path,
    runner: CliRunner,
    patched_reader,
    flag: list[str],
    coarsest_peak: int,
) -> None:
    patched_reader(_punctum_reader())
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(out),
            "--pyramid-min-size", "2", "--no-contrast", "--no-validate",
            *flag,
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output

    group = zarr.open_group(str(out / "s.ome.zarr"), mode="r")
    assert int(group["0"][:].max()) == 1000
    assert int(group["5"][:].max()) == coarsest_peak


def test_convert_rejects_an_unknown_downsample_method(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    # Caught by click's Choice before any file is opened — a typo should not
    # cost a multi-minute read to discover.
    patched_reader(FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32)))

    result = runner.invoke(
        app,
        # fmt: off
        [
            "convert", "/tmp/x.lif", str(tmp_path / "out"),
            "--downsample-method", "median",
        ],
        # fmt: on
    )
    assert result.exit_code != 0
    assert "median" in result.output
    assert "mean" in result.output and "max" in result.output


def test_convert_downsample_method_is_documented_in_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--downsample-method" in result.output


# ---------- convert (--layout bf2raw opt-in) ----------


def test_convert_layout_bf2raw_writes_bundle(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["a", "b"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "x.ome.zarr"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--layout",
            "bf2raw",
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
        ["convert", "/tmp/x.lif", str(out), "--layout", "bogus"],
    )
    assert result.exit_code != 0
    assert "layout" in result.output.lower()


# ---------- convert (--lif-mosaic per-tile, ADR-0005) ----------


def test_convert_lif_mosaic_per_tile_writes_tile_substores(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    from tests.conftest import TileScene

    scene = TileScene(
        tiles=[
            {
                "field_x": 0,
                "field_y": 0,
                "pos_x_m": 0.04,
                "pos_y_m": 0.017,
                "pos_z_m": 0.0117,
            },
            {
                "field_x": 1,
                "field_y": 0,
                "pos_x_m": 0.0405,
                "pos_y_m": 0.017,
                "pos_z_m": 0.0117,
            },
        ],
        tile_yx=(32, 32),
    )
    reader = FakeReader(
        scenes=["Position 1"],
        dims="TCZYX",
        shape=(1, 1, 1, 32, 32),
        per_tile_scenes={0: scene},
    )
    patched_reader(reader, plugin="bioio-lif")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--pyramid-min-size",
            "8",
            "--lif-mosaic",
            "per-tile",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "Position_1" / "tile_X00Y00.ome.zarr" / "zarr.json").exists()
    assert (out / "Position_1" / "tile_X01Y00.ome.zarr" / "zarr.json").exists()


def test_convert_lif_mosaic_invalid_choice_rejected(
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
            "--lif-mosaic",
            "bogus",
        ],
    )
    assert result.exit_code != 0
    assert (
        "lif-mosaic" in result.output.lower() or "lif_mosaic" in result.output.lower()
    )


def test_convert_lif_mosaic_grid_stitch_writes_single_canvas(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """--lif-mosaic grid-stitch dispatches through convert() and produces one
    canvas per scene at <output>/<scene>.ome.zarr (regression for the CLI
    plumbing to api.convert; end-to-end pixel correctness is covered in
    tests/test_lif_grid_stitch.py)."""
    from tests.conftest import TileScene

    scene = TileScene(
        tiles=[
            {
                "field_x": 0,
                "field_y": 0,
                "pos_x_m": 0.04,
                "pos_y_m": 0.017,
                "pos_z_m": 0.01,
            },
            {
                "field_x": 1,
                "field_y": 0,
                "pos_x_m": 0.041,
                "pos_y_m": 0.017,
                "pos_z_m": 0.01,
            },
            {
                "field_x": 0,
                "field_y": 1,
                "pos_x_m": 0.04,
                "pos_y_m": 0.018,
                "pos_z_m": 0.01,
            },
            {
                "field_x": 1,
                "field_y": 1,
                "pos_x_m": 0.041,
                "pos_y_m": 0.018,
                "pos_z_m": 0.01,
            },
        ],
        tile_yx=(8, 8),
    )
    reader = FakeReader(
        scenes=["Position 1"],
        dims="TCZYX",
        shape=(1, 1, 1, 8, 8),
        per_tile_scenes={0: scene},
    )
    patched_reader(reader, plugin="bioio-lif")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--pyramid-min-size",
            "4",
            "--lif-mosaic",
            "grid-stitch",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "Position_1.ome.zarr" / "zarr.json").exists()
    # No scene-named subdirectory of tile sub-stores (that's the per-tile shape).
    assert not (out / "Position_1").exists()


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


def test_inspect_json_output_includes_size_bytes(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    patched_reader(reader)
    src = tmp_path / "input.lif"
    src.write_bytes(b"\x00" * 16)

    result = runner.invoke(app, ["inspect", str(src), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["size_bytes"] == 16
    assert parsed["size_human"] == "16 B"
    # The human-readable "Size:" line is text-only — JSON output shouldn't have it.
    assert "Size:" not in result.output


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


# ---------- --reader-kwarg (issue #79) ----------


def test_convert_reader_kwarg_forwards_to_api(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeatable --reader-kwarg parses into a dict[str, str] and reaches convert()."""
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--reader-kwarg",
            "metadata_path=/writable/metadata.json",
            "--reader-kwarg",
            "other=42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["reader_kwargs"] == {
        "metadata_path": "/writable/metadata.json",
        "other": "42",
    }


def test_convert_reader_kwarg_absent_forwards_none(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting --reader-kwarg forwards ``reader_kwargs=None`` (no kwarg)."""
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    out = tmp_path / "out"
    result = runner.invoke(app, ["convert", "/tmp/x.lif", str(out)])
    assert result.exit_code == 0, result.output
    assert captured["reader_kwargs"] is None


def test_convert_reader_kwarg_missing_equals_rejected(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """A malformed --reader-kwarg (no ``=``) fails with click.BadParameter."""
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--reader-kwarg",
            "malformed-no-equals",
        ],
    )
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_convert_reader_kwarg_duplicate_key_rejected(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """Duplicate --reader-kwarg keys are rejected (fail loud vs last-wins)."""
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/x.lif",
            str(out),
            "--reader-kwarg",
            "k=v1",
            "--reader-kwarg",
            "k=v2",
        ],
    )
    assert result.exit_code != 0
    assert "more than once" in result.output


# ---------- --plate (issue #82) ----------


def test_convert_plate_option_threads_into_reader_kwargs(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--plate NAME` reaches convert() as `reader_kwargs={'plate': NAME}`."""
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/multi.lif",
            str(out),
            "--plate",
            "PlateB",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["reader_kwargs"] == {"plate": "PlateB"}


def test_convert_plate_option_merges_with_other_reader_kwargs(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--plate` merges alongside unrelated `--reader-kwarg` values."""
    captured: dict = {}

    def _fake_convert(**kwargs):
        captured.update(kwargs)
        return {"layout": "per-scene", "stores": []}

    monkeypatch.setattr(api_module, "convert", _fake_convert)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/multi.lif",
            str(out),
            "--plate",
            "PlateB",
            "--reader-kwarg",
            "other=v",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["reader_kwargs"] == {"other": "v", "plate": "PlateB"}


def test_convert_plate_option_conflicts_with_explicit_reader_kwarg(
    tmp_path: Path, runner: CliRunner, patched_reader
) -> None:
    """`--plate NAME` + `--reader-kwarg plate=OTHER` is a user error, not last-wins."""
    reader = FakeReader(scenes=["s"], dims="TCYX", shape=(1, 1, 32, 32))
    patched_reader(reader)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "convert",
            "/tmp/multi.lif",
            str(out),
            "--plate",
            "A",
            "--reader-kwarg",
            "plate=B",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_inspect_reader_kwarg_forwards_to_api(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--reader-kwarg on `inspect` reaches api.inspect()."""
    captured: dict = {}

    def _fake_inspect(input_path, *, reader_kwargs=None):
        captured["reader_kwargs"] = reader_kwargs
        return {
            "input_path": str(input_path),
            "size_bytes": 0,
            "size_human": "0 B",
            "reader_plugin": {
                "name": "bioio-fake",
                "source": "builtin",
                "distribution": "bioio-fake",
                "match_score": 100,
            },
            "n_scenes": 0,
            "scenes": [],
        }

    monkeypatch.setattr(api_module, "inspect", _fake_inspect)

    result = runner.invoke(
        app,
        [
            "inspect",
            "/tmp/x.lif",
            "--reader-kwarg",
            "metadata_path=/writable/metadata.json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["reader_kwargs"] == {"metadata_path": "/writable/metadata.json"}
