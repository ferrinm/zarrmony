"""Command-line interface for zarrmony.

Three subcommands:

- ``zarrmony convert INPUT OUTPUT`` — convert a bioimage file to OME-Zarr v0.5.
- ``zarrmony inspect INPUT`` — print a scene summary without converting.
- ``zarrmony schema dump`` — emit JSON Schema for the user-metadata model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from zarrmony import __version__
from zarrmony import api as zm_api
from zarrmony.errors import MetadataValidationError, OutputExistsError
from zarrmony.metadata.schema import export_schema_json


@click.group(name="zarrmony")
@click.version_option(__version__, prog_name="zarrmony")
def app() -> None:
    """Convert any bioimage file to OME-Zarr v0.5, preserving metadata."""


def _load_json(path: str | None) -> Any:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _parse_chunk_shape(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(x.strip()) for x in value.split(","))
    except ValueError as e:
        raise click.BadParameter(
            f"chunk-shape must be comma-separated ints (e.g. '1,1,1,512,512'); got {value!r}"
        ) from e


@app.command(name="convert")
@click.argument("input_path", metavar="INPUT", type=str)
@click.argument("output", metavar="OUTPUT", type=str)
@click.option(
    "--metadata-file",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON file with user-supplied metadata (matching the UserMetadata schema).",
)
@click.option(
    "--per-scene-metadata",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON file mapping scene_name → metadata dict for per-scene overrides.",
)
@click.option(
    "--layout",
    type=click.Choice(["per-scene", "bf2raw"]),
    default="per-scene",
    show_default=True,
    help=(
        "Output shape. 'per-scene' (default) writes one self-describing "
        "<scene>.ome.zarr store per scene under OUTPUT. 'bf2raw' writes a "
        "single bioformats2raw.layout bundle with numbered subgroups at OUTPUT."
    ),
)
@click.option(
    "--pyramid-min-size",
    type=int,
    default=256,
    show_default=True,
    help="Stop pyramid generation when the smallest spatial dim falls below this.",
)
@click.option(
    "--chunk-shape",
    callback=_parse_chunk_shape,
    default=None,
    metavar="T,C,Z,Y,X",
    help="Override auto chunk shape, comma-separated (e.g. '1,1,1,512,512').",
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "Overwrite output if it already exists. In per-scene mode this is "
        "checked per store, so existing sibling stores under OUTPUT are left "
        "alone unless they collide with a scene being written."
    ),
)
@click.option(
    "--permissive",
    is_flag=True,
    help="Skip the metadata compliance gate (for prototyping).",
)
@click.option(
    "--checksum",
    is_flag=True,
    help="Include SHA256 of the input file in the audit attrs (slower).",
)
def convert_cmd(
    input_path: str,
    output: str,
    metadata_file: str | None,
    per_scene_metadata: str | None,
    layout: str,
    pyramid_min_size: int,
    chunk_shape: tuple[int, ...] | None,
    force: bool,
    permissive: bool,
    checksum: bool,
) -> None:
    """Convert INPUT (a bioimage file) to OME-Zarr v0.5 at OUTPUT.

    By default OUTPUT is treated as a directory and one self-describing
    ``<scene>.ome.zarr`` store is written per scene. Pass ``--layout bf2raw``
    to instead write a single bioformats2raw.layout bundle at OUTPUT.
    """
    metadata = _load_json(metadata_file)
    per_scene = _load_json(per_scene_metadata)

    try:
        result = zm_api.convert(
            input_path=input_path,
            output=output,
            layout=layout,
            metadata=metadata,
            per_scene_metadata=per_scene,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            force=force,
            permissive=permissive,
            checksum=checksum,
        )
    except MetadataValidationError as e:
        raise click.ClickException(f"Metadata validation failed:\n{e}") from e
    except OutputExistsError as e:
        raise click.ClickException(str(e)) from e

    if layout == "per-scene":
        n = len(result["stores"])
        noun = "store" if n == 1 else "stores"
        click.echo(f"Wrote {n} {noun} to {output}", err=True)
    else:
        n = len(result["per_scene"])
        noun = "scene" if n == 1 else "scenes"
        click.echo(f"Wrote {n} {noun} to {output} (bf2raw bundle)", err=True)


@app.command(name="inspect")
@click.argument("input_path", metavar="INPUT", type=str)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON instead of human-readable text.",
)
def inspect_cmd(input_path: str, as_json: bool) -> None:
    """Print scenes, dims, channels, and pixel sizes for INPUT (no conversion)."""
    info = zm_api.inspect(input_path)
    if as_json:
        click.echo(json.dumps(info, indent=2, default=str))
        return

    click.echo(f"Input:  {info['input_path']}")
    click.echo(f"Plugin: {info['plugin']}")
    click.echo(f"Scenes: {info['n_scenes']}")
    for s in info["scenes"]:
        dims_str = "".join(s["dims"])
        shape_str = "x".join(str(x) for x in s["shape"])
        chans = ", ".join(s["channel_names"]) if s["channel_names"] else "(none)"
        px = s["physical_pixel_sizes"]
        if px is None:
            px_str = "(no physical pixel sizes)"
        else:
            px_str = f"Z={px['Z']} Y={px['Y']} X={px['X']}"
        click.echo(f"  [{s['index']}] {s['name']}")
        click.echo(f"      dims={dims_str} shape={shape_str} dtype={s['dtype']}")
        click.echo(f"      channels: {chans}")
        click.echo(f"      pixel sizes: {px_str}")


@app.group(name="schema")
def schema_group() -> None:
    """Inspect or export the user-metadata schema."""


@schema_group.command(name="dump")
def schema_dump_cmd() -> None:
    """Emit the JSON Schema for the user-supplied metadata model.

    Pipe to a file or to a tool like ``yq`` for YAML conversion.
    """
    click.echo(export_schema_json())
