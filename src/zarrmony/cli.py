"""Command-line interface for zarrmony.

Two subcommands:

- ``zarrmony convert INPUT OUTPUT`` — convert a bioimage file to OME-Zarr v0.5.
- ``zarrmony inspect INPUT`` — print a scene summary without converting.
"""

from __future__ import annotations

import json
from typing import Any

import click

from zarrmony import __version__
from zarrmony import api as zm_api
from zarrmony._storage import format_bytes, size_on_disk
from zarrmony.errors import OutputExistsError, PlateSelectionError
from zarrmony.geometry import DEFAULT_PYRAMID_MIN_SIZE


@click.group(name="zarrmony")
@click.version_option(__version__, prog_name="zarrmony")
def app() -> None:
    """Convert any bioimage file to OME-Zarr v0.5, preserving metadata."""


def _format_plate_summary(plate: dict[str, Any]) -> str:
    """One-liner summary of an inspect() plate_layout block."""
    name = plate.get("name") or "(unnamed)"
    n_wells_imaged = len(plate.get("wells", []))
    n_total = len(plate.get("rows", [])) * len(plate.get("columns", []))
    fields_per_well = plate.get("field_count", 0)
    n_acquisitions = len(plate.get("acquisitions", []))
    well_word = "well" if n_wells_imaged == 1 else "wells"
    field_word = "field" if fields_per_well == 1 else "fields"
    acq_word = "acquisition" if n_acquisitions == 1 else "acquisitions"
    return (
        f'Plate: "{name}" — {n_wells_imaged}/{n_total} {well_word} imaged, '
        f"{fields_per_well} {field_word} per well, "
        f"{n_acquisitions} {acq_word}"
    )


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


def _parse_reader_kwargs(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> dict[str, str] | None:
    """Parse repeatable ``--reader-kwarg KEY=VALUE`` into a ``dict[str, str]``.

    Returns ``None`` when the flag was not passed so the API sees the same
    "no kwargs" signal as an in-process caller. Values stay strings — readers
    coerce internally (e.g. ``SmartSpimReader`` casts ``metadata_path`` to a
    ``Path``). Duplicate keys raise :class:`click.BadParameter` so users don't
    silently lose a kwarg to last-wins merging; unknown kwargs are left to
    the reader constructor's native ``TypeError``.
    """
    if not value:
        return None
    parsed: dict[str, str] = {}
    for item in value:
        if "=" not in item:
            raise click.BadParameter(
                f"expected KEY=VALUE (got {item!r}); pass e.g. "
                f"'--reader-kwarg metadata_path=/path/to/sidecar.json'"
            )
        key, _, val = item.partition("=")
        if not key:
            raise click.BadParameter(
                f"reader-kwarg key must be non-empty (got {item!r})"
            )
        if key in parsed:
            raise click.BadParameter(f"reader-kwarg {key!r} was passed more than once")
        parsed[key] = val
    return parsed


@app.command(name="convert")
@click.argument("input_path", metavar="INPUT", type=str)
@click.argument("output", metavar="OUTPUT", type=str)
@click.option(
    "--layout",
    type=click.Choice(["auto", "per-scene", "bf2raw", "plate"]),
    default="auto",
    show_default=True,
    help=(
        "Output shape. 'auto' (default) picks the writer from the reader's "
        "layout_hint: flat readers write per-scene, plate-shaped readers write "
        "a plate. 'per-scene' writes one self-describing <scene>.ome.zarr per "
        "scene under OUTPUT. 'bf2raw' writes a single bioformats2raw.layout "
        "bundle with numbered subgroups at OUTPUT. 'plate' writes an OME-NGFF "
        "HCS plate store at OUTPUT (requires a plate-shaped reader)."
    ),
)
@click.option(
    "--pyramid-min-size",
    type=int,
    default=None,
    show_default=f"{DEFAULT_PYRAMID_MIN_SIZE} (from the ADR-0010 geometry policy)",
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
    "--checksum",
    is_flag=True,
    help="Include SHA256 of the input file in the audit attrs (slower).",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    show_default=True,
    help=(
        "Run OME-NGFF v0.5 validation on the written store as a final step. "
        "Requires the `zarrmony[validate]` extra; if not installed, the "
        "validator is skipped with a warning. Failures are recorded in the "
        "audit (`attrs.zarrmony.validation_warnings`) but do not delete the "
        "output."
    ),
)
@click.option(
    "--contrast-percentile",
    type=float,
    default=99.9,
    show_default=True,
    help=(
        "Data-driven omero display window: per-channel (min, Nth percentile) "
        "computed off the coarsest pyramid level and written into "
        "omero.channels[i].window.start/end. Fused into the pyramid write so "
        "raw data is read once. Pass -1 (or use --no-contrast) to disable and "
        "keep the dtype-range placeholder from issue #50."
    ),
)
@click.option(
    "--no-contrast",
    "no_contrast",
    is_flag=True,
    help="Skip percentile-based contrast; leave window.start/end at the dtype range.",
)
@click.option(
    "--lif-mosaic",
    type=click.Choice(
        ["auto-stitch", "per-tile", "grid-stitch", "stage-stitch", "bioio-lif"]
    ),
    default="auto-stitch",
    show_default=True,
    help=(
        "LIF-specific. How to write mosaic scenes with no vendor _Merged "
        "sibling. 'auto-stitch' (default) runs a per-scene cascade: "
        "stage-stitch when the scene has per-tile PosX/PosY and physical "
        "pixel size (both X and Y), else grid-stitch when tile FieldX/FieldY "
        "form a complete rectangular grid, else bioio-lif's built-in "
        "M-scan-order stitcher (which emits MosaicStitchingWarning). "
        "'stage-stitch' places each tile at its PosX/PosY stage µm position "
        "(converted via the scene's pixel size); honours the LIF-declared "
        "intended overlap; raises on missing PosX/PosY or scene pixel size "
        "and warns (MosaicPlacementWarning) when observed vs intended overlap "
        "diverges past 20%. 'grid-stitch' reassembles one canvas per scene by "
        "placing tile M=i at (field_y[i]*tile_H, field_x[i]*tile_W) from LIF "
        "FieldX/FieldY (butt joints, no overlap); raises on incomplete tile "
        "metadata. 'per-tile' writes one OME-Zarr per tile under "
        "<OUTPUT>/<scene>/tile_X{f:02d}Y{f:02d}.ome.zarr/ with stage positions "
        "in each tile's OME-XML <Plane> for external stitchers (ASHLAR, "
        "m2stitch, BigStitcher); incompatible with --layout plate. 'bioio-lif' "
        "opts back into bioio-lif's 1-pixel-overlap M-scan-order stitcher "
        "(the pre-v0.7.0 default); the cascade will pick this automatically "
        "when a scene has no <Tile> layout metadata. Other readers ignore this "
        "flag. See ADR-0005."
    ),
)
@click.option(
    "--reader-kwarg",
    "reader_kwargs",
    multiple=True,
    metavar="KEY=VALUE",
    callback=_parse_reader_kwargs,
    help=(
        "Reader-specific option forwarded to the winning plugin's open() "
        "as **kwargs. Repeatable. Values stay strings; the reader coerces "
        "internally. Motivating case: sidecar-elsewhere overrides like "
        "'--reader-kwarg metadata_path=/writable/metadata.json' for "
        "SmartSPIM exports on a read-only share. Unknown kwargs surface as "
        "the reader constructor's native TypeError — zarrmony does not "
        "validate the shape."
    ),
)
@click.option(
    "--plate",
    "plate",
    type=str,
    default=None,
    help=(
        "LIF-specific. Name of the plate template to convert on a multi-plate "
        "LIF (see `zarrmony inspect` for the available names). Required when "
        "the LIF holds ≥2 plate templates — one convert call produces one "
        "plate.zarr per ADR-0009, so run convert once per plate. Optional on "
        "single-plate LIFs; if passed, the NAME must match the plate's name. "
        "Threads through to the LIF reader as `plate=` — equivalent to "
        "'--reader-kwarg plate=NAME'; passing both is an error."
    ),
)
def convert_cmd(
    input_path: str,
    output: str,
    layout: str,
    pyramid_min_size: int | None,
    chunk_shape: tuple[int, ...] | None,
    contrast_percentile: float,
    no_contrast: bool,
    force: bool,
    checksum: bool,
    validate: bool,
    lif_mosaic: str,
    reader_kwargs: dict[str, str] | None,
    plate: str | None,
) -> None:
    """Convert INPUT (a bioimage file) to OME-Zarr v0.5 at OUTPUT.

    By default (``--layout auto``) the writer is picked from the reader's
    ``layout_hint``: flat readers write one self-describing
    ``<scene>.ome.zarr`` per scene under OUTPUT, plate-shaped readers write
    a single OME-NGFF HCS plate store at OUTPUT.
    """
    # --no-contrast wins over --contrast-percentile; either -1 sentinel or the
    # flag disables percentile-based contrast at the API boundary.
    resolved_contrast: float | None
    if no_contrast or contrast_percentile < 0:
        resolved_contrast = None
    else:
        resolved_contrast = contrast_percentile

    # --plate NAME is a convenience alias for --reader-kwarg plate=NAME. Merge
    # into reader_kwargs; refuse an overlap so the user isn't surprised by
    # last-wins semantics on a plate mismatch. Everything downstream reads
    # `reader_kwargs["plate"]`, keeping one code path for the passthrough.
    if plate is not None:
        if reader_kwargs and "plate" in reader_kwargs:
            raise click.BadParameter(
                "--plate and --reader-kwarg plate=... are mutually exclusive; "
                "pass one or the other"
            )
        reader_kwargs = {**(reader_kwargs or {}), "plate": plate}

    try:
        result = zm_api.convert(
            input_path=input_path,
            output=output,
            layout=layout,
            pyramid_min_size=pyramid_min_size,
            chunk_shape=chunk_shape,
            contrast_percentile=resolved_contrast,
            force=force,
            checksum=checksum,
            validate=validate,
            lif_mosaic=lif_mosaic,
            reader_kwargs=reader_kwargs,
        )
    except (OutputExistsError, PlateSelectionError) as e:
        raise click.ClickException(str(e)) from e

    # Dispatch on the *resolved* layout the API actually used (so --layout auto
    # reports what was written, not the user's input).
    resolved = result.get("layout")
    if resolved == "per-scene":
        n = len(result["stores"])
        noun = "store" if n == 1 else "stores"
        click.echo(f"Wrote {n} {noun} to {output} (per-scene)", err=True)
        output_bytes = sum(size_on_disk(s["store_path"]) for s in result["stores"])
    elif resolved == "plate":
        n = len(result["fields"])
        noun = "field" if n == 1 else "fields"
        click.echo(f"Wrote {n} {noun} to {output} (plate)", err=True)
        output_bytes = size_on_disk(output)
    else:
        n = len(result["per_scene"])
        noun = "scene" if n == 1 else "scenes"
        click.echo(f"Wrote {n} {noun} to {output} (bf2raw bundle)", err=True)
        output_bytes = size_on_disk(output)

    click.echo(f"Input:  {format_bytes(size_on_disk(input_path))}", err=True)
    click.echo(f"Output: {format_bytes(output_bytes)}", err=True)


@app.command(name="inspect")
@click.argument("input_path", metavar="INPUT", type=str)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON instead of human-readable text.",
)
@click.option(
    "--reader-kwarg",
    "reader_kwargs",
    multiple=True,
    metavar="KEY=VALUE",
    callback=_parse_reader_kwargs,
    help=(
        "Reader-specific option forwarded to the winning plugin's open() "
        "as **kwargs. Repeatable. Values stay strings; the reader coerces "
        "internally. Motivating case: sidecar-elsewhere overrides like "
        "'--reader-kwarg metadata_path=/writable/metadata.json' for "
        "SmartSPIM exports on a read-only share. Unknown kwargs surface as "
        "the reader constructor's native TypeError — zarrmony does not "
        "validate the shape."
    ),
)
def inspect_cmd(
    input_path: str,
    as_json: bool,
    reader_kwargs: dict[str, str] | None,
) -> None:
    """Print scenes, dims, channels, and pixel sizes for INPUT (no conversion)."""
    info = zm_api.inspect(input_path, reader_kwargs=reader_kwargs)
    if as_json:
        click.echo(json.dumps(info, indent=2, default=str))
        return

    rp = info["reader_plugin"]
    plugin_str = rp["distribution"] or rp["name"]
    click.echo(f"Input:  {info['input_path']}")
    click.echo(f"Size:   {info['size_human']}")
    click.echo(f"Plugin: {plugin_str}")
    if "plates" in info:
        names_str = ", ".join(repr(n) for n in info["plates"])
        click.echo(
            f"Plates: {len(info['plates'])} templates: {names_str} "
            f"— pass --plate NAME to convert one"
        )
    if "plate_layout" in info:
        click.echo(_format_plate_summary(info["plate_layout"]))
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
