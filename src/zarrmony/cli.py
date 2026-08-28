"""Command-line interface for zarrmony.

Two subcommands:

- ``zarrmony convert INPUT OUTPUT`` — convert a bioimage file to OME-Zarr v0.5.
- ``zarrmony inspect INPUT`` — print a scene summary without converting.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import click

from zarrmony import __version__
from zarrmony import api as zm_api
from zarrmony._storage import format_bytes, size_on_disk
from zarrmony.errors import (
    InputAccessError,
    OutputExistsError,
    PlateSelectionError,
    ReaderKwargError,
    UnsupportedFormatError,
)
from zarrmony.geometry import (
    CHUNK_TARGET_WARN_BYTES,
    DEFAULT_CHUNK_TARGET_BYTES,
    DEFAULT_COARSE_MAX_BYTES,
    DEFAULT_COARSE_MAX_LONG_AXIS,
    DEFAULT_DOWNSAMPLE_METHOD,
    DEFAULT_GEOMETRY,
    DEFAULT_ISOTROPY_TOLERANCE,
    DEFAULT_PYRAMID_MIN_SIZE,
    DEFAULT_SHARD_TARGET_BYTES,
    Geometry,
)


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


def _partial_size_line(named_human: str, files: dict[str, Any] | None) -> str:
    """A byte count that cannot be mistaken for the whole input (#116).

    A whole-slide VSI stats at 4.4 MB and converts 37 GB, which made the old
    one-number line read as a 30,000x compression ratio. When the reader
    reported a wider file set, lead with the number that was actually read and
    keep the named path's size in parentheses so the two are never confused.
    """
    if not files:
        return named_human
    noun = "file" if files["count"] == 1 else "files"
    return (
        f"{files['size_human']} across {files['count']} {noun} "
        f"(the named path alone is {named_human})"
    )


def _audit_input_files(result: dict[str, Any]) -> dict[str, Any] | None:
    """``input.files`` from whichever audit shape ``convert()`` returned.

    Per-scene returns ``{"stores": [audit, ...]}`` — every store's ``input``
    block is the same, so the first one answers for the run. bf2raw and plate
    return the single audit itself. ``None`` when the reader could not report
    a file set, or reported one no wider than the named path.
    """
    audit = result["stores"][0] if result.get("stores") else result
    if not isinstance(audit.get("input"), dict):
        return None
    if not audit["input"].get("size_is_partial"):
        return None
    return audit["input"].get("files")


def _parse_chunk_shape(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> tuple[int, ...] | None:
    if value is None:
        return None
    name = param.name.replace("_", "-") if param.name else "chunk-shape"
    try:
        return tuple(int(x.strip()) for x in value.split(","))
    except ValueError as e:
        raise click.BadParameter(
            f"{name} must be comma-separated ints (e.g. '1,1,1,512,512'); got {value!r}"
        ) from e


def _build_geometry(
    *,
    chunk_target_bytes: int | None,
    isotropy_tolerance: float | None,
    pyramid_min_size: int | None,
    coarse_max_bytes: int | None,
    coarse_max_long_axis: int | None,
    downsample_method: str | None,
    chunk_shape: tuple[int, ...] | None,
    shard_target_bytes: int | None,
    shard_shape: tuple[int, ...] | None,
) -> Geometry | None:
    """Fold the geometry flags into one :class:`Geometry`, or ``None``.

    ``None`` means "the user set no geometry flag" and lets ``convert()`` use
    :data:`~zarrmony.geometry.DEFAULT_GEOMETRY`. The CLI builds the object
    itself rather than passing the ``pyramid_min_size=`` / ``chunk_shape=``
    sugar through, because ADR-0010's ``resolve_geometry`` refuses the two
    spellings together and the newer flags (``--chunk-target-bytes``,
    ``--isotropy-tolerance``) have no sugar form — every geometry flag has to
    arrive by the same door.

    ``--chunk-shape`` bypasses the planner outright, so combining it with
    ``--chunk-target-bytes`` would silently ignore the target; that is an error
    here rather than a surprise discovered on disk.

    A large ``--chunk-target-bytes`` warns rather than fails. It is a supported
    choice — an archival store read in big sequential sweeps genuinely wants
    it — but its cost is paid by a viewer months later rather than by the
    conversion in front of the user, which is the kind of trade that deserves
    saying out loud once. See :data:`CHUNK_TARGET_WARN_BYTES`.

    The shard flags mirror the chunk pair exactly, including the exclusion, and
    default to off. Asking for shards also warns, for the opposite reason: the
    store gets cheaper to write and no harder to read *through a zarr library*,
    but a consumer that parses the codec chain itself refuses it outright.
    """
    for shape_flag, target_flag, shape, target in (
        ("--chunk-shape", "--chunk-target-bytes", chunk_shape, chunk_target_bytes),
        ("--shard-shape", "--shard-target-bytes", shard_shape, shard_target_bytes),
    ):
        if shape is not None and target is not None:
            raise click.BadParameter(
                f"{shape_flag} and {target_flag} are mutually exclusive; "
                f"{shape_flag} sets the shape directly, so the byte target the "
                f"planner would aim for is never consulted"
            )
    if shard_target_bytes is not None or shard_shape is not None:
        click.echo(
            "Warning: sharding is on. Chunks stay individually readable and "
            "every zarr-python 3 consumer — napari-ome-zarr, dask, plain "
            "__getitem__ — is unaffected, but a consumer that parses the codec "
            "chain itself sees 'sharding_indexed' where it expects 'bytes' and "
            "refuses the store. Lucida cannot read a sharded store today "
            "(ADR-0010, issue #117).",
            err=True,
        )
    if chunk_target_bytes is not None and chunk_target_bytes > CHUNK_TARGET_WARN_BYTES:
        click.echo(
            f"Warning: --chunk-target-bytes {chunk_target_bytes} is above "
            f"{CHUNK_TARGET_WARN_BYTES} ({CHUNK_TARGET_WARN_BYTES // 1024} KiB). "
            "Large chunks cut object count, but a viewer that budgets its "
            "resident tiles in bytes then holds far fewer of them at once — "
            "Lucida's 2D slice atlas drops from 121 resident chunks at the "
            "512 KiB default to 4 at 8 MiB. If object count is what you are "
            "after, --shard-target-bytes cuts it without touching the read "
            "unit. Prefer a large chunk only for archival stores read in big "
            "sequential sweeps.",
            err=True,
        )
    overrides: dict[str, Any] = {
        name: value
        for name, value in (
            ("chunk_target_bytes", chunk_target_bytes),
            ("isotropy_tolerance", isotropy_tolerance),
            ("pyramid_min_size", pyramid_min_size),
            ("coarse_max_bytes", coarse_max_bytes),
            ("coarse_max_long_axis", coarse_max_long_axis),
            ("downsample_method", downsample_method),
            ("chunk_shape", chunk_shape),
            ("shard_target_bytes", shard_target_bytes),
            ("shard_shape", shard_shape),
        )
        if value is not None
    }
    if not overrides:
        return None
    try:
        return replace(DEFAULT_GEOMETRY, **overrides)
    except ValueError as e:
        # Geometry validates at construction; surface it as a usage error
        # instead of a traceback.
        raise click.BadParameter(str(e)) from e


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
    help=(
        "Stop pyramid generation when the smaller of Y/X falls below this. "
        "Z never decides depth — a 3-plane stack would otherwise get no "
        "pyramid at all. A floor, not a cap: a volume still too large for a "
        "viewer to hold whole keeps halving past it until it reaches the "
        "--coarse-max-bytes / --coarse-max-long-axis bounds."
    ),
)
@click.option(
    "--isotropy-tolerance",
    type=float,
    default=None,
    show_default=f"{DEFAULT_ISOTROPY_TOLERANCE} (from the ADR-0010 geometry policy)",
    help=(
        "How close to the finest axis's voxel spacing an axis must be to be "
        "halved at a pyramid level. Levels therefore move toward isotropy and "
        "the scarce axis (usually Z) is spent last. 1.0 halves only "
        "exactly-isotropic axes; a large value (e.g. 1e9) halves every spatial "
        "axis at every level."
    ),
)
@click.option(
    "--coarse-max-bytes",
    type=int,
    default=None,
    show_default=(
        f"{DEFAULT_COARSE_MAX_BYTES} "
        f"({DEFAULT_COARSE_MAX_BYTES // (1024 * 1024)} MiB, from the ADR-0010 "
        f"geometry policy)"
    ),
    help=(
        "Decoded-byte budget, per timepoint and channel, for the coarse level "
        "— the level small enough that a viewer can hold the whole volume as "
        "spatial context. The pyramid keeps halving until a level fits this "
        "and --coarse-max-long-axis, even past --pyramid-min-size; depth is "
        "the greater of the two rules, so no conversion loses a level. Raise "
        "it for a consumer with more memory per frame."
    ),
)
@click.option(
    "--coarse-max-long-axis",
    type=int,
    default=None,
    show_default=(
        f"{DEFAULT_COARSE_MAX_LONG_AXIS} (from the ADR-0010 geometry policy)"
    ),
    help=(
        "Longest lateral (Y/X) extent, in voxels, a coarse level may have. "
        "The second of the two coarse-level bounds; both must hold before the "
        "pyramid stops extending."
    ),
)
@click.option(
    "--downsample-method",
    type=click.Choice(["mean", "max"]),
    default=None,
    show_default=f"{DEFAULT_DOWNSAMPLE_METHOD} (from the ADR-0010 geometry policy)",
    help=(
        "Pooling kernel for every pyramid level above 0. 'mean' (default) is "
        "right for intensity imagery and is what the OME-Zarr ecosystem "
        "assumes. 'max' preserves the peak intensity of sparse labels that "
        "mean-pooling dissolves into the background — a 15 µm soma filling "
        "1.6% of a level-5 voxel reads 1000 against ~141 rather than 114 "
        "against 100 — at the cost of biasing every level above 0 high, which "
        "makes the pyramid unusable for measurement. Applied uniformly: "
        "mixing kernels by level would show as a brightness step in viewers "
        "with no coarse/detail concept."
    ),
)
@click.option(
    "--chunk-target-bytes",
    type=int,
    default=None,
    show_default=(
        f"{DEFAULT_CHUNK_TARGET_BYTES} "
        f"({DEFAULT_CHUNK_TARGET_BYTES // 1024} KiB, from the ADR-0010 geometry policy)"
    ),
    help=(
        "Raw (uncompressed) byte target for one chunk. The planner picks the "
        "largest power-of-two chunk that fits and is closest to cubic in "
        "micrometres, per level. Raise it for archival stores read in big "
        "sequential sweeps; lower it for latency-sensitive interactive "
        "viewing. Raising it warns: a viewer budgeting resident tiles in "
        "bytes holds far fewer large ones, and object count is better cut "
        "with --shard-target-bytes (ADR-0010, issue #113)."
    ),
)
@click.option(
    "--chunk-shape",
    callback=_parse_chunk_shape,
    default=None,
    metavar="T,C,Z,Y,X",
    help=(
        "Bypass the chunk planner with an explicit shape, comma-separated "
        "(e.g. '1,1,1,512,512'). Applied verbatim to every pyramid level; "
        "mutually exclusive with --chunk-target-bytes."
    ),
)
@click.option(
    "--shard-target-bytes",
    type=int,
    is_flag=False,
    flag_value=str(DEFAULT_SHARD_TARGET_BYTES),
    default=None,
    show_default="off (chunks are written as individual objects)",
    help=(
        "Pack chunks into shards of about this many raw bytes, so one storage "
        "object holds many chunks. Pass the flag bare for the recommended "
        f"{DEFAULT_SHARD_TARGET_BYTES // (1024 * 1024)} MiB. This cuts object "
        "count and speeds up writes without coarsening reads: the chunk stays "
        "the unit a viewer fetches and budgets by, and is range-read out of "
        "the shard. Off by default because a consumer that parses the codec "
        "chain rather than using a zarr library cannot open a sharded store — "
        "Lucida cannot today (ADR-0010, issue #117)."
    ),
)
@click.option(
    "--shard-shape",
    callback=_parse_chunk_shape,
    default=None,
    metavar="T,C,Z,Y,X",
    help=(
        "Bypass the shard planner with an explicit shape, comma-separated "
        "(e.g. '1,1,1,2048,2048'). Must be a whole multiple of each level's "
        "chunk shape on every axis; mutually exclusive with "
        "--shard-target-bytes, and enables sharding on its own."
    ),
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
    help=(
        "Include SHA256 of the input in the audit attrs (slower). "
        "`input.sha256` always covers the path you named. When the reader can "
        "report the wider file set it read — a .vsi is an index beside its "
        ".ets tiles — `input.files.sha256` covers the whole set, so hashing "
        "cost tracks the pixel data rather than the index."
    ),
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
        "omero.channels[i].window.start/end. Read back off the store after "
        "the pyramid is written, so it costs no raw-pixel read. Pass -1 (or "
        "use --no-contrast) to disable and keep the dtype-range placeholder "
        "from issue #50."
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
        "SmartSPIM exports on a read-only share. The built-in 'bioio' "
        "catch-all participates too, forwarding to whichever bioio backend "
        "wins discovery — e.g. '--reader-kwarg dask_tiles=true' to keep a "
        "gigapixel slide from arriving as one whole-plane dask chunk. Leave "
        "'tile_size' off with it: zarrmony derives one that matches the "
        "planned write grid, and a pinned tile that does not divide that grid "
        "makes every write split a source tile (those two keys are coerced "
        "from their string form; every other key is passed through as a "
        "string). Unknown kwargs surface as the reader constructor's native "
        "TypeError — zarrmony does not validate the shape."
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
    isotropy_tolerance: float | None,
    coarse_max_bytes: int | None,
    coarse_max_long_axis: int | None,
    downsample_method: str | None,
    chunk_target_bytes: int | None,
    chunk_shape: tuple[int, ...] | None,
    shard_target_bytes: int | None,
    shard_shape: tuple[int, ...] | None,
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

    geometry = _build_geometry(
        chunk_target_bytes=chunk_target_bytes,
        isotropy_tolerance=isotropy_tolerance,
        pyramid_min_size=pyramid_min_size,
        coarse_max_bytes=coarse_max_bytes,
        coarse_max_long_axis=coarse_max_long_axis,
        downsample_method=downsample_method,
        chunk_shape=chunk_shape,
        shard_target_bytes=shard_target_bytes,
        shard_shape=shard_shape,
    )

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
            geometry=geometry,
            contrast_percentile=resolved_contrast,
            force=force,
            checksum=checksum,
            validate=validate,
            lif_mosaic=lif_mosaic,
            reader_kwargs=reader_kwargs,
        )
    except (
        InputAccessError,
        OutputExistsError,
        PlateSelectionError,
        ReaderKwargError,
        UnsupportedFormatError,
    ) as e:
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

    input_human = format_bytes(size_on_disk(input_path))
    input_line = _partial_size_line(input_human, _audit_input_files(result))
    click.echo(f"Input:  {input_line}", err=True)
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
        "SmartSPIM exports on a read-only share. The built-in 'bioio' "
        "catch-all participates too, forwarding to whichever bioio backend "
        "wins discovery — e.g. '--reader-kwarg dask_tiles=true' to keep a "
        "gigapixel slide from arriving as one whole-plane dask chunk. Leave "
        "'tile_size' off with it: zarrmony derives one that matches the "
        "planned write grid, and a pinned tile that does not divide that grid "
        "makes every write split a source tile (those two keys are coerced "
        "from their string form; every other key is passed through as a "
        "string). Unknown kwargs surface as the reader constructor's native "
        "TypeError — zarrmony does not validate the shape."
    ),
)
def inspect_cmd(
    input_path: str,
    as_json: bool,
    reader_kwargs: dict[str, str] | None,
) -> None:
    """Print scenes, dims, channels, and pixel sizes for INPUT (no conversion)."""
    try:
        info = zm_api.inspect(input_path, reader_kwargs=reader_kwargs)
    except (
        InputAccessError,
        PlateSelectionError,
        ReaderKwargError,
        UnsupportedFormatError,
    ) as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        click.echo(json.dumps(info, indent=2, default=str))
        return

    rp = info["reader_plugin"]
    plugin_str = rp["distribution"] or rp["name"]
    click.echo(f"Input:  {info['input_path']}")
    files = info.get("files") if info.get("size_is_partial") else None
    click.echo(f"Size:   {_partial_size_line(info['size_human'], files)}")
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
