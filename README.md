# zarrmony

[![PyPI version](https://img.shields.io/pypi/v/zarrmony.svg)](https://pypi.org/project/zarrmony/)
[![Python versions](https://img.shields.io/pypi/pyversions/zarrmony.svg)](https://pypi.org/project/zarrmony/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)
[![CI](https://github.com/ferrinm/zarrmony/actions/workflows/ci.yml/badge.svg)](https://github.com/ferrinm/zarrmony/actions/workflows/ci.yml)

Convert any bioimage file to OME-Zarr v0.5, preserving metadata.

Zarrmony reads proprietary microscopy formats (CZI, LIF, ND2, OME-TIFF, ...) via [bioio](https://bioio-devs.github.io/bioio/) and writes them as OME-Zarr v0.5, with mean-pool pyramid generation (`--downsample-method max` for sparse labels) and a full audit trail of the conversion. User-supplied metadata (study/treatment/etc.) is **not** handled by zarrmony — it is owned by [aperture-backend](https://github.com/calicolabs/aperture-backend), which associates OME-Zarr stores to a separate metadata database.

By default (`--layout auto`) the writer is chosen from the reader's `layout_hint`: a flat reader writes one self-describing `<scene>.ome.zarr` store per scene under the output directory; a plate-shaped reader writes a single OME-NGFF [HCS plate](https://ngff.openmicroscopy.org/0.5/#hcs-layout) store at the output. The legacy bundled [`bioformats2raw.layout`](https://ngff.openmicroscopy.org/0.5/#bf2raw) shape is opt-in via `--layout bf2raw` (CLI) or `layout="bf2raw"` (library).

> **Status:** v0.15 in active development. API and metadata schema are not yet stable.

## Install

```bash
pip install zarrmony
```

### Readers

Zarrmony dispatches to a reader plugin per input format. They come in three tiers:

- **Built-in** (bundled by default): CZI, LIF, ND2.
- **Optional extras** in this repo (opt-in via `pip install "zarrmony[<extra>]"`): OME-TIFF via the `ome-tiff` extra, and ~150 vendor formats via the [`bioformats` extra](#bio-formats-backed-vendor-formats) (GPL-3.0 — see below).
- **External plugins** (separate PyPI distributions, entry-point registered):
  - [`zarrmony-phenix`](https://github.com/ferrinm/zarrmony-phenix) — Opera Phenix (wraps `pyphenix.OperaPhenixReader`) — `pip install zarrmony-phenix`
  - [`zarrmony-blaze`](https://github.com/ferrinm/zarrmony-blaze) — Miltenyi UltraMicroscope Blaze (MACS iQ-processed) — `pip install zarrmony-blaze`
  - [`zarrmony-snouty`](https://github.com/ferrinm/zarrmony-snouty) — Snouty single-objective light-sheet — `pip install zarrmony-snouty`
  - [`zarrmony-smartspim`](https://github.com/ferrinm/zarrmony-smartspim) — LifeCanvas SmartSPIM stitched exports — `pip install zarrmony-smartspim`

### Extras

| Extra        | Adds                                     | When you need it                                  |
| ------------ | ---------------------------------------- | ------------------------------------------------- |
| `gcs`        | `gcsfs`                                  | Writing output to `gs://` URIs                    |
| `s3`         | `s3fs`                                   | Writing output to `s3://` URIs                    |
| `ome-tiff`   | `bioio-ome-tiff`                         | Reading OME-TIFF input                            |
| `validate`   | `ome-zarr-models`                        | Post-conversion OME-NGFF validation               |
| `bioformats` | `bioio-bioformats`                       | Reading Bio-Formats-only vendor formats (GPL-3.0) |
| `all`        | All of the above **except `bioformats`** |                                                   |
| `dev`        | pytest, ruff, pre-commit                 | Contributing                                      |

### Bio-Formats-backed vendor formats

```bash
pip install "zarrmony[bioformats]"
```

**What it buys.** Everything on the [Bio-Formats supported-formats list](https://bio-formats.readthedocs.io/en/stable/supported-formats.html) that no permissively-licensed bioio backend covers — around 150 formats. The motivating case is **Olympus/Evident cellSens VSI** whole-slide data; Zeiss ZVI and Hamamatsu NDPI are in the same bucket. No zarrmony code is involved: once `bioio-bioformats` is installed, the built-in `bioio` catch-all plugin dispatches to it, and the audit record's `distribution` field names it. Point zarrmony at the `.vsi` file itself — Bio-Formats follows the `.ets` sidecar directory automatically. The audit records that it did: `input.size_bytes` is the `.vsi` alone (a few MB of index), and `input.files` is the whole set Bio-Formats read, with `input.size_is_partial` saying which is which. `--checksum` covers both — the named path under `input.sha256`, the whole set under `input.files.sha256`.

**Licence.** `bioio-bioformats` is **GPL-3.0**; Bio-Formats is GPL. Installing this extra puts GPL code in your environment. Zarrmony itself remains Apache-2.0 and no GPL package is in its default dependency closure — which is exactly why this extra is opt-in and why it is **excluded from `all`**. Do not add it there. See [ADR-0011](docs/adr/0011-bioformats-backed-formats.md).

**Java.** Bio-Formats needs a JVM, but not one you have to install: `bffile` / `scyjava` / `cjdk` fetch their own JDK (~36 MiB, once) and the Bio-Formats maven artifacts on first use. No system Java, no maven, no `JAVA_HOME`. The first file you open is slow while that downloads; everything after is cached.

**Gigapixel inputs need tiling.** `bioio-bioformats` returns **one dask chunk per plane** by default. On a 141k × 168k slide that is a single 47.5 GB chunk, and the writer will try to hold it in memory to rechunk it. Pass:

```bash
zarrmony convert slide.vsi out/ --reader-kwarg dask_tiles=true
```

Leave `tile_size` off. Zarrmony plans the output geometry first and then asks the reader for tiles that fit it exactly, recording the choice in `config.reader_tile_size`. Pinning your own is supported and sometimes right, but a tile that does not divide the write grid makes every write split a source tile — on the reference slide that is 831,936 dask tasks against 369,600 — so the writer warns and names the tile that would have worked.

## Usage

### CLI

```bash
# Auto (default): dispatches on the reader's layout_hint.
#   flat readers (CZI, LIF, ND2, OME-TIFF) → per-scene stores under OUTPUT
#   plate-shaped readers (e.g. zarrmony-phenix) → a single HCS plate store at OUTPUT
zarrmony convert input.czi output_dir/

# Force per-scene (one <scene>.ome.zarr store per scene under OUTPUT).
zarrmony convert input.czi output_dir/ --layout per-scene

# Force HCS plate (one <plate>.ome.zarr store at OUTPUT). Requires a
# plate-shaped reader; flat readers raise LayoutMismatchError.
zarrmony convert phenix-acquisition/ output.ome.zarr --layout plate

# Bundled bioformats2raw.layout (opt-in): writes a single store at OUTPUT.
zarrmony convert input.czi output.ome.zarr --layout bf2raw

# LIF-specific: write one OME-Zarr per mosaic tile (with stage positions in
# <Plane>) instead of bioio-lif's auto-stitched 1-pixel-overlap output.
# See docs/adr/0005-lif-mosaic-write-strategy.md.
zarrmony convert mosaic.lif output_dir/ --lif-mosaic per-tile

zarrmony inspect input.czi
```

### Library

```python
from zarrmony import convert

# Auto (default): for a flat reader, returns {"input": ..., "stores": [...]};
# for a plate-shaped reader, returns the single plate audit dict (schema 3,
# with "fields" and a top-level "plate" block). Switch on audit["layout"].
result = convert("input.lif", "output_dir/")

# Bundled: returns the single bundle's audit dict.
audit = convert("input.lif", "output.ome.zarr", layout="bf2raw")

# HCS plate: writes one OME-NGFF plate store at OUTPUT.
audit = convert("phenix-acquisition/", "output.ome.zarr", layout="plate")
```

## Output geometry

Zarrmony plans an output store's **geometry** — how many pyramid levels there
are, what each level's extent is, and how each level is divided into chunks —
itself, rather than delegating chunk shape to `bioio-ome-zarr`'s memory-target
heuristic. Every constant below traces to a measurement;
[ADR-0010](./docs/adr/0010-output-geometry-policy.md) records which, and what
was rejected.

What the planner does by default:

- **Chunks are world-cubic and small.** Each level gets the largest
  power-of-two chunk whose raw size fits the 512 KiB target and whose extents
  are closest to cubic _in micrometres_, planned against that level's own voxel
  spacing. Near-isotropic uint16 data lands on the familiar `1,1,64,64,64`; a
  10:1 confocal stack (Z 5 µm, XY 0.5 µm) lands on `1,1,16,128,128` —
  80 × 64 × 64 µm — instead of a voxel-cubic 64³ spanning 320 × 32 × 32 µm. T
  and C are never chunked, so a viewer fetching one channel at one timepoint
  never pays for the others.
- **The pyramid moves toward isotropy.** A level halves every spatial axis
  whose physical spacing is within `isotropy_tolerance` of the finest
  still-halvable axis's, so the scarce axis — Z, for most volumetric light
  microscopy — is spent last. No axis halves below `axis_floor` voxels, and an
  axis already below it never halves: a 3-plane stack keeps its 3 planes at
  every level.
- **Depth is the greater of two rules.** The `pyramid_min_size` Y/X floor, and
  the depth at which a level becomes a **coarse level** — one a viewer can
  decode whole and use as spatial context, meaning `Z·Y·X·itemsize` per
  timepoint and channel is at most `coarse_max_bytes` _and_ the longest lateral
  axis is at most `coarse_max_long_axis`. Because depth is a `max()`, no
  conversion loses a level. A pyramid that bottoms out at the axis floor while
  still too large simply has no coarse level.
- **Levels above 0 are mean-pooled**, uniformly. `downsample_method="max"`
  switches the whole pyramid to max-pool for sparse-label acquisitions, where
  mean-pooling dissolves small objects into the background.
- **The same rules apply to per-scene, bf2raw and plate output**, with no
  `Z > 1` gate and no 2D exemption: every rule is written over the axes that
  are present, so a 2160² plate field is planned by exactly the rule a
  whole-brain volume is.

### Knobs

Every field lives on the frozen `zarrmony.Geometry` policy object, passed as
`convert(..., geometry=...)`. Most are also CLI flags on `zarrmony convert`.

| `Geometry` field       | CLI flag                 | Default             | What it sets                                                                                                                                                                              |
| ---------------------- | ------------------------ | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunk_target_bytes`   | `--chunk-target-bytes`   | `524288` (512 KiB)  | Raw (uncompressed) byte target for one chunk. Raise it for a bandwidth-bound consumer or to cut object count.                                                                             |
| `isotropy_tolerance`   | `--isotropy-tolerance`   | `1.5`               | How close to the finest still-halvable axis's spacing an axis must be to halve at a level. `1.0` halves only exactly-isotropic axes; a large value halves every spatial axis every level. |
| `axis_floor`           | _(library only)_         | `32`                | Minimum voxels on any axis. On Y/X it is capped by `pyramid_min_size`, so an explicitly lowered depth floor is not overridden by this default.                                            |
| `coarse_max_bytes`     | `--coarse-max-bytes`     | `67108864` (64 MiB) | Largest decoded size, per timepoint and channel, a coarse level may have.                                                                                                                 |
| `coarse_max_long_axis` | `--coarse-max-long-axis` | `2048`              | Longest lateral extent, in voxels, a coarse level may have. On single-plane data this is the bound that decides coarseness — the byte bound is inert.                                     |
| `downsample_method`    | `--downsample-method`    | `"mean"`            | Pooling kernel for every level above 0: `mean` or `max`.                                                                                                                                  |
| `pyramid_min_size`     | `--pyramid-min-size`     | `256`               | Stop halving when the smaller of Y/X would fall below this — a floor on depth, not a cap.                                                                                                 |
| `chunk_shape`          | `--chunk-shape`          | `None`              | Explicit chunk shape that bypasses the planner outright, so no byte target is consulted. `--chunk-shape` and `--chunk-target-bytes` are rejected together on the CLI.                     |
| `shard_target_bytes`   | `--shard-target-bytes`   | `None` (off)        | Raw byte target for one shard — the write unit and the storage object. Setting it turns sharding on; the bare flag resolves to 8 MiB. Must be at least `chunk_target_bytes`.              |
| `shard_shape`          | `--shard-shape`          | `None` (off)        | Explicit shard shape that bypasses the shard planner. Must be a whole multiple of the chunk on every axis. `--shard-shape` and `--shard-target-bytes` are rejected together on the CLI.   |

The two coarse-level bounds are the defaults of the viewer this output is
tuned for; they are fields rather than constants so a store can be planned for
a consumer with a different budget.

```bash
# Fewer objects, same read granularity: 512 KiB chunks packed into 8 MiB
# shards. Only for consumers that read sharded zarr v3 — see below.
zarrmony convert slide.vsi output_dir/ --shard-target-bytes

# Bigger chunks: fewer objects, coarser culling. For object storage where
# listing cost matters more than round-trip latency.
zarrmony convert input.czi output_dir/ --chunk-target-bytes 2097152

# Sparse labels: keep peak intensity in the pyramid.
zarrmony convert labels.czi output_dir/ --downsample-method max
```

```python
from zarrmony import Geometry, convert

audit = convert(
    "input.czi",
    "output_dir/",
    geometry=Geometry(chunk_target_bytes=2 * 1024 * 1024, downsample_method="max"),
)
```

`chunk_shape` and `pyramid_min_size` are also retained directly on `convert()`
as sugar that folds into a default policy, so callers written before the policy
object keep working. Passing `geometry=` together with either raises
`ValueError` rather than silently picking a winner.

The resolved policy is recorded in the audit under `config.geometry`, and what
it produced is recorded per scene / per field as `level_shapes`, `chunk_shapes`,
`shard_shapes` and `coarse_level_index` — so "does this store have a level a
viewer can hold whole?" is answerable from the store's own metadata.

### Object count, and sharding

Small chunks trade bytes-per-object for objects. A whole-brain light-sheet
store goes from 87,048 objects to ~3.2 M (~37×); a 2160² plate field goes from
4 to 39; a gigapixel slide scene reaches ~370k objects at level 0 alone. On
local disk that is irrelevant. On GCS/S3 it is listing time plus per-object
metadata cost, and at slide scale it is also conversion wall-clock — writing in
512 KiB units is what made one such scene project to nine days.

Sharding answers this without giving up read granularity, because the shard is
the write unit and the chunk is the read unit. `--shard-target-bytes` packs
whole chunks into 8 MiB storage objects: that slide scene's level 0 becomes
512² chunks inside 2048² shards, 369,600 objects down to 23,184, with each
512 KiB chunk still individually range-readable. Shards are planned by the same
world-cubic rule as chunks, per level, so an isotropic volume at the defaults
gets a `128 × 128 × 256` shard holding 16 chunks of 64³.

It is **off by default**, because it changes who can read the store. Chunks
stay individually readable and every zarr-python 3 consumer is unaffected —
napari-ome-zarr, dask, plain `__getitem__`, subsets straddling either grid, all
verified byte-identical against an unsharded store. But a consumer that parses
the codec chain itself sees `sharding_indexed` where it expects `bytes` and
refuses the store: `lucida-store` accepts only `[bytes]` or
`[bytes, compressor]`, so a sharded store fails there with
`first storage codec must be 'bytes', got 'sharding_indexed'`. The CLI warns
whenever sharding is on. See ADR-0010 for the measurements and the reversal.

## Extending zarrmony

Add support for a new bioimage format by writing a reader plugin. See
[**Writing a zarrmony reader plugin**](./docs/writing-a-reader-plugin.md)
for the Reader Protocol, matcher conventions, entry-point registration,
and a worked example.

## License

Apache-2.0. See [LICENSE](./LICENSE).
