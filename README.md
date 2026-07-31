# zarrmony

[![PyPI version](https://img.shields.io/pypi/v/zarrmony.svg)](https://pypi.org/project/zarrmony/)
[![Python versions](https://img.shields.io/pypi/pyversions/zarrmony.svg)](https://pypi.org/project/zarrmony/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)
[![CI](https://github.com/ferrinm/zarrmony/actions/workflows/ci.yml/badge.svg)](https://github.com/ferrinm/zarrmony/actions/workflows/ci.yml)

Convert any bioimage file to OME-Zarr v0.5, preserving metadata.

Zarrmony reads proprietary microscopy formats (CZI, LIF, ND2, OME-TIFF, ...) via [bioio](https://bioio-devs.github.io/bioio/) and writes them as OME-Zarr v0.5, with mean-pool pyramid generation and a full audit trail of the conversion. User-supplied metadata (study/treatment/etc.) is **not** handled by zarrmony — it is owned by [aperture-backend](https://github.com/calicolabs/aperture-backend), which associates OME-Zarr stores to a separate metadata database.

By default (`--layout auto`) the writer is chosen from the reader's `layout_hint`: a flat reader writes one self-describing `<scene>.ome.zarr` store per scene under the output directory; a plate-shaped reader writes a single OME-NGFF [HCS plate](https://ngff.openmicroscopy.org/0.5/#hcs-layout) store at the output. The legacy bundled [`bioformats2raw.layout`](https://ngff.openmicroscopy.org/0.5/#bf2raw) shape is opt-in via `--layout bf2raw` (CLI) or `layout="bf2raw"` (library).

> **Status:** v0.9 in active development. API and metadata schema are not yet stable.

## Install

```bash
pip install zarrmony
```

Optional extras:

| Extra      | Adds                     | When you need it               |
| ---------- | ------------------------ | ------------------------------ |
| `gcs`      | `gcsfs`                  | Writing output to `gs://` URIs |
| `s3`       | `s3fs`                   | Writing output to `s3://` URIs |
| `ome-tiff` | `bioio-ome-tiff`         | Reading OME-TIFF input         |
| `all`      | All of the above         |                                |
| `dev`      | pytest, ruff, pre-commit | Contributing                   |

CZI, LIF, and ND2 reader plugins are included by default.

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

## Extending zarrmony

Add support for a new bioimage format by writing a reader plugin. See
[**Writing a zarrmony reader plugin**](./docs/writing-a-reader-plugin.md)
for the Reader Protocol, matcher conventions, entry-point registration,
and a worked example.

## License

Apache-2.0. See [LICENSE](./LICENSE).
