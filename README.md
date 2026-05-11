# zarrmony

Convert any bioimage file to OME-Zarr v0.5, preserving metadata.

Zarrmony reads proprietary microscopy formats (CZI, LIF, ND2, OME-TIFF, ...) via [bioio](https://bioio-devs.github.io/bioio/) and writes them as OME-Zarr v0.5, with a configurable user-metadata gate, mean-pool pyramid generation, and a full audit trail of the conversion.

By default (`--layout auto`) the writer is chosen from the reader's `layout_hint`: a flat reader writes one self-describing `<scene>.ome.zarr` store per scene under the output directory; a plate-shaped reader writes a single OME-NGFF [HCS plate](https://ngff.openmicroscopy.org/0.5/#hcs-layout) store at the output. The legacy bundled [`bioformats2raw.layout`](https://ngff.openmicroscopy.org/0.5/#bf2raw) shape is opt-in via `--layout bf2raw` (CLI) or `layout="bf2raw"` (library).

> **Status:** v0.3 in active development. API and metadata schema are not yet stable.

## Install

```bash
pip install zarrmony
```

Optional extras:

| Extra      | Adds                     | When you need it               |
|------------|--------------------------|--------------------------------|
| `gcs`      | `gcsfs`                  | Writing output to `gs://` URIs |
| `s3`       | `s3fs`                   | Writing output to `s3://` URIs |
| `ome-tiff` | `bioio-ome-tiff`         | Reading OME-TIFF input         |
| `all`      | All of the above         |                                |
| `dev`      | pytest, ruff, pre-commit | Contributing                   |

CZI, LIF, and ND2 reader plugins are included by default.

## Usage

### CLI

```bash
# Per-scene (default): writes one <scene>.ome.zarr store per scene under OUTPUT.
zarrmony convert input.czi output_dir/ --metadata-file metadata.json

# Bundled bioformats2raw.layout (opt-in): writes a single store at OUTPUT.
zarrmony convert input.czi output.ome.zarr --layout bf2raw --metadata-file metadata.json

zarrmony inspect input.czi
zarrmony schema dump > zarrmony-metadata.schema.json
```

### Library

```python
from zarrmony import convert, UserMetadata

# Per-scene (default): returns {"input": ..., "stores": [<per-store audit>, ...]}.
result = convert(
    "input.lif",
    "output_dir/",
    metadata=UserMetadata(...),
)

# Bundled: returns the single bundle's audit dict.
audit = convert(
    "input.lif",
    "output.ome.zarr",
    layout="bf2raw",
    metadata=UserMetadata(...),
)
```

## Extending zarrmony

Add support for a new bioimage format by writing a reader plugin. See
[**Writing a zarrmony reader plugin**](./docs/writing-a-reader-plugin.md)
for the Reader Protocol, matcher conventions, entry-point registration,
and a worked example.

## License

Apache-2.0. See [LICENSE](./LICENSE).
