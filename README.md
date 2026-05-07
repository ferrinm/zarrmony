# zarrmony

Convert any bioimage file to OME-Zarr v0.5, preserving metadata.

Zarrmony reads proprietary microscopy formats (CZI, LIF, ND2, OME-TIFF, ...) via [bioio](https://bioio-devs.github.io/bioio/) and writes them as OME-Zarr v0.5 in the [`bioformats2raw.layout`](https://ngff.openmicroscopy.org/0.5/#bf2raw) shape, with a configurable user-metadata gate, mean-pool pyramid generation, and a full audit trail of the conversion.

> **Status:** v0.1 in active development. API and metadata schema are not yet stable.

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
zarrmony convert input.czi output.ome.zarr --metadata-file metadata.json
zarrmony inspect input.czi
zarrmony schema dump > zarrmony-metadata.schema.json
```

### Library

```python
from zarrmony import convert, UserMetadata

audit = convert(
    "input.lif",
    "output.ome.zarr",
    metadata=UserMetadata(...),
)
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
