"""bioformats2raw.layout wrapper writer.

After per-scene OME-Zarr images have been written into numbered subgroups
(``0/``, ``1/``, ...) by ``write_scene``, this module adds the wrapping
metadata: the top-level ``zarr.json`` flag, the ``OME/`` subgroup with its
``series`` attribute, the combined ``OME/METADATA.ome.xml``, and optionally a
``OME/source/raw.<ext>.xml`` carrying the raw vendor metadata.

Uses fsspec for file writes so local paths and remote URIs (gs://, s3://) are
handled by the same code path; the user just needs the appropriate optional
extra installed (``zarrmony[gcs]`` or ``zarrmony[s3]``).
"""

from pathlib import Path

import fsspec

from zarrmony._storage import open_root_group

NGFF_VERSION = "0.5"
BF2RAW_LAYOUT_VERSION = 3


def _normalize_store_path(store_path: str | Path) -> str:
    """Normalize to a string URI/path suitable for fsspec."""
    if isinstance(store_path, Path):
        return str(store_path)
    return store_path


def _write_text(uri: str, content: str) -> None:
    """Write text to a local path or remote URI via fsspec, creating parents."""
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "w") as f:
        f.write(content)


def write_bf2raw_wrapper(
    store_path: str | Path,
    series_paths: list[str],
    ome_xml: str,
    source_xml: str | None = None,
    source_xml_filename: str | None = None,
) -> None:
    """Write the ``bioformats2raw.layout`` wrapper around already-written scenes.

    Parameters
    ----------
    store_path
        Root of the OME-Zarr store. Per-scene images must already exist at
        ``<store_path>/<series_paths[i]>/`` for each i.
    series_paths
        Relative paths to the per-scene image groups, in the order that matches
        the Image elements in ``ome_xml``. Typical: ``["0", "1", "2", ...]``.
    ome_xml
        Combined OME-XML document, produced by ``build_combined_ome_xml``.
    source_xml
        Optional raw vendor XML (e.g. the .lif XML, the .czi XML).
    source_xml_filename
        Filename for the raw vendor XML (e.g. ``"raw.lif.xml"``). Required if
        ``source_xml`` is provided.
    """
    store_str = _normalize_store_path(store_path)

    root = open_root_group(store_str, mode="a")
    root.attrs["ome"] = {
        "version": NGFF_VERSION,
        "bioformats2raw.layout": BF2RAW_LAYOUT_VERSION,
    }

    ome_group = root.require_group("OME")
    ome_group.attrs["ome"] = {
        "version": NGFF_VERSION,
        "series": list(series_paths),
    }

    _write_text(f"{store_str.rstrip('/')}/OME/METADATA.ome.xml", ome_xml)

    if source_xml is not None:
        if source_xml_filename is None:
            raise ValueError(
                "source_xml_filename is required when source_xml is provided"
            )
        _write_text(
            f"{store_str.rstrip('/')}/OME/source/{source_xml_filename}",
            source_xml,
        )
