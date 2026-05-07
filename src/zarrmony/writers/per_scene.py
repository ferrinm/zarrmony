"""Per-scene OME-Zarr metadata sidecars.

In per-scene mode each scene produces its own self-describing store at
``<output>/<sanitized_scene_name>.ome.zarr/``. The pixel pyramid is written by
``writers.scene.write_scene``; this module adds the per-store OME-XML metadata
file and the raw vendor XML sidecar so any one store can be moved or inspected
on its own without losing provenance.
"""

from __future__ import annotations

from pathlib import Path

import fsspec


def _normalize_store_path(store_path: str | Path) -> str:
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


def write_per_scene_metadata(
    store_path: str | Path,
    *,
    ome_xml: str,
    source_xml: str | None = None,
    source_xml_filename: str | None = None,
) -> None:
    """Write ``OME/METADATA.ome.xml`` (and optionally ``OME/source/<name>``)
    inside an already-written per-scene store.

    Parameters
    ----------
    store_path
        Root of the per-scene OME-Zarr store. The pyramid arrays must already
        have been written here by ``write_scene``.
    ome_xml
        Single-Image OME-XML document for this scene (see
        ``writers.ome_xml.build_ome_xml_for_scene``).
    source_xml
        Optional raw vendor XML (e.g. the .lif XML, the .czi XML). Duplicated
        across per-scene stores by design — each store stays self-describing.
    source_xml_filename
        Filename for the raw vendor XML (e.g. ``"raw.lif.xml"``). Required if
        ``source_xml`` is provided.
    """
    store_str = _normalize_store_path(store_path).rstrip("/")
    _write_text(f"{store_str}/OME/METADATA.ome.xml", ome_xml)
    if source_xml is not None:
        if source_xml_filename is None:
            raise ValueError("source_xml_filename is required when source_xml is provided")
        _write_text(f"{store_str}/OME/source/{source_xml_filename}", source_xml)
