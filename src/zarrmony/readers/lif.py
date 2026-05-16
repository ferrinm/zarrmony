"""LIF reader plugin.

``BioImage(path)`` for a LIF file only exposes the first scene by default; the
format-specific ``bioio_lif.Reader`` exposes ``.scenes`` for full iteration.

LIF files acquired with mosaic tiling expose an ``M`` (mosaic-tile) dimension
on ``xarray_dask_data``. The OME-Zarr 0.5 axes spec only permits
``{T, C, Z, Y, X}``, so ``M`` would be rejected by the writer. ``bioio_lif``
exposes a stitched view of the same scene as ``mosaic_xarray_dask_data`` (a
delayed dask-backed xarray with M collapsed); we wrap the Reader in a small
proxy that swaps to that view whenever the current scene has an ``M`` dim.
This mirrors the CZI plugin's approach, which selects the auto-stitching
``pylibczirw`` backend for the same reason. Tile-level information (positions,
per-tile dims) is preserved verbatim in ``OME/source/raw.lif.xml``.

Exposed as ``lif_plugin`` and registered in ``readers/__init__.py`` at zarrmony
import time.
"""

from pathlib import Path
from typing import Any

from bioio_lif import Reader

from zarrmony.readers.plugin import ReaderPlugin


class _MosaicAwareLifReader:
    """Proxy that swaps ``xarray_dask_data`` → ``mosaic_xarray_dask_data``
    when the current scene has an ``M`` dim. Forwards everything else to the
    wrapped ``bioio_lif.Reader``.
    """

    def __init__(self, inner: Reader) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def xarray_dask_data(self) -> Any:
        xarr = self._inner.xarray_dask_data
        if "M" in xarr.dims:
            return self._inner.mosaic_xarray_dask_data
        return xarr

    @property
    def mosaic_summary(self) -> dict | None:
        """Audit hook — None unless the current scene was mosaic-stitched."""
        xarr = self._inner.xarray_dask_data
        if "M" not in xarr.dims:
            return None
        tile_dims = self._inner.mosaic_tile_dims
        return {
            "stitched": True,
            "tile_count": int(xarr.sizes["M"]),
            "tile_shape": {
                "Y": int(tile_dims.Y),
                "X": int(tile_dims.X),
            },
        }


def _match_lif(path: Path) -> int | None:
    return 100 if path.suffix.lower() == ".lif" else None


def _open_lif(path: Path) -> Any:
    return _MosaicAwareLifReader(Reader(str(path)))


lif_plugin = ReaderPlugin(
    name="bioio-lif",
    match=_match_lif,
    open=_open_lif,
    distribution="bioio-lif",
    source="builtin",
)
