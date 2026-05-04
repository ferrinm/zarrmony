"""Synthesize small bioimage fixture files for integration tests.

LIF and CZI cannot be synthesized in pure Python (no Python writers exist for
those proprietary formats). OME-TIFF can, via tifffile, and exercises the same
default-reader path the production CZI/LIF readers use minus the format-specific
override.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile


def make_synth_ome_tiff(
    path: Path,
    *,
    n_scenes: int = 1,
    dims: str = "TCYX",
    shape: tuple[int, ...] = (1, 2, 64, 64),
    channel_names: Sequence[str] | None = None,
    pixel_size_um: float = 0.5,
) -> Path:
    """Write a synthetic OME-TIFF at ``path``.

    Each scene is filled with ``scene_index + 1`` so tests can assert which
    scene was read.

    With ``n_scenes > 1``, multiple OME-XML Image elements are written into a
    single TIFF file (tifffile's multi-series OME mode).
    """
    if channel_names is None:
        # Default to 2 channels if dims includes C and we don't have names
        c_idx = dims.find("C")
        n_c = shape[c_idx] if c_idx >= 0 else 0
        channel_names = [f"Ch{i}" for i in range(n_c)]

    if "Y" in dims and "X" in dims:
        physical_meta = {
            "PhysicalSizeX": pixel_size_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size_um,
            "PhysicalSizeYUnit": "µm",
        }
    else:
        physical_meta = {}

    with tifffile.TiffWriter(str(path), ome=True, bigtiff=False) as tif:
        for scene_idx in range(n_scenes):
            data = np.full(shape, fill_value=scene_idx + 1, dtype=np.uint16)
            metadata = {
                "axes": dims,
                "Name": f"scene_{scene_idx}",
                "Channel": {"Name": list(channel_names)} if channel_names else None,
                **physical_meta,
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            tif.write(data, metadata=metadata)

    return path
