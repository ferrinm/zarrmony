"""Shared test fixtures and helpers."""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass

import dask.array as da
import numpy as np
import xarray as xr
from ome_types import OME
from ome_types.model import Image, Pixels, PixelType

from zarrmony.readers.plate import PlateLayout


@dataclass
class FakePhysicalPixelSizes:
    """Stand-in for bioio's PhysicalPixelSizes namedtuple."""

    Z: float | None = None
    Y: float = 1.0
    X: float = 1.0


class FakeReader:
    """Minimal bioio-like reader for testing without real proprietary files.

    Exposes the bioio-shaped surface used by zarrmony's writers and api:
    .scenes, .set_scene(), .xarray_dask_data, .physical_pixel_sizes,
    .channel_names, .metadata (raw vendor XML), .ome_metadata.
    """

    def __init__(
        self,
        scenes: Sequence[str],
        dims: str = "TCYX",
        shape: tuple[int, ...] = (1, 1, 32, 32),
        pixel_sizes: FakePhysicalPixelSizes | None = None,
        channel_names: Sequence[str] | None = None,
        raw_xml: str | None = "<root>fake source xml</root>",
        ome_metadata_fails: bool = False,
        layout_hint: str = "flat",
        plate_layout: PlateLayout | None = None,
    ) -> None:
        self.scenes = tuple(scenes)
        self._dims = dims
        self._shape = shape
        self._current_scene = 0
        self.physical_pixel_sizes = pixel_sizes or FakePhysicalPixelSizes()
        self._channel_names = list(channel_names) if channel_names is not None else []
        self._raw_xml = raw_xml
        self._ome_metadata_fails = ome_metadata_fails
        self.layout_hint = layout_hint
        self.plate_layout = plate_layout

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names

    def set_scene(self, idx: int | str) -> None:
        if isinstance(idx, str):
            idx = self.scenes.index(idx)
        self._current_scene = idx

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        # Fill with scene-index+1 so tests can assert "the right scene was read"
        arr = np.full(self._shape, fill_value=self._current_scene + 1, dtype=np.uint16)
        coords: dict[str, list[str]] = {}
        if "C" in self._dims and self._channel_names:
            coords["C"] = self._channel_names
        return xr.DataArray(da.from_array(arr), dims=list(self._dims), coords=coords)

    @property
    def metadata(self) -> ET.Element | None:
        if self._raw_xml is None:
            return None
        return ET.fromstring(self._raw_xml)

    @property
    def ome_metadata(self) -> OME:
        if self._ome_metadata_fails:
            raise RuntimeError("simulated ome_metadata failure")
        size_map = dict(zip(self._dims, self._shape, strict=True))
        scene_name = self.scenes[self._current_scene]
        return OME(
            images=[
                Image(
                    id=f"Image:{self._current_scene}",
                    name=scene_name,
                    pixels=Pixels(
                        id=f"Pixels:{self._current_scene}",
                        size_x=size_map.get("X", 1),
                        size_y=size_map.get("Y", 1),
                        size_z=size_map.get("Z", 1),
                        size_c=size_map.get("C", 1),
                        size_t=size_map.get("T", 1),
                        dimension_order="XYZCT",
                        type=PixelType.UINT16,
                    ),
                )
            ]
        )
