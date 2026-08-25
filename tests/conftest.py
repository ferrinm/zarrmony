"""Shared test fixtures and helpers."""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass

import dask.array as da
import numpy as np
import xarray as xr
from ome_types import OME
from ome_types.model import Channel, Image, Pixels, PixelType

from zarrmony.readers.plate import PlateLayout


@dataclass
class FakePhysicalPixelSizes:
    """Stand-in for bioio's PhysicalPixelSizes namedtuple.

    Y/X allow ``None`` so stage-stitch tests can simulate a scene whose reader
    can't report a physical pixel size (bioio-lif surfaces None for some
    non-calibrated confocals).
    """

    Z: float | None = None
    Y: float | None = 1.0
    X: float | None = 1.0


@dataclass
class TileScene:
    """Describes a mosaic-reassembly-eligible scene for FakeReader.

    Sets the scene up to expose ``tiles_xarray_dask_data`` (M-intact),
    ``is_mosaic_reassembly_eligible() -> True``, and a LIF-shaped ``metadata``
    blob carrying the per-tile ``<Tile FieldX/FieldY/PosX/PosY/PosZ>`` entries
    the extractor will pick up. Tile data is filled with ``m + 1`` so tests
    can assert "the right tile was written" cheaply. Used by both the
    ``lif_mosaic="per-tile"`` and ``lif_mosaic="grid-stitch"`` paths.
    """

    tiles: list[dict]  # each: {field_x, field_y, pos_x_m, pos_y_m, pos_z_m}
    tile_yx: tuple[int, int] = (32, 32)
    intended_overlap_x_pct: float | None = 10.0
    intended_overlap_y_pct: float | None = 10.0
    channels: int = 1
    z: int = 1
    t: int = 1


def _fmt_tile_attr(key: str, value: object) -> str:
    """Format one ``<Tile>`` attribute, omitting it entirely when ``value`` is None.

    Lets stage-stitch tests build a fixture where a specific tile is missing
    ``PosX``/``PosY`` (to exercise the fail-loud path) without regressing the
    grid-stitch fixtures that assume every attribute is populated.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f' {key}="{value:.10f}"'
    return f' {key}="{value}"'


def build_lif_plate_metadata(
    plates: Sequence[dict],
) -> str:
    """LIF-shaped LMSDataContainer XML carrying one or more plate templates.

    Each plate dict is ``{"name": str, "rows": [str], "columns": [str]}`` —
    every (row, col) pair becomes one column-Element (one field). Row names
    are written verbatim so tests can exercise casing normalization; column
    names are written verbatim so tests can exercise zero-padding
    normalization at the extractor boundary.

    Analogous to :func:`_build_lif_tilescan_metadata` (for mosaic fixtures) —
    the LIF plate metadata module walks ``LMSDataContainerHeader`` looking for
    Elements whose ``Children/Element`` names look like plate rows/columns, so
    the fixture only needs that structural skeleton.
    """
    plate_blocks = []
    for plate in plates:
        row_blocks = []
        for row in plate["rows"]:
            col_blocks = "".join(
                f'<Element Name="{col}"><Data /></Element>' for col in plate["columns"]
            )
            row_blocks.append(
                f'<Element Name="{row}"><Children>{col_blocks}</Children></Element>'
            )
        rows_xml = "".join(row_blocks)
        plate_blocks.append(
            f'<Element Name="{plate["name"]}">'
            f"<Children>{rows_xml}</Children>"
            f"</Element>"
        )
    plates_xml = "".join(plate_blocks)
    return (
        "<LMSDataContainerHeader><Element><Children>"
        f"{plates_xml}"
        "</Children></Element></LMSDataContainerHeader>"
    )


def _build_lif_tilescan_metadata(scene: "TileScene") -> str:
    """LIF-shaped XML blob with one ``<Image>`` carrying tile + overlap entries.

    Matches the document shape ``find_scene_xml`` keys off: scenes live under
    ``.//Image``. The single ``<Image>`` here represents the current scene's
    settings (the FakeReader instances we test against expose this for the
    one and only scene index).
    """
    tile_xml = "".join(
        "<Tile"
        + _fmt_tile_attr("FieldX", t.get("field_x"))
        + _fmt_tile_attr("FieldY", t.get("field_y"))
        + _fmt_tile_attr("PosX", t.get("pos_x_m"))
        + _fmt_tile_attr("PosY", t.get("pos_y_m"))
        + _fmt_tile_attr("PosZ", t.get("pos_z_m"))
        + " />"
        for t in scene.tiles
    )
    overlap_attrs = []
    if scene.intended_overlap_x_pct is not None:
        overlap_attrs.append(
            f'OverlapPercentageX="{scene.intended_overlap_x_pct / 100.0:.4f}"'
        )
    if scene.intended_overlap_y_pct is not None:
        overlap_attrs.append(
            f'OverlapPercentageY="{scene.intended_overlap_y_pct / 100.0:.4f}"'
        )
    stitching_xml = (
        f"<StitchingSettings {' '.join(overlap_attrs)} />" if overlap_attrs else ""
    )
    return (
        "<LMSDataContainerHeader><Element><Children><Element><Data><Image>"
        f'<Attachment Name="TileScanInfo" Application="LAS AF">{tile_xml}</Attachment>'
        f'<Attachment Name="HardwareSetting">{stitching_xml}</Attachment>'
        "</Image></Data></Element></Children></Element></LMSDataContainerHeader>"
    )


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
        mosaic_summary: dict | None = None,
        skip_reasons: dict[int, str] | None = None,
        per_tile_scenes: dict[int, "TileScene"] | None = None,
        dtype: np.typing.DTypeLike = np.uint16,
        data: np.ndarray | None = None,
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
        self.mosaic_summary = mosaic_summary
        self._skip_reasons = skip_reasons or {}
        self._per_tile_scenes = per_tile_scenes or {}
        self._dtype = np.dtype(dtype)
        self._data = data

    @property
    def skip_reason(self) -> str | None:
        return self._skip_reasons.get(self._current_scene)

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names

    @property
    def dtype(self) -> np.dtype:
        """Mirrors bioio's ``Reader.dtype`` — the pixel dtype of the current scene.

        Held as a scalar attr so tests can vary it without allocating the full
        xarray. The ``xarray_dask_data`` property honors this dtype when
        materializing the fake pixel buffer.
        """
        return self._dtype

    def set_scene(self, idx: int | str) -> None:
        if isinstance(idx, str):
            idx = self.scenes.index(idx)
        self._current_scene = idx

    @property
    def current_scene_index(self) -> int:
        return self._current_scene

    def is_mosaic_reassembly_eligible(self) -> bool:
        """True iff the current scene was configured as a mosaic-reassembly source."""
        return self._current_scene in self._per_tile_scenes

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        if self._data is not None:
            # Real pixel content, shared by every scene — for tests about what
            # the writer does to *values* (e.g. which pooling kernel built a
            # pyramid level), where a constant fill would look identical
            # whatever happened.
            arr = self._data.astype(self._dtype, copy=False)
        else:
            # Fill with scene-index+1 so tests can assert "the right scene was
            # read".
            arr = np.full(
                self._shape, fill_value=self._current_scene + 1, dtype=self._dtype
            )
        coords: dict[str, list[str]] = {}
        if "C" in self._dims and self._channel_names:
            coords["C"] = self._channel_names
        return xr.DataArray(da.from_array(arr), dims=list(self._dims), coords=coords)

    @property
    def tiles_xarray_dask_data(self) -> xr.DataArray:
        """M-intact tile xarray for per-tile-eligible scenes.

        Fills each tile slice with ``m + 1`` so per-tile tests can assert
        "tile N's pixels went to tile_X*Y*N's store" cheaply. Dims are
        ``[M, T, C, Z, Y, X]`` to mirror the bioio-lif raw shape.
        """
        scene = self._per_tile_scenes[self._current_scene]
        m = len(scene.tiles)
        shape = (
            m,
            scene.t,
            scene.channels,
            scene.z,
            scene.tile_yx[0],
            scene.tile_yx[1],
        )
        arr = np.zeros(shape, dtype=self._dtype)
        for i in range(m):
            arr[i].fill(i + 1)
        return xr.DataArray(da.from_array(arr), dims=["M", "T", "C", "Z", "Y", "X"])

    @property
    def metadata(self) -> ET.Element | None:
        if self._raw_xml is None:
            return None
        scene = self._per_tile_scenes.get(self._current_scene)
        if scene is not None:
            return ET.fromstring(_build_lif_tilescan_metadata(scene))
        return ET.fromstring(self._raw_xml)

    @property
    def ome_metadata(self) -> OME:
        if self._ome_metadata_fails:
            raise RuntimeError("simulated ome_metadata failure")
        size_map = dict(zip(self._dims, self._shape, strict=True))
        scene_name = self.scenes[self._current_scene]
        ome_channels = [
            Channel(id=f"Channel:0:{i}", name=n)
            for i, n in enumerate(self._channel_names)
        ]
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
                        channels=ome_channels,
                    ),
                )
            ]
        )
