"""Tests for the built-in reader plugins (matchers + import-time registration).

The registry contract itself (dispatch, ties, entry points, error handling)
lives in ``test_plugin_registry.py``. This file focuses on the format-specific
matchers and the integration check that the real built-in registry resolves
each extension to the correct plugin.
"""

import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from zarrmony.errors import MosaicStitchingWarning
from zarrmony.readers import czi as czi_mod
from zarrmony.readers import default as default_mod
from zarrmony.readers import lif as lif_mod
from zarrmony.readers import nd2 as nd2_mod
from zarrmony.readers import plugin as plugin_mod
from zarrmony.readers.plugin import (
    ReaderPlugin,
    get_reader,
    list_plugins,
    register_plugin,
)


@pytest.fixture(autouse=True)
def restore_registry() -> Iterator[None]:
    """Snapshot and restore the plugin registry around each test."""
    snapshot = dict(plugin_mod._PLUGINS)
    loaded = plugin_mod._ENTRY_POINTS_LOADED
    yield
    plugin_mod._PLUGINS.clear()
    plugin_mod._PLUGINS.update(snapshot)
    plugin_mod._ENTRY_POINTS_LOADED = loaded


# --- Built-in plugin registration ----------------------------------------


def test_builtin_plugins_are_registered_at_import_time() -> None:
    by_name = {p.name: p for p in list_plugins()}
    for name in ("bioio", "bioio-czi", "bioio-lif", "bioio-nd2"):
        assert name in by_name, f"{name!r} not registered at import time"
        assert by_name[name].source == "builtin"


# --- Matcher behavior (cheap, no Reader construction) --------------------


def test_default_matcher_returns_zero_for_any_path() -> None:
    # Catch-all: lowest possible score so any extension-specific plugin wins.
    assert default_mod._match_default(Path("/tmp/anything.tif")) == 0
    assert default_mod._match_default(Path("/tmp/no_extension")) == 0


@pytest.mark.parametrize(
    ("matcher", "ext"),
    [
        (czi_mod._match_czi, ".czi"),
        (lif_mod._match_lif, ".lif"),
        (nd2_mod._match_nd2, ".nd2"),
    ],
)
def test_format_matcher_claims_extension(matcher, ext: str) -> None:
    assert matcher(Path(f"/tmp/foo{ext}")) == 100
    assert matcher(Path(f"/tmp/foo{ext.upper()}")) == 100  # case-insensitive
    assert matcher(Path("/tmp/foo.tif")) is None


def test_uri_style_paths_extract_extension() -> None:
    # Path() handles gs:// strings as plain text; .suffix walks the last segment.
    assert czi_mod._match_czi(Path("gs://my-bucket/folder/sample.czi")) == 100


# --- Built-in dispatch end-to-end ----------------------------------------


class _Sentinel:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_builtin_extension_dispatch_with_real_registry() -> None:
    """Replace each built-in's ``open`` with a sentinel and confirm dispatch
    walks the real registry to the right plugin per extension. The real
    matchers run; only ``open`` is faked so we don't need real fixture files.
    """
    sentinels = {
        "bioio": _Sentinel("default"),
        "bioio-czi": _Sentinel("czi"),
        "bioio-lif": _Sentinel("lif"),
        "bioio-nd2": _Sentinel("nd2"),
    }
    for name, sentinel in sentinels.items():
        original = plugin_mod._PLUGINS[name]
        replaced = ReaderPlugin(
            name=original.name,
            match=original.match,
            open=lambda _p, s=sentinel: s,
            distribution=original.distribution,
            source=original.source,
        )
        register_plugin(replaced, replace=True)

    cases = [
        ("/tmp/foo.tif", "bioio", 0),
        ("/tmp/foo.CZI", "bioio-czi", 100),
        ("/tmp/foo.lif", "bioio-lif", 100),
        ("/tmp/foo.nd2", "bioio-nd2", 100),
    ]
    for path, expected_name, expected_score in cases:
        reader, plugin, score = get_reader(path)
        assert plugin.name == expected_name, path
        assert score == expected_score, path
        assert reader is sentinels[expected_name], path


# --- _MosaicAwareLifReader proxy ----------------------------------------


@dataclass
class _FakeTileDims:
    Y: int
    X: int


class _FakeBioioLifReader:
    """Stand-in for ``bioio_lif.Reader`` exposing the mosaic surface we use."""

    def __init__(
        self,
        scenes_with_m: dict[int, bool],
        tile_count: int = 12,
        tile_yx: tuple[int, int] = (5048, 5048),
        scene_names: list[str] | None = None,
        metadata_xml: str | None = None,
    ) -> None:
        if scene_names is not None:
            assert len(scene_names) == len(scenes_with_m)
            self.scenes = tuple(scene_names)
        else:
            self.scenes = tuple(f"scene_{i}" for i in scenes_with_m)
        self._has_m = scenes_with_m
        self._current = 0
        self._tile_yx = tile_yx
        self._tile_count = tile_count
        self._metadata_xml = metadata_xml
        # Sentinel — not relevant to mosaic logic, but verifies forwarding.
        self.physical_pixel_sizes = "fake-px"
        self.channel_names = ["DAPI", "GFP"]

    @property
    def metadata(self) -> ET.Element | None:
        if self._metadata_xml is None:
            return None
        return ET.fromstring(self._metadata_xml)

    def set_scene(self, idx: int) -> None:
        self._current = idx

    @property
    def current_scene_index(self) -> int:
        return self._current

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        if self._has_m[self._current]:
            arr = np.zeros(
                (self._tile_count, 1, 1, self._tile_yx[0], self._tile_yx[1]),
                dtype=np.uint16,
            )
            return xr.DataArray(da.from_array(arr), dims=["M", "C", "Z", "Y", "X"])
        arr = np.zeros((1, 1, 1, 64, 64), dtype=np.uint16)
        return xr.DataArray(da.from_array(arr), dims=["T", "C", "Z", "Y", "X"])

    @property
    def mosaic_xarray_dask_data(self) -> xr.DataArray:
        # A stitched view: M collapsed, Y/X grown.
        stitched_y = self._tile_yx[0] * 4
        stitched_x = self._tile_yx[1] * 3
        arr = np.zeros((1, 1, 1, stitched_y, stitched_x), dtype=np.uint16)
        return xr.DataArray(da.from_array(arr), dims=["T", "C", "Z", "Y", "X"])

    @property
    def mosaic_tile_dims(self) -> _FakeTileDims:
        return _FakeTileDims(Y=self._tile_yx[0], X=self._tile_yx[1])


def test_proxy_returns_stitched_view_for_mosaic_scene() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True, 1: False}))
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning, match="bioio-lif is auto-stitching"):
        xarr = proxy.xarray_dask_data

    assert "M" not in xarr.dims
    assert list(xarr.dims) == ["T", "C", "Z", "Y", "X"]


def test_proxy_passes_through_non_mosaic_scene() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True, 1: False}))
    proxy.set_scene(1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicStitchingWarning)
        xarr = proxy.xarray_dask_data

    assert list(xarr.dims) == ["T", "C", "Z", "Y", "X"]
    assert xarr.shape == (1, 1, 1, 64, 64)


def test_proxy_does_not_warn_when_merged_sibling_present() -> None:
    # inspect() walks every scene and accesses xarray_dask_data; the warning
    # would otherwise lie ("no sibling found") even when the sibling exists.
    inner = _FakeBioioLifReader(
        {0: True, 1: False},
        scene_names=["Position 1", "Position 1_Merged"],
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicStitchingWarning)
        xarr = proxy.xarray_dask_data

    assert list(xarr.dims) == ["T", "C", "Z", "Y", "X"]


def test_proxy_warning_names_scene_and_tile_count() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}, tile_count=12))
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning) as captured:
        _ = proxy.xarray_dask_data

    msg = str(captured[0].message)
    assert "'scene_0'" in msg
    assert "12 mosaic tiles" in msg
    assert "1-pixel" in msg


def test_proxy_mosaic_summary_for_mosaic_scene() -> None:
    proxy = lif_mod._MosaicAwareLifReader(
        _FakeBioioLifReader({0: True}, tile_count=12, tile_yx=(5048, 5048))
    )
    proxy.set_scene(0)

    assert proxy.mosaic_summary == {
        "stitched": True,
        "stitcher": "bioio-lif",
        "overlap_assumption_px": 1,
        "tile_count": 12,
        "tile_shape": {"Y": 5048, "X": 5048},
    }


def test_proxy_mosaic_summary_none_for_non_mosaic_scene() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: False}))
    proxy.set_scene(0)

    assert proxy.mosaic_summary is None


# --- mosaic_summary: tile positions + intended overlap (issue #34) --------


def _mosaic_metadata_xml(
    *,
    tiles: list[tuple[int, int, float, float, float]],
    overlap_x: str | None = "0.10",
    overlap_y: str | None = "0.10",
) -> str:
    """A whole-document LIF metadata tree with a single scene carrying tiles.

    Matches the bioio-lif shape ``find_scene_xml`` keys off: scene XML lives
    under ``.//Image``. The first ``<Image>`` is the (only) scene's settings.
    """
    tile_xml = "".join(
        f'<Tile FieldX="{fx}" FieldY="{fy}" PosX="{px:.10f}" PosY="{py:.10f}" PosZ="{pz:.10f}" />'
        for fx, fy, px, py, pz in tiles
    )
    overlap_attrs = []
    if overlap_x is not None:
        overlap_attrs.append(f'OverlapPercentageX="{overlap_x}"')
    if overlap_y is not None:
        overlap_attrs.append(f'OverlapPercentageY="{overlap_y}"')
    stitching_xml = (
        f"<StitchingSettings {' '.join(overlap_attrs)} />" if overlap_attrs else ""
    )
    return (
        "<LMSDataContainerHeader><Element><Children><Element><Data><Image>"
        f'<Attachment Name="TileScanInfo" Application="LAS AF">{tile_xml}</Attachment>'
        f'<Attachment Name="HardwareSetting">{stitching_xml}</Attachment>'
        "</Image></Data></Element></Children></Element></LMSDataContainerHeader>"
    )


def test_proxy_mosaic_summary_includes_tiles_and_overlap_when_extractable() -> None:
    tiles = [
        (0, 0, 0.0400, 0.0170, 0.0117),
        (1, 0, 0.0405, 0.0170, 0.0117),
    ]
    inner = _FakeBioioLifReader(
        {0: True},
        tile_count=2,
        tile_yx=(512, 512),
        metadata_xml=_mosaic_metadata_xml(
            tiles=tiles, overlap_x="0.10", overlap_y="0.15"
        ),
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)

    summary = proxy.mosaic_summary
    assert summary is not None
    assert summary["intended_overlap_x_pct"] == 10.0
    assert summary["intended_overlap_y_pct"] == 15.0
    assert summary["tiles"] == [
        {
            "field_x": 0,
            "field_y": 0,
            "pos_x_m": 0.04,
            "pos_y_m": 0.017,
            "pos_z_m": 0.0117,
        },
        {
            "field_x": 1,
            "field_y": 0,
            "pos_x_m": 0.0405,
            "pos_y_m": 0.017,
            "pos_z_m": 0.0117,
        },
    ]
    # Original shape preserved.
    assert summary["tile_count"] == 2
    assert summary["tile_shape"] == {"Y": 512, "X": 512}


def test_proxy_mosaic_summary_omits_tile_keys_when_metadata_absent() -> None:
    # No metadata XML — extractor declines; the audit falls back to today's shape.
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}))
    proxy.set_scene(0)

    summary = proxy.mosaic_summary
    assert summary is not None
    assert "tiles" not in summary
    assert "intended_overlap_x_pct" not in summary
    assert "intended_overlap_y_pct" not in summary


# --- MosaicStitchingWarning text: overlap-aware vs generic ----------------


def test_proxy_warning_quotes_intended_overlap_when_extractable() -> None:
    inner = _FakeBioioLifReader(
        {0: True},
        tile_count=4,
        metadata_xml=_mosaic_metadata_xml(
            tiles=[(0, 0, 0.04, 0.017, 0.0117)],
            overlap_x="0.10",
            overlap_y="0.10",
        ),
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning) as captured:
        _ = proxy.xarray_dask_data

    msg = str(captured[0].message)
    assert "10% intended overlap" in msg
    assert "1-pixel" in msg
    # Generic 5–15% fallback wording should NOT appear when the real value is known.
    assert "5–15%" not in msg


def test_proxy_warning_falls_back_to_generic_when_overlap_missing() -> None:
    # No metadata XML at all — warning uses the existing generic wording.
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}, tile_count=4))
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning) as captured:
        _ = proxy.xarray_dask_data

    msg = str(captured[0].message)
    assert "5–15%" in msg
    assert "1-pixel" in msg
    assert "intended overlap" not in msg


# --- ADR-0005 escape-hatch sentence + per-tile reader surface ---------------


def test_proxy_warning_quotes_per_tile_escape_hatch() -> None:
    """The MosaicStitchingWarning body must name the lif_mosaic='per-tile'
    escape hatch so users hitting a bad stitch see the corrected output path
    in the same message (ADR-0005)."""
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}, tile_count=4))
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning) as captured:
        _ = proxy.xarray_dask_data

    msg = str(captured[0].message)
    assert 'lif_mosaic="per-tile"' in msg


def test_proxy_warning_quotes_grid_stitch_escape_hatch() -> None:
    """The MosaicStitchingWarning body must also name the lif_mosaic='grid-stitch'
    escape hatch (#39). Grid-stitch fixes M-scan-order tile placement while
    preserving one-store-per-scene, so users with tooling that expects a single
    canvas see it as an alternative to per-tile."""
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}, tile_count=4))
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning) as captured:
        _ = proxy.xarray_dask_data

    msg = str(captured[0].message)
    assert 'lif_mosaic="grid-stitch"' in msg


def test_proxy_warning_names_m_scan_arrangement_bug() -> None:
    """The MosaicStitchingWarning body must explicitly name the M-scan-order
    tile placement bug (#39). Prior text only mentioned the overlap-stripe
    issue, leaving users no in-context signal that arrangement was ALSO broken."""
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}, tile_count=4))
    proxy.set_scene(0)

    with pytest.warns(MosaicStitchingWarning) as captured:
        _ = proxy.xarray_dask_data

    msg = str(captured[0].message)
    assert "M-scan order" in msg
    assert "FieldX/FieldY" in msg


def test_proxy_tiles_xarray_dask_data_returns_m_intact_view() -> None:
    """The per-tile writer path needs the raw M-intact xarray; the auto-stitch
    swap and its MosaicStitchingWarning must NOT fire on this accessor."""
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}, tile_count=8))
    proxy.set_scene(0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MosaicStitchingWarning)
        xarr = proxy.tiles_xarray_dask_data

    assert "M" in xarr.dims
    assert xarr.sizes["M"] == 8


def test_proxy_is_mosaic_reassembly_eligible_true_for_mosaic_no_sibling() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: True}))
    proxy.set_scene(0)
    assert proxy.is_mosaic_reassembly_eligible() is True


def test_proxy_is_mosaic_reassembly_eligible_false_for_non_mosaic_scene() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: False}))
    proxy.set_scene(0)
    assert proxy.is_mosaic_reassembly_eligible() is False


def test_proxy_is_mosaic_reassembly_eligible_false_when_merged_sibling_present() -> (
    None
):
    """A mosaic scene with a vendor _Merged sibling is NOT reassembly-eligible —
    the merged sibling is the source of truth for that scene, so per-tile or
    grid-stitch output would be redundant."""
    inner = _FakeBioioLifReader(
        {0: True, 1: False},
        scene_names=["Position 1", "Position 1_Merged"],
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)
    assert proxy.is_mosaic_reassembly_eligible() is False


def test_proxy_forwards_arbitrary_attrs() -> None:
    inner = _FakeBioioLifReader({0: True})
    proxy = lif_mod._MosaicAwareLifReader(inner)

    assert proxy.scenes is inner.scenes
    assert proxy.channel_names == ["DAPI", "GFP"]
    assert proxy.physical_pixel_sizes == "fake-px"
    proxy.set_scene(0)
    assert inner._current == 0


# --- skip_reason: vendor _Merged sibling preference ---------------------


def test_skip_reason_none_for_non_mosaic_scene() -> None:
    proxy = lif_mod._MosaicAwareLifReader(_FakeBioioLifReader({0: False}))
    proxy.set_scene(0)

    assert proxy.skip_reason is None


def test_skip_reason_none_for_mosaic_without_merged_sibling() -> None:
    # Mosaic scene 'Position 1' with no 'Position 1_Merged' sibling.
    inner = _FakeBioioLifReader(
        {0: True, 1: False},
        scene_names=["Position 1", "AnotherScene"],
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)

    assert proxy.skip_reason is None


def test_skip_reason_set_when_merged_sibling_present() -> None:
    # Mosaic scene 'Position 1' with vendor-stitched 'Position 1_Merged' sibling.
    inner = _FakeBioioLifReader(
        {0: True, 1: False},
        scene_names=["Position 1", "Position 1_Merged"],
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)

    reason = proxy.skip_reason
    assert reason is not None
    assert "'Position 1_Merged'" in reason


def test_skip_reason_does_not_apply_to_the_merged_sibling_itself() -> None:
    # The merged sibling is NOT a mosaic scene, so it has no skip_reason.
    inner = _FakeBioioLifReader(
        {0: True, 1: False},
        scene_names=["Position 1", "Position 1_Merged"],
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(1)

    assert proxy.skip_reason is None


def test_skip_reason_requires_exact_suffix_match() -> None:
    # Different stem ('Other_Merged') does not pair with 'Position 1'.
    inner = _FakeBioioLifReader(
        {0: True, 1: False},
        scene_names=["Position 1", "Other_Merged"],
    )
    proxy = lif_mod._MosaicAwareLifReader(inner)
    proxy.set_scene(0)

    assert proxy.skip_reason is None
