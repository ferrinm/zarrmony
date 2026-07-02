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

The bioio-lif stitcher is **not content-aware**: it hardcodes a 1-pixel
inter-tile overlap and ignores the LIF metadata's actual stage XY positions.
For acquisitions with normal 5–15% overlap, the output has double-coverage
stripes at every tile seam. Two-tier handling:

1. If the LIF contains a sibling scene named ``<scene>_Merged`` (Leica's
   convention for vendor-stitched output), the proxy advertises a
   ``skip_reason`` for the mosaic scene. ``convert()`` honors that hook,
   skips the mosaic scene, and writes only the merged sibling — emitting a
   :class:`MosaicMergedSiblingWarning` so the user can see the substitution.
2. Otherwise the proxy falls back to ``mosaic_xarray_dask_data`` and emits a
   :class:`MosaicStitchingWarning` recording the stitcher + 1-pixel overlap
   assumption in the audit's ``mosaic`` block, so users can route around it
   later (an external stitcher: ASHLAR, m2stitch, BigStitcher).

The mosaic audit block also surfaces per-tile stage positions and the LIF-
declared intended overlap (extracted via :mod:`zarrmony.metadata.lif_tiles`)
when available, and the :class:`MosaicStitchingWarning` quotes that overlap
in its text so the user can predict the stripe width.

ADR-0005 adds three opt-in writer paths that consume the raw M-intact tile
xarray via :attr:`tiles_xarray_dask_data` and sidestep the bioio-lif stitcher
entirely: ``lif_mosaic="per-tile"`` writes one OME-Zarr per tile carrying the
stage position in each tile's ``<Plane>``; ``lif_mosaic="grid-stitch"``
reassembles a single canvas by placing tile M=i at
``(field_y[i]*tile_H, field_x[i]*tile_W)`` from the LIF ``FieldX``/``FieldY``
indices, fixing bioio-lif's M-scan-order placement bug on a butt-jointed
canvas; ``lif_mosaic="stage-stitch"`` places each tile at its ``PosX``/``PosY``
stage µm position (converted via the scene's physical pixel size) so the
canvas honours the LIF-declared intended overlap instead of butt joints. All
three share the eligibility predicate :meth:`is_mosaic_reassembly_eligible`
(mosaic scene with no ``_Merged`` sibling). The proxy stays thin — reader
exposes the raw tiles surface, ``api.convert()`` owns strategy dispatch.

Exposed as ``lif_plugin`` and registered in ``readers/__init__.py`` at zarrmony
import time.
"""

import warnings
from pathlib import Path
from typing import Any

from bioio_lif import Reader

from zarrmony.errors import MosaicStitchingWarning
from zarrmony.metadata._lif_scene import find_scene_xml
from zarrmony.metadata.lif_tiles import extract_tile_layout
from zarrmony.readers.plugin import ReaderPlugin

_STITCHER_NAME = "bioio-lif"
_OVERLAP_ASSUMPTION_PX = 1
_MERGED_SUFFIX = "_Merged"


class _MosaicAwareLifReader:
    """Proxy that swaps ``xarray_dask_data`` → ``mosaic_xarray_dask_data``
    when the current scene has an ``M`` dim and no vendor ``_Merged`` sibling
    is available. Forwards everything else to the wrapped ``bioio_lif.Reader``.
    """

    def __init__(self, inner: Reader) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _merged_sibling(self) -> str | None:
        scene_name = self._inner.scenes[self._inner.current_scene_index]
        sibling = f"{scene_name}{_MERGED_SUFFIX}"
        return sibling if sibling in self._inner.scenes else None

    @property
    def skip_reason(self) -> str | None:
        """Non-None when the current scene should not be written.

        Set when the current scene is a mosaic AND a sibling scene named
        ``<scene>_Merged`` is present in ``scenes`` — the merged sibling will
        be written by its own loop iteration, so writing the unstitched
        tiles (or our imprecise auto-stitch of them) would be redundant.
        """
        if "M" not in self._inner.xarray_dask_data.dims:
            return None
        sibling = self._merged_sibling()
        if sibling is None:
            return None
        return (
            f"vendor-merged sibling scene {sibling!r} is present; "
            f"writing that instead of bioio-lif's auto-stitched tiles"
        )

    def _tile_layout(self) -> dict | None:
        """Extracted LIF tile + overlap metadata for the current scene, or ``None``.

        Pure metadata read — used by both :attr:`mosaic_summary` (audit) and
        :attr:`xarray_dask_data` (warning text). Fail-safe end-to-end: a missing
        ``metadata`` surface, a raising ``scene_root``, or any extractor failure
        all yield ``None``.
        """
        scene_xml = find_scene_xml(self._inner)
        if scene_xml is None:
            return None
        return extract_tile_layout(scene_xml)

    @property
    def xarray_dask_data(self) -> Any:
        xarr = self._inner.xarray_dask_data
        if "M" not in xarr.dims:
            return xarr
        scene_name = self._inner.scenes[self._inner.current_scene_index]
        if self._merged_sibling() is None:
            warnings.warn(
                self._stitching_warning_text(scene_name, int(xarr.sizes["M"])),
                MosaicStitchingWarning,
                stacklevel=2,
            )
        return self._inner.mosaic_xarray_dask_data

    @property
    def tiles_xarray_dask_data(self) -> Any:
        """The raw M-intact xarray for the current mosaic scene (no auto-stitch).

        ``xarray_dask_data`` swaps the M-intact view for ``mosaic_xarray_dask_data``
        (and emits :class:`MosaicStitchingWarning`); this property bypasses both
        so the mosaic-reassembly writer paths in ``convert()`` can iterate tiles
        without re-triggering the auto-stitch warning. Returns the inner
        reader's raw ``xarray_dask_data`` unchanged — if ``M`` is absent the
        caller should not be on a reassembly branch in the first place, so we
        don't try to invent one. Used by ``api.convert(..., lif_mosaic="per-tile")``
        and ``api.convert(..., lif_mosaic="grid-stitch")``.
        """
        return self._inner.xarray_dask_data

    def is_mosaic_reassembly_eligible(self) -> bool:
        """True when the current scene is a mosaic with no vendor ``_Merged`` sibling.

        Both non-default ``lif_mosaic`` writer paths (per-tile, grid-stitch)
        are only relevant for this case — a mosaic scene whose tiles bioio-lif
        would otherwise auto-stitch. Mosaic scenes with a ``_Merged`` sibling
        still get skipped (their pixels come from the merged sibling's own
        scene loop iteration); non-mosaic scenes write through the standard
        per-scene path under any ``lif_mosaic`` value.
        """
        if "M" not in self._inner.xarray_dask_data.dims:
            return False
        return self._merged_sibling() is None

    def _stitching_warning_text(self, scene_name: str, tile_count: int) -> str:
        """Compose the :class:`MosaicStitchingWarning` body.

        Names both known auto-stitch pathologies — the 1-pixel overlap seam
        (quoted against the LIF-declared intended overlap when extractable) and
        the M-scan-order tile placement — then lists both zarrmony escape
        hatches (per-tile first: pixel-correct; grid-stitch second: fixes
        arrangement on a single canvas), then external stitchers as a last
        resort. Ordering is deliberate: per-tile is the honest correctness
        answer, grid-stitch preserves the one-store-per-scene invariant when
        the user's downstream tooling needs a single canvas.
        """
        layout = self._tile_layout()
        overlap_x = layout.get("intended_overlap_x_pct") if layout else None
        if overlap_x is not None:
            overlap_clause = (
                f"LIF metadata declares {overlap_x:g}% intended overlap; "
                f"bioio-lif stitched with a 1-pixel overlap — expect "
                f"~{overlap_x:g}%-wide double-coverage stripes at every seam"
            )
        else:
            overlap_clause = (
                "bioio-lif assumes a 1-pixel inter-tile overlap. "
                "For acquisitions with non-trivial overlap (typical 5–15%), "
                "the output has double-coverage stripes at tile seams and is "
                "unfit for quantitative analysis at tile boundaries"
            )
        return (
            f"scene {scene_name!r}: bioio-lif is auto-stitching "
            f"{tile_count} mosaic tiles. The LIF stage XY positions are NOT "
            f"used, and tiles are placed by M-scan order — not by their "
            f"declared FieldX/FieldY grid indices, so visual layout may not "
            f"match acquisition. {overlap_clause}. No vendor-stitched sibling "
            f"('{scene_name}{_MERGED_SUFFIX}') was found; consider re-running "
            f'with lif_mosaic="stage-stitch" to place tiles at their true '
            f"stage µm positions (honours declared overlap; one canvas per "
            f'scene), lif_mosaic="grid-stitch" to fix arrangement on a butt-'
            f"jointed canvas by FieldX/FieldY indices (seams remain if the "
            f'acquisition had overlap), lif_mosaic="per-tile" to write each '
            f"tile as its own OME-Zarr (pixel-correct, no seams — external "
            f"stitcher must reassemble), or an external stitcher (ASHLAR, "
            f"m2stitch, BigStitcher) if boundary correctness matters."
        )

    @property
    def mosaic_summary(self) -> dict | None:
        """Audit hook — None unless the current scene was mosaic-stitched.

        Merges :func:`extract_tile_layout`'s output when available so the audit
        records the actual tile positions and the LIF-declared intended overlap
        alongside the stitcher/overlap-assumption fields. Extractor misses
        (missing/malformed scene XML, no ``<Tile>`` elements) leave the
        new keys absent — today's shape is preserved on fall-back.
        """
        xarr = self._inner.xarray_dask_data
        if "M" not in xarr.dims:
            return None
        tile_dims = self._inner.mosaic_tile_dims
        summary: dict = {
            "stitched": True,
            "stitcher": _STITCHER_NAME,
            "overlap_assumption_px": _OVERLAP_ASSUMPTION_PX,
            "tile_count": int(xarr.sizes["M"]),
            "tile_shape": {
                "Y": int(tile_dims.Y),
                "X": int(tile_dims.X),
            },
        }
        layout = self._tile_layout()
        if layout is not None:
            summary.update(layout)
        return summary


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
