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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from bioio_lif import Reader

from zarrmony.errors import MosaicStitchingWarning
from zarrmony.metadata._lif_scene import find_scene_xml
from zarrmony.metadata.lif_plate import extract_plate_layouts
from zarrmony.metadata.lif_tiles import extract_tile_layout
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin

_STITCHER_NAME = "bioio-lif"
_OVERLAP_ASSUMPTION_PX = 1
_MERGED_SUFFIX = "_Merged"

# The ADR-0004 plate writer treats every FOV as belonging to a single
# acquisition when the reader doesn't distinguish acquisitions itself. LIF's
# plate template records the well/field grid but not per-acquisition passes
# (a rescan of the same plate becomes a second LIF file, not a second
# acquisition inside one file), so the LIF reader wires every field to id=1.
_DEFAULT_ACQUISITION_ID = 1
_DEFAULT_ACQUISITION_NAME = "acquisition"


def _plate_layout_from_dict(plate: dict) -> PlateLayout:
    """Materialize a :class:`PlateLayout` from :func:`extract_plate_layouts` output.

    The extractor speaks in raw dicts (dependency-free) so it can lift into
    ``bioio-lif`` later; this glue turns those dicts into the writer's typed
    surface. Every plate is wired to a single default acquisition per ADR-0004
    v1 (multi-acquisition is a deferred v2 concern).
    """
    fields = [
        PlateField(
            scene_index=f["scene_index"],
            row=f["row"],
            column=f["column"],
            field_name=f["field_name"],
            acquisition_id=_DEFAULT_ACQUISITION_ID,
        )
        for f in plate["fields"]
    ]
    return PlateLayout(
        name=plate["name"],
        rows=list(plate["rows"]),
        columns=list(plate["columns"]),
        acquisitions=[
            Acquisition(id=_DEFAULT_ACQUISITION_ID, name=_DEFAULT_ACQUISITION_NAME)
        ],
        fields=fields,
    )


class _MosaicAwareLifReader:
    """Proxy that swaps ``xarray_dask_data`` → ``mosaic_xarray_dask_data``
    when the current scene has an ``M`` dim and no vendor ``_Merged`` sibling
    is available. Forwards everything else to the wrapped ``bioio_lif.Reader``.
    """

    def __init__(self, inner: Reader) -> None:
        self._inner = inner
        self._plate_cache: tuple[list[str], PlateLayout | None] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _plate_state(self) -> tuple[list[str], PlateLayout | None]:
        """(available_plates, plate_layout) — cached per reader instance.

        Walks the LIF ``LMSDataContainer`` XML once via
        :func:`extract_plate_layouts`. The plate structure is a function of the
        file, not of the current scene, so a single walk suffices for the
        reader's lifetime. Fail-closed end-to-end: a missing ``metadata``
        surface, an unserializable XML element, or an extractor error all
        yield ``([], None)`` so the reader stays flat.

        Multi-plate LIF handling is deferred to #82: for this tracer bullet
        ``plate_layout`` is populated only when exactly one plate template is
        present. Multi-plate files still surface their plate names via
        ``available_plates`` so the follow-up ``--plate NAME`` selector has
        something to key off, but ``layout_hint`` stays ``"flat"`` and the
        existing per-scene write path is unchanged.
        """
        if self._plate_cache is not None:
            return self._plate_cache
        state: tuple[list[str], PlateLayout | None] = ([], None)
        try:
            metadata = getattr(self._inner, "metadata", None)
            if metadata is not None and hasattr(metadata, "tag"):
                source_xml = ET.tostring(metadata, encoding="unicode")
                plates = extract_plate_layouts(source_xml)
                names = [p["name"] for p in plates]
                layout: PlateLayout | None = None
                if len(plates) == 1:
                    layout = _plate_layout_from_dict(plates[0])
                state = (names, layout)
        except Exception:  # noqa: BLE001 — metadata never breaks a conversion
            state = ([], None)
        self._plate_cache = state
        return state

    @property
    def available_plates(self) -> list[str]:
        """Names of every plate template in the LIF XML, document order.

        Empty for non-plate LIFs. Consumed by #82's ``--plate NAME`` selector
        to disambiguate multi-plate files and by the error message that names
        available plates when no selector was passed.
        """
        return list(self._plate_state()[0])

    @property
    def plate_layout(self) -> PlateLayout | None:
        """The single detected plate's :class:`PlateLayout`, or ``None``.

        Populated only when the LIF holds exactly one plate template. Multi-
        plate files leave this ``None`` (and ``layout_hint`` at ``"flat"``)
        until #82 wires the ``--plate NAME`` disambiguator.
        """
        return self._plate_state()[1]

    @property
    def layout_hint(self) -> str:
        """``"plate"`` when a single plate template resolved, else ``"flat"``.

        Non-plate LIFs and multi-plate LIFs both report ``"flat"`` for now —
        multi-plate resolution ships in #82.
        """
        return "plate" if self._plate_state()[1] is not None else "flat"

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

        Fires only when bioio-lif's own stitcher actually runs — under the
        v0.7.0 cascade default (``lif_mosaic="auto-stitch"``) that's either
        the cascade fallback (no ``<Tile>`` layout metadata to feed
        stage-stitch or grid-stitch) or an explicit
        ``lif_mosaic="bioio-lif"`` request. The warning names both known
        pathologies — the 1-pixel overlap seam (quoted against the
        LIF-declared intended overlap when extractable) and the M-scan-order
        placement that ignores ``FieldX``/``FieldY`` — and points at
        ``lif_mosaic="per-tile"`` as the remaining zarrmony alternative
        (stage-stitch and grid-stitch aren't listed because the cascade
        already tried them or the user opted out of both).
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
            f"('{scene_name}{_MERGED_SUFFIX}') was found. Under the default "
            f'lif_mosaic="auto-stitch" cascade this fallback runs only when '
            f"neither stage-stitch nor grid-stitch is eligible (no per-tile "
            f"<Tile> layout in the scene XML); consider "
            f'lif_mosaic="per-tile" to write each tile as its own OME-Zarr '
            f"(pixel-correct, no seams — external stitcher must reassemble), "
            f"or an external stitcher (ASHLAR, m2stitch, BigStitcher) if "
            f"boundary correctness matters."
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
