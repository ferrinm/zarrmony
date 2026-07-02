"""Pure per-tile layout extractor + grid reassembly for Leica LIF scene XML.

A mosaic LIF scene carries two pieces of acquisition metadata that bioio-lif's
stitcher ignores:

* ``<Tile FieldX FieldY PosX PosY PosZ />`` elements inside the scene's
  ``<Attachment Name="TileScanInfo">`` — the per-tile (row, column) grid index
  and the stage position (meters) of each tile's origin.
* ``<StitchingSettings OverlapPercentageX OverlapPercentageY .../>`` — the
  user-configured intended overlap fraction (LIF stores 0.10 to mean 10%).

:func:`extract_tile_layout` returns a structured dict surfacing both, suitable
for the audit's ``mosaic`` block. It is fail-closed (mirrors
:mod:`zarrmony.metadata.lif_channels`): oversized input, DTDs / entity
definitions, external entities, and any malformed XML yield ``None``. A scene
with no ``<Tile>`` elements also yields ``None`` — the audit/warning fall back
to today's generic shape rather than carrying a nonsense empty list.

The extractor is dependency-free and intended to lift into ``bioio-lif`` later.

For the grid-stitch writer path (``lif_mosaic="grid-stitch"``),
:func:`reassemble_grid` and its validation helpers place raw M-intact tiles onto
a butt-jointed canvas by their declared ``field_x``/``field_y`` slots — the fix
for bioio-lif's M-scan-order placement bug that shows up in real acquisitions.
Grid reassembly is strict: incomplete/malformed tile metadata raises with a
clear message pointing at ``lif_mosaic="per-tile"`` as the graceful escape.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import dask
import dask.array as da
import numpy as np
import xarray as xr

# LIF stage positions are stored in meters; OME + zarrmony's pixel-size axis
# scales are in micrometers, so stage-stitch converts through µm.
_STAGE_METERS_TO_UM = 1_000_000.0

# A stage-based placement is a sanity check, not a stitcher. Real acquisitions
# have small pixel-size / stage-calibration jitter; 20% is loose enough to not
# fire on well-calibrated confocals but tight enough to flag an obviously wrong
# pixel-size (unit swap, half-scale, etc.).
_STAGE_OVERLAP_TOLERANCE_FRAC = 0.20

# A confocal scene blob is a few hundred KB; 32 MiB is generous headroom while
# still bounding parse work. Mirrors :mod:`lif_channels`.
_MAX_BYTES = 32 * 1024 * 1024

# A LIF scene blob is plain element/attribute XML. Any DOCTYPE or entity
# declaration is malformed or hostile (billion-laughs / XXE). Reject textually
# before parsing — stdlib ``ElementTree`` *does* expand internal entities.
_DOCTYPE_OR_ENTITY = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


class _EntityRejectingTarget:
    """ExpatBuilder target that refuses DTDs and entity definitions.

    Belt-and-suspenders alongside the textual pre-scan: if a declaration slipped
    past the regex, expat's ``entity_decl`` / ``unparsed_entity_decl``
    callbacks fire here and abort the parse instead of expanding anything.
    """

    def __init__(self) -> None:
        self._builder = ET.TreeBuilder()

    def entity_decl(self, *_args, **_kwargs):  # pragma: no cover - defensive
        raise ValueError("entity declarations are not permitted")

    def unparsed_entity_decl(self, *_args, **_kwargs):  # pragma: no cover
        raise ValueError("entity declarations are not permitted")

    def start_doctype_decl(self, *_args, **_kwargs):  # pragma: no cover
        raise ValueError("DOCTYPE is not permitted")

    def start(self, tag, attrib):
        return self._builder.start(tag, attrib)

    def end(self, tag):
        return self._builder.end(tag)

    def data(self, text):
        return self._builder.data(text)

    def close(self):
        return self._builder.close()


def _safe_parse(scene_xml: str) -> ET.Element | None:
    """Parse ``scene_xml`` into a root element, fail-closed (never raises)."""
    if not isinstance(scene_xml, str) or not scene_xml:
        return None
    if len(scene_xml.encode("utf-8", "ignore")) > _MAX_BYTES:
        return None
    if _DOCTYPE_OR_ENTITY.search(scene_xml):
        return None
    try:
        parser = ET.XMLParser(target=_EntityRejectingTarget())
        parser.feed(scene_xml)
        return parser.close()
    except Exception:
        return None


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return int(value) if math.isfinite(value) and value.is_integer() else None


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tile_entries(root: ET.Element) -> list[dict]:
    """One dict per ``<Tile>`` element, in document order.

    Each tile carries the grid index (``field_x``, ``field_y``) and the stage
    position (``pos_x_m``, ``pos_y_m``, ``pos_z_m``) verbatim from LIF — meters,
    floats. Any attribute that fails to parse degrades to ``None`` rather than
    dropping the tile, so a partially-stamped tile still names its grid slot.
    """
    tiles: list[dict] = []
    for tile in root.iter("Tile"):
        attrib = tile.attrib
        tiles.append(
            {
                "field_x": _to_int(attrib.get("FieldX")),
                "field_y": _to_int(attrib.get("FieldY")),
                "pos_x_m": _to_float(attrib.get("PosX")),
                "pos_y_m": _to_float(attrib.get("PosY")),
                "pos_z_m": _to_float(attrib.get("PosZ")),
            }
        )
    return tiles


def _overlap_pct(root: ET.Element, attr: str) -> float | None:
    """The first ``StitchingSettings``'s ``attr`` as a percent (0.10 → 10.0).

    LIF stores ``OverlapPercentageX/Y`` as a *fraction* (0.10 == 10%) despite
    the misleading name; the surface key ``intended_overlap_x_pct`` represents
    actual percent, so we multiply by 100. Returns ``None`` when no
    ``StitchingSettings`` element exists or the attribute is absent/unparseable.
    """
    for settings in root.iter("StitchingSettings"):
        fraction = _to_float(settings.attrib.get(attr))
        if fraction is None:
            return None
        return fraction * 100.0
    return None


def extract_tile_layout(scene_xml: str) -> dict | None:
    """Extract tile positions and intended overlap from a LIF scene XML string.

    Returns ``{"tiles": [...], "intended_overlap_x_pct": float|None,
    "intended_overlap_y_pct": float|None}`` when the scene has at least one
    ``<Tile>`` element; otherwise ``None`` (also for missing/malformed/oversized
    input). Each tile dict has the keys ``field_x``, ``field_y``, ``pos_x_m``,
    ``pos_y_m``, ``pos_z_m``; positions are LIF's stage coordinates in meters.

    Fail-closed: any unexpected structural surprise yields ``None`` so the audit
    falls back cleanly. Metadata never crashes a conversion.
    """
    root = _safe_parse(scene_xml)
    if root is None:
        return None
    try:
        tiles = _tile_entries(root)
        if not tiles:
            return None
        return {
            "tiles": tiles,
            "intended_overlap_x_pct": _overlap_pct(root, "OverlapPercentageX"),
            "intended_overlap_y_pct": _overlap_pct(root, "OverlapPercentageY"),
        }
    except Exception:
        return None


# --- grid-stitch reassembly (lif_mosaic="grid-stitch") --------------------
#
# ``bioio_lif``'s auto-stitcher places tiles in M-scan order — the order they
# appear on the ``M`` dim of the raw ``xarray_dask_data``. Real acquisitions
# rarely record tiles row-major from ``(0, 0)``, so the auto-stitched canvas
# routinely puts the wrong tile at each grid slot. Grid-stitch fixes this by
# reading each tile's declared ``field_x``/``field_y`` and placing it at
# ``(field_y*tile_H, field_x*tile_W)`` on a butt-jointed canvas.
#
# This slice does not handle overlap — canvas is ``(rows*tile_H, cols*tile_W)``,
# so an acquisition with real overlap gets a missing-pixel gap at every seam
# (the mirror of auto-stitch's double-coverage stripe). The audit records
# ``overlap_assumption_px: 0`` so downstream consumers can spot the mismatch
# against ``intended_overlap_*_pct``.


def grid_shape(tiles: list[dict]) -> tuple[int | None, int | None]:
    """Best-effort ``(rows, cols)`` from a tile layout, or ``(None, None)``.

    Used by the audit's ``placement_shape`` field — must not raise even when
    the layout is malformed (a partial grid still records a partial audit).
    Returns ``(None, None)`` if any ``field_x``/``field_y`` is missing.
    """
    xs = [t.get("field_x") for t in tiles]
    ys = [t.get("field_y") for t in tiles]
    if any(v is None for v in xs) or any(v is None for v in ys):
        return None, None
    return max(ys) + 1, max(xs) + 1


def validate_grid_layout(tiles: list[dict], m_size: int) -> tuple[int, int]:
    """Assert ``tiles`` covers a complete rectangular grid; return ``(rows, cols)``.

    Raises :class:`ValueError` with a clear message on any invariant failure:
    empty tile list, count mismatch with the M dim, missing ``field_x``/
    ``field_y``, duplicate slots, or non-contiguous grid coverage. Each error
    message points at ``lif_mosaic="per-tile"`` as the graceful-degrade escape,
    since per-tile tolerates the same metadata shortcomings that grid-stitch
    cannot recover from.
    """
    escape = 'pass lif_mosaic="per-tile" to write per-tile stores instead.'
    if not tiles:
        raise ValueError(
            "lif_mosaic='grid-stitch': no <Tile> entries were extracted from "
            "the scene XML; cannot place tiles without per-tile grid indices. "
            f"{escape}"
        )
    if len(tiles) != m_size:
        raise ValueError(
            f"lif_mosaic='grid-stitch': tile-metadata count ({len(tiles)}) does "
            f"not match the array's M dim ({m_size}). {escape}"
        )
    missing = [
        i
        for i, t in enumerate(tiles)
        if t.get("field_x") is None or t.get("field_y") is None
    ]
    if missing:
        raise ValueError(
            f"lif_mosaic='grid-stitch': tiles at M index(es) {missing} are "
            f"missing FieldX and/or FieldY on the scene XML's <Tile> elements. "
            f"{escape}"
        )
    slots = {(t["field_x"], t["field_y"]) for t in tiles}
    if len(slots) != m_size:
        raise ValueError(
            f"lif_mosaic='grid-stitch': tiles have duplicate (field_x, field_y) "
            f"pairs; expected {m_size} unique slots, got {len(slots)}. {escape}"
        )
    xs = [t["field_x"] for t in tiles]
    ys = [t["field_y"] for t in tiles]
    cols = max(xs) + 1
    rows = max(ys) + 1
    expected = {(x, y) for x in range(cols) for y in range(rows)}
    if slots != expected:
        raise ValueError(
            f"lif_mosaic='grid-stitch': (field_x, field_y) pairs do not cover a "
            f"complete rectangular grid; expected {rows}x{cols} slots, got "
            f"{sorted(slots)}. {escape}"
        )
    return rows, cols


def reassemble_grid(tiles_xarr: xr.DataArray, tile_layout: dict | None) -> xr.DataArray:
    """Reassemble raw M-intact tiles onto a butt-jointed canvas by grid index.

    Given the reader's raw M-intact xarray (dims include ``M``, ``Y``, ``X``)
    and the ``extract_tile_layout()`` output, returns a new xarray with the
    ``M`` dim consumed — tile at M=i is placed at
    ``(field_y[i]*tile_H, field_x[i]*tile_W)`` on the canvas. T/C/Z axes and
    their coords are preserved untouched. The returned array remains
    dask-backed — no eager materialization.

    Strict: raises :class:`ValueError` (via :func:`validate_grid_layout`) when
    the tile layout is missing, incomplete, or malformed. Callers that want a
    graceful fallback should use ``lif_mosaic="per-tile"`` (surfaced in every
    error message).
    """
    if tile_layout is None:
        raise ValueError(
            "lif_mosaic='grid-stitch' requires per-tile grid metadata "
            "(<Tile FieldX FieldY .../> under <Attachment "
            'Name="TileScanInfo"> in the scene XML), but none could be '
            "extracted for the current scene. Pass "
            'lif_mosaic="per-tile" to write per-tile stores instead.'
        )
    tiles = tile_layout.get("tiles") or []
    m_size = int(tiles_xarr.sizes["M"])
    rows, cols = validate_grid_layout(tiles, m_size)

    idx_map = {(t["field_x"], t["field_y"]): i for i, t in enumerate(tiles)}
    non_m_dims = [d for d in tiles_xarr.dims if d != "M"]
    y_axis = non_m_dims.index("Y")
    x_axis = non_m_dims.index("X")

    row_arrays = []
    for y in range(rows):
        row_tiles = [tiles_xarr.isel(M=idx_map[(x, y)]).data for x in range(cols)]
        row_arrays.append(da.concatenate(row_tiles, axis=x_axis))
    canvas_data = da.concatenate(row_arrays, axis=y_axis)

    # Y/X coords carry per-tile pixel positions — they no longer size-match
    # the enlarged canvas, so drop them along with any other spatial coord
    # whose dims touch Y or X. Non-spatial coords (T/C/Z, channel names, etc.)
    # survive untouched.
    preserved_coords = {
        name: coord
        for name, coord in tiles_xarr.coords.items()
        if "M" not in coord.dims and "Y" not in coord.dims and "X" not in coord.dims
    }
    return xr.DataArray(canvas_data, dims=non_m_dims, coords=preserved_coords)


# --- stage-stitch reassembly (lif_mosaic="stage-stitch") ------------------
#
# Grid-stitch fixes tile ORDERING by placing M=i at its declared FieldX/FieldY
# grid slot, but its canvas is butt-jointed — a scene acquired with the typical
# 5–15% overlap ends up ~10% too narrow (missing the overlap region). Stage-
# stitch honors the acquisition's true intended overlap by converting each
# tile's ``PosX``/``PosY`` (meters) → µm → pixels via the scene's physical
# pixel size, normalising to a common (0, 0), and snapping to integer pixel
# offsets. Later-placed tiles overwrite earlier-placed tiles in the overlap
# region (documented, deterministic; no blending in this slice).
#
# Missing inputs fail LOUD (not silent fallback): callers should route through
# grid-stitch when stage positions or pixel size are unavailable. A sanity
# check compares the placement's observed overlap against the LIF-declared
# intended overlap and emits :class:`MosaicPlacementWarning` on a >20% mismatch
# — catches pixel-size / unit-conversion bugs without failing conversion.


def _validate_stage_tiles(tiles: list[dict]) -> None:
    """Assert every tile carries ``pos_x_m``/``pos_y_m``; raise clear error otherwise.

    ``pos_z_m`` is optional (stage-stitch places on Y/X only). ``field_x``/
    ``field_y`` are only used for the observed-overlap adjacency inference,
    not the placement itself, so missing grid indices don't error here.
    """
    escape = (
        'Pass lif_mosaic="grid-stitch" to place tiles by FieldX/FieldY grid '
        "indices instead (no stage-position dependency, but butt joints — "
        "seams remain if the acquisition had overlap)."
    )
    if not tiles:
        raise ValueError(
            "lif_mosaic='stage-stitch': no <Tile> entries were extracted from "
            f"the scene XML; cannot place tiles without stage positions. "
            f"{escape}"
        )
    missing = []
    for i, t in enumerate(tiles):
        missing_keys = [k for k in ("pos_x_m", "pos_y_m") if t.get(k) is None]
        if missing_keys:
            missing.append((i, missing_keys))
    if missing:
        details = "; ".join(f"M={i}: missing {', '.join(keys)}" for i, keys in missing)
        raise ValueError(
            f"lif_mosaic='stage-stitch': one or more tiles is missing stage "
            f"position attributes ({details}). The LIF scene XML must carry "
            f"<Tile PosX=.. PosY=..> for every tile. {escape}"
        )


def _median(values: list[float]) -> float:
    """Median of ``values`` — assumes non-empty (caller checks)."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _observed_overlap_pct(
    tiles: list[dict],
    offsets: list[tuple[int, int]],
    tile_h: int,
    tile_w: int,
) -> dict[str, float | None]:
    """Median observed overlap % on each axis from placed-tile pairs.

    Adjacency is inferred from ``field_x``/``field_y`` (a horizontal neighbour
    is the tile at ``(fx+1, fy)``; vertical is ``(fx, fy+1)``). If any tile is
    missing grid indices, the axis with no valid pair yields ``None`` — the
    caller treats a ``None`` observation as "cannot sanity-check", not "0%
    overlap". Overlap is computed as ``(tile_w - dx) / tile_w * 100`` (positive
    == tiles overlap; negative == there's a gap between them).
    """
    slot_to_m: dict[tuple[int, int], int] = {}
    for i, t in enumerate(tiles):
        fx = t.get("field_x")
        fy = t.get("field_y")
        if fx is None or fy is None:
            continue
        slot_to_m[(fx, fy)] = i

    x_overlaps: list[float] = []
    y_overlaps: list[float] = []
    for (fx, fy), i in slot_to_m.items():
        east = slot_to_m.get((fx + 1, fy))
        if east is not None and tile_w > 0:
            dx = offsets[east][1] - offsets[i][1]
            x_overlaps.append((tile_w - dx) / tile_w * 100.0)
        south = slot_to_m.get((fx, fy + 1))
        if south is not None and tile_h > 0:
            dy = offsets[south][0] - offsets[i][0]
            y_overlaps.append((tile_h - dy) / tile_h * 100.0)
    return {
        "x": float(_median(x_overlaps)) if x_overlaps else None,
        "y": float(_median(y_overlaps)) if y_overlaps else None,
    }


def compute_stage_placements(
    tile_layout: dict | None,
    *,
    pixel_size_x_um: float | None,
    pixel_size_y_um: float | None,
    tile_h: int,
    tile_w: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], dict[str, float | None]]:
    """Compute per-tile integer ``(y_px, x_px)`` offsets, canvas ``(H, W)``, observed overlap.

    Converts each tile's ``pos_x_m``/``pos_y_m`` (LIF's stage coordinates in
    meters) to micrometres, divides by the scene's physical pixel size (µm/px),
    normalises so the minimum is ``(0, 0)`` and snaps to the nearest integer
    pixel. Canvas shape is ``max(offset + tile_shape)`` across all tiles.

    Fails loud when the inputs stage-stitch depends on are missing (see
    :func:`_validate_stage_tiles` and the pixel-size branch below) — the whole
    point of stage-stitch is honouring the declared stage geometry; a silent
    fallback would defeat that. Each error message points at
    ``lif_mosaic="grid-stitch"`` as the graceful escape.
    """
    if pixel_size_x_um is None or pixel_size_y_um is None:
        missing = []
        if pixel_size_x_um is None:
            missing.append("X")
        if pixel_size_y_um is None:
            missing.append("Y")
        raise ValueError(
            f"lif_mosaic='stage-stitch': scene physical pixel size is None for "
            f"axis {missing} — cannot convert stage µm positions to pixels. "
            f'Pass lif_mosaic="grid-stitch" to place tiles by FieldX/FieldY '
            f"grid indices instead (no pixel-size dependency)."
        )
    if tile_layout is None:
        raise ValueError(
            "lif_mosaic='stage-stitch' requires per-tile stage positions "
            '(<Tile PosX PosY .../> under <Attachment Name="TileScanInfo"> '
            "in the scene XML), but none could be extracted for the current "
            'scene. Pass lif_mosaic="grid-stitch" to place tiles by '
            "FieldX/FieldY grid indices instead."
        )
    tiles = tile_layout.get("tiles") or []
    _validate_stage_tiles(tiles)

    xs_px_raw = [t["pos_x_m"] * _STAGE_METERS_TO_UM / pixel_size_x_um for t in tiles]
    ys_px_raw = [t["pos_y_m"] * _STAGE_METERS_TO_UM / pixel_size_y_um for t in tiles]
    x0 = min(xs_px_raw)
    y0 = min(ys_px_raw)
    offsets: list[tuple[int, int]] = [
        (int(round(y - y0)), int(round(x - x0)))
        for y, x in zip(ys_px_raw, xs_px_raw, strict=True)
    ]

    canvas_h = max(oy + tile_h for oy, _ox in offsets)
    canvas_w = max(ox + tile_w for _oy, ox in offsets)
    observed = _observed_overlap_pct(tiles, offsets, tile_h, tile_w)
    return offsets, (canvas_h, canvas_w), observed


def _place_tiles_numpy(
    tile_arrays: list[np.ndarray],
    offsets: list[tuple[int, int]],
    canvas_shape: tuple[int, ...],
    dtype: np.dtype,
    y_axis: int,
    x_axis: int,
) -> np.ndarray:
    """Numpy placement kernel — later tiles overwrite earlier in overlap regions.

    Runs inside the :func:`dask.delayed` wrapper so ``tile_arrays`` may arrive
    as numpy arrays (dask has already resolved the per-tile dependencies by
    then). The overwrite policy is deterministic (iteration order == M order)
    and documented on :func:`reassemble_stage` so downstream callers can predict
    which tile "wins" a shared pixel.
    """
    canvas = np.zeros(canvas_shape, dtype=dtype)
    for tile, (oy, ox) in zip(tile_arrays, offsets, strict=True):
        tile_np = np.asarray(tile)
        ty = tile_np.shape[y_axis]
        tx = tile_np.shape[x_axis]
        idx: list[slice] = [slice(None)] * canvas.ndim
        idx[y_axis] = slice(oy, oy + ty)
        idx[x_axis] = slice(ox, ox + tx)
        canvas[tuple(idx)] = tile_np
    return canvas


def reassemble_stage(
    tiles_xarr: xr.DataArray,
    offsets: list[tuple[int, int]],
    canvas_shape_yx: tuple[int, int],
) -> xr.DataArray:
    """Reassemble raw M-intact tiles onto a canvas at the given pixel offsets.

    Given the reader's raw M-intact xarray (dims include ``M``, ``Y``, ``X``)
    and the per-tile ``(y_px, x_px)`` offsets from
    :func:`compute_stage_placements`, returns a new xarray with the ``M`` dim
    consumed. Later-placed tiles overwrite earlier-placed tiles in overlap
    regions (iteration order == M order — deterministic, no blending in this
    slice). T/C/Z axes and their coords are preserved untouched. The returned
    array remains dask-backed — placement is wrapped in :func:`dask.delayed`
    so no eager compute happens until the writer materialises pixels.
    """
    m_size = int(tiles_xarr.sizes["M"])
    if len(offsets) != m_size:
        raise ValueError(
            f"lif_mosaic='stage-stitch': placement offsets count ({len(offsets)}) "
            f"does not match array M dim ({m_size})"
        )
    non_m_dims = [d for d in tiles_xarr.dims if d != "M"]
    y_axis = non_m_dims.index("Y")
    x_axis = non_m_dims.index("X")

    per_tile_shape = tuple(int(s) for s in tiles_xarr.isel(M=0).shape)
    canvas_shape = list(per_tile_shape)
    canvas_shape[y_axis] = canvas_shape_yx[0]
    canvas_shape[x_axis] = canvas_shape_yx[1]
    canvas_shape_t = tuple(canvas_shape)
    dtype = tiles_xarr.dtype

    tile_dependencies = [tiles_xarr.isel(M=i, drop=True).data for i in range(m_size)]

    delayed_canvas = dask.delayed(_place_tiles_numpy)(
        tile_dependencies, offsets, canvas_shape_t, dtype, y_axis, x_axis
    )
    canvas_data = da.from_delayed(delayed_canvas, shape=canvas_shape_t, dtype=dtype)

    # See reassemble_grid: Y/X coords carry per-tile positions and can't
    # be reused on a canvas of a different size.
    preserved_coords = {
        name: coord
        for name, coord in tiles_xarr.coords.items()
        if "M" not in coord.dims and "Y" not in coord.dims and "X" not in coord.dims
    }
    return xr.DataArray(canvas_data, dims=non_m_dims, coords=preserved_coords)


def stage_overlap_discrepancy(
    observed: dict[str, float | None],
    intended_x_pct: float | None,
    intended_y_pct: float | None,
    *,
    tolerance_frac: float = _STAGE_OVERLAP_TOLERANCE_FRAC,
) -> list[tuple[str, float, float]]:
    """List ``(axis, observed_pct, intended_pct)`` tuples exceeding tolerance.

    Compares each axis's observed overlap (from
    :func:`_observed_overlap_pct`) against the LIF-declared intended overlap
    from ``StitchingSettings``. An axis is flagged when both values exist AND
    ``|observed − intended| / intended > tolerance_frac``. When either value
    is ``None`` the axis is silently skipped — the sanity check cannot fire
    without both a placement and a declared target to compare against.
    """
    findings: list[tuple[str, float, float]] = []
    for axis, intended, obs_key in (
        ("x", intended_x_pct, "x"),
        ("y", intended_y_pct, "y"),
    ):
        if intended is None or intended == 0:
            continue
        obs = observed.get(obs_key)
        if obs is None:
            continue
        if abs(obs - intended) / abs(intended) > tolerance_frac:
            findings.append((axis, obs, intended))
    return findings
