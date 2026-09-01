"""OME-Zarr → OME-Zarr geometry migration (ADR-0012, issue #91).

``convert`` turns a vendor file into a store planned by the current ADR-0010
geometry policy. This module turns a store planned by *some other* geometry into
one planned by the current policy, without going back to the vendor file — which
for a 1.3 TiB light-sheet acquisition is the difference between a streaming pass
over the converted pixels and a re-read of the source through Bio-Formats.

Four things distinguish it from ``convert``-over-the-old-store, and they are the
whole reason it is a separate command rather than a reader plugin:

* **It reads zarr, not pixels through a reader.** The store's own metadata says
  what its layout, axes, spacing, channels and OME-XML are, so nothing has to be
  re-derived and nothing is lost. Going through ``bioio-ome-zarr`` would work for
  the voxels and would silently replace the vendor provenance in the audit with
  "the reader was ``bioio-ome-zarr`` and the input was the previous store".
* **The pass is read-once.** The unit of work is the element-wise LCM of the
  source's write grid and the planned write grid, clamped to the level extent
  (:func:`read_once_tile`) — the smallest block that is simultaneously a whole
  number of source objects and a whole number of target objects. Every source
  object is therefore touched exactly once for any input chunking, and the
  old-geometry case ADR-0010 called out (full-width single-plane slabs rebuilt
  into 64³ chunks, read in Z bands of 64) falls out of the rule rather than
  being special-cased.
* **It resumes.** Progress is a high-water mark per ``(image, level)`` over a
  deterministic tile order, so the state is O(1) whether the level holds 171
  tiles or 120,000. Because a tile is a whole multiple of the write grid, a
  crash can only tear objects the high-water mark has not yet claimed, and
  re-running rewrites them whole.
* **A partial target is not an OME-Zarr.** The ``attrs.ome`` every consumer
  needs is withheld until the run finishes, so an interrupted store is refused
  by everything that reads OME-Zarr, for free and without their cooperation.
  Resume state lives beside it under ``attrs.zarrmony_rechunk`` and is deleted
  when the run completes.

The source is opened read-only and never written to. ``force`` means what it
means for ``convert``: overwrite the *target* if it already exists.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import fsspec
import numpy as np
from bioio_ome_zarr.writers import Channel

from zarrmony import __version__
from zarrmony._constants import NGFF_VERSION
from zarrmony._storage import (
    format_bytes,
    open_root_group,
    prepare_output_path,
    size_on_disk,
)
from zarrmony._validate import run_validation
from zarrmony.audit import AUDIT_SCHEMA_VERSION, write_audit_record
from zarrmony.errors import (
    RechunkSourceError,
    RechunkStateError,
    RechunkVerificationError,
    WorkingSetTooLargeError,
)
from zarrmony.geometry import (
    DEFAULT_GEOMETRY,
    DownsampleMethod,
    Geometry,
    plan_level_chunk_shapes,
    plan_level_shard_shapes,
)
from zarrmony.writers.pyramid import (
    coarse_level_index,
    compute_level_shapes,
    downsample_block,
    pool_factors,
)
from zarrmony.writers.scene import (
    _CONTRAST_METHOD,
    ZarrmonyWriter,
    _channel_contrast_ops,
    _dtype_window,
    _pair_contrast_results,
    _to_json_scalar,
    _update_omero_window_start_end,
    _write_grid,
)

VerifyMode = Literal["none", "sample", "full"]
#: A layout one store can be in — the same three ``convert`` writes.
StoreLayout = Literal["per-scene", "bf2raw", "plate"]
#: Plus the fourth thing ``rechunk`` accepts: a plain directory of sibling
#: ``*.ome.zarr`` stores, which is not a layout so much as a pile of them.
RechunkLayout = Literal["per-scene", "bf2raw", "plate", "sibling-directory"]

#: Root attribute holding resume state while a target is being written. Its
#: presence, together with the *absence* of ``attrs.ome``, is what makes a
#: partial target unmistakable — and unopenable — as a finished store.
STATE_KEY = "zarrmony_rechunk"
STATE_VERSION = 1

#: How often the high-water marks are flushed to the target's attrs mid-level.
#: Each flush rewrites one small JSON object, so the cost is negligible against
#: a multi-hour pass; the number bounds how much work a crash can discard.
#: Level boundaries checkpoint unconditionally on top of this.
CHECKPOINT_SECONDS = 30.0

#: Fraction of detected physical RAM the read-once tile may occupy by default.
#: Not a hardcoded byte figure: the tile is set by the *source's* chunking, and
#: on the reference light-sheet volume it is 7.85 GiB per channel unsharded and
#: 15.7 GiB sharded — numbers that are unremarkable on the conversion host and
#: impossible on a laptop, so the budget has to scale with the machine. The
#: headroom below 1.0 is not decoration: the measured peak RSS on the volumetric
#: path ran 2.0x the gather itself, because compression buffers and the shard
#: encoder live alongside the block.
DEFAULT_WORKING_SET_FRACTION = 0.5

#: Sentinel for "inherit the contrast percentile from the source's own audit".
#: Distinct from ``None``, which means "no contrast pass" — a source converted
#: with ``--no-contrast`` records ``None`` and must keep its dtype-range window.
INHERIT: Any = object()

_PLANNABLE_AXES = frozenset("TCZYX")


# --------------------------------------------------------------------------
# Paths and small helpers
# --------------------------------------------------------------------------


def _join(base: str | Path, key: str) -> str:
    """Append a group key to a store path, preserving any URI scheme."""
    b = str(base).rstrip("/")
    return f"{b}/{key}" if key else b


def _jsonable(value: Any) -> Any:
    """Coerce planner output (tuples, numpy scalars) into zarr-writable JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return str(value)
    return value


def _physical_memory_bytes() -> int | None:
    """Total physical RAM, or ``None`` where the platform will not say.

    ``os.sysconf`` answers on Linux and macOS, which is every host zarrmony has
    ever run a conversion on, and costs no new dependency. A platform that
    declines simply gets no default budget — the working set is still computed
    and printed, it is just not compared against anything.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    if pages < 0 or page_size < 0:
        return None
    return int(pages) * int(page_size)


# --------------------------------------------------------------------------
# Reading the source store
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceImage:
    """One multiscale image inside a source store, as its own metadata reports it.

    ``key`` is the group path relative to the store root — ``""`` for a
    single-image ``.ome.zarr``, ``"0"`` for a bf2raw series member, ``"B/03/0"``
    for a plate field. It is the identity everything downstream keys on: the
    target group to write, the resume high-water marks, and the audit record to
    patch.
    """

    key: str
    name: str | None
    axes: tuple[dict[str, Any], ...]
    dims: tuple[str, ...]
    dtype: np.dtype
    spacings: tuple[float, ...]
    level_paths: tuple[str, ...]
    level_shapes: tuple[tuple[int, ...], ...]
    level_chunks: tuple[tuple[int, ...], ...]
    level_shards: tuple[tuple[int, ...], ...] | None
    level_write_grids: tuple[tuple[int, ...], ...]
    level0_transforms: tuple[dict[str, Any], ...]
    ome: dict[str, Any]

    @property
    def base_shape(self) -> tuple[int, ...]:
        return self.level_shapes[0]

    @property
    def base_write_grid(self) -> tuple[int, ...]:
        return self.level_write_grids[0]


def _scale_of(transforms: Sequence[Any]) -> list[float] | None:
    for t in transforms or ():
        if isinstance(t, dict) and t.get("type") == "scale":
            return [float(v) for v in t["scale"]]
    return None


def read_source_image(base: str | Path, key: str) -> SourceImage:
    """Read one image's geometry and metadata straight off the store.

    Everything the geometry planners need — axes, level-0 shape, dtype and µm
    spacing — is in the store's own ``multiscales`` block and array metadata, so
    no reader is opened and nothing is inferred. The spacing composes the
    multiscale-level ``coordinateTransformations`` with the level-0 dataset's,
    as NGFF specifies; zarrmony writes only the latter, but a store from another
    writer may split them.
    """
    group_path = _join(base, key)
    group = open_root_group(group_path, mode="r")
    ome = dict(group.attrs.get("ome") or {})
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        raise RechunkSourceError(
            f"no multiscales metadata at {group_path!r}; rechunk reads a store's "
            f"own OME-Zarr metadata and this group declares none"
        )
    ms = dict(multiscales[0])

    axes = tuple(dict(a) for a in (ms.get("axes") or ()))
    if not axes:
        raise RechunkSourceError(
            f"the multiscales block at {group_path!r} declares no axes, so there "
            f"is nothing to say which extent is Z, Y or X"
        )
    dims = tuple(str(a.get("name", "")).upper() for a in axes)
    unplannable = sorted({d for d in dims if d not in _PLANNABLE_AXES})
    if unplannable:
        raise RechunkSourceError(
            f"{group_path!r} carries axes {unplannable} that the geometry policy "
            f"does not plan for; rechunk handles the OME-NGFF T/C/Z/Y/X axes"
        )

    datasets = list(ms.get("datasets") or ())
    if not datasets:
        raise RechunkSourceError(
            f"the multiscales block at {group_path!r} lists no datasets"
        )
    level_paths = tuple(str(d["path"]) for d in datasets)

    level0_transforms = tuple(
        dict(t) for t in (datasets[0].get("coordinateTransformations") or ())
    )
    scale = _scale_of(level0_transforms)
    if scale is None:
        raise RechunkSourceError(
            f"level 0 of {group_path!r} carries no scale transform, so its "
            f"physical voxel spacing is unknown and the pyramid cannot be planned"
        )
    outer = _scale_of(ms.get("coordinateTransformations") or ())
    if outer is not None:
        scale = [s * o for s, o in zip(scale, outer, strict=True)]

    arrays = []
    for path in level_paths:
        try:
            arrays.append(group[path])
        except KeyError as e:
            raise RechunkSourceError(
                f"{group_path!r} lists a level at {path!r} that is not in the store"
            ) from e

    shards = tuple(
        tuple(int(s) for s in a.shards) if a.shards is not None else None
        for a in arrays
    )
    return SourceImage(
        key=key,
        name=ms.get("name"),
        axes=axes,
        dims=dims,
        dtype=np.dtype(arrays[0].dtype),
        spacings=tuple(float(s) for s in scale),
        level_paths=level_paths,
        level_shapes=tuple(tuple(int(s) for s in a.shape) for a in arrays),
        level_chunks=tuple(tuple(int(c) for c in a.chunks) for a in arrays),
        level_shards=shards if any(s is not None for s in shards) else None,  # type: ignore[arg-type]
        level_write_grids=tuple(_write_grid(a) for a in arrays),
        level0_transforms=level0_transforms,
        ome=ome,
    )


def detect_layout(base: str | Path) -> RechunkLayout:
    """Which of the three output layouts this store is, from its root attrs.

    The store says what it is, so there is no ``--layout`` override: a plate is a
    plate because it carries ``attrs.ome.plate``, and forcing a different answer
    would either invent structure that is not there or discard structure that is.
    ``"sibling-directory"`` is the fourth case — a plain directory of
    ``*.ome.zarr`` stores, which ``convert --layout per-scene`` produces and
    which is therefore the shape most existing output trees are in.
    """
    if sibling_stores(base):
        return "sibling-directory"
    try:
        root = open_root_group(base, mode="r")
    except Exception as e:  # noqa: BLE001 - zarr raises several unrelated types
        raise RechunkSourceError(
            f"cannot open {str(base)!r} as a zarr group: {type(e).__name__}: {e}"
        ) from e
    ome = root.attrs.get("ome") or {}
    if "plate" in ome:
        return "plate"
    if "bioformats2raw.layout" in ome:
        return "bf2raw"
    if "multiscales" in ome:
        return "per-scene"
    raise RechunkSourceError(
        f"{str(base)!r} is a zarr group but its root carries no OME-Zarr layout "
        f"metadata — no 'multiscales', no 'bioformats2raw.layout', no 'plate'. "
        f"If it is a directory of sibling .ome.zarr stores, none of them opened."
    )


def _store_layout(base: str | Path) -> StoreLayout:
    """:func:`detect_layout` for something already known to be one store.

    A directory of siblings is not a layout a single store can have, so reaching
    it here means a ``.ome.zarr``-named child that is itself a pile of stores —
    nesting zarrmony does not produce and will not silently flatten.
    """
    layout = detect_layout(base)
    if layout == "sibling-directory":
        raise RechunkSourceError(
            f"{str(base)!r} is a directory of sibling .ome.zarr stores rather "
            f"than a single store; rechunk fans out over one level of siblings, "
            f"not over nested directories of them"
        )
    return layout


def sibling_stores(base: str | Path) -> list[str]:
    """Names of ``*.ome.zarr`` children directly under a plain directory.

    Empty when ``base`` is itself a zarr group (it has a root ``zarr.json``), so
    a bf2raw bundle whose series subgroups happen to be named that way is never
    mistaken for a directory of siblings.
    """
    fs, path = fsspec.core.url_to_fs(str(base))
    path = path.rstrip("/")
    if not fs.exists(path) or not fs.isdir(path):
        return []
    if fs.exists(f"{path}/zarr.json"):
        return []
    names = []
    for entry in fs.ls(path, detail=False):
        name = str(entry).rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".ome.zarr") and fs.exists(f"{path}/{name}/zarr.json"):
            names.append(name)
    return sorted(names)


def discover_images(base: str | Path, layout: StoreLayout) -> list[str]:
    """Group keys of every multiscale image in the store, in written order.

    Order matters: the audit's ``per_scene`` / ``fields`` list is positional, so
    walking the plate's own ``wells`` and each well's own ``images`` — rather
    than listing the store — is what keeps the rechunked audit's per-image
    records aligned with the source's.
    """
    root = open_root_group(base, mode="r")
    ome = root.attrs.get("ome") or {}
    if layout == "per-scene":
        return [""]
    if layout == "bf2raw":
        try:
            series = list(root["OME"].attrs.get("ome", {}).get("series") or ())
        except KeyError:
            series = []
        if not series:
            raise RechunkSourceError(
                f"{str(base)!r} declares bioformats2raw.layout but its OME group "
                f"lists no series, so there is no way to know which subgroups are "
                f"images or what order they were written in"
            )
        return [str(s) for s in series]
    if layout == "plate":
        keys: list[str] = []
        for well in ome["plate"].get("wells") or ():
            well_path = str(well["path"])
            well_group = root[well_path]
            images = (well_group.attrs.get("ome") or {}).get("well", {}).get("images")
            for image in images or ():
                keys.append(f"{well_path}/{image['path']}")
        if not keys:
            raise RechunkSourceError(
                f"the plate at {str(base)!r} lists no imaged wells"
            )
        return keys
    raise RechunkSourceError(f"unknown layout: {layout!r}")


# --------------------------------------------------------------------------
# The read-once tile
# --------------------------------------------------------------------------


def read_once_tile(
    source_grid: Sequence[int],
    write_grid: Sequence[int],
    extent: Sequence[int],
) -> tuple[int, ...]:
    """The smallest block that reads each source object once and writes whole ones.

    Element-wise ``lcm(source, target)``, clamped to the level extent. Both
    halves are load-bearing and neither can be relaxed:

    * A multiple of the **source** grid means no source object is ever needed by
      two different tiles, which is the streaming guarantee — one pass over the
      input, no matter how the input was chunked.
    * A multiple of the **target** grid means no storage object is ever written
      by two different tiles, which is what makes a crash recoverable: the only
      objects that can be torn belong to the tile in flight, and re-running
      rewrites that tile whole.

    Clamping to the extent does not break either property. An axis whose LCM
    exceeds its extent holds exactly one tile, so it has no interior boundary
    for either grid to straddle.

    The old-geometry case ADR-0010 predicted is this rule's easiest instance
    rather than a special case: a source of ``[1,1,1,Y,X]`` full-width slabs
    against a ``[1,1,64,64,64]`` write grid gives ``lcm`` of ``[1,1,64,Y,X]`` —
    a Z band of 64 planes, exactly the "read Z in bands of 64" the ADR describes.
    It is also where the working set comes from, and why it is checked before
    anything is written.
    """
    return tuple(
        min(int(e), math.lcm(max(1, int(s)), max(1, int(w))))
        for s, w, e in zip(source_grid, write_grid, extent, strict=True)
    )


def pooled_tile(
    write_grid: Sequence[int],
    parent_grid: Sequence[int],
    factors: Sequence[int],
    extent: Sequence[int],
) -> tuple[int, ...]:
    """The output tile for a pyramid level pooled from its parent on disk.

    Same two constraints as :func:`read_once_tile`, one level up: the tile must
    be a whole multiple of this level's write grid, and the parent region it
    pools — the tile scaled by the coarsen factors — must be a whole multiple of
    the *parent's* write grid. The second gives ``tile * f ≡ 0 (mod parent)``,
    which holds exactly when the tile is a multiple of
    ``parent // gcd(parent, f)``.

    At the default policy this is almost always just the write grid itself
    (equal grids, factor 2), so the parent block is 2× per halved axis and the
    working set stays at a few multiples of one storage object. It matters on a
    pyramid whose grid changes shape partway down, which the reference volume's
    does when its long axis flips from X to Y.
    """
    out = []
    for grid, parent, factor, ext in zip(
        write_grid, parent_grid, factors, extent, strict=True
    ):
        parent, factor = max(1, int(parent)), max(1, int(factor))
        needed = parent // math.gcd(parent, factor)
        out.append(min(int(ext), math.lcm(max(1, int(grid)), needed)))
    return tuple(out)


def _tile_count(shape: Sequence[int], tile: Sequence[int]) -> int:
    return math.prod(
        math.ceil(int(e) / max(1, int(t))) for e, t in zip(shape, tile, strict=True)
    )


def _iter_tiles(
    shape: Sequence[int], tile: Sequence[int]
) -> Iterator[tuple[slice, ...]]:
    """Every tile of ``shape``, in C order — the order the high-water mark counts.

    ``itertools.product`` varies the last axis fastest and is stable across
    interpreter versions, so tile *n* of a resumed run is the same tile *n* the
    interrupted one was writing. That is the whole basis of an O(1) resume state.
    """
    starts = [
        range(0, int(e), max(1, int(t))) for e, t in zip(shape, tile, strict=True)
    ]
    for origin in itertools.product(*starts):
        yield tuple(
            slice(o, min(o + int(t), int(e)))
            for o, t, e in zip(origin, tile, shape, strict=True)
        )


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImagePlan:
    """What the current policy plans for one source image, and what it will cost."""

    image: SourceImage
    level_shapes: tuple[tuple[int, ...], ...]
    chunk_shapes: tuple[tuple[int, ...], ...]
    shard_shapes: tuple[tuple[int, ...], ...] | None
    coarse_index: int | None
    write_grids: tuple[tuple[int, ...], ...]
    tiles: tuple[tuple[int, ...], ...]
    tile_counts: tuple[int, ...]
    working_set_bytes: tuple[int, ...]

    @property
    def peak_working_set_bytes(self) -> int:
        return max(self.working_set_bytes)

    @property
    def is_noop(self) -> bool:
        """Whether the store already has exactly the geometry being planned.

        Compared against the arrays on disk rather than against the source's
        audit, because the audit records what a past run intended and the arrays
        record what is there. A no-op is reported and skipped rather than
        rewritten — which is what lets a fan-out over a directory of stores be
        re-run after an interruption without tracking which ones finished.
        """
        src = self.image
        if src.level_shapes != self.level_shapes:
            return False
        if src.level_chunks != self.chunk_shapes:
            return False
        planned_shards = self.shard_shapes
        if (src.level_shards is None) != (planned_shards is None):
            return False
        return planned_shards is None or src.level_shards == planned_shards

    def fingerprint(self) -> dict[str, Any]:
        """The identity a resumed run must still agree with.

        Deliberately the *resolved plan* rather than the ``Geometry`` that
        produced it. Two zarrmony versions can hold the same policy and plan
        different shapes from it — that is what a planner fix is — and what a
        half-written store cannot survive is a change in the shapes, not a
        change in the policy's spelling.
        """
        src = self.image
        return _jsonable(
            {
                "key": src.key,
                "source_level_shapes": src.level_shapes,
                "source_level_chunks": src.level_chunks,
                "source_level_shards": src.level_shards,
                "source_dtype": str(src.dtype),
                "level_shapes": self.level_shapes,
                "chunk_shapes": self.chunk_shapes,
                "shard_shapes": self.shard_shapes,
            }
        )


def plan_image(image: SourceImage, geometry: Geometry) -> ImagePlan:
    """Resolve the current policy against one source image.

    Uses the same four planner calls ``write_scene`` makes, in the same order and
    with the same inputs, so a rechunked store's shapes are the shapes a
    re-conversion would have picked — that equality is one half of the
    acceptance criterion and it is bought here, by not having a second planner.
    """
    level_shapes = compute_level_shapes(
        image.base_shape, image.dims, image.spacings, image.dtype, geometry
    )
    coarse = coarse_level_index(level_shapes, image.dims, image.dtype, geometry)
    chunk_shapes = plan_level_chunk_shapes(
        level_shapes, image.dims, image.spacings, image.dtype, geometry
    )
    shard_shapes = plan_level_shard_shapes(
        chunk_shapes, level_shapes, image.dims, image.spacings, image.dtype, geometry
    )
    write_grids = tuple(
        tuple(int(s) for s in (shard_shapes[i] if shard_shapes else chunk_shapes[i]))
        for i in range(len(level_shapes))
    )
    shapes = tuple(tuple(int(s) for s in shape) for shape in level_shapes)

    itemsize = int(image.dtype.itemsize)
    tiles: list[tuple[int, ...]] = [
        read_once_tile(image.base_write_grid, write_grids[0], shapes[0])
    ]
    working: list[int] = [math.prod(tiles[0]) * itemsize]
    for i in range(1, len(shapes)):
        factors = pool_factors(shapes[i - 1], shapes[i])
        tile = pooled_tile(write_grids[i], write_grids[i - 1], factors, shapes[i])
        parent_block = tuple(
            min(int(p), int(t) * int(f))
            for p, t, f in zip(shapes[i - 1], tile, factors, strict=True)
        )
        tiles.append(tile)
        # Both blocks are resident at once: the parent region is read whole and
        # the pooled result is built beside it before the write.
        working.append((math.prod(parent_block) + math.prod(tile)) * itemsize)

    return ImagePlan(
        image=image,
        level_shapes=shapes,
        chunk_shapes=tuple(tuple(int(c) for c in c_) for c_ in chunk_shapes),
        shard_shapes=(
            tuple(tuple(int(s) for s in s_) for s_ in shard_shapes)
            if shard_shapes is not None
            else None
        ),
        coarse_index=coarse,
        write_grids=write_grids,
        tiles=tuple(tiles),
        tile_counts=tuple(_tile_count(shapes[i], tiles[i]) for i in range(len(shapes))),
        working_set_bytes=tuple(working),
    )


def _check_working_set(plans: Sequence[ImagePlan], budget: int | None) -> None:
    """Refuse a run whose read-once tile cannot fit, before anything is written."""
    if budget is None:
        return
    for plan in plans:
        peak = plan.peak_working_set_bytes
        if peak <= budget:
            continue
        src = plan.image
        label = src.key or "the image"
        raise WorkingSetTooLargeError(
            f"reading {label} once needs a {list(plan.tiles[0])} block of "
            f"{format_bytes(plan.working_set_bytes[0])} (peak across levels "
            f"{format_bytes(peak)}), above the {format_bytes(budget)} budget. "
            f"That size is set by the source's own grid of "
            f"{list(src.base_write_grid)} meeting the planned "
            f"{list(plan.write_grids[0])}: the block has to be a whole number of "
            f"both, or a source object would be read twice. Raise the budget if "
            f"the machine has the memory, plan a target grid that is shorter on "
            f"the axis the source spans whole, or run this on a larger host."
        )


# --------------------------------------------------------------------------
# Resume state
# --------------------------------------------------------------------------


def _read_state(output: str | Path) -> dict[str, Any] | None:
    """Resume state on an existing target, or ``None`` if there is none to read."""
    fs, path = fsspec.core.url_to_fs(str(output))
    if not fs.exists(f"{path.rstrip('/')}/zarr.json"):
        return None
    try:
        root = open_root_group(output, mode="r")
    except Exception:  # noqa: BLE001 - an unreadable target is simply not resumable
        return None
    state = root.attrs.get(STATE_KEY)
    return dict(state) if isinstance(state, dict) else None


def _write_state(output: str | Path, state: dict[str, Any]) -> None:
    """Flush resume state to the target root, re-reading the group each time.

    Re-opening rather than holding one :class:`zarr.Group` is deliberate.
    ``Attributes.__setitem__`` is a read-modify-write against the *cached*
    metadata on the group object, so a long-lived handle that missed an
    intervening write by another handle would silently drop it on its next
    flush — and every other attrs write in this module (the restored
    ``attrs.ome``, the omero window, the audit) goes through its own handle.
    One small JSON read per checkpoint is not a cost worth reasoning about.
    """
    root = open_root_group(output, mode="a")
    root.attrs[STATE_KEY] = _jsonable(state)


def _diff_fingerprints(old: Any, new: Any, path: str = "") -> str | None:
    """The first field where two plan fingerprints disagree, as a readable path."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            where = f"{path}.{key}" if path else str(key)
            if key not in old:
                return f"{where} (absent in the partial target)"
            if key not in new:
                return f"{where} (absent in the new plan)"
            found = _diff_fingerprints(old[key], new[key], where)
            if found:
                return found
        return None
    if old != new:
        return f"{path or 'plan'}: {old!r} on disk, {new!r} now"
    return None


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _channels_from_source(image: SourceImage) -> list[Channel] | None:
    """Rebuild writer ``Channel``s from the source store's own omero block.

    Only shapes what the writer emits up front; the finished store's omero block
    is copied from the source wholesale afterwards, with the display window
    recomputed. Passing them here anyway means the intermediate store is
    self-consistent at every moment rather than only at the end.
    """
    omero = (image.ome.get("omero") or {}).get("channels") or []
    if not omero:
        return None
    fallback = _dtype_window(image.dtype)
    return [
        Channel(
            label=str(c.get("label", f"C:{i}")),
            color=str(c.get("color", "FFFFFF")),
            window=dict(c.get("window") or fallback),
        )
        for i, c in enumerate(omero)
    ]


def _make_writer(
    store_path: str, plan: ImagePlan, *, resuming: bool = False
) -> ZarrmonyWriter:
    """Create the target image group and its empty pyramid arrays.

    The same :class:`~zarrmony.writers.scene.ZarrmonyWriter` ``convert`` uses,
    given the same planner output, so the target's ``zarr.json`` files — codec
    chain, fill value, dimension names, ``sharding_indexed`` configuration — are
    what a re-conversion would have written rather than an approximation of it.

    ``resuming`` binds to the arrays a previous run already created instead of
    creating them. It is not an optimization: the writer's initialization opens
    the root group in ``"w"`` mode, so re-running it on a resumed target would
    empty the store and throw away everything the high-water mark says is done.
    """
    image = plan.image
    writer = ZarrmonyWriter(
        store=store_path,
        level_shapes=[list(s) for s in plan.level_shapes],
        dtype=image.dtype,
        zarr_format=3,
        image_name=image.name,
        channels=_channels_from_source(image),
        axes_names=[str(a.get("name")) for a in image.axes],
        axes_types=[a.get("type") for a in image.axes],
        axes_units=[a.get("unit") for a in image.axes],
        physical_pixel_size=list(image.spacings),
        chunk_shape=[list(c) for c in plan.chunk_shapes],
        shard_shape=(
            [list(s) for s in plan.shard_shapes]
            if plan.shard_shapes is not None
            else None
        ),
    )
    if resuming:
        writer.attach()
    else:
        writer.initialize()
    return writer


def _write_level_zero(
    source_array: Any,
    target_array: Any,
    plan: ImagePlan,
    done: int,
    flush: Callable[[int], None],
) -> int:
    """Copy level 0 tile by tile, resuming after the ``done``-th tile.

    A plain zarr slice assignment rather than a dask graph. That is not
    austerity: handing a whole level to ``da.rechunk`` is the pathology #111 and
    #113 measured, where a single volume builds millions of tasks before writing
    a byte. Here the graph is one tile wide and zarr's own async chunk I/O does
    the parallelism, so the task count is bounded by the tile no matter how big
    the level is.

    The voxels make no round trip through a codec they did not already make —
    the block read out of the source is the block handed to the target — which
    is what "bit-identical level 0" rests on, and what the verification pass
    then checks in place.
    """
    last_flush = time.monotonic()
    for index, region in enumerate(_iter_tiles(plan.level_shapes[0], plan.tiles[0])):
        if index < done:
            continue
        target_array[region] = source_array[region]
        done = index + 1
        now = time.monotonic()
        if now - last_flush >= CHECKPOINT_SECONDS:
            flush(done)
            last_flush = now
    return done


def _write_level(
    parent_array: Any,
    target_array: Any,
    plan: ImagePlan,
    level: int,
    geometry: Geometry,
    done: int,
    flush: Callable[[int], None],
) -> int:
    """Pool one level from the level below it, on disk, tile by tile.

    Same shape of loop as level 0 and the same resume unit, with the pooling
    done by :func:`~zarrmony.writers.pyramid.downsample_block` so the kernel and
    the ``trim_excess`` behaviour are the pyramid's own. Reading the parent back
    off the store rather than holding it is what
    :meth:`~zarrmony.writers.scene.ZarrmonyWriter.write_pyramid` already does for
    the same reason (issue #111); the levels are exact under either, so the
    result matches a re-conversion's.
    """
    factors = pool_factors(plan.level_shapes[level - 1], plan.level_shapes[level])
    parent_shape = plan.level_shapes[level - 1]
    last_flush = time.monotonic()
    for index, region in enumerate(
        _iter_tiles(plan.level_shapes[level], plan.tiles[level])
    ):
        if index < done:
            continue
        parent_region = tuple(
            slice(r.start * f, min(int(p), r.stop * f))
            for r, f, p in zip(region, factors, parent_shape, strict=True)
        )
        pooled = downsample_block(
            np.asarray(parent_array[parent_region]), factors, geometry
        )
        expected = tuple(r.stop - r.start for r in region)
        if pooled.shape != expected:
            raise RechunkVerificationError(
                f"pooling level {level - 1} into level {level} produced a "
                f"{pooled.shape} block where the planned level shape needs "
                f"{expected}; this is a planner/pooling disagreement, not a "
                f"data problem, and the target is incomplete"
            )
        target_array[region] = pooled
        done = index + 1
        now = time.monotonic()
        if now - last_flush >= CHECKPOINT_SECONDS:
            flush(done)
            last_flush = now
    return done


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _verify_seed(fingerprint: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def verify_level_zero(
    source_array: Any,
    target_array: Any,
    plan: ImagePlan,
    mode: VerifyMode,
    seed: int,
) -> dict[str, Any]:
    """Read level 0 back and compare it against the source at the same offsets.

    The failure this exists to catch is **placement** — a tile's voxels landing
    at another tile's offset. Note what would not catch it: hashing the block as
    read and again as written. Level 0 is a pure copy, so both digests agree
    while the data sits in the wrong place. Only a read-back keyed by global
    coordinates is decisive.

    ``"sample"`` reads one target-chunk-sized block out of every written tile, at
    a chunk-aligned offset drawn from a generator seeded by the plan
    fingerprint. One chunk is decisive for the tile that contains it, so no tile
    goes unchecked, and the cost is the chunk count rather than the byte count —
    on the reference volume, 171 chunks against a 262 GB level, under a
    thousandth of a percent. Re-running checks the same offsets, which makes a
    reported pass reproducible.

    ``"full"`` reads every tile back for the case where that is worth the second
    pass over the data. ``"none"`` skips it.
    """
    if mode == "none":
        return {"mode": "none", "blocks_checked": 0, "passed": None}

    shape = plan.level_shapes[0]
    tile = plan.tiles[0]
    rng = np.random.default_rng(seed)
    checked = 0
    for region in _iter_tiles(shape, tile):
        if mode == "full":
            probe = region
        else:
            probe = _sample_block(region, plan.chunk_shapes[0], rng)
        if not np.array_equal(
            np.asarray(source_array[probe]), np.asarray(target_array[probe])
        ):
            origin = [int(s.start) for s in probe]
            raise RechunkVerificationError(
                f"level-0 voxels at offset {origin} differ between the source "
                f"and the rechunked store. Level 0 is copied verbatim, so a "
                f"mismatch means a block was written at the wrong offset. The "
                f"target is left in place for inspection."
            )
        checked += 1
    return {"mode": mode, "blocks_checked": checked, "passed": True}


def _sample_block(
    region: Sequence[slice], chunk: Sequence[int], rng: np.random.Generator
) -> tuple[slice, ...]:
    """One chunk-aligned block drawn from inside ``region``."""
    probe = []
    for span, length in zip(region, chunk, strict=True):
        length = max(1, int(length))
        extent = span.stop - span.start
        n = max(1, math.ceil(extent / length))
        offset = span.start + int(rng.integers(n)) * length
        probe.append(slice(offset, min(offset + length, span.stop)))
    return tuple(probe)


# --------------------------------------------------------------------------
# Metadata copy
# --------------------------------------------------------------------------


def merge_image_ome(source_ome: dict[str, Any], target_ome: dict[str, Any]) -> dict:
    """The finished image's ``attrs.ome``: the source's, with geometry replaced.

    Built by starting from the source and overwriting what changed rather than
    starting from the writer's output and copying things back. The two differ in
    what happens to a key neither zarrmony writer knows about — ``omero.rdefs``,
    a viewer's own annotation, a downstream tool's marker — and starting from the
    source is what preserves it.

    What geometry changed is exactly the dataset list: one entry per new level,
    each with the scale the new level shapes imply. Level 0's transforms are put
    back verbatim from the source, since level 0's voxels are unchanged and so is
    its physical spacing; anything the source carried there beyond the scale — a
    translation, say — survives with them.
    """
    merged = deepcopy(source_ome)
    merged["version"] = target_ome.get("version", NGFF_VERSION)
    target_ms = (target_ome.get("multiscales") or [{}])[0]
    datasets = deepcopy(target_ms.get("datasets") or [])
    source_ms = (merged.get("multiscales") or [{}])[0]
    source_datasets = source_ms.get("datasets") or []
    if datasets and source_datasets:
        source_l0 = source_datasets[0].get("coordinateTransformations")
        if source_l0:
            datasets[0] = {**datasets[0], "coordinateTransformations": source_l0}
    source_ms["datasets"] = datasets
    if "omero" not in merged and "omero" in target_ome:
        merged["omero"] = deepcopy(target_ome["omero"])
    merged["multiscales"] = [source_ms, *list(merged.get("multiscales") or [])[1:]]
    return merged


def copy_sidecars(source: str | Path, output: str | Path) -> list[str]:
    """Copy the ``OME/`` subtree — METADATA.ome.xml, raw vendor XML — verbatim.

    Geometry-independent by construction: OME-XML carries ``SizeX/Y/Z/C/T``,
    pixel type, physical sizes, channels, planes and instrument, and says nothing
    about how any of it is chunked. So there is nothing to regenerate, and
    regenerating it would risk losing whatever the original converter put there.
    Copied byte for byte, including the ``OME`` group's own ``zarr.json`` where
    there is one, which is what carries a bf2raw bundle's ``series`` list.

    Runs at the end, after the pixels: a partial target has no reason to look
    more finished than it is.
    """
    src_fs, src_path = fsspec.core.url_to_fs(str(source))
    src_root = f"{src_path.rstrip('/')}/OME"
    if not src_fs.exists(src_root):
        return []
    dst_fs, dst_path = fsspec.core.url_to_fs(str(output))
    dst_root = f"{dst_path.rstrip('/')}/OME"
    copied = []
    for entry in sorted(src_fs.find(src_root)):
        relative = str(entry)[len(src_root) + 1 :]
        destination = f"{dst_root}/{relative}"
        parent = destination.rsplit("/", 1)[0]
        dst_fs.makedirs(parent, exist_ok=True)
        with src_fs.open(entry, "rb") as fh:
            payload = fh.read()
        with dst_fs.open(destination, "wb") as fh:
            fh.write(payload)
        copied.append(f"OME/{relative}")
    return copied


def _restore_container_metadata(
    source: str | Path, output: str | Path, layout: StoreLayout
) -> None:
    """Write the root (and, for a plate, the well) attrs the layout is defined by.

    Withheld until here on purpose. Until the root carries ``attrs.ome``, the
    target is a zarr group full of arrays and not an OME-Zarr: every consumer
    refuses it without being told to, which is a stronger guarantee than a flag
    file and costs nothing to maintain. For per-scene output the root *is* the
    image, so its restore happens with the image's; for the two container
    layouts the root's own block is copied straight across, being pure structure
    — a plate's rows, columns and wells, a bundle's layout marker — with no
    geometry in it at all.
    """
    if layout == "per-scene":
        return
    source_root = open_root_group(source, mode="r")
    target_root = open_root_group(output, mode="a")
    source_ome = dict(source_root.attrs.get("ome") or {})
    if layout == "plate":
        for well in source_ome.get("plate", {}).get("wells") or ():
            well_path = str(well["path"])
            row = well_path.split("/")[0]
            target_root.require_group(row)
            well_group = target_root.require_group(well_path)
            well_group.attrs["ome"] = deepcopy(
                dict(source_root[well_path].attrs.get("ome") or {})
            )
    target_root.attrs["ome"] = source_ome


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def _minimal_audit(source: str | Path, layout: str, keys: Sequence[str]) -> dict:
    """The audit skeleton for a source store that carries none of its own.

    A ``bioformats2raw`` bundle, a store from another writer, a zarrmony store
    predating the audit block: all rechunk fine, and all need somewhere for the
    new geometry to be recorded. ``reader_plugin`` is ``null`` rather than
    invented — nothing here knows which reader produced those pixels — and
    ``input`` describes the store that was read, which is the honest answer when
    it is the only input there is.
    """
    record: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "version": __version__,
        "layout": layout,
        "reader_plugin": None,
        "input": {
            "path": str(source),
            "exists": True,
            "size_bytes": size_on_disk(source),
            "size_human": format_bytes(size_on_disk(source)),
        },
        "output": {"ome_ngff_version": NGFF_VERSION},
        "config": {},
        "conversion_started_at": None,
        "conversion_finished_at": None,
        "metadata_warnings": [],
    }
    blank = [{"image_key": key} for key in keys]
    if layout == "plate":
        record["fields"] = blank
        record["plate"] = {}
    else:
        record["per_scene"] = blank
    return record


def _image_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-image list of whichever audit shape this is (ADR-0004)."""
    if "fields" in audit:
        return audit["fields"]
    return audit.setdefault("per_scene", [])


def build_rechunked_audit(
    *,
    source_audit: dict[str, Any] | None,
    source: str | Path,
    output: str | Path,
    layout: str,
    plans: Sequence[ImagePlan],
    geometry: Geometry,
    contrast: dict[str, Any] | None,
    contrast_stats: dict[str, list[dict[str, Any]]],
    verification: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    resumed: bool,
    validate: bool,
    force: bool,
) -> dict[str, Any]:
    """The rechunked store's ``attrs.zarrmony``: the source's, amended in place.

    The record stays where ADR-0008's BigQuery ingest reads it, with the same
    keys meaning the same things, because nesting it under a new namespace would
    break every consumer to record a fact none of them asked about.

    Three groups of keys, and which group a key is in follows from one question —
    is it still true?

    * **Untouched** because it still is: ``input`` (the vendor file, its size,
      its digest), ``reader_plugin`` (the plugin that produced these pixels —
      level 0 is bit-identical, so it did), the original conversion timestamps,
      ``metadata_warnings``, and each image's ``acquisition`` / ``objective`` /
      ``channels``. Re-pointing ``input`` at the intermediate store would trade a
      provenance chain reaching the microscope for one reaching a directory.
    * **Overwritten** because the rechunk changed it: ``config.geometry``, every
      image's ``level_shapes`` / ``chunk_shapes`` / ``shard_shapes`` /
      ``coarse_level_index`` / ``contrast``, and ``version``.
      ``config.reader_tile_size`` becomes ``null``: no reader was asked for a
      tile, and leaving the old value would claim otherwise.
    * **Added**: ``rechunks``, appended to rather than replaced, because a
      rechunked store can be rechunked again and the intermediate geometry is
      part of how the store got here.
    """
    audit = (
        deepcopy(source_audit)
        if source_audit
        else _minimal_audit(source, layout, [p.image.key for p in plans])
    )
    had_audit = source_audit is not None
    source_geometry = deepcopy((audit.get("config") or {}).get("geometry"))
    previous_version = audit.get("version")

    audit["audit_schema_version"] = AUDIT_SCHEMA_VERSION
    audit["version"] = __version__
    audit["layout"] = layout
    audit["output"] = {"ome_ngff_version": NGFF_VERSION}

    config = dict(audit.get("config") or {})
    config["geometry"] = geometry.to_audit()
    config["reader_tile_size"] = None
    config["contrast_percentile"] = contrast["percentile"] if contrast else None
    config["validate"] = validate
    config["force"] = force
    audit["config"] = config

    records = _image_records(audit)
    for index, plan in enumerate(plans):
        record = records[index] if index < len(records) else {}
        if index >= len(records):
            records.append(record)
        record["image_key"] = plan.image.key
        record["level_shapes"] = [list(s) for s in plan.level_shapes]
        record["chunk_shapes"] = [list(c) for c in plan.chunk_shapes]
        record["shard_shapes"] = (
            [list(s) for s in plan.shard_shapes]
            if plan.shard_shapes is not None
            else None
        )
        record["coarse_level_index"] = plan.coarse_index
        per_channel = contrast_stats.get(plan.image.key)
        if contrast and per_channel is not None:
            record["contrast"] = {
                "percentile": contrast["percentile"],
                "method": contrast["method"],
                "per_channel": per_channel,
            }
        else:
            record.pop("contrast", None)

    entry: dict[str, Any] = {
        "operation": "rechunk",
        "zarrmony_version": __version__,
        "previous_zarrmony_version": previous_version,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "resumed": resumed,
        "source": {
            "path": str(source),
            "layout": layout,
            "had_audit": had_audit,
            "size_bytes": size_on_disk(source),
            "conversion_finished_at": audit.get("conversion_finished_at"),
            "images": [
                {
                    "image_key": p.image.key,
                    "level_shapes": [list(s) for s in p.image.level_shapes],
                    "chunk_shapes": [list(c) for c in p.image.level_chunks],
                    "shard_shapes": (
                        [list(s) for s in p.image.level_shards]
                        if p.image.level_shards is not None
                        else None
                    ),
                    "dtype": str(p.image.dtype),
                }
                for p in plans
            ],
        },
        "source_geometry": source_geometry,
        "working_set": {
            "peak_bytes": max(p.peak_working_set_bytes for p in plans),
            "per_image": [
                {
                    "image_key": p.image.key,
                    "tile": list(p.tiles[0]),
                    "bytes": p.working_set_bytes[0],
                    "tiles": list(p.tile_counts),
                }
                for p in plans
            ],
        },
        "contrast": contrast,
        "verification": verification,
    }
    audit.setdefault("rechunks", []).append(entry)
    audit["store_path"] = str(output)
    return audit


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------


def _resolve_geometry(
    geometry: Geometry | None,
    downsample_method: DownsampleMethod | None,
    source_audit: dict[str, Any] | None,
) -> tuple[Geometry, bool]:
    """Current policy for the nine shape fields; the source's kernel for the tenth.

    ``downsample_method`` is the one ``Geometry`` field that describes *pixels*
    rather than storage shape, and it is a property of the specimen rather than
    of the policy — a sparse-label acquisition converted with ``max`` is one
    whose small objects mean-pooling dissolves. Migrating it to today's default
    would rebuild its whole pyramid with the wrong kernel, silently, and would
    also break the acceptance claim that upper levels match a re-conversion:
    a re-conversion of *that* data would use ``max`` too. So it is inherited,
    and every other field takes current policy.

    Returns the resolved policy and whether the kernel came from the source.
    """
    base = geometry if geometry is not None else DEFAULT_GEOMETRY
    if downsample_method is not None:
        return replace(base, downsample_method=downsample_method), False
    inherited = ((source_audit or {}).get("config") or {}).get("geometry") or {}
    method = inherited.get("downsample_method")
    if method and method != base.downsample_method:
        return replace(base, downsample_method=method), True
    return base, bool(method)


def _resolve_contrast(
    contrast_percentile: Any, source_audit: dict[str, Any] | None
) -> tuple[float | None, bool]:
    """The percentile to recompute the omero window at, and whether it was inherited.

    The window is recomputed rather than copied because it is a *function of the
    pyramid*: ``write_scene`` measures it on the coarsest level, and the rechunk
    gives that level a new shape. Copying the old numbers would attach a window
    measured on one volume to a different one.

    The percentile itself is inherited, so a source converted with
    ``--no-contrast`` keeps its dtype-range window rather than silently gaining a
    data-driven one. A source with no audit has no percentile to inherit and
    keeps its window exactly as it stands — the most faithful copy available.
    """
    if contrast_percentile is not INHERIT:
        return contrast_percentile, False
    if not source_audit:
        return None, False
    config = source_audit.get("config") or {}
    if "contrast_percentile" not in config:
        return None, False
    value = config["contrast_percentile"]
    return (float(value) if value is not None else None), True


def _rechunk_store(
    source: str | Path,
    output: str | Path,
    *,
    layout: StoreLayout,
    geometry: Geometry | None,
    downsample_method: DownsampleMethod | None,
    contrast_percentile: Any,
    force: bool,
    resume: bool,
    verify: VerifyMode,
    validate: bool,
    max_working_set_bytes: int | None,
    working_set_fraction: float,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Migrate one store (not a directory of them) to the current geometry."""

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    source_root = open_root_group(source, mode="r")
    source_audit = source_root.attrs.get("zarrmony")
    source_audit = dict(source_audit) if isinstance(source_audit, dict) else None

    geometry_resolved, kernel_inherited = _resolve_geometry(
        geometry, downsample_method, source_audit
    )
    keys = discover_images(source, layout)
    images = [read_source_image(source, key) for key in keys]
    plans = [plan_image(image, geometry_resolved) for image in images]

    if all(plan.is_noop for plan in plans):
        say(f"{source}: already at the target geometry, nothing to do")
        return {
            "source": str(source),
            "output": None,
            "layout": layout,
            "skipped": True,
            "reason": "already-at-target-geometry",
            "images": len(plans),
        }

    budget = max_working_set_bytes
    if budget is None:
        physical = _physical_memory_bytes()
        budget = int(physical * working_set_fraction) if physical else None
    _check_working_set(plans, budget)

    peak = max(p.peak_working_set_bytes for p in plans)
    say(
        f"{source}: {len(plans)} image(s), reading in "
        f"{sum(p.tile_counts[0] for p in plans):,} tiles of up to "
        f"{format_bytes(peak)}"
        + (f" against a {format_bytes(budget)} budget" if budget else "")
    )

    fingerprint = {
        "source": str(source),
        "layout": layout,
        "images": [plan.fingerprint() for plan in plans],
    }

    existing = _read_state(output)
    resumed = False
    if existing and existing.get("status") == "in-progress" and resume and not force:
        differing = _diff_fingerprints(existing.get("fingerprint"), fingerprint)
        if differing:
            raise RechunkStateError(
                f"{output} holds an unfinished rechunk written against a "
                f"different plan — {differing}. Resuming would interleave two "
                f"geometries in one store. Pass force=True to discard it and "
                f"start over."
            )
        resumed = True
        state = existing
        say(f"{output}: resuming an interrupted rechunk")
    else:
        prepare_output_path(output, force=force)
        state = {
            "state_version": STATE_VERSION,
            "status": "in-progress",
            "started_at": datetime.now().astimezone().isoformat(),
            "fingerprint": fingerprint,
            "withheld_ome": {},
            "progress": {},
        }

    started_at = datetime.fromisoformat(state["started_at"])

    # Every target array first, and the resume state only afterwards. The writer
    # sets the image group's attrs wholesale on initialize, and for per-scene
    # output that group *is* the root — so writing state before the writers ran
    # would hand a fresh target a state key the next initialize erases.
    writers: dict[str, ZarrmonyWriter] = {}
    for plan in plans:
        key = plan.image.key
        writer = _make_writer(_join(output, key), plan, resuming=resumed)
        writers[key] = writer

        # The writer stamps `attrs.ome` up front. Stash it and take it back off
        # the store so the target cannot be read as an OME-Zarr until the run
        # finishes; it goes back, merged with the source's, at the end. On a
        # resumed run the previous attempt already did this, so there is nothing
        # to find and the stash in the state carries over untouched.
        image_group = open_root_group(_join(output, key), mode="a")
        if "ome" in image_group.attrs:
            state["withheld_ome"][key] = dict(image_group.attrs["ome"])
            del image_group.attrs["ome"]

        marks = list(state["progress"].get(key) or ())
        marks += [0] * (len(plan.level_shapes) - len(marks))
        state["progress"][key] = marks
    _write_state(output, state)

    for plan in plans:
        key = plan.image.key
        writer = writers[key]
        marks = state["progress"][key]
        source_group = open_root_group(_join(source, key), mode="r")
        label = key or Path(str(output)).name

        for level in range(len(plan.level_shapes)):
            if marks[level] >= plan.tile_counts[level]:
                continue
            say(
                f"  {label} level {level}: {plan.tile_counts[level] - marks[level]:,}"
                f" of {plan.tile_counts[level]:,} tiles to write"
            )

            def flush(done: int, _key: str = key, _level: int = level) -> None:
                state["progress"][_key][_level] = done
                _write_state(output, state)

            if level == 0:
                marks[0] = _write_level_zero(
                    source_group[plan.image.level_paths[0]],
                    writer.datasets[0],
                    plan,
                    marks[0],
                    flush,
                )
            else:
                marks[level] = _write_level(
                    writer.datasets[level - 1],
                    writer.datasets[level],
                    plan,
                    level,
                    geometry_resolved,
                    marks[level],
                    flush,
                )
            flush(marks[level])

    # Verification runs on the arrays alone, before any `attrs.ome` goes back, so
    # a store that fails it never spends a moment looking complete.
    seed = _verify_seed(fingerprint)
    verification = {"mode": verify, "blocks_checked": 0, "passed": None}
    if verify != "none":
        say(f"Verifying level 0 ({verify})")
        total = 0
        for plan in plans:
            source_group = open_root_group(_join(source, plan.image.key), mode="r")
            result = verify_level_zero(
                source_group[plan.image.level_paths[0]],
                writers[plan.image.key].datasets[0],
                plan,
                verify,
                seed,
            )
            total += int(result["blocks_checked"])
        verification = {"mode": verify, "blocks_checked": total, "passed": True}

    contrast_value, contrast_inherited = _resolve_contrast(
        contrast_percentile, source_audit
    )
    contrast_stats: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        key = plan.image.key
        store_path = _join(output, key)
        image_group = open_root_group(store_path, mode="a")
        withheld = state["withheld_ome"].get(key)
        if withheld is not None:
            image_group.attrs["ome"] = merge_image_ome(plan.image.ome, withheld)

        if contrast_value is None:
            continue
        channel_count = len((plan.image.ome.get("omero") or {}).get("channels") or ())
        if channel_count == 0:
            continue
        coarse = writers[key].read_level(-1)
        stats = _pair_contrast_results(
            da.compute(
                *_channel_contrast_ops(
                    coarse, list(plan.image.dims), channel_count, contrast_value
                )
            ),
            channel_count,
        )
        _update_omero_window_start_end(store_path, stats)
        contrast_stats[key] = [
            {
                "channel_index": i,
                "start": _to_json_scalar(low),
                "end": _to_json_scalar(high),
            }
            for i, (low, high) in enumerate(stats)
        ]

    _restore_container_metadata(source, output, layout)
    sidecars = copy_sidecars(source, output)

    # A per-scene target is a single image store; the container layouts are
    # validated at the root, which is what `convert` does for each of them.
    validation_findings = run_validation(output, layout, validate)

    audit = build_rechunked_audit(
        source_audit=source_audit,
        source=source,
        output=output,
        layout=layout,
        plans=plans,
        geometry=geometry_resolved,
        contrast=(
            {
                "percentile": contrast_value,
                "method": _CONTRAST_METHOD,
                "inherited": contrast_inherited,
            }
            if contrast_value is not None
            else None
        ),
        contrast_stats=contrast_stats,
        verification=verification,
        started_at=started_at,
        finished_at=datetime.now().astimezone(),
        resumed=resumed,
        validate=validate,
        force=force,
    )
    audit["validation_warnings"] = validation_findings
    audit["rechunks"][-1]["sidecars_copied"] = sidecars
    audit["rechunks"][-1]["downsample_method_inherited"] = kernel_inherited
    write_audit_record(output, audit)

    # Last: the state key is what said "unfinished". Nothing above may depend on
    # it having already gone, and nothing below needs it.
    root = open_root_group(output, mode="a")
    del root.attrs[STATE_KEY]

    return {
        "source": str(source),
        "output": str(output),
        "layout": layout,
        "skipped": False,
        "reason": None,
        "resumed": resumed,
        "images": len(plans),
        "audit": audit,
    }


def rechunk(
    source: str | Path,
    output: str | Path,
    *,
    geometry: Geometry | None = None,
    downsample_method: DownsampleMethod | None = None,
    contrast_percentile: Any = INHERIT,
    force: bool = False,
    resume: bool = True,
    verify: VerifyMode = "sample",
    validate: bool = True,
    max_working_set_bytes: int | None = None,
    working_set_fraction: float = DEFAULT_WORKING_SET_FRACTION,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rewrite an OME-Zarr store's geometry to the current policy (ADR-0012).

    ``source`` is an existing store — a ``.ome.zarr`` image, a
    ``bioformats2raw`` bundle, an HCS plate — or a plain directory of sibling
    ``*.ome.zarr`` stores, which is what ``convert --layout per-scene`` produces
    and therefore the shape most existing output trees are in. A directory fans
    out: each child is an independent unit of work with its own resume state and
    its own line of output, and children already at the target geometry are
    skipped, so re-running the same command after an interruption finishes the
    job rather than redoing it.

    ``source`` is opened read-only and never modified. ``force`` overwrites the
    *target*, exactly as it does for :func:`~zarrmony.api.convert`; there is no
    in-place mode, because the new pyramid can have a different number of levels
    and there is no instant at which such a store would be both readable and
    correct.

    :param geometry: The policy to migrate *to*. Defaults to
        :data:`~zarrmony.geometry.DEFAULT_GEOMETRY` — the point of the command.
    :param downsample_method: Overrides the pooling kernel. Left unset, the
        kernel is inherited from the source's audit rather than reset to the
        current default; see :func:`_resolve_geometry`.
    :param contrast_percentile: Percentile for the recomputed omero display
        window. Defaults to :data:`INHERIT` — the source's own percentile, so a
        store converted with contrast off keeps its dtype-range window. Pass
        ``None`` to force it off, or a float to override.
    :param resume: Continue an interrupted target instead of refusing it.
        Progress is a high-water mark per ``(image, level)``, so this is cheap
        and exact rather than a guess from which objects exist — zarr writes no
        object that is entirely fill value, so "absent" and "not yet written"
        are indistinguishable on disk.
    :param verify: ``"sample"`` (default) reads one chunk back out of every
        written tile and compares it against the source at the same offset;
        ``"full"`` reads all of level 0 back; ``"none"`` skips it.
    :param max_working_set_bytes: Absolute cap on the read-once tile. Defaults
        to ``working_set_fraction`` of detected physical RAM.
    :param progress: Called with human-readable status lines as the run goes.
    :returns: ``{"source", "output", "layout", "stores": [...]}`` where each
        entry reports one migrated (or skipped) store.
    """
    children = sibling_stores(source)
    if children:
        results = []
        for name in children:
            results.append(
                _rechunk_store(
                    _join(source, name),
                    _join(output, name),
                    layout=_store_layout(_join(source, name)),
                    geometry=geometry,
                    downsample_method=downsample_method,
                    contrast_percentile=contrast_percentile,
                    force=force,
                    resume=resume,
                    verify=verify,
                    validate=validate,
                    max_working_set_bytes=max_working_set_bytes,
                    working_set_fraction=working_set_fraction,
                    progress=progress,
                )
            )
        return {
            "source": str(source),
            "output": str(output),
            "layout": "sibling-directory",
            "stores": results,
        }

    layout = _store_layout(source)
    result = _rechunk_store(
        source,
        output,
        layout=layout,
        geometry=geometry,
        downsample_method=downsample_method,
        contrast_percentile=contrast_percentile,
        force=force,
        resume=resume,
        verify=verify,
        validate=validate,
        max_working_set_bytes=max_working_set_bytes,
        working_set_fraction=working_set_fraction,
        progress=progress,
    )
    return {
        "source": str(source),
        "output": str(output),
        "layout": layout,
        "stores": [result],
    }


__all__ = [
    "DEFAULT_WORKING_SET_FRACTION",
    "INHERIT",
    "STATE_KEY",
    "ImagePlan",
    "SourceImage",
    "build_rechunked_audit",
    "copy_sidecars",
    "detect_layout",
    "discover_images",
    "merge_image_ome",
    "plan_image",
    "pooled_tile",
    "read_once_tile",
    "read_source_image",
    "rechunk",
    "sibling_stores",
    "verify_level_zero",
]
