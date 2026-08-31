# Geometry acceptance run

Runbook for [issue #90](https://github.com/ferrinm/zarrmony/issues/90) — the acceptance test for the [ADR-0010](../adr/0010-output-geometry-policy.md) output-geometry series. The reference SmartSPIM dataset is re-converted from source under the new geometry, and the defect the ADR was written against is confirmed gone in a live viewer.

Two halves: a multi-hour conversion that a human has to launch on a machine with the source mounted, and a visual check in Lucida that a human has to eyeball. Everything in between is scripted.

The commands below take the source export from `$SRC`. Set it to the reference dataset's path before running; dataset identifiers and share paths are tracked internally rather than in this public repo.

## Pre-flight (already done)

The predicted geometry below was computed by running today's `main` planner over the reference dataset's shape and voxel spacing — `compute_level_shapes` and `plan_level_chunk_shapes` at `(1, 3, 3627, 8835, 7452)` uint16, Z 2.0 / Y 1.8 / X 1.8 µm. Every number matches the ADR and the issue, so the acceptance criteria are checks on the run, not open questions about the policy.

| level | shape                      | chunk                | chunks/level | MiB per (t,c) | long axis |
| ----- | -------------------------- | -------------------- | ------------ | ------------- | --------- |
| 0     | `(1, 3, 3627, 8835, 7452)` | `(1, 1, 64, 64, 64)` | 2,780,973    | 455,467.1     | 8835      |
| 1     | `(1, 3, 1813, 4417, 3726)` | `(1, 1, 64, 64, 64)` | 359,310      | 56,911.3      | 4417      |
| 2     | `(1, 3, 906, 2208, 1863)`  | `(1, 1, 64, 64, 64)` | 47,250       | 7,108.4       | 2208      |
| 3     | `(1, 3, 453, 1104, 931)`   | `(1, 1, 64, 64, 64)` | 6,480        | 888.1         | 1104      |
| 4     | `(1, 3, 226, 552, 465)`    | `(1, 1, 64, 64, 64)` | 864          | 110.6         | 552       |
| 5     | `(1, 3, 113, 276, 232)`    | `(1, 1, 64, 64, 64)` | 120          | 13.8          | 276       |

- **`coarse_level_index` = 5.** Level 4 is 110.6 MiB per (t,c), over the 64 MiB bound; level 5 is 13.8 MiB with a 276 long axis, under both.
- **Level 0 chunk is 512 KiB raw**, from `[1,1,1,1125,7452]` (15.99 MiB) before.
- **3,194,997 chunk objects** across all levels, against the ADR's ~3.2 M estimate.
- **1,637.3 GB raw** across all levels; the ~283 GB estimate implies ~5.8× compression.

Level 5 also satisfies the two `SourceCoarseConfig` bounds ADR-0010 does not model — a 512 KiB chunk against the 16 MiB `max_chunk_bytes` cap, and 40 chunks per (t,c) against the 4096 `max_chunk_count_per_tc` cap — so all four of Lucida's source-coarse conditions hold, not just the two the policy encodes. `scripts/verify_geometry.py` checks all four.

## What to watch during the run

The chunk change turns a streaming write into a rechunking one, and that is the part of this run with no prior measurement behind it:

- **The source is one TIFF per Z plane.** `zarrmony-smartspim` builds its dask array with one plane per chunk — `(1, 1, 1, 8835, 7452)`, 131.6 MB each. Under the old geometry Z stayed 1 in the target chunk, so each plane was split laterally and written without ever touching its neighbours. At `Z=64` the writer's `arr.rechunk(target)` now has to combine 64 planes per output block. Dask should plan that as split-then-merge (thin lateral strips first, concatenated in Z), which keeps per-task memory small — but it is worth watching RSS for the first few minutes rather than assuming.
- **The graph is ~37× bigger.** 87 k chunks became ~3.2 M, in a single-process `da.compute` on the threaded scheduler. Scheduling overhead, not I/O, may end up dominating wall-clock.

If either bites, cap concurrency with `DASK_NUM_WORKERS` and record what happened — it is a design input for the `zarrmony rechunk` tool in [#91](https://github.com/ferrinm/zarrmony/issues/91), not a reason to change the geometry.

**Do a timed smoke run first.** Build a small SmartSPIM-shaped export by symlinking one channel directory with a Z subrange into scratch, convert that, and extrapolate:

```bash
mkdir -p /scratch/geom_smoke/Ex_488_Ch0_stitched
ls "$SRC"/Ex_488_Ch0_stitched/*.tif \
  | head -128 \
  | xargs -I{} ln -s {} /scratch/geom_smoke/Ex_488_Ch0_stitched/
cp "$SRC"/metadata_*.json /scratch/geom_smoke/

time zarrmony convert /scratch/geom_smoke /scratch/geom_smoke_out
```

The reader globs `metadata_*.json`, so the sidecar keeps its original name in the scratch directory. 128 planes is two full Z-blocks — enough to exercise the rechunk that the old geometry never triggered — and ~1/85 of the full read (3627 planes × 3 channels), so a smoke run taking _t_ suggests very roughly 85 _t_ for the real thing. Enough to size the job, not to quote.

## Environment

```bash
uv pip install 'zarrmony[validate]' zarrmony-smartspim
zarrmony --version
python -c "from zarrmony.readers.plugin import list_plugins; print([p.name for p in list_plugins()])"
```

The `validate` extra matters: without `ome-zarr-models` installed the validator is skipped with a warning rather than run, and "store validates against the OME-NGFF validator" is one of the acceptance criteria.

Confirm the source reads as expected before committing hours to it:

```bash
zarrmony inspect "$SRC"
```

Expect scene `volume`, shape `(1, 3, 3627, 8835, 7452)` uint16, Z 2.0 / Y 1.8 / X 1.8 µm, channels Ex488 / Ex561 / Ex639. If the metadata sidecar is missing or the export sits on a read-only share, point at a copy with `--reader-kwarg metadata_path=/writable/metadata_<dataset>.json`.

## Convert

Default geometry — no geometry flags. That is the point of the run: what the policy does unprompted.

```bash
OUT=<destination for the store>

time zarrmony convert \
  "$SRC" \
  "$OUT"
```

The reader's `layout_hint` is `flat`, so this writes one store at `$OUT/volume.ome.zarr`.

## Sharding

Optional, off by default, and **now measured on this volume** — see #126, and the table at the end of this section. Add `--shard-target-bytes` to the convert above; bare, it resolves to 8 MiB. What the planner returns for this volume:

- **The chunk does not move.** `(1, 1, 64, 64, 64)` at every level, identical to the unsharded run, so `coarse_level_index` is still 5 and every read-side expectation below carries over unchanged. Sharding changes the write unit and nothing else.
- **16 chunks per shard, 8.00 MiB exactly, at every level** — `(1, 1, 128, 128, 256)` at levels 0–2, flipping to `(1, 1, 128, 256, 128)` at 3–5 as per-level spacing changes which axis is longest in micrometres. No tail-level raggedness, unlike the 2D arm. **3,194,997 objects down to 210,345**, 15.2×.

Three things differ from the 2D acceptance run, and each is a reason not to assume #124's result transfers wholesale:

- **`TileAlignmentWarning` will fire on Y and X, and the hint is not actionable.** `_align_reader_tiles` early-returns for any plugin that is not the default `bioio` one, since `tile_size` is that plugin's convention — so `config.reader_tile_size` will be `null` in the audit, which is correct behaviour rather than a defect. The adapter hands back one whole Z-plane per dask block, which divides neither the 128 nor the 256 write grid. The suggested `tile_size=128,256` is generated from the write grid without knowing which plugin is loaded, and this plugin does not accept it. The condition is pre-existing, not caused by sharding: unsharded, the same two axes offend against the 64³ grid.
- **Size the host for the Z gather, not for the pyramid.** Filling one chunk row needs 64 planes; one shard row needs 128, at ~125.6 MiB per plane per channel — roughly 47 GiB of gather before compression buffers. Measured peak was **94.2 GiB**, 2.0× that arithmetic, where the 2D arm overshot its own gather by only 1.35×. The multiplier is not stable between the two paths; size against 94.2 GiB, not against the gather. Use a large-memory node.
- **A speedup here is not purely the write-unit effect.** Sharding also cuts the per-plane split factor from 16,263 chunks to 2,100 shards, 7.7× on top of the 15.2× object reduction. #124's 2D arm isolates the write-unit effect because its reader tiles already matched the grid; this one does not. Record it as such.
- **Object count is an upper bound here, not an exact prediction.** Zarr does not write a shard whose chunks are all fill value, so the run landed at **209,211 shards against the 210,345 planned**, every absent one on a trailing row covering a 3-voxel sliver of a padded axis. A shortfall is not a truncated write; an excess would be the alarming direction.

Measured, against the unsharded run of #90 — same source, same default geometry, same `(1, 1, 64, 64, 64)` chunk and same six levels, so this isolates sharding:

|             | no shards       | 8 MiB shards    |             |
| ----------- | --------------- | --------------- | ----------- |
| store bytes | 261,959,557,972 | 262,010,912,050 | **+0.02 %** |
| objects     | 3,182,337       | 209,220         | **15.2×**   |
| wall-clock  | 30 h 04 m 43 s  | 10 h 09 m 54 s  | **0.34×**   |
| peak RSS    | 117.2 GiB       | 94.20 GiB       | −20 %       |
| CPU         | 117 %           | 168 %           | +44 %       |

The size question that earlier editions of this section told you not to predict has an answer, and a reason: **compression runs per chunk and sharding does not change the chunk**, so the payload is identical by construction and the 51.4 MB delta is the shard index — 16 B per chunk slot plus 4 B per shard. Plane tile versus cube never entered into it. See ADR-0010's #126 follow-up.

Downstream, `scripts/verify_geometry.py` is shard-blind — the report never mentions sharding at all and its object count comes from the chunk grid, so pass `--no-object-count` and count with `find` instead (#129). And Lucida cannot open a sharded store at all (#117), so the `lucida dataset health` generated-coarse gate below **cannot run** against a sharded store; use `napari-ome-zarr` and the OME-NGFF validator for the visual check, and note that napari pins 3D display to the coarsest level regardless of zoom, so the pyramid check is only meaningful in 2D.

## Verify the store

```bash
python scripts/verify_geometry.py "$OUT/volume.ome.zarr" --expect-coarse-level 5
```

Exit status is 0 only when every check passes. The script recomputes the predicted geometry from the store's own recorded policy rather than from the table above, so it is checking the conversion, not a transcription. Its Markdown output covers four of the seven acceptance criteria — validation, level shapes, level-0 chunk shape, `coarse_level_index = 5` — plus store size, object count and wall-clock from the audit timestamps. Paste it onto the issue.

## Verify in Lucida

```bash
lucida dataset open "$OUT/volume.ome.zarr"
lucida dataset health <dataset>
```

`plan_generated_coarse_for_image` early-returns `None` for any image whose manifest carries a `coarse_level_index`, so **`Generated coarse: … (levels 0, …)`** is the observable signal that the source coarse tier resolved and the server is not building its own. A non-zero level count means Lucida did not accept level 5 — capture the health output before doing anything else, because that is the interesting failure.

Then open the dataset in the web viewer and put it under a 3D camera:

- **The behaviour to eyeball.** The admission window is `max(64, 4 × concurrency) = 64` requests, drained centre-out. Under the old geometry those 64 requests bought 64 whole-brain Z planes — 128 µm of Z across the full 15.9 × 13.4 mm, almost all off-screen. They should now buy roughly a 256³ cube: about 461 × 461 × 512 µm centred on the camera. The reported symptom was "the data budget is maxed out with a few slices instead of the 3D volume in the middle of the viewer"; that is what should be gone.
- **Expect it to look dimmer.** With a real coarse level present, the coarse tier is zarrmony's mean-pooled level 5 rather than Lucida's max-pooled generated tier. For sparse structures mean-pooling reads dimmer. That is the documented consequence of the default, not a regression — `--downsample-method max` is the escape hatch if a viewer genuinely needs the old appearance.

## Record on the issue

The verification script's output covers the measurements. Add, in prose:

- Wall-clock, and the machine it ran on (cores, RAM, whether the source and destination were local).
- Peak RSS if you watched it, and whether `DASK_NUM_WORKERS` had to be capped.
- Store size and object count against the ~283 GB / ~3.2 M estimates.
- The `lucida dataset health` generated-coarse line.
- A screenshot under a 3D camera, and one sentence on whether the fetch budget resolved into a centred volume.
