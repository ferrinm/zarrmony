# `zarrmony rechunk` migrates a store to the current geometry as a sibling command with its own pipeline

**Status:** Accepted (2026-08-31).

`zarrmony rechunk SOURCE OUTPUT` reads an existing OME-Zarr store and writes a new one planned by the current [ADR-0010](./0010-output-geometry-policy.md) policy, without going back to the vendor file. It is a **sibling of `convert`, not a mode of it**: it reads the source's zarr metadata directly rather than through a reader plugin, and it owns copy fidelity, resume and layout walking itself. Level-0 voxels are copied through byte-for-byte; every level above is pooled from the level below by the same code a fresh conversion uses, so a rechunked store is indistinguishable from a re-conversion at every level. The layout is detected from the source's own `attrs.ome`; there is no `--layout` override. The pass is streaming — each source block is read exactly once — by working in tiles that are the element-wise LCM of the source's write grid and the target's, budgeted against detected physical RAM. Resume is a high-water mark per `(image, level)` over a deterministic tile order, and an unfinished target carries no `attrs.ome` at all, so it cannot be opened as an OME-Zarr. Provenance is the source's own audit record, amended in place, plus an append-only `rechunks` list that distinguishes a rechunked store from a converted one. Audit schema **14 → 15**.

## Why this needs writing down

[ADR-0010](./0010-output-geometry-policy.md) deferred this deliberately, and its Considered Options entry ("Build an OME-Zarr → OME-Zarr `zarrmony rechunk` command as part of this work") already contains the argument for why the migration is cheap and lossless. What it did not contain is the argument for how to build it, and every one of the surfaces it named as "distinct" — metadata-copy fidelity, `--force` semantics, resumability across a multi-hour job over millions of objects — turns out to be a decision that can be made two defensible ways.

The forcing case is the one this repository has actually converted: a whole-brain volume at 3.18 M objects and 30 hours unsharded, and a whole-slide scene projected at ~6 days. A migration tool whose failure mode is "start again" is not a migration tool at those durations. Equally, a tool that silently produces something _nearly_ like a re-conversion is worse than no tool, because the whole reason to migrate is that the current geometry is the one downstream consumers are tuned for.

## Considered Options

### One: a mode of `convert` versus a sibling command

- **A sibling command with its own pipeline (accepted).** `rechunk` opens the source store's zarr groups directly, reads `multiscales` and the arrays' `zarr.json`, and reuses `ZarrmonyWriter` — the same writer `convert` uses, given the same planner output, so the target's codec chain, fill value, dimension names and `sharding_indexed` configuration are what a re-conversion would have written rather than an approximation of it. `convert()`'s signature and its audit semantics are untouched.
- **An OME-Zarr _reader plugin_ feeding `convert`.** Rejected, and it is the option that looks cheapest for about ten minutes. `bioio-ome-zarr` exists, so `convert store.ome.zarr new/` nearly works today. It fails on the three things that make this a migration rather than a conversion. Level 0 would round-trip through the reader's dask array and the writer's rechunk instead of being copied, so "bit-identical" would be a hope rather than a guarantee. The audit record would describe the _intermediate store_ as the input — `input.path` pointing at a directory, `reader_plugin` naming `bioio-ome-zarr` — which severs the provenance chain to the microscope at exactly the moment it matters. And `convert` has no resume, so adding one would mean adding it to the path every conversion takes, for a property only this command needs.
- **A `--from-zarr` flag on `convert`.** Rejected as the worst of both: the same audit and fidelity problems, plus a `convert` whose flag list now contains options that are meaningless for its primary use (`--verify`, `--resume`) and whose reader-facing flags (`--reader-kwarg`, `--lif-mosaic`) are meaningless for the new one.

### Two: how the layout is decided

- **Detect it from the source's `attrs.ome` (accepted).** `multiscales` means one image; `bioformats2raw.layout: 3` means a bundle with numbered subgroups; `plate` means HCS. A directory that is none of those but contains sibling `*.ome.zarr` children fans out, one independent unit of work and one resume state per child — which is what `convert --layout per-scene` produces and therefore the shape most existing output trees are in.
- **A `--layout` override.** Rejected. On `convert` the flag is meaningful because the reader's `layout_hint` is a _suggestion_ about data whose final shape is not yet fixed. Here the shape is a fact already on disk. An override could only mean "read this plate as if it were something else", which is not a migration and has no correct answer.

### Three: what is copied verbatim and what is recomputed

Copied: `multiscales.axes`, `multiscales.name`, level-0 `coordinateTransformations`, `OME/METADATA.ome.xml`, `OME/source/raw.<ext>.xml`, the bf2raw wrapper's `attrs.ome.series`, and a plate's `attrs.ome.plate` plus each well's `attrs.ome.well.images`. Recomputed: `multiscales.datasets` (the level list and its scales change), every array's `zarr.json`, each image's `level_shapes` / `chunk_shapes` / `shard_shapes` / `coarse_level_index`, and `config.geometry`.

The omero display window is the one that needed deciding, because it belongs to neither group cleanly.

- **Recompute the window, inheriting the percentile (accepted).** The window is measured off the _coarsest pyramid level_, and the coarsest level is a different level after the migration — the pyramid's depth is exactly what the policy changed. Copying the old window would carry over a number derived from an array that no longer exists in the store. Copying the _percentile_ and re-measuring is the only reading under which the recorded `contrast.percentile` stays true. A source whose audit records `contrast_percentile: null` — converted with contrast off — inherits the `null` and keeps its dtype-range window, which is why the sentinel for "inherit" has to be distinct from `None`.
- **Copy the window verbatim.** Rejected for a store zarrmony wrote. Retained for the one case where recomputing is unfounded: a source with **no audit record at all** (a store zarrmony did not write) has its omero windows copied as-is, because there is no percentile to inherit and inventing one would silently restyle someone else's store.

### Four: the read-once traversal, and what bounds it

The unit of work is `elementwise_lcm(source_write_grid, target_write_grid)`, clamped to the level's extent. That is the smallest block that is a whole number of source blocks _and_ a whole number of target blocks, which is what makes "each source block is read exactly once" true by construction rather than by luck. On the case ADR-0010 predicted — full-width single-plane slabs into 64³ chunks — it resolves to Z bands of 64 at full width, exactly as forecast.

- **Drive the tile loop explicitly; never hand a whole level to `dask.rechunk` (accepted).** This is [#111](./0010-output-geometry-policy.md#follow-up-issues-111-114) restated. A single volume handed to `da.rechunk` builds millions of tasks before writing a byte. Here the graph is one tile wide and zarr's own async chunk I/O does the parallelism, so task count is bounded by the tile regardless of level size.
- **The budget is a fraction of detected physical RAM, default 0.5, with an absolute override.** A fixed byte default was drafted first and rejected on review: any figure small enough to protect a laptop is absurdly restrictive on the 192-core hosts this data is actually converted on, and any figure large enough for those hosts protects nothing. `os.sysconf` gives the real number; the fraction is the knob. Exceeding it is an **up-front refusal naming the image, the level and the size**, not an eventual out-of-memory kill.
- **Restrict the supported chunkings to those where the LCM is small.** Rejected. The LCM can be large — coprime edges are the pathological case — but a refusal the user can act on is a better answer than a supported subset that does not include their store.

### Five: `--force`, and the absence of an in-place mode

`rechunk SRC DST` writes to a new path. The source is opened read-only and never modified. `--force` means what it means on `convert`: overwrite an existing DST.

- **In-place migration.** Rejected. The new pyramid can have a _different number of levels_, so there is no instant during an in-place rewrite at which the store is both readable and correct. A tool whose crash window leaves a corrupt store is not a migration tool.

### Six: resume

Progress is a **high-water mark per `(image, level)`** over a deterministic tile order, checkpointed on a ~30 second timer and at every level boundary, into `attrs["zarrmony_rechunk"]` on the target root.

- **Why a high-water mark rather than probing the store.** Zarr writes no object that is entirely fill value, so "absent" and "not yet written" are indistinguishable on disk — a probe-based resume would either redo written work or skip unwritten work, and there is no way to tell which. Because a tile is a whole multiple of the write grid, a crash can only tear objects the mark has not yet claimed, and re-running rewrites those whole.
- **The target carries no `attrs.ome` until the run finishes.** The writer stamps it on initialize; `rechunk` immediately stashes it in the resume state and removes it, putting it back — merged with the source's copied fields — only after the arrays are written and verified. This is what makes "a partially written target is never mistakable for a complete store" a structural property rather than a convention: a partial target is not an OME-Zarr, and no reader will open it as one.
- **A plan fingerprint guards the resume.** Resuming against a _different_ target geometry would interleave two geometries in one store. The state carries a fingerprint of the resolved plan; a mismatch refuses, naming the field that differs. `--force` discards and starts over.
- **One consequence that cost a debugging session and is recorded so it does not recur:** `OMEZarrWriter._open_root` opens with `mode="w"`. Re-running `initialize()` on a resumed target therefore _empties the group_, discarding every level the high-water mark says is done. `ZarrmonyWriter` gains an `attach()` that binds to the existing arrays and writes nothing, and the resume path uses it. Anything else that reconstructs a writer over an existing store has the same hazard.

### Seven: the audit record

The record stays at `attrs.zarrmony`, with the same keys meaning the same things, because nesting it under a new namespace would break ADR-0008's BigQuery ingest to record a fact none of its consumers asked about. Which group a key falls into follows from one question — _is it still true?_

- **Untouched:** `input` (the vendor file, its size, its digest), `reader_plugin` (level 0 is bit-identical, so that plugin did produce these pixels), the original conversion timestamps, `metadata_warnings`, and each image's `acquisition` / `objective` / `channels`. Re-pointing `input` at the intermediate store would trade a provenance chain reaching the microscope for one reaching a directory.
- **Overwritten:** `config.geometry`, each image's `level_shapes` / `chunk_shapes` / `shard_shapes` / `coarse_level_index` / `contrast`, and `version`. `config.reader_tile_size` becomes `null` — no reader was asked for a tile, and leaving the old value would claim otherwise.
- **Added:** `rechunks`, an **append-only list**. A rechunked store can be rechunked again, and the intermediate geometry is part of how the store got here. Each entry carries the operation, both zarrmony versions, timestamps, whether the run resumed, the source's path and pre-migration shapes, the source geometry, the peak working set and per-image tile plan, the contrast decision, and the verification result.
- **The discriminator is `"rechunks" in attrs.zarrmony`** — presence of the key, not a field to be kept in sync with it. Schema **15**, because a consumer that has never seen `rechunks` and reads `config.geometry` as "the geometry this data was converted at" is now wrong, and a version bump is how it finds out.

### Eight: which geometry field does _not_ take the current policy

`downsample_method` **inherits from the source's `config.geometry`**; the other nine `Geometry` fields take current policy defaults. An explicit `downsample_method=` overrides.

- It is the one `Geometry` field that describes _pixels_ rather than storage shape, and it is a property of the specimen rather than of the policy — a sparse-label acquisition converted with `max` is one whose small objects mean-pooling dissolves. Migrating it to today's default would rebuild its whole pyramid with the wrong kernel, silently, in a command the user ran to change chunk sizes.
- **A no-op is refused, not written.** If the resolved geometry matches what is already on disk, the store is skipped without writing anything. This is what makes fan-out over a directory idempotent: re-running after an interruption finishes the job rather than redoing it, and children already migrated are left alone.

### Nine: how "bit-identical" is established

Two mechanisms, because the two claims are different claims.

- **Equivalence is proved by test, not at runtime.** `convert@new` and `convert@old + rechunk` produce byte-identical arrays at _every_ level, over a matrix of shapes, dtypes and chunkings. That is a property of the code and belongs in the test suite; re-deriving it on every run would mean re-converting from source, which is the cost the command exists to avoid.
- **Level-0 bit-identity is checked at runtime**, by `verify="none" | "sample" | "full"`, default `sample`. `sample` reads one chunk back out of every written tile at an offset seeded off the plan fingerprint — deterministic, so a failure reproduces. `full` reads all of level 0 back, roughly doubling read cost. The mode, the number of blocks checked and the result are recorded in the `rechunks` entry, so "was this verified?" is answerable from the store rather than from shell history. Verification runs **before** `attrs.ome` goes back, so a store that fails it never spends a moment looking complete.

## Consequences

- **Existing stores have a migration path that does not touch the vendor file.** ADR-0010's Consequences said "existing stores are unaffected and are not migrated"; that stands for the default, but the escape hatch now exists. A store on a read-only share whose source file is gone can still move to the current geometry.
- **The streaming guarantee is a property of the tile rule, and it degrades honestly.** Where the two grids share little, the LCM tile is large, and the command says so up front and refuses rather than discovering it under memory pressure hours in. Users who hit the refusal have two real answers — raise the budget, or accept a chunking closer to the source's.
- **Resume changes what "interrupt" costs, and only at tile granularity.** A run killed mid-tile redoes that tile, not that level. On the reference volume that is minutes against thirty hours.
- **`ZarrmonyWriter` now has two entry points and they are not interchangeable.** `initialize()` creates; `attach()` binds. Every future caller has to pick, and picking wrong on an existing store destroys it. The docstrings say so at both sites.
- **`run_validation` moved out of `api.py` into `_validate.py`, and `prepare_output_path` into `_storage.py`.** Both are now shared by `convert` and `rechunk`. This was forced by the import graph — `_rechunk` cannot import `api`, which re-exports it — but it is also the right home for both: a store's spec compliance does not depend on which command wrote it.
- **The audit's `input` block can now describe a file that no longer exists anywhere near the store.** That is deliberate and is the point of keeping it, but a consumer that treats `input.path` as "a path I can open" was always wrong and is now more visibly wrong.
- **Two commands can write the same audit shape, so the schema is the contract.** A consumer distinguishing them checks for `rechunks`. A consumer that does not care — most of them, since the geometry keys mean the same thing either way — needs no change beyond accepting version 15.
- **`--verify sample` costs a read of one chunk per tile on every run, by default.** Roughly a percent of the pass on realistic tile counts, and it is the difference between "the copy is correct" being a claim about the code and being a claim about this store. `--verify none` is there for anyone who disagrees.
