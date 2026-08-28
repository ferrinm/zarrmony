# Olympus VSI acceptance run

Runbook for [issue #103](https://github.com/ferrinm/zarrmony/issues/103) — the acceptance test for [ADR-0011](../adr/0011-bioformats-backed-formats.md). Three Olympus/Evident cellSens whole-slide `.vsi` datasets are converted to OME-Zarr through `bioio-bioformats`, confirming that the "install the GPL extra instead of writing an adapter" decision holds up end-to-end on real gigapixel data.

It needs a human: three multi-hour jobs, ~770 GB of intermediate accounting, and a visual check in Lucida. Everything in between is scripted.

The commands below take each slide from `$SRC`. Set it before running; dataset identifiers and share paths are tracked internally rather than in this public repo. The three slides are referred to as **A**, **B** and **C**.

## The data

Olympus VS200 slide scanner, cellSens ASW 4.2. 103 GB total across three structurally identical slides (35 / 34 / 34 GB).

Each dataset is a 4.5 MB `.vsi` TIFF-like index (`II*\0` plus Olympus `IS` private tags), with the actual pixels in a sidecar directory — `_<name>_/stackNNNNN/frame_t*.ets`, SIS tiled pyramids, 512×512 source tiles. `stack10002` is the 36 GB main image, `stack10000` the overview, `stack1` the label. **Point zarrmony at the `.vsi`**; Bio-Formats follows the sidecar directory itself.

Four scenes per slide:

| scene                                 | Y × X               |     C | dtype     | µm/px     |
| ------------------------------------- | ------------------- | ----: | --------- | --------- |
| `label`                               | 18232 × 1675 RGB    |     1 | uint8     | 2.738     |
| `overview`                            | 30683 × 40960       |     1 | `>u2`     | 1.625     |
| **`20x_DAPI_N, FITC, TRITC, Cy5_01`** | **141267 × 168124** | **4** | **`>u2`** | **0.325** |
| `macro image`                         | 375 × 504 RGB       |     1 | uint8     | —         |

Main scene: 20×, Z=1, T=1, channels `DAPI_N / FITC / TRITC / Cy5`, 10 source pyramid levels. The dtype is **big-endian**. `X`/`Y` extents differ slightly per slide — A is 140757 × 171855, B is 141267 × 168124, C is 140752 × 171215 (Y × X) — so every number below that depends on extent is quoted for B.

Scene names sanitize to these store directory names in per-scene layout: `label`, `overview`, `20x_DAPI_N_FITC_TRITC_Cy5_01`, `macro_image`.

## Pre-flight (already done)

The geometry below was computed by running today's `main` planner over each slide's shape, dtype and voxel spacing — `compute_level_shapes` + `plan_level_chunk_shapes` + `coarse_level_index` with the default `Geometry`. So the acceptance criteria are checks on the run, not open questions about the policy.

**Main scene, slide B** — `(1, 4, 1, 141267, 168124)`, `>u2`, 0.325 µm/px:

| level | shape                       | chunk                 | chunks/level | raw      |
| ----- | --------------------------- | --------------------- | ------------ | -------- |
| 0     | `(1, 4, 1, 141267, 168124)` | `(1, 1, 1, 512, 512)` | 363,216      | 190.0 GB |
| 1     | `(1, 4, 1, 70633, 84062)`   | `(1, 1, 1, 512, 512)` | 91,080       | 47.5 GB  |
| 2     | `(1, 4, 1, 35316, 42031)`   | `(1, 1, 1, 512, 512)` | 22,908       | 11.9 GB  |
| 3     | `(1, 4, 1, 17658, 21015)`   | `(1, 1, 1, 512, 512)` | 5,880        | 3.0 GB   |
| 4     | `(1, 4, 1, 8829, 10507)`    | `(1, 1, 1, 512, 512)` | 1,512        | 0.7 GB   |
| 5     | `(1, 4, 1, 4414, 5253)`     | `(1, 1, 1, 512, 512)` | 396          | 0.2 GB   |
| 6     | `(1, 4, 1, 2207, 2626)`     | `(1, 1, 1, 512, 512)` | 120          |          |
| 7     | `(1, 4, 1, 1103, 1313)`     | `(1, 1, 1, 512, 512)` | 36           |          |
| 8     | `(1, 4, 1, 551, 656)`       | `(1, 1, 1, 512, 512)` | 16           |          |
| 9     | `(1, 4, 1, 275, 328)`       | `(1, 1, 1, 275, 328)` | 4            |          |

- **`coarse_level_index` = 7** on all three slides.
- **512 KiB level-0 chunks**, landing exactly on the `.ets` 512×512 source tile grid — so level-0 writes should be tile-aligned with no read amplification. A good omen, not a guarantee; confirm it in the run.
- **10 planned levels, matching the source's own 10 zoom levels** almost exactly (an independent `slideio` read reports the source's smallest level as 328 × 275).
- **253.3 GB raw / 485,168 objects** for B's main scene. A is 258.0 GB / 493,484 and C is 257.1 GB / 492,384 — same level count, same chunk shape, same coarse index.

**The other three scenes** (same for all slides; they are fixed-size):

| scene         | levels | coarse index | raw     | objects |
| ------------- | -----: | -----------: | ------- | ------: |
| `overview`    |      7 |            5 | 3.35 GB |   6,408 |
| `label`       |      5 |            4 | 40.7 MB |      98 |
| `macro image` |      1 |            0 | 0.2 MB  |       1 |

So a full four-scene conversion of B is **~256.7 GB raw / ~491,700 objects**, and the three slides together are **~772 GB raw** before compression. The source is 34 GB per slide because the `.ets` tiles are compressed and WSI fluorescence is mostly background, so the real on-disk figure should land far below the raw estimate — but check free space against the raw number before starting, and decide the codec deliberately.

## What to watch during the run

**`dask_tiles` is not optional here.** `bioio-bioformats` returns one dask chunk per plane by default — `chunksize (1, 1, 1, 141267, 168124)`, 47.5 GB. `writers/scene.py`'s `arr.rechunk(tgt_chunks)` has to materialize a chunk to split it, so the conversion will OOM before it writes anything. `--reader-kwarg dask_tiles=true` is the whole fix.

**Do not pass `tile_size` with it.** Zarrmony plans the write grid first and reopens the reader asking for tiles that match it (#112), recording the result in `config.reader_tile_size`. An earlier revision of this runbook recommended `tile_size=1024,1024`; that was the worst of the available options and this document is the reason it spread. 1024² source tiles into the planned 512² write grid split on every write: **831,936 dask tasks against the 369,600 an aligned run needs**, plus each source tile re-read once per output chunk it feeds. Larger tiles produce a _larger_ graph, which is why nobody caught it by reasoning about it.

**The kwargs apply to every scene.** There is no scene selector; `dask_tiles` on the 375 × 504 macro image is harmless but the whole four-scene set is converted in one call.

**First open pays for the JVM.** `bffile`/`scyjava`/`cjdk` fetch a JDK (~36 MiB) and the Bio-Formats maven artifacts on first use. Do that once, on the `inspect` below, so it is not confused with conversion time.

**Big-endian source.** The main and overview scenes are `>u2`. Confirm the written store's byte order is sane and that a reader is not paying a byte-swap per chunk — see the endianness check below.

### Pre-flight on the machine that will run it

Read access first, because every other failure mode is easier to recognise than this one. VSI is multi-file — the `.vsi` is an index and the pyramid lives in a sibling `_<name>_/stackNNNNN/*.ets` tree — so Bio-Formats needs to both **open the file** and **list its directory**:

```bash
head -c 4 "$SRC/<slide-B>.vsi" | xxd   # expect 4949 2a00  (II*\0, little-endian TIFF)
ls -d "$SRC"/_*_/                      # the sibling pyramid directory must be listable
```

Run this as the user and on the machine that will host the JVM. Do not run the conversion against a share the JVM cannot read directly:

- On **macOS**, network and removable volumes under `/Volumes` — SMB, and FUSE mounts such as sshfs — are gated by the privacy layer, not by the file mode. `open()` returns `EPERM` ("Operation not permitted") while `stat` still succeeds, so `ls` on the file looks fine and only the read fails. Grant Full Disk Access to the application hosting the process (Terminal, Emacs, VS Code) under System Settings > Privacy & Security and restart **that application** — restarting only the Python process does not pick up a new grant.
- Bio-Formats surfaces this as `java.io.FileNotFoundException: <path> (Operation not permitted)`, which bioio then reports as `UnsupportedFileFormatError` recommending you install an extra. That recommendation is wrong for this failure. zarrmony now raises `InputAccessError` instead (ADR-0011), but the underlying trap is worth knowing.
- Reading a whole-slide VSI over sshfs also means hundreds of thousands of small random reads across the `.ets` files. Prefer running on a host with the storage mounted natively.

```bash
zarrmony inspect "$SRC/<slide-B>.vsi"
```

Expect four scenes, `Plugin: bioio-bioformats`, channels `DAPI_N / FITC / TRITC / Cy5` on the main scene and 0.325 µm/px. This is also what warms the JDK/maven cache.

Then confirm the tiling knob actually reaches the backend, which is the one thing that decides whether the real run survives:

```python
from zarrmony.readers.plugin import get_reader

reader, plugin, score = get_reader(
    f"{SRC}/<slide-B>.vsi",
    reader_kwargs={"dask_tiles": "true", "tile_size": "512,512"},
)
print(plugin.name, score)
reader.set_scene(2)                       # the 20x main scene
arr = reader.xarray_dask_data.data
print(arr.shape, arr.dtype, arr.chunksize, arr.npartitions)
```

`chunksize` must be `(1, 1, 1, 512, 512)`, not the full plane. If it is the full plane, stop — the run will OOM.

`tile_size` is pinned **here only**, to prove the kwarg reaches the backend. Leave it off the real run: `convert()` derives the same value from the planned write grid and a hand-pinned one that later disagrees with the geometry is exactly the failure this check is not looking for.

## Environment

```bash
uv pip install 'zarrmony[validate,bioformats]'
zarrmony --version
python -c "import bioio_bioformats; print('bioformats backend present')"
```

The `validate` extra matters: without `ome-zarr-models` installed the validator is skipped with a warning rather than run, and "store validates against the OME-NGFF validator" is an acceptance criterion. The `bioformats` extra is **GPL-3.0** — see ADR-0011; that is a property of the environment you are building here, not of zarrmony.

## Convert

One slide at a time, timed, output to a filesystem with the room computed above:

```bash
time zarrmony convert \
  "$SRC/<slide-B>.vsi" \
  "$OUT/slide-B" \
  --reader-kwarg dask_tiles=true
```

`--layout` is left at `auto`: a plain `BioImage` has no `layout_hint`, so it falls back to `"flat"` and writes one `<scene>.ome.zarr` per scene under `$OUT/slide-B`.

Two things for the runner to settle and record:

- **Which scenes to keep.** All four convert. `label` and `macro image` are RGB uint8 thumbnails costing 41 MB and 99 objects between them — cheap enough to keep, but decide whether a separate store per thumbnail is the shape downstream wants, and say so on the issue.
- **Codec.** ~257 GB raw per slide against a 34 GB source. Decide deliberately rather than discovering the default.

## Sharding

Optional, off by default, and now measured on this exact input — #124 re-ran the main scene at 512²-in-2048² against the unsharded baseline on the same host:

|            | 8 MiB chunks, no shards | 512² in 2048² shards   |
| ---------- | ----------------------- | ---------------------- |
| wall-clock | 3 h 02 m 00 s           | 3 h 04 m 54 s (+1.6 %) |
| store size | 139.0 GiB               | 138.92 GiB             |
| objects    | 31,634                  | 31,634                 |
| peak RSS   | 12.35 GiB               | 15.33 GiB              |
| CPU        | 152 %                   | 177 %                  |

Add `--shard-target-bytes` to the convert above; bare, it resolves to 8 MiB. Leave `tile_size` unpinned — `plan_write_grid` derives `[2048, 2048]`, the level-0 shard, and the run emitted **zero** `TileAlignmentWarning` across three hours with write amplification 1.0005. Budget headroom rather than hours: the cost of sharding here is +24 % resident set and +19 % CPU, not elapsed time.

Three things change downstream, and all three have bitten:

- **`verify_geometry.py` is shard-blind.** The report never mentions sharding — no shard shape in the checks, no shard column in the levels table — and reports the inner chunk as if it were the store's only chunking, so its object prediction is the unsharded one (493,484 here against 31,156 on disk) and every check still passes. Pass `--no-object-count`, which suppresses the false failure and skips a multi-minute tree walk, and count with `find "$STORE" -type f | wc -l` instead. Do not paste the report as evidence of geometry without saying in the surrounding text that the store is sharded. Tracked in #124.
- **Lucida cannot open a sharded store.** `lucida-store`'s codec-chain parser rejects `sharding_indexed`, so the "Verify in Lucida" section below cannot run. That is the documented cost of the flag (#117), not a regression — use `napari-ome-zarr` for the visual check. The level readout in napari's status bar is the useful one: zooming in should step it monotonically down to 0, which proves a 512 KiB chunk is being range-read out of an 8 MiB shard.
- **The shard shrinks on coarse levels.** Levels 0–6 get `[1,1,1,2048,2048]` (16 chunks, 8.00 MiB), then 1536² (9, 4.50 MiB), 1024² (4, 2.00 MiB) and 274 × 335 (1, 0.18 MiB). The planner returns the smallest whole-chunk multiple covering the level rather than a mostly-empty 2048², so per-level shard shapes are expected to differ. Take object counts only after the pyramid completes; a count read mid-run will not match the grid.

## Verify the store

`scripts/verify_geometry.py` recomputes the predicted geometry from the store's _own_ recorded policy and compares it against what is on disk, then checks all four of Lucida's source-coarse bounds:

```bash
python scripts/verify_geometry.py \
  "$OUT/slide-B/20x_DAPI_N_FITC_TRITC_Cy5_01.ome.zarr" \
  --expect-coarse-level 7
```

Exit status gates the run; stdout is Markdown ready to paste onto the issue. Run it on the `overview` store too (`--expect-coarse-level 5`).

Then the things the geometry verifier does not cover:

```python
import json, os, zarr
from ome_types import from_xml

store = f"{OUT}/slide-B/20x_DAPI_N_FITC_TRITC_Cy5_01.ome.zarr"
attrs = dict(zarr.open_group(store, mode="r").attrs)

# Channel names and pixel size survived into the multiscales metadata.
ms = attrs["ome"]["multiscales"][0]
print([a["name"] for a in ms["axes"]])                      # ['t','c','z','y','x']
print(ms["datasets"][0]["coordinateTransformations"])       # expect 0.325 on y and x
print([c["label"] for c in attrs["ome"]["omero"]["channels"]])

# The audit names the backend the ADR is about, and the coarse level.
print(attrs["zarrmony"]["reader_plugin"])                    # distribution: bioio-bioformats
print(attrs["zarrmony"]["per_scene"][0]["coarse_level_index"])   # 7

# OME instrument metadata: 3 detectors, 3 objectives.
ome = from_xml(open(os.path.join(store, "OME", "METADATA.ome.xml")).read())
print(ome.instruments)

# Endianness: what did the bytes codec actually record for level 0?
level0 = json.load(open(os.path.join(store, "0", "zarr.json")))
print(level0["data_type"], level0["codecs"])
```

The `bytes` codec records `endian: little` on a normal store. A big-endian **source** does not by itself make the output big-endian — the question is whether the conversion swapped once on read (fine) or the store advertises `big` and every reader swaps per chunk (not fine).

- **Validator.** `convert` runs it automatically with the `validate` extra installed; confirm no `ValidationWarning` in the run output and that `attrs.zarrmony.validation_warnings` is empty.

## Verify in Lucida

Open the main-scene store. What to check:

- It resolves the **source** coarse level (7) rather than planning a server-generated one.
- Pan and zoom across the full 168k × 141k extent at several zoom levels without stalling.
- All four channels render with sensible contrast; the ADR-0007 colors are distinguishable.
- Nothing looks byte-swapped — a big-endian source read as little-endian produces obvious high-frequency noise, not subtle error.

## Record on the issue

Per slide: store size on disk and object count against the ~257 GB / ~492k raw estimates, the compression ratio achieved, conversion wall-clock, and the `config.reader_tile_size` the audit record shows.

- The endianness answer.
- Whether any `TileAlignmentWarning` was emitted. It should not be — the derived tile matches the grid by construction — so one means a scene whose geometry the derivation could not plan, and it is worth the issue comment.
- The scene-keeping decision.
- **Any Bio-Formats-specific rough edges.** These are the input to what the `bioformats` extra's README section needs to warn about — the reason for running this against real data rather than a fixture.
