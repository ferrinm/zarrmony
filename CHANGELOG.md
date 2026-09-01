# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The acceptance verifier can see a shard.** `scripts/verify_geometry.py`
  collected each level's `chunks` and never its `shards`, so on a sharded store
  it predicted the object count off the read grid while `count_objects` walked
  the write grid — 493,484 predicted against 31,634 on disk for the store
  measured in #124, a 15.84× disagreement the report did not explain. It now
  predicts through `count_storage_objects` on the shard grid where there is
  one, labels the figure `Shard objects`, and prints the chunk grid beside it
  as the read unit it has become. The levels table gains `shard` and `shards`
  columns, so a pyramid whose shard shape changes partway down (the reference
  volume flips its long axis from X to Y at level 3) no longer reads as
  uniform. A new `shard shapes` check compares what is on disk against what the
  store's own recorded policy plans, and fails when they disagree in either
  direction — including a store that is sharded while its audit says it is not.
  An unsharded store's report is unchanged apart from the new check below.
  (#129)
- **The verifier compares object counts as `objects <= grid`, not equality, and
  says how far short.** Zarr writes no object that is entirely fill value, so a
  level with a padded trailing axis legitimately comes in under its own grid —
  #126's volume was 1,134 shards short of 210,345, every absent one on a
  3-voxel sliver. A shortfall passes and is quantified so a reader can tell
  "all-fill objects skipped" from "the geometry is wrong"; an excess fails.
  Counting is now classified against the level arrays' own names, because a
  store also holds one `zarr.json` per level plus the group's, and
  `OME/METADATA.ome.xml` with the source metadata beside it — six files on the
  smallest possible store, none of them pixels, and enough to put a correct
  store over its grid. `tests/test_verify_geometry.py` covers the script, which
  had no tests. (#129)

- **Every byte count zarrmony prints or records now carries the unit it was
  actually computed in.** `format_bytes` divides by 1024 and used to label the
  result `KB`/`MB`/`GB`/`TB`; it now says `KiB`/`MiB`/`GiB`/`TiB`. The
  arithmetic has not moved — the same 359,438,211,506-byte input reads
  `334.8 GiB` where it read `334.8 GB`, which was understating it by 7.4%
  against its own label and matched neither `ls -lh` (binary, bare `G`) nor
  Finder (decimal since macOS 10.6). This is the rendering of `size_human` in
  the audit record, the `Input:` / `Output:` / `Size:` lines of `convert` and
  `inspect`, and the store sizes `scripts/verify_geometry.py` prints into an
  acceptance report — so anyone diffing two audits across this change will see
  every size string move, with the `size_bytes` beside it unchanged. It also
  makes those figures comparable against `--chunk-target-bytes` and
  `--shard-target-bytes`, whose help text and ADRs were already written in
  KiB/MiB. `audit_schema_version` stays at 14: no key was added, removed or
  retyped, and the reason is recorded beside the constant. (#128)

## [0.16.0] - 2026-08-31

The planner already knew how many objects a conversion would write; now it
says so. Everything in this release is about what zarrmony tells the person
running it — one new warning before a six-day run starts, and a pass over the
existing text to remove everything in it that only a reader of this repository
could act on.

### Added

- **A conversion that will write a very large number of objects now says so
  before it starts.** The planner resolves every level's shape and chunk shape
  before a byte moves, so the object count is arithmetic rather than a
  measurement — and it used to go unsaid until the run failed to finish. With
  sharding off, a scene whose whole pyramid plans more than
  `STORAGE_OBJECT_WARN_COUNT` (100,000) objects raises `ObjectCountWarning` at
  plan time, before the arrays exist. The message carries the projected count,
  which scene it is about, the ~6 days the reference whole-slide scene took at
  493,484 objects (#113), `--shard-target-bytes` with the count it would give
  instead — and that a sharded store needs `sharding_indexed` support to open,
  which `lucida-store` does not have. Without that last clause the warning
  would trade a slow conversion for a store the user's viewer cannot open.
  Nothing is refused, no default moves, and the ordinary `warnings` filters are
  the override — there is no suppression flag. Quiet when sharding is on, since
  then the object count _is_ the shard count. (#122)
- `zarrmony.geometry.count_storage_objects(level_shapes, write_grids)` — the
  arithmetic on its own, taking one write grid per level so it returns the
  shard count on a sharded store and the chunk count on an unsharded one. It
  exists as a shared helper because `scripts/verify_geometry.py` predicts the
  count off the chunk grid and so misreports every sharded store (#124, tracked
  on #129); both callers can now reach the same figure. The result is an upper
  bound — zarr writes no all-fill object — so assert `objects <= count`.

### Changed

- **Text the CLI prints no longer cites ADRs, issue numbers or a downstream
  project.** The sharding and `--chunk-target-bytes` warnings, and the `--help`
  text for eight flags, carried references only a reader of this repository
  could use — and named `lucida-store` as the consumer that cannot open a
  sharded store, which dates the message the day that consumer gains the
  feature. The measured facts stay (121 resident chunks at 512 KiB against 4 at
  8 MiB; a consumer that parses the codec chain itself refuses a sharded
  store); the citations move to the docstring beside the code and to the ADR.
  `tests/test_cli.py` pins it, as `tests/test_object_count_warning.py` does for
  the warning added above.
- The VSI runbook and ADR-0011 no longer print the reference slide's main scene
  name; it is `<main-scene>`, with a note that the literal value is tracked
  internally. Scene names read as technical facts, which is what made this one
  survive review — a slide scanner builds them from the magnification, the
  filter panel and an acquisition index, so the string is a stable handle for
  one acquisition even though no token in it is sensitive alone. The three
  scanner-generated scene names around it are unchanged.
- `scripts/check_no_internal_paths.py` gained a rule for that shape (a
  magnification prefix followed by three or more joined tokens) and, for the
  first time, tests — `tests/test_check_no_internal_paths.py` pins every rule's
  negative cases as well as its positive ones, since a pattern that fires on
  ordinary prose gets suppressed until it stops protecting anything.
  CONTRIBUTING.md documents the scene-name case explicitly.

## [0.15.1] - 2026-08-31

Documentation only; no code change, and no reason to upgrade except to read
the corrected text. Two things closed since 0.15.0, and both replace a claim
with a measurement rather than merely qualifying it. 0.15.0 described
sharding's benefit in a way that implied a measurement nobody had taken — that
measurement has now been taken, on both a slide and a volume. And the ADR-0010
acceptance run (#90) finished, so the geometry series' two viewer criteria rest
on numbers instead of predictions.

### Fixed

- **0.15.0's sharding wall-clock was measured with 8 MiB _chunks_, not shards
  — and the sharded figure is now in hand.** The 3 h 02 m attributed to
  sharding came from a whole-slide conversion run at
  `--chunk-target-bytes 8388608`, no shards, on a pre-#112 build. That scene
  has since been re-converted at 512²-in-2048² on the same host at stock
  geometry (#124):

  | | 8 MiB chunks, no shards | 512² in 2048² shards |
  | --- | --- | --- |
  | wall-clock | 3 h 02 m 00 s | 3 h 04 m 54 s (+1.6 %) |
  | store size | 139.0 GiB | 138.92 GiB |
  | objects | 31,634 | 31,634 |
  | peak RSS | 12.35 GiB | 15.33 GiB |
  | CPU | 152 % | 177 % |

  So the transfer #113 predicted holds, and the inference is retired: sharding
  costs 1.6 % of wall-clock for a read unit 16× smaller. The object count
  landed on the planner's prediction exactly at all ten levels — 31,156 shard
  objects for the main scene against 493,484 chunks, 15.84× — the
  `TileAlignmentWarning` count over the whole run was **zero**, and write
  amplification was 1.0005, which settles the read-modify-write concern
  ADR-0010 raised against `lock=True`. README and ADR-0010 now carry the
  measured figures.
- **0.15.0 also predicted a sharded store would be larger, and it is not.**
  "512 KiB chunks compress worse than 8 MiB ones" is sound in principle and
  does not bite at this ratio: the two stores are 88 MB apart on 139 GiB. The
  prediction is removed from the README and corrected in ADR-0010.
- **The volumetric case is measured too, and it shows the caveat above was
  aimed at the wrong mechanism.** The reference whole-brain volume has been
  written both ways at the same `[1,1,64,64,64]` chunk and the same six levels
  — unsharded under #90, sharded under #126 — which isolates sharding from the
  chunk target:

  | | no shards | 8 MiB shards | |
  | --- | --- | --- | --- |
  | store bytes | 261,959,557,972 | 262,010,912,050 | +0.02 % |
  | objects | 3,182,337 | 209,220 | 15.2× |
  | wall-clock | 30 h 04 m 43 s | 10 h 09 m 54 s | 0.34× |

  **Compression runs per chunk, and sharding does not change the chunk**, so
  the compressed payload is identical by construction and the 51.4 MB delta is
  the shard index: 16 B per chunk slot plus 4 B per shard predicts 51.96 MB,
  within 1.2 %. Whether the chunk is a plane tile or a cube never entered into
  it — that is a question about `chunk_target_bytes`, a separate field. Note
  also that a volume's shard count is an upper bound rather than an exact
  prediction, since zarr skips a shard whose chunks are all fill value: 209,211
  landed against 210,345 planned.
- **`scripts/verify_geometry.py` turns out to be shard-blind in a way the
  0.15.0 notes understated.** It was known to predict object count from the
  chunk grid. Run against a real sharded store for the first time, the report
  **never mentions sharding at all** — no shard shape in the checks, no shard
  column in the levels table — and every check passes while it prints the
  unsharded object prediction. Use `--no-object-count` and measure the count
  directly until #129 lands the fix; do not quote the report as evidence of a
  store's geometry without saying in the surrounding text that it is sharded.
- **The reference volume's sharded object count in 0.15.0's Migration note was
  wrong** — "a 256³ shard holding 64 chunks of 64³ … ~47k objects" was the
  pre-implementation illustration from ADR-0010, which #117 had already
  corrected everywhere else. 256³ uint16 is 32 MiB, four times the 8 MiB
  target, so the planner returns `128 × 128 × 256`: 16 chunks per shard, not
  64. Re-derived against the shipped planner, the reference volume goes from
  3,194,997 objects to **210,345** (15.2×), not ~47k. The figure for the
  whole-slide scene — 369,600 level-0 objects to 23,184 — was already correct.

### Changed

- **`docs/references/geometry-acceptance-run.md` replaces "put it under a 3D
  camera and eyeball it" with a scripted procedure.** The criterion is a
  bounding-box check on integers: `lucida plan visible-chunks` dumps the
  selected chunk coordinates, `lucida camera` drives the 3D camera headlessly,
  and lucida-core already sorts the chunk list centre-out from the camera, so
  the first 64 entries of the `visible` tier are the admission window. The
  runbook also now warns that a stale Lucida CLI fails the coarse-tier gate
  confusingly — client and server can report the same version string while the
  client predates `dataset health` — and records the diagnostic's own
  `Web planner equivalent: false` caveat.

### HITL validation

**The ADR-0010 acceptance run is complete (#90).** The reference whole-brain
volume was re-converted from source under the default geometry, and both viewer
criteria now have measurements behind them rather than predictions.

- **Lucida resolves zarrmony's coarse tier and builds nothing of its own.**
  `lucida dataset health` reports
  `Generated coarse: healthy (levels 0, ready 0, pending 0, failed 0, unavailable 0)`
  with `Generated cache: 0 on disk`. `plan_generated_coarse_for_image`
  early-returns for any image whose manifest carries a `coarse_level_index`, so
  zero generated levels is the direct signal that the server accepted the
  mean-pooled level 5 instead of grinding out a max-pooled tier at concurrency 1
  over the full source. This is ADR-0010's coarse-level stopping rule observed
  end-to-end for the first time.
- **The 64-request admission window resolves into a centred block.** Under a 3D
  camera inside the volume at level 0, the window spans 6 × 6 × 3 chunks —
  691.2 × 691.2 × 384 µm, 32 MiB — drawn from 188,223 candidates in a
  13.4 × 15.9 × 7.25 mm volume. Under the old `[1,1,1,1125,7452]` chunk the
  level-0 grid was 1 × 8 × 3627, so the same 64 requests could only ever take
  1 × 8 × 8: the whole specimen, eight planes thick, at 1.0 GiB. The reported
  symptom — "the data budget is maxed out with a few slices instead of the 3D
  volume in the middle of the viewer" — was unavoidable at any budget under that
  grid, because a centred cube was not expressible in it. So the measurement
  confirms the mechanism end to end, but the causal argument rests on the chunk
  shape alone.

## [0.15.0] - 2026-08-27

The ADR-0010 output geometry series, start to finish: the policy itself
(#83–#88) plus the follow-ups it turned up once it met real data (#100–#117).
zarrmony now plans chunk shapes and pyramid levels itself instead of delegating
to `bioio-ome-zarr`'s memory-target heuristic; sharding arrives as an opt-in
answer to the object count that policy costs; and the Bio-Formats extra opens
the ~150 vendor formats that exercised all of it. A **minor** bump — on-disk
geometry changes for every new conversion, including plate output, but no API
breaks. Read the Migration section before comparing a new store against an old
one.

### Added

- Frozen `Geometry` policy object (ADR-0010), exported as
  `zarrmony.Geometry`, carrying every output-geometry choice in one
  immutable value: `chunk_target_bytes` (512 KiB), `isotropy_tolerance`
  (1.5), `axis_floor` (32), `coarse_max_bytes` (64 MiB),
  `coarse_max_long_axis` (2048), `downsample_method` (`"mean"`),
  `pyramid_min_size` (256) and an explicit `chunk_shape` override
  (`None`). `convert()` accepts it as `geometry=`, and it is threaded
  end-to-end through layout dispatch into the per-scene, bf2raw and
  plate writers — no loose geometry keyword survives in the chain.
  Values are validated at construction, so a bad policy fails at the
  call site rather than after a multi-minute read. This slice was
  behaviour-preserving on its own — the on-disk geometry it produced was
  byte-identical to v0.14.0 — with the five planner fields plumbed and
  audited but inert until the ADR-0010 chunk planner and anisotropy-aware
  pyramid land. (#83)
- World-cubic chunk planner (ADR-0010). `zarrmony.geometry.plan_chunk_shape()`
  and `plan_level_chunk_shapes()` pick, for each pyramid level, the largest
  power-of-two chunk whose raw size fits `chunk_target_bytes` and whose
  *physical* extents are closest to cubic — cubic in micrometres, not in
  voxels. Each level is planned against its own µm spacing (derived by
  the new `spacings_for_level` helper as
  `spacing_0 × shape_0 / shape_level`), so a level that halved Y and X
  does not inherit level 0's chunk. Chunks never exceed the level extent,
  and T and C stay at 1 so a viewer fetching one channel at one timepoint
  never pays for the others. Near-isotropic uint16 lands on the familiar
  64³; a 10:1 confocal stack (Z 5 µm, XY 0.5 µm) lands on
  `1,1,16,128,128` — 80 × 64 × 64 µm — instead of a 64³ that would span
  320 × 32 × 32 µm. (#84)
- `zarrmony convert --chunk-target-bytes N` exposes the byte target on the
  CLI. Mutually exclusive with `--chunk-shape`, which bypasses the planner
  outright. (#84)
- Anisotropy-aware pyramid levels (ADR-0010). A level halves every spatial
  axis whose physical spacing is within `isotropy_tolerance` (1.5) of the
  finest still-halvable axis's, so levels move *toward* isotropy and the
  scarce axis — Z, for most volumetric light microscopy — is spent last. On
  the reference SmartSPIM volume (Z 2.0 / Y 1.8 / X 1.8 µm) every axis is
  within tolerance, so levels go `(3627,8835,7452) → (1813,4417,3726) →
  (906,2208,1863) → (453,1104,931) → (226,552,465)` where Z previously stayed
  at 3627 forever; on a 10:1 confocal stack Y and X halve alone until their
  spacing has caught up with Z's, taking the coarsest level from 10:1
  anisotropic to 1.25:1. A per-axis floor of 32 voxels (`axis_floor`) applies:
  an axis never halves below it, so a 3-plane stack keeps its 3 planes at
  every level. (#85)
- `zarrmony convert --isotropy-tolerance F` exposes the tolerance on the CLI.
  `1.0` halves only exactly-isotropic axes; a large value halves every spatial
  axis at every level. (#85)
- Per-scene / per-field audit records gain `chunk_shapes`: one chunk shape
  per level, positionally aligned with the existing `level_shapes`. (#84)
- Coarse-level stopping rule (ADR-0010). Pyramid depth is now the **greater**
  of the `pyramid_min_size` Y/X rule and the depth at which a level becomes a
  *coarse level* — one a viewer can decode whole and use as spatial context:
  `Z·Y·X·itemsize ≤ coarse_max_bytes` (64 MiB) per timepoint and channel, and
  `max(Y, X) ≤ coarse_max_long_axis` (2048). Both `Geometry` fields were
  present but inert since #83 and are now read. Because depth is a `max()`,
  the change is monotone — no conversion loses a level — and the per-axis
  32-voxel floor still holds, so a pyramid that bottoms out while still too
  large simply has no coarse level. On the reference SmartSPIM volume this
  adds level 5 at `(113, 276, 232)`: 13.8 MiB per (t,c) with a 276-voxel long
  axis, where the Y/X rule stopped at level 4's 110 MiB. Both bounds are
  Lucida's `SourceCoarseConfig` defaults, adopted knowingly. (#86)
- `zarrmony convert --coarse-max-bytes N` and `--coarse-max-long-axis N`
  expose both bounds on the CLI, so a store can be planned for a consumer
  with a different budget. (#86)
- Per-scene / per-field audit records gain `coarse_level_index`: the index
  into that record's `level_shapes` of the shallowest coarse level, or `null`
  when no level reaches the bounds. "Does this store have a level a viewer can
  hold whole?" is now answerable from the store's own metadata instead of from
  a viewport. (#86)
- `Geometry(downsample_method=...)` now selects the pooling kernel every
  pyramid level above 0 is built with: `"mean"` (unchanged default) or
  `"max"`. Present but inert since #83; read by `build_pyramid` as of this
  slice. Mean-pool stays the default and the whole-pyramid answer — it is what
  the OME-Zarr ecosystem assumes, and max-pool biases every level above 0 high,
  lifting background toward the noise maximum at high factors. `"max"` exists
  for sparse-label acquisitions, where mean-pooling dissolves small objects
  into the background: a 15 µm soma filling 1.6 % of a level-5 voxel
  mean-pools to 114 against a background of 100, and max-pools to 1000
  against ~141. Applied **uniformly** to every level — a mean-detail /
  max-coarse hybrid was rejected because viewers with no coarse/detail concept
  (napari, vizarr) would show a brightness step at the last level. Both
  kernels preserve the input dtype. (#87)
- `zarrmony convert --downsample-method [mean|max]` exposes the kernel on the
  CLI. (#87)
- README gains an **Output geometry** section: what the planner does, every
  `Geometry` field with its default and its CLI flag, what the audit records,
  the object-count trade, and where sharding does and does not help.
  (#89, #117)
- **Sharding, opt-in and off by default.** `Geometry.shard_target_bytes` /
  `--shard-target-bytes` and `Geometry.shard_shape` / `--shard-shape` pack
  whole chunks into larger storage objects, reversing ADR-0010's rejection of
  the feature. The shard is the write unit and the storage object; the chunk
  stays the read unit, individually range-readable. That resolves what looked
  like a trade-off between a conversion that finishes and a store a viewer can
  use: on a 141k × 172k × 4-channel slide scene, 512 KiB chunks inside 8 MiB
  shards give level 0 the 8 MiB write unit that took a projected nine days
  down to 3 h 02 m, the 23,184 objects that run wrote instead of 369,600, and
  the 512 KiB read unit that fills a viewer's residency budget with 121 chunks
  instead of 4. **Those two figures were measured with 8 MiB _chunks_, not
  shards** — the transfer rested on #113's finding that cost scales with the
  write unit, and has since been confirmed directly: the same scene at
  512²-in-2048² took 3 h 04 m 54 s, +1.6 % (#124, see 0.15.1). Shards are
  planned per level by the same world-cubic rule as
  chunks, over whole-chunk multiples, so isotropic data at the defaults gets a
  `128 × 128 × 256` shard holding 16 chunks of 64³; `--shard-target-bytes`
  with no value resolves to 8 MiB. **It is off by default because it changes
  who can read the store.** Every zarr-python 3 consumer is unaffected —
  napari-ome-zarr, dask, plain `__getitem__` and subsets straddling either
  grid were verified byte-identical against an unsharded store — but a
  consumer that parses the codec chain itself sees `sharding_indexed` where it
  expects `bytes` and refuses to open the array, which today includes
  `lucida-store`. The CLI warns whenever sharding is on, naming exactly that.
  (#117)
- **RGB scenes convert.** Bio-Formats models a colour plane as one *channel*
  of three interleaved *samples* (`C=1, S=3`), but NGFF has no samples axis,
  so `normalize_axes` rejected the sixth dim and zarrmony could not convert
  any RGB image — brightfield scans, and the `label`/`macro` thumbnails that
  every whole-slide format carries beside its fluorescence scan. Since
  `convert()` has no scene selector, one 375×504 RGB thumbnail aborted the
  entire multi-scene conversion with `UnsupportedAxesError`. The new
  `transforms.fold_samples_axis` turns `S` into `C` before the order check —
  a rename, not a reshape, so it costs no dask rechunk on a gigapixel scene —
  and labels the result `Red`/`Green`/`Blue` (`Alpha` for RGBA). Those get
  the literal primaries rather than the ADR-0007 palette: those five hues
  encode fluorescence emission bands, and compositing a photograph's red
  sample in cyan would reproduce the wrong picture. A degenerate `S=1` is
  dropped instead of folded, and `C>1` with `S>1` raises rather than
  inventing an ordering. The audit's `axis_normalization` block gains
  `rgb_samples_folded`, with `input_dims` still showing the reader's own
  pre-fold dims — the `S` is the only record that the input was in colour.
- **`bioformats` optional extra** — `pip install "zarrmony[bioformats]"`
  installs `bioio-bioformats`, which the built-in `bioio` catch-all plugin
  then dispatches to with no zarrmony code, unlocking the ~150 vendor
  formats on the Bio-Formats supported-formats list that no permissively
  licensed bioio backend covers (Olympus/Evident cellSens VSI, Zeiss ZVI,
  Hamamatsu NDPI, …). `bioio-bioformats` is **GPL-3.0** and zarrmony is
  Apache-2.0, so the extra is opt-in in the strict sense: it is deliberately
  **not** part of `all`, not part of `dev`, and never in the default install
  — the user assembles the GPL environment themselves and zarrmony's own
  dependency closure stays Apache-2.0. Bio-Formats needs a JVM but not a
  system one: `bffile`/`scyjava`/`cjdk` fetch their own JDK (~36 MiB) and the
  maven artifacts on first use. New [ADR-0011](docs/adr/0011-bioformats-backed-formats.md)
  records the rule and the licence constraint, and qualifies ADR-0003 —
  Bio-Formats-covered formats get the extra, not a `zarrmony-<vendor>`
  adapter package. (#100, #102)
- **`reader_kwargs` now reach the built-in `bioio` plugin.** `_open_default`
  forwards `**kwargs` to `BioImage`, which forwards them to whichever bioio
  backend wins discovery — so `--reader-kwarg`/`reader_kwargs=` finally
  works for the catch-all path, not just for external plugins. The
  motivating case is gigapixel 2D: `bioio-bioformats` returns one dask chunk
  per plane (47.5 GB on the reference whole-slide input), which the writer
  cannot rechunk without materialising it, so
  `--reader-kwarg dask_tiles=true --reader-kwarg tile_size=1024,1024` is the
  difference between a conversion and an OOM. `dask_tiles` and `tile_size`
  are coerced from their CLI string form because the backends that accept
  them are third-party; every other key keeps the documented
  string-passthrough contract, and unknown keys still surface as the backend
  constructor's native `TypeError`. (#101)
- **Install hint on unreadable input.** When the default plugin's `BioImage`
  raises `UnsupportedFileFormatError`, zarrmony now wraps it in
  `UnsupportedFormatError` naming the `bioformats` extra (bioio's original
  chained as `__cause__`). Suppressed when `bioio-bioformats` is already
  installed. (#102)
- **A post-mortem that tells "no reader" apart from "cannot read".** bioio
  reports every dispatch failure as `UnsupportedFileFormatError` recommending
  an extra, discarding the real cause into a log line — so a read-denied
  mount, a dead JVM and a genuinely unknown format are indistinguishable, and
  the recommended fix is wrong for two of the three. On that failure path the
  default plugin now checks whether the input can actually be opened and its
  directory listed, raising the new `InputAccessError` when it cannot
  (naming macOS's privacy layer for an `EPERM`, which is what a denied SMB or
  FUSE volume under `/Volumes` looks like), flagging a readable file in an
  unlistable directory (how a multi-file format such as VSI fails when its
  sibling pyramid is out of reach), and folding bioio's discarded per-backend
  errors into the message. When `bioio-bioformats` is installed the message
  now says so and reports what was tried instead of falling back to bioio's
  "install an extra".
- `docs/writing-a-reader-plugin.md` opens with a "do you actually need a
  plugin?" section pointing at the Bio-Formats list first, and
  `docs/references/vsi-acceptance-run.md` is the runbook for the ADR-0011
  acceptance conversion. (#102, #103)

### Fixed

- **`input.size_bytes` and `--checksum` no longer describe an index file and
  call it the input.** Both stat and hash the path handed to `convert`, but for
  a multi-file vendor format that path holds no pixels: a whole-slide VSI is
  4.4 MB of `.vsi` beside 37.2 GB of `.ets` tiles in a sibling directory, so
  the audit reported `Input: 4.4 MB` against `Output: 139.0 GB` and the
  recorded SHA256 would not have changed if every tile were replaced.
  zarrmony now asks the reader what it actually read —
  `IFormatReader.getUsedFiles()`, via `bffile`'s `BioFile.used_files()` — and
  records `input.files` (`count`, `size_bytes`, a listing capped at 64 with
  `listing_truncated`) plus `input.size_is_partial`. Under `--checksum`,
  `input.files.sha256` is a manifest digest over the whole set: SHA256 of
  sorted `<relpath>\0<file sha256>` lines, relative to the set's common
  ancestor, so moving a dataset between filers does not change it but adding,
  renaming or corrupting a member does. `size_bytes` and `sha256` keep their
  old meaning — the named path — so nothing pinned to schema 13 reads a
  different quantity under an old key. Both new keys are **absent**, not
  `false`, when no reader could report a file set: unknown and known-complete
  are different claims. `inspect` reports the same block and never hashes it,
  and the CLI's `Input:` / `Size:` lines now read
  `37.2 GB across 6 files (the named path alone is 4.4 MB)`. (#116)
- **Reader tiles now nest in the planned write grid instead of being split
  into it.** Nothing connected the reader's `tile_size` to the geometry the
  ADR-0010 planner picks, so `write_pyramid` absorbed the difference with a
  `rechunk` — and on the whole-slide path that rechunk was always a *split*,
  which is not the same price as a merge. On a 141k × 172k × 4-channel scene,
  1024² source tiles into the planned 512² write grid built **831,936** dask
  tasks against the 369,600 an aligned run needs, and re-read each source tile
  once per output chunk it fed; merging in the other direction costs 1.06–1.25×.
  Asking for *larger* tiles therefore produced a *larger* graph, which is why
  this survived review — and why the recommendation in this repo's own README,
  ADR-0011 and VSI runbook, `--reader-kwarg tile_size=1024,1024`, was the worst
  of the available options. `convert()` now plans the write grid from metadata
  alone, before touching a pixel, and reopens the reader asking for tiles that
  divide it; the grid is `shards or chunks`, so the derived tile follows the
  write unit under sharding rather than the read unit. Files with several
  scenes get the element-wise minimum, which is provably split-free and is not
  dragged down by a `label` thumbnail whose grid already spans it, and each
  scene is planned at **its own** dtype — a whole-slide VSI is `>u2`
  fluorescence beside `uint8` RGB thumbnails, and itemsize divides the byte
  target. **Pass
  `--reader-kwarg dask_tiles=true` alone now**; a pinned `tile_size` is still
  honoured, and the writer warns — naming the tile that would have worked —
  whenever source blocks split, whatever the reason. The tile zarrmony asked
  for is recorded as `config.reader_tile_size`. (#112)
- **Gigapixel conversions no longer stall in graph construction.** The writer
  built every pyramid level as one lazy dask graph and handed the lot to a
  single `da.compute`, so the coarsest level's graph reached back through every
  intermediate level to the reader and every level's rechunk was constructed up
  front. On a 141k × 172k × 4-channel whole-slide scene that graph never
  finished being *built*: py-spy showed hours inside `Task.__init__`,
  `blockwise.cull` and `rechunk._compute_rechunk` at ~115 % CPU with almost
  nothing written, and what did get written cost 4.2× more source reads than
  the input's own size. `ZarrmonyWriter.write_pyramid` now writes one level at
  a time, pooling each from the level just written and re-opened off the store,
  so the graph held at any moment is bounded by a single level's task count and
  raw pixels are read exactly once, by level 0. Correctness rests on both
  pooling kernels being exact over a block of any shape, which is why the
  pyramid could always have been built this way — a level pooled from disk is
  bit-identical to the same level pooled inside one graph, asserted for `mean`
  and `max` alike. The cost is reading each written level back once: the
  pyramid's own bytes, roughly a third of level 0. (#111)
- **Data-driven contrast no longer blocks the write.** The per-channel min and
  percentile were fused into the pyramid's single `da.compute` as `extra_ops`,
  on the theory that sharing a graph meant sharing the chunk reads. Measured
  against a whole-slide input it did the opposite: a controlled A/B differing
  only in `contrast_percentile` wrote **0 bytes in 610 s** with contrast on —
  while reading 200–370 MB/min — against a first chunk at 291 s and 12–38
  chunks/min with it off. Contrast is now computed after the pyramid is on
  disk, reading back the coarsest level, which `geometry.coarse_max_bytes`
  already caps at 64 MiB. Same values, no effect on the write. (#114)
- **Pyramids build on readers that hand back lazy blocks.**
  `bioio-bioformats` assembles its dask graph out of `LazyBioArray` handles
  rather than materialised arrays. Level 0 wrote fine — zarr only needs
  `__array__` — but every level above it goes through `dask.array.coarsen`,
  which reshapes each block, so the conversion died mid-write with
  `AttributeError: 'LazyBioArray' object has no attribute 'reshape'` (and the
  contrast pass hit the same wall needing `.mean`). `write_scene` now maps
  `np.asarray` over the blocks when the array's `_meta` cannot reshape. The
  check is on the block prototype's capabilities, not the reader's identity —
  a dask array whose blocks cannot reshape is broken for any backend — and it
  reads nothing, so every reader that already yields ndarrays keeps its graph
  untouched.

### Changed

- `convert()`'s `chunk_shape` and `pyramid_min_size` are **retained** as
  sugar that folds into the geometry policy, so no existing caller
  breaks. Passing `geometry=` together with either raises `ValueError`
  rather than silently picking a winner. `--pyramid-min-size` on the CLI
  is unchanged apart from its help text now naming the ADR-0010 default.
  (#83)
- `write_scene()` and `write_plate()` replace their `pyramid_min_size` /
  `chunk_shape` keywords with a single `geometry: Geometry` argument
  (defaulting to `DEFAULT_GEOMETRY`). These are internal writer entry
  points, not part of the public `zarrmony` surface. (#83)
- **BREAKING (audit schema 8 → 9):** the audit `config` block replaces
  the `pyramid_min_size` / `chunk_shape` pair with a single `geometry`
  sub-dict recording the *resolved* policy. Consumers reading
  `config.pyramid_min_size` must read `config.geometry.pyramid_min_size`.
  Per-level shapes stay on each scene / field record's `level_shapes`;
  per-level chunk shapes and the coarse level index join them in a later
  ADR-0010 slice. (#83)
- **Chunk shapes on disk change for every conversion.** zarrmony now
  plans each level's chunk itself and hands the writer an explicit
  per-level list, instead of delegating to bioio-ome-zarr's memory-target
  heuristic. That heuristic filled the rightmost axis first under a 16 MiB
  budget — right for a 2D plane, but on anything with a Z extent it
  converged on full-width single-plane slabs (e.g. `1,1,2,2048,1536`,
  12 MiB) that no frustum cull can trim; a 512³ region fetch touched
  ~1.6 % useful bytes. Existing stores are unaffected and still readable;
  re-convert to pick up the new geometry. Pass `chunk_shape=` (or
  `--chunk-shape`) to keep an exact shape. (#84)
- **Audit schema 9 → 10 (additive):** per-scene / per-field records gain
  `chunk_shapes`. Consumers pinned to 9 can widen their pin without
  changes. (#84)
- **Pyramid level shapes on disk change for volumetric conversions.** Z now
  downsamples alongside Y and X whenever its spacing is within
  `isotropy_tolerance`, so a level is ⅛ of its parent rather than ¼ and a
  volume gains a genuinely coarse level instead of a stack of full-depth
  slabs. Per-axis factors flow into `coordinateTransformations` automatically
  — `OMEZarrWriter` derives each dataset's scale from its shape relative to
  level 0's — so a store's NGFF metadata stays correct with no extra
  handling. 2D and plate output are unchanged by this rule: a single-plane Z
  cannot halve, and depth was still decided by the Y/X `pyramid_min_size` rule
  alone until #86 below. (#85)
- Pyramid depth additionally stops once Y and X are both at their floor,
  rather than continuing to halve Z alone — those levels are not new
  resolutions to a viewer zooming out. This can only remove levels that did
  not exist before #85. (#85)
- `compute_level_shapes()` takes the scene's per-axis µm spacing and a
  `Geometry` in place of its `min_size` keyword, and `build_pyramid()` drops
  its `dims` argument — coarsen factors now come from consecutive level
  shapes, so no hardcoded `{axis: 2}` remains. Both are internal writer entry
  points, not part of the public `zarrmony` surface. (#85)
- **Volumes too large for a viewer to hold gain one or more levels.** Depth no
  longer stops at the `pyramid_min_size` Y/X floor when the resulting level is
  still, say, 110 MiB per timepoint and channel; it keeps halving toward the
  coarse-level bounds, down to the 32-voxel axis floor if that is what it
  takes. Existing conversions can only gain levels, never lose them. Most 2D
  and plate output is unaffected — the Y/X rule already reaches a level well
  inside both bounds. (#86)
- `compute_level_shapes()` takes a required `dtype` argument (4th positional,
  matching `plan_chunk_shape`): the coarse-level byte bound is about decoded
  bytes, so the same shape in uint8 reaches it a level earlier than in uint16.
  Internal writer entry point, and unreleased since #85. (#86)
- **Audit schema 10 → 11 (additive):** per-scene / per-field records gain
  `coarse_level_index`. Consumers pinned to 10 can widen their pin without
  changes. (#86)
- `build_pyramid()` takes a third positional `geometry` argument (defaulting to
  `DEFAULT_GEOMETRY`) to read `downsample_method` from. Internal writer entry
  point, not part of the public `zarrmony` surface. **No audit schema bump** —
  `config.geometry.downsample_method` has been recorded since #83, so a
  consumer pinned to schema 11 can already tell a max-pooled store from a
  mean-pooled one. (#87)
- Under `downsample_method="max"`, the data-driven omero contrast window
  (`--contrast-percentile`, #53) opens higher: it is the min and Nth percentile
  of the coarsest pyramid level, and that level is now max-pooled. This is the
  correct window for the pyramid actually written — the default `"mean"` path
  is unchanged. (#87)
- **2D and plate output get the same geometry as everything else**, with no
  `Z > 1` gate and no plate exemption — per-scene, bf2raw and plate conversions
  of the same field now provably plan identical `level_shapes`, `chunk_shapes`
  and `coarse_level_index`, and write identical arrays. The visible change is
  on single-plane fields: a 2160² uint16 Phenix field was one
  `(1,1,1,2160,2160)` chunk of 8.9 MiB — more than a viewer's entire 8 MB
  per-frame decoded budget in one object, and nothing a frustum cull could trim
  — and is now 25 chunks of exactly 512 KiB at level 0. On a single plane the
  *lateral* coarse bound is what decides which level is coarse (8.9 MiB is an
  eighth of `coarse_max_bytes`), so `--coarse-max-long-axis` is the knob that
  matters for plate stores and `--coarse-max-bytes` is inert. (#88)
- **`audit_schema_version` bumps `11 → 12`.** Additive. Each scene / plate
  field record gains `shard_shapes` — one shard shape per pyramid level,
  positionally aligned with `level_shapes` and `chunk_shapes`, or `null` for
  the whole record when sharding is off, which is every default conversion —
  and `config.geometry` gains `shard_target_bytes` and `shard_shape`. Worth a
  bump rather than a silent addition because object layout, and whether a
  consumer needs `sharding_indexed` support to open the store at all, are no
  longer inferable from `chunk_shapes`. The bump also covers
  `axis_normalization.rgb_samples_folded` (#107), which landed against 11.
  Consumers pinned to 11 can widen their pin. (#117)
- **`audit_schema_version` bumps `12 → 13`.** Additive. `config` gains
  `reader_tile_size` — the `(Y, X)` tile zarrmony asked the reader for so its
  blocks would nest in the write grid, or `null` when it left the reader's own
  blocking alone. Recorded rather than left to be rediscovered because two runs
  of the same file produce identical stores and differ only in what they cost.
  Consumers pinned to 12 can widen their pin. (#112)
- **`audit_schema_version` bumps `13 → 14`.** Additive. `input` gains `files`
  and `size_is_partial` — what the reader says it actually read, when it can
  say. `input.size_bytes` and `input.sha256` are unchanged and still describe
  the path the user named. Consumers pinned to 13 can widen their pin, but
  anything deriving a compression ratio or a "did we convert the whole file"
  check from `size_bytes` should now branch on `size_is_partial` first. (#116)
- `ZarrmonyWriter.write_pyramid()` takes the **base array** plus a `geometry=`
  keyword and returns `None`, where it used to take the full list of per-level
  dask arrays plus `extra_ops=` and return the computed extras. It derives the
  levels itself now (#111), so the caller no longer builds a lazy pyramid to
  hand it. The new `read_level(i)` re-opens a written level as a dask array.
  `build_pyramid()` is unchanged and still the right tool for a scene that fits
  in memory; the one-step-at-a-time primitive both paths share is the new
  `writers.pyramid.downsample_step()`. Neither name is exported from the
  `zarrmony` namespace.
- `bioio-base` is now a declared dependency rather than an undeclared
  transitive one — `readers/default.py` imports `UnsupportedFileFormatError`
  from it and `bioio` does not re-export it. No resolution change in
  practice; `bioio` already required it.

### Migration

Nothing in the API breaks — `chunk_shape` and `pyramid_min_size` keep working
as sugar over the new policy object, and existing stores are unaffected and
still readable. What changes is the geometry of every store written from here
on, so a new conversion will not match an old one. Five things to expect when
comparing them.

- **Object count rises ~37× on volumetric data, and that is the one number this
  policy makes materially worse.** The reference whole-brain light-sheet store
  goes from **87,048 objects to ~3.2 M** (2.78 M of them at level 0) — the
  512 KiB chunk target trades bytes-per-object for objects. Plate output rises
  less: a 2160² field goes from 4 chunk objects to 39 (~10×), a 1080² field
  from 3 to 14 (~4.7×), and a 384-well plate at 4 fields × 3 channels from
  roughly 23k objects to roughly 106k. This is irrelevant on local disk. **On
  object storage it is listing and per-object metadata cost** — budget for it
  before re-converting a large store to GCS/S3, and reach for
  `--shard-target-bytes` rather than `--chunk-target-bytes`: shards cut the
  object count without enlarging the read unit, which is what a viewer's
  residency budget actually cares about. See the next bullet for the catch.
- **Sharding answers that object count, and you have to ask for it.** At the
  8 MiB default an isotropic volume gets a `128 × 128 × 256` shard holding 16
  chunks of 64³, which brings the reference store from 3,194,997 objects to
  210,345 with every chunk still individually range-readable; the reference
  whole-slide scene's level 0 goes from 369,600 to 23,184. Both are the
  planner's arithmetic. The 8 MiB **write unit** is also what took that scene
  from a projected nine days to 3 h 02 m — that run used 8 MiB chunks, and the
  sharded re-conversion has since matched it at 3 h 04 m 54 s (#124, see
  0.15.1). It stays off by
  default because it changes who can read the store: `lucida-store`'s
  codec-chain parser accepts only `[bytes]` or `[bytes, compressor]` and
  rejects `sharding_indexed`, so a sharded store fails to open there with
  `first storage codec must be 'bytes', got 'sharding_indexed'` — an error
  that reads as corruption rather than as an unsupported feature. Every
  zarr-python 3 consumer is unaffected. Turn it on with
  `--shard-target-bytes` when you know your reader supports it (ADR-0010
  follow-up, #117).
- **The pyramid looks different at its coarsest level, for anyone who was
  seeing a server-generated coarse tier.** A volumetric store now contains a
  real coarse level, so a viewer that previously fell back to generating its own
  max-pooled tier stops doing that and reads the written one instead. The
  written one is **mean-pooled**, so sparse structures — labelled somata,
  puncta, thin processes — appear **dimmer** at the coarsest level than they did
  before. This is an appearance change, not a data loss: mean-pool is what the
  OME-Zarr ecosystem assumes and is the only pooling that keeps the pyramid
  usable for measurement. Convert with `--downsample-method max` to keep the
  previous look throughout the pyramid.
- **The pyramid gains a level but shrinks in relative overhead; do not budget
  for a size reduction.** Because Z now halves alongside Y and X, each level is
  **⅛** of the previous rather than ¼: on the reference volume, levels 1–5 add
  ~14 % over level 0 where levels 1–4 previously added ~33 %. Smaller chunks
  compress somewhat worse than full-width slabs, so the net is a wash to
  modestly smaller overall rather than a guaranteed saving.
- **New stores are not comparable byte-for-byte with v0.14.0 ones.** Chunk
  shapes, pyramid depth and — for volumetric data — level shapes all change, so
  a checksum or object-listing diff against an old store will differ everywhere.
  Re-convert from source to pick up the new geometry; pass `chunk_shape=` /
  `--chunk-shape` to reproduce an exact previous shape. An OME-Zarr → OME-Zarr
  `zarrmony rechunk` migration command is a follow-up (ADR-0010).

## [0.14.0] - 2026-08-10

### Added

- LIF plate detection (tracer bullet, ADR-0009). The built-in `bioio-lif`
  reader walks the `LMSDataContainer` XML at open time and, when the file
  contains exactly one plate template, sets `layout_hint="plate"` and
  populates `plate_layout` with the well/field grid. Under the default
  `zarrmony convert --layout auto`, a single-plate LIF now writes a
  spec-conformant OME-NGFF HCS `plate.zarr` (previously per-scene stores).
  `zarrmony inspect` surfaces a `plate_layout` block for these files.
  Row letters are normalized to uppercase and column strings to width-2
  zero-padded (`zarrmony-phenix` convention); the plate template's full
  row/column list is preserved even when only some wells were imaged.
  Non-plate (flat) LIF conversion is unchanged. Multi-plate LIFs continue
  to convert as flat until #82 wires the `--plate NAME` selector — their
  plate names surface via `reader.available_plates` so #82 has something
  to key off. (#81)
- Multi-plate LIF selection via `--plate NAME` (ADR-0009). The
  `bioio-lif` reader now accepts a `plate=` kwarg (surfaced on the CLI
  as `zarrmony convert --plate NAME`) that narrows `reader.scenes` to a
  single plate template's fields and remaps `PlateField.scene_index` to
  positions in that filtered list, so a multi-plate LIF converts as one
  spec-conformant `plate.zarr` per invocation. `zarrmony convert` on a
  multi-plate LIF **without** `--plate` now raises `PlateSelectionError`
  naming every available plate (previously it silently produced per-scene
  stores for all plates concatenated). `zarrmony inspect` surfaces a new
  additive `plates` block listing every plate template found on
  multi-plate LIFs. Passing `--plate NAME` on a single-plate LIF is
  accepted as a belt-and-suspenders check — a mismatch raises with both
  the requested and the actual plate name. Motivating case: the user's
  real 30 GB `PFF-HEK293-seeding-07172026.lif` packing `BFP-SNCA-WT`
  and `BFP-SNCA-A53T` in one file — now converts as two separate plate
  stores with one `convert` call each. (#82)

### Changed

- **BREAKING:** `zarrmony convert` on a multi-plate LIF without
  `--plate NAME` now raises `PlateSelectionError` instead of silently
  routing to per-scene output. Callers who were relying on the flat
  fallback for a multi-plate LIF should pick a specific plate per
  invocation (running `convert` once per plate) or pass
  `--layout per-scene` explicitly to opt out of plate detection.

## [0.13.0] - 2026-08-05

### Added

- Generic reader-kwargs passthrough on `convert()` and `inspect()`: both
  gain an optional `reader_kwargs: dict[str, Any] | None = None` that is
  forwarded verbatim to the winning plugin's `open()` as `**kwargs`. The
  CLI grows a matching repeatable `--reader-kwarg KEY=VALUE` option on
  `zarrmony convert` and `zarrmony inspect`. Motivating case: reaching
  `zarrmony-smartspim` v0.2.0's `metadata_path=` sidecar override from
  the public API and CLI without instantiating the reader directly, so a
  read-only SmartSPIM export directory can be paired with a sidecar JSON
  stored elsewhere:

  ```python
  from zarrmony import inspect
  inspect(
      "/mnt/readonly/<dataset>",
      reader_kwargs={"metadata_path": "/gdrive/.../metadata_<dataset>.json"},
  )
  ```

  ```bash
  zarrmony inspect /mnt/readonly/<dataset> \
    --reader-kwarg metadata_path=/writable/metadata_<dataset>.json
  ```

  Values from the CLI stay as strings; readers coerce internally
  (`SmartSpimReader` casts `metadata_path` to a `Path`). Duplicate CLI
  keys fail loud (rather than silent last-wins). Unknown kwargs surface
  as the reader constructor's native `TypeError` — zarrmony deliberately
  does not validate the shape, deferring a plugin-side kwarg schema
  until a second plugin needs the same discovery mechanism. (#79)

### Changed

- `ReaderPlugin.open`'s type widens from `Callable[[Path], ReaderProtocol]`
  to `Callable[..., ReaderProtocol]`. Existing plugins whose `open()`
  accepts only a path see `**{}` and behave identically — backwards
  compatible for every registered built-in (`bioio`, `bioio-czi`,
  `bioio-lif`, `bioio-nd2`) and every external plugin
  (`zarrmony-smartspim`, `zarrmony-snouty`, `zarrmony-blaze`,
  `zarrmony-phenix`). `get_reader(path)` grows an optional
  `reader_kwargs=` and forwards it via `plugin.open(p, **(reader_kwargs
  or {}))`.
- Reader-plugin authoring guide gains a subsection under §4 documenting
  how to declare a reader-specific keyword argument on `open()` and how
  the passthrough reaches it from the API and CLI.

## [0.12.0] - 2026-08-05

### Added

- CZI microscope-model extraction from the raw vendor XML: new
  `zarrmony.metadata.czi_acquisition.extract_czi_acquisition` reads
  `Metadata/Information/Instrument/Microscopes/Microscope[@Name]` from
  `reader.metadata` and returns `{"microscope": "Zeiss <Model>"}`
  (`"Zeiss Axioscan 7"`, `"Zeiss LSM 980"`, etc.). Fills the model gap
  bioio-czi's OME projection leaves — the OME `<Microscope>` element on
  CZI files ships with `Manufacturer="Zeiss"` but empty `Model`, so
  `per_scene[i].acquisition.microscope` used to land as just `"Zeiss"`
  and was indistinguishable across Axio Scan, LSM 900/980, Elyra,
  Celldiscoverer, and Lattice Lightsheet. XXE-hardened parser (rejects
  DOCTYPE / ENTITY declarations, size-capped) matches the LIF extractor. (#77)
- ND2 microscope extraction from `nd2.ND2File(...).text_info()`: new
  `zarrmony.metadata.nd2_acquisition.extract_nd2_acquisition` reopens the
  ND2 file via the Nikon SDK and reads the free-text `capturing` field
  (typically `"NIS-Elements 5.42\nNikon Instruments Inc.\nTi2-E"`), dropping
  vendor/software lines and prepending `"Nikon "` to the microscope stand
  when needed. Fills the `microscope` slot bioio-nd2's OME projection omits
  entirely — previously ND2 scenes shipped with no `microscope` key at all.
  Fail-safe when the `nd2` package isn't installed or the SDK raises. (#78)

### Changed

- `_audit_acquisition_for_scene` gains a vendor-specific tier between the
  LIF extractor and the OME projection: `bioio_czi`-shaped readers dispatch
  to the CZI vendor extractor, `bioio_nd2` to the ND2 one. Dispatch keys
  on the reader class's `__module__` so no reader-side changes are needed.
  Precedence remains uniform `setdefault` — vendor beats OME (so
  `"Zeiss Axioscan 7"` wins over `"Zeiss"`), LIF still beats vendor, and
  the reader `acquisition_audit` hook still fills only gaps none of the
  above populated. Same 4-tier composition applied to `inspect()`. (#77, #78)
- Plugin authoring guide §2 `acquisition_audit` subsection updated to
  document the new 4-tier precedence order.

## [0.11.0] - 2026-08-05

### Added

- Non-LIF `imaging_method` extraction: `extract_acquisition_from_ome` now
  projects `<Channel AcquisitionMode>` into `per_scene[i].acquisition.imaging_method`
  as a deduped `list[str]` in first-seen channel order. Scenes whose channels
  record different modes (bright-field reference + confocal detail, e.g.)
  surface every mode encountered. OME's enum vocabulary maps to the same
  ADR-0008 token set the LIF extractor already emits — `WideField` →
  `widefield_fluorescence`, `LaserScanningConfocalMicroscopy` → `confocal`,
  `SpinningDiskConfocal` / `SweptFieldConfocal` → `spinning_disk_confocal`,
  `SPIM` → `light_sheet`, plus `TIRF`, `STED`, `MultiPhotonMicroscopy` →
  `multiphoton`, `BrightField`, `StructuredIllumination`, etc. Unmapped
  values (`Other`, or any future enum member not in the map) are dropped
  rather than emitted verbatim so the token vocabulary stays bounded. (#76)
- Soft-optional `reader.acquisition_audit` hook: any reader can inject
  fields into `per_scene[i].acquisition` by exposing an attribute (or
  `@property`) returning a dict with any subset of `{date, microscope,
  microscope_serial, imaging_method}`. Reserved for readers whose modality
  is known by construction but neither the LIF scene-XML extractor nor OME's
  `Channel.AcquisitionMode` surface produces it (SmartSPIM / Blaze light-sheet
  TIFFs whose source files have no OME modality tag). Same fail-safe shape
  as `layout_hint` / `channel_names` — accessed via `getattr` guarded by
  `try/except` so a raising hook degrades to no extras. Callable and
  plain-attribute forms both supported. (#76)
- Plugin authoring guide §2 gains an `acquisition_audit` row + a full
  subsection covering the shape and precedence semantics.
- LIF acquisition extractor grows widefield detection: per-channel
  `WideFieldChannelInfo.ContrastingMethodName` (`FLUO` →
  `widefield_fluorescence`, `BF` / `TL-BF` → `bright_field`, `DIC` → `dic`,
  `PH` → `phase_contrast`, `POL` → `polarised_light`, `DF` → `dark_field`)
  and `ATLCameraSettingDefinition` presence as a widefield-family fallback.
  Multi-channel scenes with mixed contrasting methods surface every distinct
  token, matching the OME `Channel.AcquisitionMode` behaviour. Fixes missing
  `imaging_method` on Leica Thunder / DMi8 / AF 6000LX LIFs where
  `DataSourceTypeName` is the generic `"Camera"`.
- LIF acquisition extractor grows `<TimeStampList>` date parsing: the LAS X
  3.x per-scene shape carries space-separated hex FILETIME values in the
  element text (e.g. `"1dc28bfd6199e60 1dc28bfd9148ee0 …"`); the first value
  is projected to the scene's ISO 8601 UTC start time. Older
  `<TimeStamp HighInteger= LowInteger= />` handling is preserved and still
  wins when both encodings are present. Fixes missing `date` on Thunder /
  SP8 / STELLARIS LIFs whose per-scene XML uses the newer surface.

### Changed

- `per_scene[i].acquisition` composition is now uniformly layered across
  three sources with `setdefault` precedence (first source that populates a
  key wins; later sources fill only remaining gaps): LIF scene-XML extractor
  (LIF scenes only) → OME projection from `reader.ome_metadata` → reader
  `acquisition_audit` hook. Previously the LIF extractor ran alone if it
  fired (wholesale winner), so LIF scenes with an incomplete `HardwareSetting`
  header dropped `imaging_method` entirely even when bioio-lif's
  `Channel.AcquisitionMode` had it. Now the OME projection can fill LIF's
  gaps. The hook can never override a source-file-derived extraction —
  OME reporting `["confocal"]` beats a hook claiming `["light_sheet"]`,
  matching the principle that source-file metadata is more trustworthy
  than caller-supplied hints. (#76)
- Same 3-tier composition applied to `inspect()`'s per-scene acquisition
  path so pre-flight tooling sees identical output to the convert-time
  audit.

### HITL validation

Real-data inspect() passes across every currently-supported reader:

- **LIF Thunder AF 6000LX widefield** — `["widefield_fluorescence"]`, plus
  `["widefield_fluorescence", "bright_field"]` for a multi-modal scene, and
  `date` now populated via the new `TimeStampList` parser.
- **LIF SP8 confocal**, **LIF STELLARIS 8 confocal** — `["confocal"]`,
  microscope + serial, `date` now populated (previously missing).
- **CZI Zeiss Axioscan (widefield fluorescence)** —
  `["widefield_fluorescence"]` via OME `Channel.AcquisitionMode`.
- **ND2 Nikon spinning disk** — `["spinning_disk_confocal"]` via OME
  `Channel.AcquisitionMode`.
- **Snouty HT-SOLS light-sheet** — `["light_sheet"]` via `acquisition_audit`
  hook (external plugin).
- **Blaze Miltenyi UltraMicroscope light-sheet** — `["light_sheet"]` via
  hook (external plugin).
- **Phenix Opera Phenix** — `["spinning_disk_confocal"]` via hook
  (external plugin).

Known partial gaps (deferred to follow-up tickets, not blocking):
`microscope` on CZI ships as just `"Zeiss"` (bioio-czi omits Model on
Axioscan) and is entirely missing on ND2 (bioio-nd2 doesn't populate the
OME `<Microscope>` element); Phenix has no `date` yet. Each is filed
separately.

## [0.10.0] - 2026-08-04

### Added

- Top-level `attrs.zarrmony.output` audit block, initially with a single key
  `ome_ngff_version = "0.5"`, sourced from the writer's `NGFF_VERSION`
  constant. Gives Aperture BigQuery ingest a single stable audit path for the
  NGFF version instead of hardcoding `"0.5"` or reading a different
  attribute. `NGFF_VERSION` is now imported from a shared
  `zarrmony._constants` module by both `writers/plate.py` and
  `writers/bf2raw.py`, eliminating the risk of drift between the two writer
  paths and the audit. (ADR-0008, #70)
- Per-scene channel identity in every reader path: `per_scene[i].channels`
  (flat layout) / `fields[i].channels` (plate layout) now carry one dict per
  channel in acquisition order, with the ADR-0008 9-key shape
  (`{index, name, dye?, fluor?, excitation_nm?, emission_low_nm?,
  emission_high_nm?, color?, lut_name?}`). LIF persists the extractor's
  already-parsed identity verbatim (dropping the LIF-internal `detector`
  field); CZI / ND2 / OME-TIFF / default project from
  `reader.ome_metadata.images[0].pixels.channels`. Missing per-channel fields
  are omitted (never emitted as `null`) so consumers can distinguish "reader
  didn't extract this field" from "reader tried and got nothing." Per-tile
  (`lif_mosaic="per-tile"`) writes stamp the same block onto every tile store
  so each remains fully self-describing. (ADR-0008, #61)
- New helper module `zarrmony.metadata.audit_channels` (public functions
  `from_lif_extracted`, `from_ome_channels`) implements the shared projection
  used by all reader paths. `zarrmony.metadata.lif_channels.resolve_channel_colors`
  is the renamed / promoted public form of the previously-internal color
  batch resolver, so the audit's `color` value matches what the writer wrote.
- Non-LIF per-scene objective + acquisition extraction (ADR-0008 / #63, #64,
  #65). CZI / ND2 / OME-TIFF / any other bioio reader whose OME surface
  exposes `instruments[0].objectives` / `instruments[0].microscope` /
  `images[i].acquisition_date` now populates the same `objective` and
  `acquisition` audit blocks the LIF path already emitted. Shared
  extractor module: `zarrmony.metadata.ome_extractors` (public
  `extract_objective_from_ome`, `extract_acquisition_from_ome`) — walks
  `reader.ome_metadata` in fail-safe mode, returning the ADR-shaped dict
  or `None`. `microscope` combines `Manufacturer` + `Model` into
  `"Nikon Ti2"`-style strings; `microscope_serial` comes from the OME
  `Microscope.serial_number`. `imaging_method` is deliberately NOT
  populated by the shared OME projection — OME has no first-class
  modality field; format-specific tokens ride on the LIF-only surface for
  now. Also extends `inspect()` so pre-flight tooling sees the same
  acquisition block via the OME fallback for non-LIF readers. (#63, #64, #65)
- LIF per-scene acquisition/instrument extraction (ADR-0008 / #62). Every
  LIF scene's audit and `inspect()` output now surfaces an
  `acquisition: {date?, microscope?, microscope_serial?, imaging_method?}`
  dict, with each key optional and the block omitted entirely when nothing
  extractable is present. `microscope` prefers the LIF
  `HardwareSetting.SystemTypeName` (Leica's brand-and-model string, e.g.
  `"STELLARIS 8"`); `microscope_serial` comes from
  `SystemSerialNumber`; `date` is decoded from the `<TimeStamp>` FILETIME
  ticks into an ISO 8601 UTC string; `imaging_method` is a `list[str]` of
  OME-conventional modality tokens (`"confocal"`, `"widefield_fluorescence"`,
  `"spinning_disk_confocal"`, etc.). CZI / ND2 / OME-TIFF follow-ons in
  #63–#65 land the same block shape on those reader paths.
  New helper module `zarrmony.metadata.acquisition` (public
  `extract_acquisition`). (#62)
- Plate audits now carry `attrs.zarrmony.fields[i].well_id` — the
  concatenated `<row-letter><col-number>` well identifier (e.g. `"A03"`,
  `"B12"`) matching Aperture BigQuery's expected `well_id` column format. Also
  surfaces `attrs.zarrmony.plate.plate_id` and `inspect().plate_layout.plate_id`
  when the plate reader populates `PlateLayout.plate_id` (a new optional
  field on the dataclass). Missing `plate_id` → key omitted (no `null`);
  missing extraction path in a reader → key absent. The NGFF-spec `attrs.ome.plate`
  block does NOT gain `plate_id` (audit-only surface, since the NGFF 0.5 plate
  schema does not define such a key). No built-in reader supplies `plate_id`
  today — external plate reader adapters (Opera Phenix etc.) populate it via
  `PlateLayout(plate_id=...)`. (ADR-0008, #66)

### Changed

- `AUDIT_SCHEMA_VERSION` bumped from `7` → `8` to signal the new top-level
  `output` block. Reading old stores is unaffected (the field is additive);
  consumers pinned to schema `7` should widen their pin. (#70)

## [0.9.0] - 2026-07-23

### Added

- Data-driven omero display window: `convert()` (and the `zarrmony convert`
  CLI) now computes per-channel `(min, 99.9th percentile)` on the pyramid
  write pass and writes them into `omero.channels[i].window.start` / `.end`,
  so uint16/uint32/float fluorescence stores open with sensible auto-contrast
  in napari / OMERO on first click instead of appearing black. Fused into the
  same `da.compute` call as the pyramid writes — raw data is read once. The
  quantile is computed on the coarsest pyramid level (recorded in the audit
  as `contrast.method = "coarsest-pyramid-level"`) so the sort stays cheap
  even for 80+ GB LIF inputs. New keyword: `convert(..., contrast_percentile=99.9)`
  — pass a different float in `(0, 100)` to shift the tail, or `None` to skip
  the extra ops and keep the dtype-range placeholder from issue #50. CLI
  mirrors: `--contrast-percentile FLOAT` and `--no-contrast`. The audit
  records the resolved percentile under `config.contrast_percentile` and the
  per-channel bounds under `per_scene[i].contrast.per_channel[]`. Also
  backfills `_channels_for_current_scene` in the plate writer to pass
  `window=_dtype_window(reader.dtype)` (closing the #50 gap on plate FOVs).
  (#53)
- Per-scene objective-lens extraction for Leica LIF conversions. When a LIF
  scene's XML carries objective attributes (on
  `<ATLConfocalSettingDefinition>` and/or `<Objective>` elements), the audit
  record now surfaces them under
  `attrs.zarrmony.per_scene[i].objective` with any subset of the keys
  `nominal_magnification`, `numerical_aperture`, `immersion`, `model`,
  `working_distance_um`. Missing individual fields are omitted from the dict
  (never `null` / `0`); scenes with no objective info at all omit the
  `objective` key entirely rather than persist an empty dict. (#52)
- Per-scene `OME/METADATA.ome.xml` now emits a top-level
  `<Instrument><Objective/></Instrument>` populated from the same LIF
  extraction, plus per-image `<InstrumentRef/>` and `<ObjectiveSettings/>`
  references so downstream tooling (napari, OMERO, `ome_types.from_xml`)
  reads the objective as standards-compliant OME metadata. Non-LIF readers
  and LIF scenes with no objective info emit the pre-#52 no-instrument
  shape unchanged. Both per-scene and per-tile (`lif_mosaic="per-tile"`)
  write paths participate. (#52)
- `convert(..., channel_colors="source-file")` — new opt-in mode that trusts
  the source file's stored per-channel color when present (LIF
  `<ChannelDescription LUTName>`, OME-XML `<Channel Color>`), falling through
  to the emission-band scheme for channels without a stored color.
  `config.channel_colors` in the audit record preserves the literal string
  `"source-file"` verbatim so downstream tools can distinguish it from
  `None` and from a per-channel dict. (ADR-0007, #51)
- `ChannelColorCollisionWarning` — new warning class. Fires once per
  conversion when two or more channels resolve to the same display color
  (e.g. two far-red dyes both landing on white); later-in-order channels
  round-robin through `UNKNOWN_PALETTE` skipping already-taken colors, and
  the warning body names the reassigned channels. Suppress with the standard
  `warnings.filterwarnings("ignore", category=ChannelColorCollisionWarning)`
  or override deterministically with `channel_colors={<name>: <hex6>}`. (#51)
- `zarrmony.metadata.channel_colors.EMISSION_BANDS`, `EXCITATION_BANDS`,
  `DYE_TO_BAND`, `UNKNOWN_PALETTE`, and the batch entry point
  `assign_colors(channels, *, source_file_colors=None, overrides=None)`. (#51)
- `zarrmony.metadata.lif_channels.extract_channels` now surfaces an eighth
  per-channel field, `lut_name` (the raw
  `<ChannelDescription LUTName>` attribute), consumed by
  `channel_colors="source-file"`. Additive to the identity dict; existing
  consumers reading the seven original keys are unaffected. (#51)
- ADR `docs/adr/0007-emission-band-channel-colors.md` documenting the
  emission-band scheme, why colorblind CMY over dye-true-color, exact band
  boundaries, the `source-file` opt-in, the collision policy, and the
  interpretation of the "555 nm" ambiguity in the reporting session. (#51)

### Changed

- `AUDIT_SCHEMA_VERSION` bumped from `6` → `7` to signal the new optional
  `per_scene[i].objective` sub-dict. Reading old stores is unaffected (the
  field is additive); consumers pinned to schema `6` should widen their pin.
  (#52)

### Changed (BREAKING)

- Default per-channel display colors now come from the fluorophore's emission
  midpoint (falling back to excitation, then dye-name substring) mapped onto
  the Nature Methods "Points of View" CMY / green palette
  (`cyan / green / yellow / magenta / white`). Existing files with red-family
  channels will render with different colors than they did under v0.8: on the
  reporter's real file (DAPI 405 nm / Alexa 546 555 nm / Alexa 647 651 nm),
  colors now render as cyan / magenta / white instead of the collision-prone
  substring-match output. Users who need the old dye-brand true-color scheme
  should pin their preferred hex per channel via
  `convert(channel_colors={<name>: "<hex6>"})`. (ADR-0007, #51)
- `zarrmony.metadata.channel_colors.NAME_COLORS` and `PALETTE` are removed.
  Callers importing them will fail at import time — the failure is loud on
  purpose so a silent behavior change is impossible. Replace `NAME_COLORS`
  with `DYE_TO_BAND` (semantics: name → band color, not name → dye color)
  and `PALETTE` with `UNKNOWN_PALETTE`. (#51)
- `zarrmony.metadata.channel_colors.color_for_channel` is removed. Use
  `assign_colors(...)` (batch) or the single-channel helpers
  `color_for_emission_nm`, `color_for_excitation_nm`, `color_for_dye_name`
  which cleanly separate the three fallback stages. (#51)

### Fixed

- OMERO display window (`omero.channels[i].window`) now spans the array's
  dtype range instead of the hardcoded 0–255. Both `Channel(...)` construction
  sites (`api._lif_scene_channels` and `writers.scene._default_channels`, plus
  the shared `api._channels_for_scene` name-based path) pass `window=` derived
  from the reader dtype: integer dtypes use `np.iinfo(min, max)` and float
  dtypes use `0.0`/`1.0` (OMERO convention for normalized floats). uint16 /
  uint32 / float32 stores no longer open black in napari and OMERO because the
  display window was clamping intensities into an 8-bit band. `start` / `end`
  mirror `min` / `max` — percentile-based auto-contrast is out of scope for
  this fix. (#50)

## [0.8.0] - 2026-07-10

### Added

- `size_human` field alongside `size_bytes` in the audit record's `input`
  block (persisted at `attrs.zarrmony.input.size_human`) and in the
  `zarrmony.api.inspect()` return dict. Formatted via `format_bytes`
  (base‑1024, `ls -lh`‑style `KB`/`MB`/`GB` labels) so consumers reading
  `zarr.json` or `zarrmony inspect --json` output can read the size at a
  glance without post-processing. (#49)

### Changed

- `AUDIT_SCHEMA_VERSION` bumped from `5` → `6` to signal the new
  `input.size_human` field. Reading old stores is unaffected (the field is
  additive); consumers pinned to schema `5` should widen their pin. (#49)

## [0.7.1] - 2026-07-02

### Fixed

- `MosaicStitchingWarning` no longer fires on every mosaic scene during a
  LIF conversion when the cascade actually routes the scene through
  `stage-stitch` or `grid-stitch`. `_scene_channel_count()` used to read the
  scene's C size via `reader.xarray_dask_data`, which on the LIF proxy trips
  the auto-stitch warning as a side effect; it now prefers the
  side-effect-free `reader.channel_names` and only falls back to the xarray
  dims read when the reader doesn't expose channel names. On a 48-scene
  mosaic LIF that cascades to stage-stitch, stderr goes from 48 false-alarm
  warnings to zero. Explicit `lif_mosaic="bioio-lif"` still fires the
  warning per scene (that path legitimately runs the 1-pixel-overlap
  stitcher). (#46)

## [0.7.0] - 2026-07-02

### Changed (BREAKING)

- `lif_mosaic="auto-stitch"` (still the default) no longer means "use
  bioio-lif's 1-pixel-overlap M-scan-order stitcher"; it now dispatches a
  per-scene cascade — try `stage-stitch` when the scene has per-tile
  `PosX`/`PosY` and both physical pixel sizes, fall back to `grid-stitch`
  when the `FieldX`/`FieldY` grid is complete, and only fall through to
  bioio-lif's built-in stitcher when no `<Tile>` layout is present. Existing
  scripts passing `--lif-mosaic auto-stitch` (or the API `lif_mosaic=
  "auto-stitch"`) will now produce different — and, per the validation done
  under #41 against a real Leica LIF, better — pixels. To restore the
  pre-v0.7.0 behavior for a specific run, pass the new
  `lif_mosaic="bioio-lif"` value; that path still fires
  `MosaicStitchingWarning`. Under `layout="plate"`, the cascade skips
  `stage-stitch` (not wired to the plate writer) and lands on
  `grid-stitch` or `bioio-lif`. (#41)
- `MosaicStitchingWarning` text no longer suggests `lif_mosaic="grid-stitch"`
  or `"stage-stitch"` as alternatives — the cascade already tried them
  before falling through to this branch. The warning still names
  `lif_mosaic="per-tile"` as the remaining zarrmony alternative.

### Added

- `lif_mosaic="bioio-lif"` — explicit value that opts back into bioio-lif's
  1-pixel-overlap M-scan-order stitcher. The cascade will also select this
  automatically for scenes with no `<Tile>` metadata. (#41)
- `mosaic.cascade_selected: true` on the audit block whenever the
  `auto-stitch` cascade picked the concrete stitcher (as opposed to the
  user requesting one explicitly). Pairs with the existing
  `mosaic.stitcher` field so downstream analysis can distinguish
  "grid-stitch was picked because stage was ineligible" from "user
  explicitly asked for grid-stitch". (#41)
- Non-throwing `is_stage_layout_complete()` and `is_grid_layout_complete()`
  predicates in `zarrmony.metadata.lif_tiles`, plus a
  `select_auto_stitch_cascade()` selector that composes them. Both writer
  paths (`_convert_per_scene` and `write_plate`) use the same selector so
  the cascade decision stays consistent. (#41)

## [0.6.0] - 2026-07-02

### Added

- Fourth value on `lif_mosaic`: `"stage-stitch"` reassembles a single canvas
  per scene by placing each tile at its LIF `PosX`/`PosY` stage µm position
  (converted to pixels via the scene's physical pixel size), normalised to a
  common `(0, 0)` and snapped to integer pixels. Later-M tiles overwrite
  earlier tiles in overlap regions (deterministic later-wins; no blending in
  this slice), so the canvas honours the acquisition's declared 5–15% overlap
  instead of grid-stitch's butt joints. Strict on inputs: raises `ValueError`
  naming what's missing (per-tile `PosX`/`PosY` or scene physical pixel size
  X/Y) with `lif_mosaic="grid-stitch"` named as the graceful escape.
  Sanity-checks the placement by comparing observed vs LIF-declared overlap
  on each axis; emits `MosaicPlacementWarning` (new warning class) when they
  diverge by >20% — catches pixel-size / unit-conversion bugs without failing
  conversion. Audit carries `mosaic.stitcher="zarrmony-stage"`,
  `mosaic.tile_pixel_offsets=[{m_index, y_px, x_px}, ...]`, and
  `mosaic.observed_overlap_pct={x, y}` alongside the existing
  `intended_overlap_*_pct` fields. Powered by new pure-function helpers
  `zarrmony.metadata.lif_tiles.compute_stage_placements`,
  `reassemble_stage`, and `stage_overlap_discrepancy`. (#40)

### Fixed

- `lif_mosaic="grid-stitch"` and `"stage-stitch"` no longer crash with
  `CoordinateValidationError: conflicting sizes for dimension 'Y'` when the
  reader attaches per-tile Y/X pixel-space coords. Both reassemblers now
  drop stale Y/X coords along with the M-indexed coords, so a canvas whose
  Y/X size no longer matches a single tile stays consistent. Surfaced on a
  real Leica LIF (3×3 mosaic, 2048×2048 tiles → 6144×6144 canvas); the
  synthetic fixtures happened not to carry Y/X coords so the fault didn't
  show up in CI. (#41)

## [0.5.0] - 2026-07-01

### Added

- LIF mosaic scenes (no vendor `_Merged` sibling) can now be written as one
  OME-Zarr per tile via the new `convert(..., lif_mosaic="per-tile")` API
  kwarg and matching CLI option `--lif-mosaic {auto-stitch,per-tile,grid-stitch}`
  (default `auto-stitch`, LIF-specific — other readers ignore the flag).
  Per-tile output shape: `<output>/<sanitized_scene>/tile_X{f:02d}Y{f:02d}.ome.zarr/`,
  zero-padded grid coordinates from the LIF `<Tile FieldX>`/`FieldY>` attrs.
  Each tile sub-store is a self-describing OME-Zarr v0.5 image whose
  `OME/METADATA.ome.xml` carries `<Plane>` `PositionX/Y/Z` (meters → µm per
  OME convention) so external stitchers (ASHLAR, m2stitch, BigStitcher) can
  re-stitch from the tile stores alone. The scene-named parent directory is
  a plain directory — NOT a zarr group — so tools that recurse into
  multiscales stores see two unambiguous images per tile group, not nested
  groups. Reuses `writers.scene.write_scene` per tile (one pixel-writing
  path under maintenance). See
  [ADR-0005](docs/adr/0005-lif-mosaic-write-strategy.md) for the design
  rationale. (#36)
- Third value on `lif_mosaic`: `"grid-stitch"` reassembles a single canvas
  per scene by placing tile M=i at `(field_y[i]*tile_H, field_x[i]*tile_W)`
  from the LIF `FieldX`/`FieldY` indices (butt joints, no overlap handling).
  Fixes bioio-lif's M-scan-order placement bug — where tiles are stitched in
  the order they appear on the raw `M` dim rather than at their declared
  grid slots — while preserving the "one scene = one OME-Zarr store" invariant
  that per-tile breaks. Strict on metadata: raises `ValueError` naming the
  missing field(s) and pointing at `lif_mosaic="per-tile"` as the graceful
  escape when the tile layout is incomplete/malformed (no silent fallback).
  Composes with `layout="plate"` — a mosaic FOV in a plate well is
  reassembled and written as one image per FOV, honoring the plate spec.
  Audit carries `mosaic.stitcher="zarrmony-grid"`,
  `mosaic.overlap_assumption_px=0` (pair with `intended_overlap_*_pct` to
  diagnose missing-pixel seams), and `mosaic.placement_shape={rows, cols}`.
  Powered by new pure-function helpers `zarrmony.metadata.lif_tiles.reassemble_grid`,
  `validate_grid_layout`, and `grid_shape`. (#39)
- `writers.ome_xml.attach_stage_position_plane()` — stamps a single
  `<Plane TheC=0 TheZ=0 TheT=0 PositionX/Y/Z .../>` on an OME `Image`. Used
  by the per-tile path; also handy for downstream code that wants to add
  stage positions to a one-image OME-XML document.

### Changed

- `MosaicStitchingWarning` text now names both auto-stitch pathologies (the
  1-pixel-overlap seams AND the M-scan-order tile placement, so users have an
  in-context signal that arrangement is broken too) and both escape hatches
  (per-tile first: pixel-correct; grid-stitch second: fixes arrangement on a
  single canvas), then external stitchers as a last resort. The warning still
  fires only in auto-stitch mode; per-tile and grid-stitch bypass the
  bioio-lif stitcher entirely.
- `audit_schema_version` bumped to 5 to mark the new per-tile and grid-stitch
  audit keys. In per-tile mode each tile's audit carries `mosaic.per_tile=true`,
  `mosaic.tile_index`, `mosaic.tile_count`, and the full `mosaic.tile_stores`
  index (one entry per tile with `field_x`, `field_y`, `store_path`,
  `pos_x_m`, `pos_y_m`, `pos_z_m`). In grid-stitch mode the audit carries
  `mosaic.stitcher="zarrmony-grid"`, `mosaic.overlap_assumption_px=0`, and
  `mosaic.placement_shape={rows, cols}`. Auto-stitch audits keep their
  pre-v0.5 shape — no `per_tile`, no `tile_stores`, `stitcher="bioio-lif"` —
  so existing consumers switch on `mosaic.per_tile` (per-tile discriminator)
  and `mosaic.stitcher` (grid-stitch vs auto-stitch discriminator) to decide
  which shape they're looking at.
- Reader eligibility predicate `_MosaicAwareLifReader.is_per_tile_eligible()`
  renamed to `is_mosaic_reassembly_eligible()` — same semantics (mosaic scene
  with no `_Merged` sibling), now shared by both non-default `lif_mosaic`
  writer paths.

### Migration

- No action required for users who don't pass the new flag. The default
  remains `auto-stitch` and produces byte-for-byte identical output to
  v0.4.1 (the audit schema bump is additive — no existing field changes
  meaning).
- For users converting LIF mosaics to S3/GCS and seeing 1-pixel-overlap
  stripes at tile seams: re-run with `--lif-mosaic per-tile` (CLI) or
  `lif_mosaic="per-tile"` (library). Note that `layout="plate"` +
  `lif_mosaic="per-tile"` is incompatible (a plate FOV is one image by
  spec) and raises `LayoutMismatchError`; convert as flat to get per-tile
  stores from a mosaic scene.
- For users whose downstream tooling expects a single canvas per scene but
  who need correct tile arrangement: re-run with `--lif-mosaic grid-stitch`
  (CLI) or `lif_mosaic="grid-stitch"` (library). Grid-stitch preserves the
  one-store-per-scene invariant AND composes with `layout="plate"` (mosaic
  FOVs get reassembled and written as one image per FOV). Incomplete tile
  metadata raises `ValueError` naming the missing field(s) — fall back to
  `per-tile` if you can't fix the source LIF.
- Audit consumers (Lucida, anything reading `attrs.zarrmony`) should
  branch on `mosaic.per_tile` (present → per-tile output; look at
  `mosaic.tile_stores` for sibling sub-store URLs) and `mosaic.stitcher`
  (`"zarrmony-grid"` → grid-stitch; `"bioio-lif"` → auto-stitch;
  `mosaic.placement_shape` present only for grid-stitch).

## [0.4.1] - 2026-06-30

### Added

- LIF mosaic scenes now surface per-tile stage positions and the LIF-declared
  intended overlap in the audit's `mosaic` block (`tiles[].{field_x, field_y,
  pos_x_m, pos_y_m, pos_z_m}`, `intended_overlap_x_pct`,
  `intended_overlap_y_pct`), parsed from the scene XML's `<Tile>` and
  `<StitchingSettings>` elements by the new pure-stdlib extractor
  `zarrmony.metadata.lif_tiles.extract_tile_layout`. Pixels and on-disk layout
  are unchanged — the audit JSON is the surface that grows. (#34)
- `MosaicStitchingWarning` text now quotes the LIF-declared intended overlap
  percentage when available (e.g. *"LIF metadata declares 10% intended overlap;
  bioio-lif stitched with a 1-pixel overlap — expect ~10%-wide double-coverage
  stripes at every seam"*) and falls back to the previous generic 5–15% wording
  when extraction misses. (#34)

## [0.4.0] - 2026-06-29

### Added

- Leica `.lif` **confocal** channel identity is now preserved in the OME-Zarr
  output: each channel's fluorophore (dye), excitation/emission wavelengths, and
  detector are read from the scene metadata and written as OME-XML `<Channel>`
  elements and omero channel labels (e.g. `ALEXA 594 (590 nm)`) instead of
  display-LUT color names (`Blue`, `Gray`, …).
- `size_on_disk(path)` and `format_bytes(n)` helpers in `zarrmony._storage`.
  Handle single files, local directory trees (recursive), and remote fsspec
  URIs; render byte counts as `2.3 GB` style using powers of 1024.
- `zarrmony.inspect()` adds a `size_bytes` key to its return dict (recursive
  byte count for the input, computed via `size_on_disk()`); the CLI's
  `Size:` line and `--json` output both surface it.
- `zarrmony convert` prints `Input:` / `Output:` size lines to stderr below
  the `Wrote N stores to OUTPUT ...` message.
- Optional OME-NGFF v0.5 validation as the final step of `convert()`. Wired
  via [`ome-zarr-models`](https://github.com/ome-zarr-models/ome-zarr-models-py)
  (install with `pip install zarrmony[validate]`); enabled by default and
  toggleable via `convert(..., validate=False)` or `--no-validate`. Findings
  are recorded in `attrs.zarrmony.validation_warnings` and surfaced via the
  new `ValidationWarning` class. Output is *not* deleted on validation
  failure — the validator has known gaps (the `omero` block is not
  validated) so false-positive deletion would be worse than re-inspecting.

### Changed

- `audit.py:_file_forensics` now uses `size_on_disk()` so directory-tree
  inputs (`.zarr` stores, multi-file `.czi` series, `.lif` companion dirs)
  report the full recursive byte count instead of the top-level inode size.
- `audit_schema_version` bumped to 4 (added top-level `validation_warnings`).

### Removed

- **BREAKING (#33):** Entire user-supplied metadata surface. User metadata is
  now owned by [aperture-backend](https://github.com/calicolabs/aperture-backend),
  which associates OME-Zarr stores to a separate metadata database.
  Removed from zarrmony:
  - `zarrmony.UserMetadata` (the Pydantic model) and the
    `zarrmony.metadata.model` / `zarrmony.metadata.schema` modules.
  - `MetadataValidationError` (no longer raised).
  - `convert()`'s `metadata=`, `per_scene_metadata=`, `per_well_metadata=`,
    and `permissive=` parameters.
  - The CLI options `--metadata-file` / `-m`, `--per-scene-metadata`, and
    `--permissive`; the `zarrmony schema dump` subcommand and `schema`
    command group.
  - `write_plate(..., per_well_user_metadata=...)` parameter,
    `zarrmony.writers.plate.resolve_per_well_metadata`, and the
    `attrs.zarrmony.user_metadata` block on well groups and audit records
    (per-scene, bf2raw, and plate audits all drop `user_metadata`).

  Migration: drop these parameters and flags from all callers. The
  file-derived metadata (OME-XML, LIF channel identities, audit records'
  reader / bioio distribution / validation findings / etc.) is unchanged
  and remains zarrmony's responsibility.

## [0.3.6] - 2026-05-15

### Fixed

- `bioio-lif` reader plugin no longer emits `MosaicStitchingWarning` for
  mosaic scenes that have a vendor `_Merged` sibling present. The warning
  text claimed "no vendor-stitched sibling found" — false in that case.
  `convert()` was unaffected (it short-circuits via `skip_reason` before
  touching `xarray_dask_data`), but `inspect()` walks every scene and
  triggered the misleading warning.

## [0.3.5] - 2026-05-15

### Added

- `MosaicStitchingWarning` (in `zarrmony.errors`) — emitted by the
  `bioio-lif` reader plugin whenever it auto-stitches a mosaic. The
  bioio-lif stitcher hardcodes a 1-pixel inter-tile overlap and ignores
  the LIF metadata's actual stage XY positions; for acquisitions with
  normal 5–15% overlap the output has double-coverage stripes at every
  tile seam. The warning text names the scene, the tile count, and
  recommends an external stitcher (ASHLAR, m2stitch, BigStitcher) when
  no `_Merged` sibling is present and correctness at tile boundaries
  matters.
- `MosaicMergedSiblingWarning` (in `zarrmony.errors`) — emitted by
  `convert()` (per-scene mode) when a LIF mosaic scene is skipped because
  a sibling scene named `<scene>_Merged` is present. The merged sibling
  is written instead, so the imprecise auto-stitch is never invoked.
- `skip_reason` hook on the reader interface (optional). When the active
  scene's `reader.skip_reason` is non-None, `convert()` (per-scene mode)
  emits a warning and skips that scene without writing a store.

### Changed

- `bioio-lif` reader plugin now prefers a vendor-stitched sibling scene
  (Leica's `<scene>_Merged` convention) over its own auto-stitch when
  one is present in the LIF. The mosaic scene is omitted from output;
  only the merged sibling is written. When no `_Merged` sibling exists,
  the plugin falls back to `mosaic_xarray_dask_data` as before, with a
  `MosaicStitchingWarning`.
- Per-scene audit gains an optional `mosaic` block recording tile count
  and tile shape when stitching was applied.
- The `mosaic` audit block now also records `stitcher: "bioio-lif"` and
  `overlap_assumption_px: 1` so downstream consumers can tell which
  stitcher produced the pixels and what overlap assumption it baked in.

### Fixed

- LIF files with mosaic tiling no longer raise `UnsupportedAxesError`. The
  `M` dimension is collapsed inside the reader plugin before the writer
  sees the xarray.

## [0.3.4] - 2026-05-13

### Changed

- Internal/dev tooling alignment with the Calico
  `github-template-python-library` template, in preparation for an internal
  Calico fork (`calicolabs-zarrmony`). No user-facing API changes.
  - Drop minimum Python version from 3.13 to 3.11; CI matrix now tests
    both 3.11 and 3.13 across ubuntu and macos.
  - Switch build backend from `hatchling` to `setuptools` +
    `setuptools_scm`. Version is now derived from git tags via
    `dynamic = ["version"]`; `__version__` resolved at runtime via
    `importlib.metadata.version()`.
  - Switch Python formatter from `ruff format` to `black` (default
    settings, line length 88). `ruff` continues to handle linting.
  - Adopt `prettier` (pinned via `package.json`) for YAML / Markdown /
    JSON formatting. CI gains a separate `format` job that runs both
    `black --check` and `npx prettier --check`.
  - Rename `LICENSE` → `LICENSE.md` and add `©` to the copyright line per
    Calico external-release policy. `license-files` in pyproject updated.
  - Add `.github/CODEOWNERS` with `* @ferrinm`.
  - Add PyPI version, Python version, license, and CI status badges to
    the README.
  - Add `INTERNAL_FORK.md` documenting the public/private fork
    relationship, expected diff at fork time, deliberate template
    deviations, and the manual sync procedure.

## [0.3.3] - 2026-05-11

### Changed

- Documentation completion for the v0.3 plate output design (issue #12):
  - `docs/writing-a-reader-plugin.md` gains §9 "Writing a plate-shaped
    reader" covering `layout_hint="plate"`, the `PlateLayout` /
    `PlateField` / `Acquisition` dataclasses, the structural validation
    rules zarrmony enforces (rows/columns membership, `scene_index`
    uniqueness, no duplicate well paths, `acquisition_id`, single-
    acquisition v1 limit), `field_name` semantics (vendor-native,
    preserved in audit + multiscales `name`, NOT the on-disk path), the
    v1 single-acquisition / single-plate limits and how to fall back to
    flat output, and the `parse_well_key` / `resolve_per_well_metadata`
    / `summarize_plate_layout` helpers. The previous "Reserved:
    `layout_hint`" subsection is rewritten — the hint actively drives
    `layout="auto"` dispatch as of v0.3.
  - `README.md` "Usage" examples gain `--layout auto` (the v0.3 default),
    `--layout plate` (with a `per_well_metadata` example), and a
    description of the plate-mode return shape, alongside the existing
    `per-scene` and `bf2raw` examples.

## [0.3.2] - 2026-05-11

### Added

- `inspect()` returns an additive top-level `plate_layout` key when the reader
  exposes one (`layout_hint == "plate"` and `plate_layout is not None`). The
  value mirrors the audit's `plate` block (`name`, `rows`, `columns`,
  `wells`, `acquisitions`, `field_count`). Flat-reader callers see the
  existing return shape unchanged (no new keys).
- `zarrmony inspect` CLI prints a one-line plate summary header before the
  per-scene table when a plate layout is present, e.g.
  `Plate: "synthetic-2x2" — 3/4 wells imaged, 1 field per well, 1 acquisition`.
- `zarrmony.writers.plate.summarize_plate_layout(plate_layout)` helper exposes
  the plate-attr-shaped summary builder (no I/O) for plugin authors and
  external tooling.

## [0.3.1] - 2026-05-11

### Added

- `convert(per_well_metadata={"B04": {...}, ...})` for plate mode: a dict
  keyed by compact well coordinate (leading-alpha row + trailing-numeric
  column, e.g. `"B04"`, `"AA01"` for 1536-well plates). Keys are validated
  against the plate's canonical `rows`/`columns` (zero-padding from the
  plate is the source of truth — `"B1"` against a `"01"`-padded plate is
  rejected). Each override persists to the well group's
  `attrs.zarrmony.user_metadata` on disk and to the audit's
  `plate.wells[i].user_metadata` block. The on-disk `attrs.ome.plate`
  block stays spec-clean.
- `zarrmony.writers.plate.parse_well_key` and `resolve_per_well_metadata`
  helpers exposed for plugin authors and downstream tooling that needs to
  validate well-keyed dicts against a `PlateLayout`.

### Changed

- `per_scene_metadata` rejection in plate mode now points users at
  `per_well_metadata` (was: a generic "follow-up slice" message).

## [0.3.0] - 2026-05-11

### Added

- `layout="plate"` writer (and `--layout plate`): `convert()` produces a
  spec-conformant OME-NGFF 0.5 HCS plate store. Each FOV lands at
  `<plate>/<row>/<column>/<seq_int>/` (sequential int paths assigned per well
  in field order); plate-level metadata is written to `attrs.ome.plate` on
  the root group; row groups are structural; well groups carry
  `attrs.ome.well.images`; a single combined `OME/METADATA.ome.xml` lives at
  the plate root (no per-FOV sidecars). The `bioformats2raw.layout` marker
  is intentionally NOT emitted on plate stores. See ADR-0004.
- `layout="auto"` (the new default): `convert()` and the CLI dispatch on
  `reader.layout_hint` — `"flat"` → `per-scene`, `"plate"` → `plate`. Explicit
  overrides still work; see Migration below.
- `ReaderProtocol.plate_layout: PlateLayout | None`. New
  `zarrmony.readers.plate` module exposes `PlateLayout`, `PlateField`, and
  `Acquisition` dataclasses for plate-shaped readers (and re-exported from
  `zarrmony.readers.plugin`). The `writing-a-reader-plugin.md` guide will
  grow a "Writing a plate-shaped reader" section in a follow-up slice (#12).
- `PlateLayoutError` (`writers.plate` validation; raised before any pixel
  writes when a layout is internally inconsistent), `LayoutMismatchError`
  (`convert()` raises when explicit `layout='plate'` is passed against a
  non-plate-shaped reader), and `LayoutDowngradeWarning` (`convert()`
  emits when explicit `per-scene`/`bf2raw` is passed against a plate-shaped
  reader; `writers.plate` emits when `reader.scenes` contains scenes
  unreferenced by any `PlateField`).
- ADR-0004 (`docs/adr/0004-plate-output-design.md`) documenting the plate
  output design, including the dispatch matrix, locked decisions, and
  rejected alternatives.

### Changed

- **BREAKING:** `convert()`'s default `layout` flips from `"per-scene"` to
  `"auto"`. The CLI default flips from `--layout per-scene` to
  `--layout auto`. For a flat reader the resolved behavior is unchanged
  (auto → per-scene); plate-shaped readers now produce a plate store by
  default instead of being rejected.
- **BREAKING:** `audit_schema_version` bumps `2 → 3`. Plate-layout audits
  use a `fields: [...]` list (each entry extends the existing per-scene
  record with `row`, `column`, `field_path`, `field_name`, `acquisition_id`)
  and a top-level `plate` block (`name`, `rows`, `columns`, `wells`,
  `acquisitions`, `field_count`). Flat-layout audits keep using
  `per_scene: [...]`. Audit consumers switch on the top-level `layout`
  discriminator to pick the right key.
- The audit's `config.layout` records the *resolved* layout (what was
  actually written: `per-scene` / `bf2raw` / `plate`), never `auto`.

### Migration

- **Default layout flip.** If you relied on the old default for a
  plate-shaped reader (rejected outright in v0.2), the v0.3 default now
  writes a plate store. To force the old per-scene behavior, pass
  `layout="per-scene"` explicitly; you'll get a `LayoutDowngradeWarning`
  noting that plate metadata is being dropped.
- **Audit schema 3.** Detect with the top-level `audit_schema_version`
  field. Schema 3 plate audits replace `per_scene` with `fields` and add a
  top-level `plate` block. Switch on `audit["layout"]` to pick the right
  key:

  ```python
  if audit["layout"] == "plate":
      records = audit["fields"]   # row, column, field_path, ...
      plate = audit["plate"]      # name, rows, columns, wells, ...
  else:
      records = audit["per_scene"]
  ```
- **Forcing `layout="plate"` against a flat reader** now raises
  `LayoutMismatchError` (previously this combination was a generic
  `ZarrmonyError`). Catch the new type if you depend on the failure mode.

## [0.2.1] - 2026-05-08

### Added

- Plugin-author guide at `docs/writing-a-reader-plugin.md` covering the
  Reader Protocol, matcher score conventions, entry-point registration,
  the `zarrmony-<vendor>` distribution naming convention, and a
  registry-isolation pattern for tests. Linked from the README under a
  new "Extending zarrmony" section.
- Direct contract tests for the reader plugin registry
  (`tests/test_plugin_registry.py`) covering duplicate registration,
  matcher-exception resilience, score-tie ordering, the entry-point loader
  (success, failed load, wrong type, name collision), and
  `NoMatchingPluginError` message contents. Pulls `readers/plugin.py`
  coverage to 100% and runs without any bioio fixture files.

## [0.2.0] - 2026-05-08

### Added

- `audit_schema_version` field at the top level of every audit record
  (currently `2`).
- Built-in bioio readers are now first-class `ReaderPlugin` instances
  registered via the new `zarrmony.readers.plugin` registry. `convert()` and
  `inspect()` dispatch through `plugin.get_reader()`, returning
  `(reader, plugin, match_score)`.

### Changed

- **BREAKING:** the audit record's `reader_plugin` is now a nested dict with
  `name`, `version`, `source`, `distribution`, and `match_score`. The flat
  `reader_plugin: str` and `reader_plugin_version: str | None` top-level
  fields are removed.
- **BREAKING:** `inspect()` returns a nested `reader_plugin` dict (same shape
  as the audit record) instead of a flat `plugin: str` field.
- `audit.build_audit_record()` signature: `reader_plugin` now takes a
  `ReaderPlugin | None`; new keyword args `match_score` and `distribution`
  (the latter overrides the plugin's static `distribution` so the catch-all
  default plugin can surface the actual bioio sub-package, e.g.
  `bioio-ome-tiff`).
- The CZI, LIF, and ND2 built-in readers are now `ReaderPlugin` instances
  (`czi_plugin`, `lif_plugin`, `nd2_plugin`) registered through the new
  registry alongside `default_plugin`. They live next to `default.py` at
  `zarrmony/readers/{czi,lif,nd2}.py` (the `readers/overrides/` subpackage
  is gone).

### Removed

- **BREAKING:** `zarrmony.readers._OVERRIDES`, `register_override()`,
  `ReaderFactory`, and the legacy 2-tuple `zarrmony.readers.get_reader()`
  are gone. There is now exactly one way for a reader to exist in zarrmony:
  register a `ReaderPlugin` via `zarrmony.readers.plugin.register_plugin()`
  (or expose one through the `zarrmony.readers` entry-point group). Callers
  should import `get_reader` from `zarrmony.readers.plugin`.
- **BREAKING:** the legacy 2-tuple `open_default_reader`,
  `open_czi_reader`, `open_lif_reader`, `open_nd2_reader` factory functions
  are removed. Use the corresponding `*_plugin` instances (or call them
  directly as `plugin.open(path)`) instead.
- **BREAKING:** the `zarrmony.readers.overrides` subpackage is removed;
  imports of `zarrmony.readers.overrides.{czi,lif,nd2}` should move to
  `zarrmony.readers.{czi,lif,nd2}`.

### Migration

Reading pre-0.2 audit records: the old top-level `reader_plugin: str`
is now `reader_plugin.name`, and the old `reader_plugin_version: str | None`
is now `reader_plugin.version`. Detect the schema with the new top-level
`audit_schema_version` field (absent or `1` = legacy flat shape; `2` =
nested dict).

## [0.1.4] - 2026-05-07

### Added

- Initial package scaffolding (`src/zarrmony/` layout, ruff, pre-commit, CI on Ubuntu + macOS).
- Apache-2.0 license.
- PyPI release workflow (trusted publishing via OIDC).
- `--layout {per-scene,bf2raw}` option on `zarrmony convert` (and a matching
  `layout=` argument on `convert()`).
- `zarrmony.naming.sanitize_scene_name` and `resolve_scene_dirnames` for
  filesystem-safe per-scene store directory names with collision fallback.

### Changed

- **BREAKING:** `convert()`'s default output layout flipped from
  `bioformats2raw.layout` to **per-scene**. By default, `output` is now treated
  as a directory and one self-describing `<sanitized_scene_name>.ome.zarr`
  store is written per scene. The bundled `bioformats2raw.layout` shape
  remains available via `layout="bf2raw"` (`--layout bf2raw`).
- **BREAKING:** `convert()`'s return shape in the new default layout is
  `{"input": ..., "output": ..., "layout": "per-scene", "stores": [<per-store
  audit>, ...]}`. `layout="bf2raw"` continues to return the single bundle's
  audit dict.
- In per-scene mode, `force` / `--force` is checked per output store: a
  pre-existing sibling store under the output directory is left untouched
  unless it collides with a scene being written.
