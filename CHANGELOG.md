# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
