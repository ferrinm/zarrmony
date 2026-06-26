# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
