# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
