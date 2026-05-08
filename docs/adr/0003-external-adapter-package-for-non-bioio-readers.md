# External adapter package for non-bioio readers

Non-bioio reader plugins (Phenix and future custom-instrument readers) ship as separate distributable packages, not as in-tree extras of zarrmony. Specifically, Opera Phenix support is a new `zarrmony-phenix` package that depends on both `pyphenix` and `zarrmony` and contains a thin adapter wrapping `pyphenix.OperaPhenixReader` to satisfy zarrmony's Reader Protocol. PyPhenix itself is not modified.

## Considered Options

- **Fold all plugins in-tree as optional extras (`zarrmony[phenix]`).** Rejected: Phenix carries real domain logic (FFC math, plate-coordinate parsing, mosaic stitching) and dependencies (PIL, custom XML parsing) that don't belong in zarrmony's import graph. The bundled bioio overrides are 10 lines each and exist purely to pin a backend choice — they're qualitatively different and stay in-tree.
- **Add a `zarrmony.readers` entry point directly to PyPhenix.** Rejected: PyPhenix is a napari plugin. Its purpose is interactive visualization; coupling it to zarrmony's Reader Protocol pollutes that purpose and forces PyPhenix releases to track zarrmony Protocol changes. A dedicated adapter package is the integration seam — both upstreams stay focused.

## Consequences

- One additional repo and release process to maintain (`zarrmony-phenix`).
- Users wanting Phenix→OME-Zarr conversion install `zarrmony-phenix` (which transitively pulls PyPhenix) without dragging in Qt/napari.
- This pattern is the template for any future custom-instrument support: the instrument-specific reader code lives in its own package, and a thin `zarrmony-<vendor>` adapter wraps it for zarrmony.
- The audit record's `distribution` field identifies which adapter package produced a given dataset, so the boundary is recoverable from any converted output.