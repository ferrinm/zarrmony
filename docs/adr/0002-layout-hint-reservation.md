# `layout_hint` reservation for plate output

The Reader Protocol declares `layout_hint: Literal["flat", "plate"] = "flat"` even though zarrmony's writer currently emits flat output only. Plate-shaped readers (Phenix and similar HCS instruments) set `layout_hint = "plate"` from day one; the writer ignores it today. When OME-NGFF Plate output lands in a future zarrmony milestone, the writer dispatch reads this field — no Protocol break, no plugin changes.

## Considered Options

- **Build HCS Plate output now, in the same milestone as the plugin system.** Rejected: plate output is a _writer_ concern, not a _reader_ concern. Bundling them conflates two architectural changes, and the writer side (HCS-spec compliance, plate metadata, well aggregation, multi-acquisition support) is at least as much work as the plugin system itself. Ship the plugin system first; ship HCS separately.
- **Flatten Phenix to scenes and lose plate semantics entirely.** Rejected: Phenix users would see N independent images instead of "row B col 4 field 2", and downstream tools (napari-ome-zarr, MoBIE) couldn't light up plate-aware UI. Worse, migrating later means re-converting every Phenix dataset produced in the interim.
- **Add `layout_hint` only when the writer needs it.** Rejected: it's five lines of Protocol surface now vs. a Protocol break (and a forced version bump for every published plugin) later. The cost asymmetry favors reservation.

## Consequences

- Future readers of the codebase will see a Protocol field with no consumer in `convert()` and no writer dispatch on it. This ADR exists to explain that it is intentional forward-compat, not dead code.
- The Phenix adapter (`zarrmony-phenix`) sets `layout_hint = "plate"` from its first release. In v0.1 of zarrmony, this field is informational only; once HCS writer support lands, the same Phenix release starts producing plate-shaped output without modification.
- Until HCS is implemented, plate-shaped readers fall back to flat-scenes output. Phenix scene names should encode plate coordinates (e.g., `"B04-f02"`) so the information is recoverable from a flat conversion if a user runs one before the HCS writer ships.
