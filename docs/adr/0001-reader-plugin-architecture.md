# Reader plugin architecture

Zarrmony's reader system is a unified plugin registry: every reader (built-in or external) is a `ReaderPlugin(name, match, open)` registered via Python entry points (`zarrmony.readers` group) or runtime `register_plugin()`. Dispatch walks all registered plugins, calls each `match(path) -> int | None` (cheap, side-effect-free), picks the highest non-`None` score, and calls `open(path) -> Reader`. The returned reader satisfies a small `typing.Protocol` covering the seven attributes `convert()` actually consumes (`scenes`, `set_scene`, `xarray_dask_data`, `physical_pixel_sizes`, `channel_names`, `ome_metadata`, `metadata`) plus a forward-compat `layout_hint`.

## Considered Options

- **Inherit from `bioio_base.Reader`.** Rejected: zarrmony only consumes ~7 attributes; coupling to bioio's full surface drags in obligations zarrmony doesn't use, and forces non-bioio readers (Phenix) into an inheritance hierarchy that doesn't fit. The `Protocol` makes the actual contract explicit and structural; bioio readers conform automatically via duck typing.
- **Extension-only dispatch (today's `_OVERRIDES` dict).** Rejected: Phenix input is a *directory* identified by `Index.idx.xml` inside it, not a file extension. Predicate-based matching is required for that and any future directory-shaped formats.
- **First-match-wins predicate dispatch.** Rejected: registration order becomes load-bearing, which is fragile under entry-point discovery. Priority scores let a specific plugin (Phenix-TIFF-directory) outrank a generic one (any TIFF) without depending on import order.
- **Combined `match` + `open` in one call.** Rejected: instantiating every reader to ask "can you handle this?" is expensive (header parsing, file opening). Separated cheap matchers keep `inspect()` fast and let dispatch enumerate candidates without side effects.

## Consequences

- The existing `_OVERRIDES` dict and `register_override()` API are removed; the bioio overrides (czi/lif/nd2/default) become first-class `ReaderPlugin` instances registered at zarrmony import time. There is exactly one way for a reader to exist in zarrmony.
- Plugin-author trust model: matchers that raise are logged and treated as no-match (one buggy third-party plugin doesn't break unrelated conversions); `open()` that raises is fatal (silently falling through would mask bugs and produce mysteriously-different outputs depending on which plugins are installed).
- Reader resource lifetime: an optional `close()` method is called in a `finally` block. Plugins holding non-GC resources (file handles, network connections) implement it; others ignore.
- Built-ins are registered before entry points are walked, so equal-priority ties resolve to built-ins — installing a third-party plugin can't accidentally hijack CZI conversion unless its author consciously bids a higher score.