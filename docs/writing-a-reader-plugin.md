# Writing a zarrmony reader plugin

Zarrmony reads bioimages through a plugin registry. This guide is for authors
who want to teach zarrmony about a new format — typically by wrapping an
existing reader library (bioio backend, vendor SDK, custom parser) so that
`zarrmony convert <your-format>` Just Works.

By the end you will know how to write a `ReaderPlugin`, register it via Python
entry points, name and version your distribution, and test it in isolation.

## 1. What is a plugin?

A zarrmony reader plugin is a single value:

```python
from zarrmony.readers.plugin import ReaderPlugin

my_plugin = ReaderPlugin(
    name="zarrmony-myformat",     # identifies the plugin in audit records
    match=match_myformat,         # cheap predicate: does this path look like mine?
    open=open_myformat,           # factory that returns a ReaderProtocol-compatible object
    distribution="zarrmony-myformat",  # PyPI distribution name (optional)
    source="entry_point",         # how this plugin reached the registry
)
```

When zarrmony is asked to convert a path, it walks every registered plugin,
calls each `match(path)`, and `open()`s the highest-scoring one. The returned
reader can be anything that satisfies the small structural Reader Protocol
described below — there is no base class to inherit from.

Plugins reach the registry one of three ways:

- **Built-in** — registered at `import zarrmony` (`source="builtin"`).
- **Entry point** — declared in your `pyproject.toml` under
  `zarrmony.readers` (`source="entry_point"`). The recommended path for any
  third-party plugin.
- **Runtime** — registered programmatically via `register_plugin()`
  (`source="runtime"`). Useful in tests.

See [ADR-0001](./adr/0001-reader-plugin-architecture.md) for the design
rationale.

## 2. The Reader Protocol

Your `open()` factory must return an object satisfying
`zarrmony.readers.plugin.ReaderProtocol`. It is a `typing.Protocol` — duck
typing, no inheritance — covering the surface that `convert()` and
`inspect()` actually consume.

### Required attributes

| Attribute                       | Type                                          | What zarrmony does with it                                                                                                                   |
| ------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `scenes`                        | `list[str]`                                   | Iterates to enumerate scenes; names are sanitized into per-scene store directory names.                                                      |
| `set_scene(index: int) -> None` | method                                        | Switches the active scene before reading data. Called once per scene.                                                                        |
| `xarray_dask_data`              | `xarray.DataArray` (dask-backed)              | The image array for the current scene. Dimensions should follow OME-Zarr 0.5 axis order; channels named via the `C` coordinate when present. |
| `physical_pixel_sizes`          | object with `.X`, `.Y`, `.Z` floats (microns) | Scale transform written into the multiscales metadata. Mirrors `bioio_base.PhysicalPixelSizes`; `None` on any axis is fine.                  |

### Soft-optional attributes

These are accessed via `getattr` or `try/except`. Omit them and zarrmony falls
back gracefully — the conversion still produces a valid OME-Zarr; it just
loses fidelity for the missing piece.

| Attribute                         | Fallback when missing or raising                                                                                                                                                                                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `channel_names: list[str]`        | Channels labelled `C:0`, `C:1`, …; the audit records `channel_names: null`.                                                                                                                                                                                      |
| `ome_metadata`                    | A minimal OME `Image` element synthesised from the scene shape; an audit warning records the failure. Return `OME` (from `ome-types`), an `xml.etree.ElementTree.Element`, or an XML string.                                                                     |
| `metadata`                        | No `OME/source/raw.<format>.xml` is written. Anything that serialises with `str()` works; native `OME`, `Element`, or `str` skip the round-trip.                                                                                                                 |
| `acquisition_audit: dict \| None` | No reader-side extras contribute to `per_scene[i].acquisition`; the block is composed from the LIF scene-XML extractor (LIF only) and the OME projection alone. See [§ acquisition_audit](#acquisition_audit) below for the shape and gap-fill semantics.        |
| `close()`                         | Skipped. Implement it if your reader holds non-GC resources (file handles, network sessions). zarrmony's intent is to call this in a `finally` block once it lands ([ADR-0001](./adr/0001-reader-plugin-architecture.md)); writing it now is forward-compatible. |

### `acquisition_audit`

Use this hook when your reader knows an acquisition-block field by
construction but no source-file surface (LIF scene XML, OME per-channel
`AcquisitionMode`) carries it. Typical case: a stitched TIFF reader for a
single-modality microscope (SmartSPIM = light-sheet, Blaze = light-sheet)
where the exported file is a plain OME-TIFF with no modality tag but the
reader knows the modality is fixed.

Shape (any subset of the ADR-0008 keys — every key optional):

```python
class MyReader:
    layout_hint = "flat"

    @property
    def acquisition_audit(self) -> dict | None:
        return {"imaging_method": ["light_sheet"]}
```

A plain instance attribute (`self.acquisition_audit = {...}`) works too;
either shape is accepted. Returning `None` or a non-dict yields no extras.

**Precedence — `setdefault` layering.** Zarrmony composes
`per_scene[i].acquisition` in four tiers, first source wins per key:

1. LIF scene-XML extractor (LIF scenes only).
2. Vendor-specific extractors (CZI raw XML → microscope model; ND2 SDK
   `text_info().capturing` → microscope model). Fills gaps bioio's OME
   projection leaves — bioio-czi emits only `"Zeiss"` and bioio-nd2 omits
   `<Microscope>` entirely.
3. OME projection from `reader.ome_metadata` — populates `date`,
   `microscope`, `microscope_serial`, and `imaging_method` (from per-channel
   `<Channel AcquisitionMode>`).
4. `reader.acquisition_audit` — fills only keys none of the above
   populated.

The hook can never override a source-file-derived extraction. If bioio's
OME projection reports `Channel.AcquisitionMode = "SpinningDiskConfocal"`,
your `acquisition_audit` returning `["light_sheet"]` won't take effect —
OME is the more trustworthy source when it fires. Reserve the hook for
cases where source-file extraction produces nothing.

**Fail-safe.** A hook that raises yields no extras; conversion continues
normally with whatever the earlier tiers produced.

### `layout_hint` and `plate_layout`

`layout_hint: Literal["flat", "plate"]` drives the default
`layout="auto"` dispatch in `convert()` and the CLI: `"flat"` → per-scene
writer, `"plate"` → HCS plate writer. Flat-image readers can leave the
attribute unset (the Protocol default is `"flat"`). Plate-shaped readers
(Phenix and similar) set `layout_hint = "plate"` and additionally populate
`plate_layout: PlateLayout | None` with the structured plate shape — see
§9 below for the full plate-reader contract.

## 3. Writing the matcher

`match(path: Path) -> int | None` must be **cheap** and **side-effect-free**.
It runs against every registered plugin on every conversion — opening files,
hitting the network, or parsing megabytes of header is forbidden. Return
`None` for "not mine"; return an integer score for "mine, with this
priority".

### Score conventions

- **Built-ins use 100.** All extension-keyed bioio overrides
  (`bioio-czi`, `bioio-lif`, `bioio-nd2`) return `100` on a match.
- **The catch-all default plugin returns 0.** Anything that returns a
  positive score outranks the bioio fallback.
- **Third-party plugins**: return `≥ 100` if you want to _win_ a tie against
  a built-in (e.g. you ship a faster CZI reader); return `< 100` if you want
  to _defer_ to built-ins for files you could also handle. Built-ins register
  before entry points are walked, so equal scores resolve to the built-in via
  stable sort — installing your plugin can't accidentally hijack CZI.

### Examples

**Extension match** (bioio-czi, bioio-lif, bioio-nd2 all use this shape):

```python
def match_myformat(path: Path) -> int | None:
    return 100 if path.suffix.lower() == ".myf" else None
```

**Directory-marker match** (e.g. Phenix: a directory containing
`Index.idx.xml`):

```python
def match_phenix(path: Path) -> int | None:
    if path.is_dir() and (path / "Index.idx.xml").is_file():
        return 100
    return None
```

**Magic-byte sniff** (when extension isn't reliable). Read a fixed-size
prefix only — never the whole file:

```python
def match_myformat(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            header = f.read(8)
    except OSError:
        return None
    return 100 if header.startswith(b"\x89MYF\r\n\x1a\n") else None
```

If your matcher raises, zarrmony logs a warning and treats it as no-match —
one buggy plugin can't break unrelated conversions ([ADR-0001](./adr/0001-reader-plugin-architecture.md),
"trust model"). Don't rely on this; raising for an unrelated path is still a
bug. Use it as a backstop, not a control-flow tool.

## 4. Writing the `open()` factory

`open(path: Path) -> ReaderProtocol` is allowed to be expensive. It is only
called on the winning plugin, exactly once per conversion. Open file handles,
parse headers, instantiate vendor SDK objects — whatever you need.

```python
def open_myformat(path: Path):
    return MyReader(str(path))
```

If `open()` raises, the conversion fails fast. This is intentional: silently
falling through to a different plugin would mask bugs and produce mysteriously
different outputs depending on which plugins are installed.

### Accepting reader-specific keyword arguments

`open()` is invoked as `plugin.open(path, **reader_kwargs)` — plugins that
accept only a path see `**{}` and behave identically to the single-argument
form (backwards compatible with every existing plugin). Declare additional
keyword parameters when your reader has options a caller can't discover from
the file alone. The motivating case is a sidecar-elsewhere override:

```python
def open_smartspim(path: Path, *, metadata_path: Path | str | None = None):
    return SmartSpimReader(path, metadata_path=metadata_path)
```

Callers reach the kwarg through `convert()` / `inspect()`'s `reader_kwargs`
dict, or the CLI's repeatable `--reader-kwarg KEY=VALUE`:

```python
from zarrmony import inspect
inspect(
    "/mnt/readonly/<dataset>",
    reader_kwargs={"metadata_path": "/writable/metadata_<dataset>.json"},
)
```

```bash
zarrmony inspect /mnt/readonly/<dataset> \
  --reader-kwarg metadata_path=/writable/metadata_<dataset>.json
```

Values from the CLI stay as strings; the reader coerces internally
(`SmartSpimReader` casts `metadata_path` to a `Path`). Unknown kwargs
surface as the reader constructor's native `TypeError` — zarrmony does not
intercept or validate the shape, so a typo like
`--reader-kwarg meta_path=…` fails loudly at open time. There is
deliberately no plugin-side kwarg schema in v1; that is deferred until a
second plugin needs the same discovery mechanism.

If your reader doesn't directly satisfy the Reader Protocol — for example,
you're wrapping a vendor SDK whose attribute names don't line up — write a
thin adapter:

```python
class MyAdapter:
    layout_hint = "flat"

    def __init__(self, path: Path):
        self._sdk = VendorSDK.open(str(path))
        self.scenes = [s.name for s in self._sdk.list_series()]
        self._active = 0

    def set_scene(self, index: int) -> None:
        self._active = index

    @property
    def xarray_dask_data(self):
        return self._sdk.read_series_lazy(self._active)  # an xarray.DataArray

    @property
    def physical_pixel_sizes(self):
        return self._sdk.list_series()[self._active].pixel_sizes

    def close(self) -> None:
        self._sdk.close()
```

For a complete worked example see §6.

## 5. Registering via entry points

Declare your plugin under the `zarrmony.readers` entry-point group in your
`pyproject.toml`:

```toml
[project]
name = "zarrmony-myformat"
dependencies = ["zarrmony>=0.2", "myformat-sdk>=1.0"]

[project.entry-points."zarrmony.readers"]
myformat = "zarrmony_myformat.plugin:my_plugin"
```

The right-hand side is `module:attribute`, where `attribute` is the
`ReaderPlugin` instance. The entry-point key (`myformat` above) is for your
own organisation; zarrmony uses the plugin's `name` field everywhere user-
visible.

Once installed, zarrmony discovers your plugin automatically on first
dispatch — no further wiring required. Verify with:

```python
from zarrmony.readers.plugin import list_plugins
print([p.name for p in list_plugins()])
# -> ['bioio', 'bioio-czi', 'bioio-lif', 'bioio-nd2', 'zarrmony-myformat']
```

## 6. Worked example: the built-in CZI plugin

The CZI plugin (`src/zarrmony/readers/czi.py`) is the smallest end-to-end
example in the tree. It pins the `bioio_czi.Reader` backend rather than
letting bioio's discovery pick one, so audit records always name the same
backend and CZI input fails fast at import time if the plugin isn't
installed.

```python
"""CZI reader plugin."""

from pathlib import Path
from typing import Any

from bioio_czi import Reader

from zarrmony.readers.plugin import ReaderPlugin


def _match_czi(path: Path) -> int | None:
    return 100 if path.suffix.lower() == ".czi" else None


def _open_czi(path: Path) -> Any:
    return Reader(str(path))


czi_plugin = ReaderPlugin(
    name="bioio-czi",
    match=_match_czi,
    open=_open_czi,
    distribution="bioio-czi",
    source="builtin",
)
```

Walking it line by line:

- **`_match_czi`** — extension check, returns `100` or `None`. No I/O.
- **`_open_czi`** — single-line factory. `bioio_czi.Reader` already
  satisfies the Reader Protocol (it exposes `scenes`, `set_scene`,
  `xarray_dask_data`, `physical_pixel_sizes`, `channel_names`,
  `ome_metadata`, and `metadata`), so no adapter is needed.
- **`distribution="bioio-czi"`** — the PyPI package name that this plugin
  is shipped from. Surfaces in the audit record so a converted dataset can
  always be traced back to the package that produced it.
- **`source="builtin"`** — set explicitly so the audit record can
  distinguish bundled plugins from external ones. Entry-point plugins
  should set `source="entry_point"`; runtime-registered plugins (typically
  in tests) leave it as the default `"runtime"`.

The CZI plugin is registered in `src/zarrmony/readers/__init__.py`; your
plugin reaches the registry via the entry point declared in §5 instead.

## 7. Distribution naming convention

For PyPI distributions: name the package **`zarrmony-<vendor>`** (e.g.
`zarrmony-phenix`, `zarrmony-mycompany`). This makes plugins discoverable by
a simple `pip search` / GitHub search and lines up with the in-tree
adapter pattern documented in [ADR-0003](./adr/0003-external-adapter-package-for-non-bioio-readers.md).

The plugin's `name` field (the audit-record identifier) is your call —
convention is to match the distribution name (`name="zarrmony-phenix"`),
but built-ins follow `bioio-<format>` for historical compatibility. Pick
something stable; users will see it in audit records and downstream tools
will key off it.

## 8. Testing your plugin

Test the matcher and `open()` in isolation, then run an end-to-end
conversion against a small fixture file.

### Unit-testing the matcher

The matcher is a pure function — test it directly:

```python
from pathlib import Path
from zarrmony_myformat.plugin import _match_myformat

def test_match_extension():
    assert _match_myformat(Path("/data/sample.myf")) == 100
    assert _match_myformat(Path("/data/sample.czi")) is None
    # case-insensitivity
    assert _match_myformat(Path("/data/SAMPLE.MYF")) == 100
```

### Unit-testing `open()`

`open()` typically just calls a constructor; the interesting tests live in
the upstream SDK. A smoke test that the returned object satisfies the
Protocol surface is enough:

```python
def test_open_returns_reader_protocol(tmp_path):
    fixture = tmp_path / "sample.myf"
    write_minimal_fixture(fixture)
    reader = _open_myformat(fixture)
    assert hasattr(reader, "scenes")
    assert hasattr(reader, "set_scene")
    assert hasattr(reader, "xarray_dask_data")
    assert hasattr(reader, "physical_pixel_sizes")
```

### Registry isolation in tests

Tests that register plugins should isolate themselves from the real
registry. Snapshot and restore around each test:

```python
import pytest
from zarrmony.readers import plugin as plugin_mod
from zarrmony_myformat.plugin import my_plugin

@pytest.fixture
def isolated_registry():
    snapshot_plugins = dict(plugin_mod._PLUGINS)
    snapshot_loaded = plugin_mod._ENTRY_POINTS_LOADED
    plugin_mod._PLUGINS.clear()
    plugin_mod._ENTRY_POINTS_LOADED = True  # skip entry-point discovery
    try:
        yield
    finally:
        plugin_mod._PLUGINS.clear()
        plugin_mod._PLUGINS.update(snapshot_plugins)
        plugin_mod._ENTRY_POINTS_LOADED = snapshot_loaded


def test_my_plugin_wins_dispatch(isolated_registry, tmp_path):
    from zarrmony.readers.plugin import register_plugin, get_reader
    register_plugin(my_plugin)
    fixture = tmp_path / "sample.myf"
    write_minimal_fixture(fixture)
    _reader, plugin, score = get_reader(fixture)
    assert plugin.name == "zarrmony-myformat"
    assert score == 100
```

The same pattern is used in zarrmony's own
[`tests/test_plugin_registry.py`](../tests/test_plugin_registry.py) — read
that file for additional coverage ideas (matcher exceptions, score-tie
ordering, entry-point loader edge cases).

### End-to-end conversion

The strongest test is converting a tiny real fixture and asserting on the
output store:

```python
from zarrmony import convert

def test_convert_myformat_end_to_end(tmp_path, isolated_registry):
    from zarrmony.readers.plugin import register_plugin
    register_plugin(my_plugin)

    out_dir = tmp_path / "out"
    result = convert("tests/fixtures/tiny.myf", out_dir)
    assert result["stores"]
    assert (out_dir / "scene-0.ome.zarr" / "zarr.json").is_file()
```

Keep your fixture small (a few KB if possible). Vendor SDKs often have a
"create empty file with N×M×C×Z×T axes" helper — use it.

## 9. Writing a plate-shaped reader

If your format describes a high-content-screening plate (multiple wells,
multiple fields per well — Phenix, Operetta, ImageXpress, etc.), opt into
the HCS plate writer instead of producing one-image-per-scene flat output.
Set `layout_hint = "plate"` and populate `plate_layout` with a
`PlateLayout` describing the rows, columns, fields-of-view, and
acquisitions. Under `--layout auto` (the default), zarrmony then writes a
single OME-NGFF 0.5 [`plate`](https://ngff.openmicroscopy.org/0.5/#hcs-layout)
store at the output path. See [ADR-0004](./adr/0004-plate-output-design.md)
for the design rationale.

### The `PlateLayout` dataclass

```python
from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout

plate_layout = PlateLayout(
    name="experiment-2026-05-11",
    rows=["A", "B", "C", "D", "E", "F", "G", "H"],          # every physical row
    columns=[f"{c:02d}" for c in range(1, 13)],              # every physical column
    acquisitions=[Acquisition(id=1, name="baseline")],       # 0 or 1 acquisition (v1)
    fields=[
        PlateField(scene_index=0, row="B", column="04", field_name="F001", acquisition_id=1),
        PlateField(scene_index=1, row="B", column="04", field_name="F002", acquisition_id=1),
        PlateField(scene_index=2, row="C", column="05", field_name="F001", acquisition_id=1),
        # ...
    ],
)
```

The dataclasses are also re-exported from `zarrmony.readers.plugin` for
convenience. `PlateLayout`, `PlateField`, and `Acquisition` are all
`@dataclass(frozen=True)`.

### Field semantics and the structural rules zarrmony enforces

Zarrmony validates the `PlateLayout` before any pixels are written; a
violation raises `PlateLayoutError` (from `zarrmony.errors`). The rules:

- **Every `PlateField.row` must appear in `PlateLayout.rows`**, and every
  `PlateField.column` in `PlateLayout.columns`. List every _physical_ row
  and column even when only some are imaged — sparse-plate semantics are
  the reader's responsibility (a 96-well plate with six imaged wells still
  declares `rows=["A".."H"]` and `columns=["01".."12"]`).
- **`scene_index` must be a valid index into `reader.scenes`**, and must
  be unique across all fields (a duplicate would silently double-write the
  same source data into two different well paths).
- **No two `PlateField` entries may produce the same well path.** Within a
  well, fields are written to sequential integer paths (`<row>/<col>/0`,
  `<row>/<col>/1`, …) in the order they appear in `plate_layout.fields`.
- **`acquisition_id`, when set, must point at a declared `Acquisition.id`**.
- **At most one `Acquisition`** in v1 (multi-acquisition is reserved for v2;
  see ADR-0004 §Considered Options).

If `reader.scenes` contains scenes that no `PlateField` references,
zarrmony emits a `LayoutDowngradeWarning` and skips them — useful to
detect when an adapter forgets to expose some FOVs.

### `field_name` is vendor-native, not the on-disk path

`PlateField.field_name` is the **vendor's human-readable label** (e.g.
Phenix's `F001`, `F002`). It is preserved in the audit record and used as
the multiscales `name` for that FOV's image. It is **not** the on-disk
directory: zarrmony assigns sequential integer paths (`0`, `1`, …) per
well so well groups satisfy the OME-NGFF requirement that field paths be
sortable as integers. If you need to round-trip vendor labels back from
the converted store, read `attrs.zarrmony.audit.fields[i].field_name` or
the multiscales `name`.

### `convert()` consumes `plate_layout` directly

There is no string-parsing fallback. If you populate `plate_layout`, the
plate writer uses it; if you leave it `None`, zarrmony cannot dispatch to
plate even when `layout_hint="plate"`. Building the layout once, in your
adapter, against the vendor's own well/field tables is the source of
truth.

### v1 limits and when to fall back to flat

The v1 plate writer is **single-acquisition, single-plate**. If your input
violates either:

- **Multiple acquisitions**: degrade gracefully today by exposing only one
  acquisition's worth of fields and emitting a warning, OR set
  `layout_hint = "flat"` so callers get per-scene output. Don't try to
  pack multiple acquisitions into one `PlateLayout` — the writer asserts
  `len(acquisitions) <= 1`.
- **Multiple physical plates per input**: `plate_layout` is a single
  `PlateLayout`, not a list. Either expose just the first plate (with a
  warning) or set `layout_hint = "flat"`. The choice is the adapter's,
  not zarrmony's.

In both cases, picking `layout_hint = "flat"` for now means callers get
one `<scene>.ome.zarr` per FOV under the output directory — full pixel
fidelity, no plate metadata. Users can also force this on a per-call
basis with `layout="per-scene"` (a `LayoutDowngradeWarning` will fire to
flag that the plate metadata is being dropped).

### Helpers exposed for adapters

For adapters and downstream tooling that need to validate user input
against a `PlateLayout`:

- **`zarrmony.writers.plate.parse_well_key(key)`** splits a compact well
  key like `"B04"` or `"AA01"` into `(row, col)`. Casing and zero-padding
  are preserved verbatim — caller validates against the plate's
  canonical spellings.
- **`zarrmony.writers.plate.summarize_plate_layout(plate_layout)`**
  returns the OME-NGFF `plate`-shaped summary dict (no I/O), mirroring
  the on-disk `attrs.ome.plate` and the audit `plate` block. Used by
  `inspect()` to surface plate context before a conversion runs.

### Worked example shape

A minimal plate-shaped adapter:

```python
from pathlib import Path
from typing import Any

from zarrmony.readers.plate import Acquisition, PlateField, PlateLayout
from zarrmony.readers.plugin import ReaderPlugin


class MyPlateReader:
    layout_hint = "plate"

    def __init__(self, path: Path):
        self._sdk = VendorSDK.open(str(path))
        # Order scenes so PlateField.scene_index points back into this list.
        self.scenes = [fov.id for fov in self._sdk.iter_fovs()]
        self.plate_layout = self._build_plate_layout()
        self._active = 0

    def _build_plate_layout(self) -> PlateLayout:
        plate = self._sdk.plate
        return PlateLayout(
            name=plate.name,
            rows=[r.label for r in plate.rows],
            columns=[c.label for c in plate.columns],
            acquisitions=[Acquisition(id=1, name=self._sdk.acquisition_name)],
            fields=[
                PlateField(
                    scene_index=i,
                    row=fov.well.row,
                    column=fov.well.column,
                    field_name=fov.label,
                    acquisition_id=1,
                )
                for i, fov in enumerate(self._sdk.iter_fovs())
            ],
        )

    def set_scene(self, index: int) -> None:
        self._active = index

    @property
    def xarray_dask_data(self):
        return self._sdk.read_fov_lazy(self._active)

    @property
    def physical_pixel_sizes(self):
        return self._sdk.iter_fovs()[self._active].pixel_sizes

    def close(self) -> None:
        self._sdk.close()


def _match_myplate(path: Path) -> int | None:
    return 100 if path.is_dir() and (path / "plate.xml").is_file() else None


def _open_myplate(path: Path) -> Any:
    return MyPlateReader(path)


myplate_plugin = ReaderPlugin(
    name="zarrmony-myplate",
    match=_match_myplate,
    open=_open_myplate,
    distribution="zarrmony-myplate",
    source="entry_point",
)
```

For a real adapter, see the `zarrmony-phenix` package (tracked in
issue #13).
