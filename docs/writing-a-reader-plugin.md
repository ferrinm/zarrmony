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

| Attribute | Type | What zarrmony does with it |
|-----------|------|----------------------------|
| `scenes` | `list[str]` | Iterates to enumerate scenes; names are sanitized into per-scene store directory names. |
| `set_scene(index: int) -> None` | method | Switches the active scene before reading data. Called once per scene. |
| `xarray_dask_data` | `xarray.DataArray` (dask-backed) | The image array for the current scene. Dimensions should follow OME-Zarr 0.5 axis order; channels named via the `C` coordinate when present. |
| `physical_pixel_sizes` | object with `.X`, `.Y`, `.Z` floats (microns) | Scale transform written into the multiscales metadata. Mirrors `bioio_base.PhysicalPixelSizes`; `None` on any axis is fine. |

### Soft-optional attributes

These are accessed via `getattr` or `try/except`. Omit them and zarrmony falls
back gracefully — the conversion still produces a valid OME-Zarr; it just
loses fidelity for the missing piece.

| Attribute | Fallback when missing or raising |
|-----------|----------------------------------|
| `channel_names: list[str]` | Channels labelled `C:0`, `C:1`, …; the audit records `channel_names: null`. |
| `ome_metadata` | A minimal OME `Image` element synthesised from the scene shape; an audit warning records the failure. Return `OME` (from `ome-types`), an `xml.etree.ElementTree.Element`, or an XML string. |
| `metadata` | No `OME/source/raw.<format>.xml` is written. Anything that serialises with `str()` works; native `OME`, `Element`, or `str` skip the round-trip. |
| `close()` | Skipped. Implement it if your reader holds non-GC resources (file handles, network sessions). zarrmony's intent is to call this in a `finally` block once it lands ([ADR-0001](./adr/0001-reader-plugin-architecture.md)); writing it now is forward-compatible. |

### Reserved: `layout_hint`

`layout_hint: Literal["flat", "plate"]` is reserved for future HCS Plate
writer dispatch ([ADR-0002](./adr/0002-layout-hint-reservation.md)). Plate-
shaped readers (Phenix and similar) should set it to `"plate"` from day one;
the writer ignores it in v0.2 and falls back to flat per-scene output. Once
HCS lands, the same plugin starts producing plate-shaped output without
modification. Flat-image readers can leave the attribute unset (the Protocol
default is `"flat"`).

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
- **Third-party plugins**: return `≥ 100` if you want to *win* a tie against
  a built-in (e.g. you ship a faster CZI reader); return `< 100` if you want
  to *defer* to built-ins for files you could also handle. Built-ins register
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
from zarrmony import convert, UserMetadata

def test_convert_myformat_end_to_end(tmp_path, isolated_registry):
    from zarrmony.readers.plugin import register_plugin
    register_plugin(my_plugin)

    out_dir = tmp_path / "out"
    result = convert(
        "tests/fixtures/tiny.myf",
        out_dir,
        metadata=UserMetadata(...),
    )
    assert result["stores"]
    assert (out_dir / "scene-0.ome.zarr" / "zarr.json").is_file()
```

Keep your fixture small (a few KB if possible). Vendor SDKs often have a
"create empty file with N×M×C×Z×T axes" helper — use it.
