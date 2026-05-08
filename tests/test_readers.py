"""Tests for the built-in reader plugins (matchers + import-time registration).

The registry contract itself (dispatch, ties, entry points, error handling)
lives in ``test_plugin_registry.py``. This file focuses on the format-specific
matchers and the integration check that the real built-in registry resolves
each extension to the correct plugin.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from zarrmony.readers import czi as czi_mod
from zarrmony.readers import default as default_mod
from zarrmony.readers import lif as lif_mod
from zarrmony.readers import nd2 as nd2_mod
from zarrmony.readers import plugin as plugin_mod
from zarrmony.readers.plugin import (
    ReaderPlugin,
    get_reader,
    list_plugins,
    register_plugin,
)


@pytest.fixture(autouse=True)
def restore_registry() -> Iterator[None]:
    """Snapshot and restore the plugin registry around each test."""
    snapshot = dict(plugin_mod._PLUGINS)
    loaded = plugin_mod._ENTRY_POINTS_LOADED
    yield
    plugin_mod._PLUGINS.clear()
    plugin_mod._PLUGINS.update(snapshot)
    plugin_mod._ENTRY_POINTS_LOADED = loaded


# --- Built-in plugin registration ----------------------------------------


def test_builtin_plugins_are_registered_at_import_time() -> None:
    by_name = {p.name: p for p in list_plugins()}
    for name in ("bioio", "bioio-czi", "bioio-lif", "bioio-nd2"):
        assert name in by_name, f"{name!r} not registered at import time"
        assert by_name[name].source == "builtin"


# --- Matcher behavior (cheap, no Reader construction) --------------------


def test_default_matcher_returns_zero_for_any_path() -> None:
    # Catch-all: lowest possible score so any extension-specific plugin wins.
    assert default_mod._match_default(Path("/tmp/anything.tif")) == 0
    assert default_mod._match_default(Path("/tmp/no_extension")) == 0


@pytest.mark.parametrize(
    ("matcher", "ext"),
    [
        (czi_mod._match_czi, ".czi"),
        (lif_mod._match_lif, ".lif"),
        (nd2_mod._match_nd2, ".nd2"),
    ],
)
def test_format_matcher_claims_extension(matcher, ext: str) -> None:
    assert matcher(Path(f"/tmp/foo{ext}")) == 100
    assert matcher(Path(f"/tmp/foo{ext.upper()}")) == 100  # case-insensitive
    assert matcher(Path("/tmp/foo.tif")) is None


def test_uri_style_paths_extract_extension() -> None:
    # Path() handles gs:// strings as plain text; .suffix walks the last segment.
    assert czi_mod._match_czi(Path("gs://my-bucket/folder/sample.czi")) == 100


# --- Built-in dispatch end-to-end ----------------------------------------


class _Sentinel:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_builtin_extension_dispatch_with_real_registry() -> None:
    """Replace each built-in's ``open`` with a sentinel and confirm dispatch
    walks the real registry to the right plugin per extension. The real
    matchers run; only ``open`` is faked so we don't need real fixture files.
    """
    sentinels = {
        "bioio": _Sentinel("default"),
        "bioio-czi": _Sentinel("czi"),
        "bioio-lif": _Sentinel("lif"),
        "bioio-nd2": _Sentinel("nd2"),
    }
    for name, sentinel in sentinels.items():
        original = plugin_mod._PLUGINS[name]
        replaced = ReaderPlugin(
            name=original.name,
            match=original.match,
            open=lambda _p, s=sentinel: s,
            distribution=original.distribution,
            source=original.source,
        )
        register_plugin(replaced, replace=True)

    cases = [
        ("/tmp/foo.tif", "bioio", 0),
        ("/tmp/foo.CZI", "bioio-czi", 100),
        ("/tmp/foo.lif", "bioio-lif", 100),
        ("/tmp/foo.nd2", "bioio-nd2", 100),
    ]
    for path, expected_name, expected_score in cases:
        reader, plugin, score = get_reader(path)
        assert plugin.name == expected_name, path
        assert score == expected_score, path
        assert reader is sentinels[expected_name], path
