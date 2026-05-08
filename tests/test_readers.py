"""Tests for the reader plugin registry.

The actual format-specific ``Reader`` constructors (bioio_lif.Reader,
bioio_czi.Reader) require real LIF/CZI fixture files. We test the matchers
directly (cheap, side-effect-free predicates) and exercise dispatch end-to-end
with fake ``ReaderPlugin`` instances registered in an isolated registry.
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
    NoMatchingPluginError,
    ReaderPlugin,
    get_reader,
    list_plugins,
    register_plugin,
    unregister_plugin,
)


class _Fake:
    def __init__(self, path: str | Path, tag: str) -> None:
        self.path = str(path)
        self.tag = tag


def _fake_plugin(
    name: str,
    *,
    score: int | None = 100,
    tag: str | None = None,
) -> ReaderPlugin:
    tag = tag or name
    return ReaderPlugin(
        name=name,
        match=lambda _p: score,
        open=lambda p: _Fake(p, tag),
        distribution=None,
        source="runtime",
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


# --- Dispatch (uses fake plugins in an isolated registry) ----------------


def test_get_reader_returns_highest_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin_mod, "_PLUGINS", {})
    monkeypatch.setattr(plugin_mod, "_ENTRY_POINTS_LOADED", True)
    register_plugin(_fake_plugin("low", score=10))
    register_plugin(_fake_plugin("high", score=50))

    reader, plugin, score = get_reader("/tmp/anything.xyz")
    assert reader.tag == "high"
    assert plugin.name == "high"
    assert score == 50


def test_equal_score_resolves_to_first_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin_mod, "_PLUGINS", {})
    monkeypatch.setattr(plugin_mod, "_ENTRY_POINTS_LOADED", True)
    register_plugin(_fake_plugin("first", score=100))
    register_plugin(_fake_plugin("second", score=100))

    _reader, plugin, _ = get_reader("/tmp/anything.xyz")
    assert plugin.name == "first"


def test_no_match_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin_mod, "_PLUGINS", {})
    monkeypatch.setattr(plugin_mod, "_ENTRY_POINTS_LOADED", True)
    register_plugin(_fake_plugin("nope", score=None))

    with pytest.raises(NoMatchingPluginError):
        get_reader("/tmp/anything.xyz")


def test_matcher_that_raises_is_treated_as_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin_mod, "_PLUGINS", {})
    monkeypatch.setattr(plugin_mod, "_ENTRY_POINTS_LOADED", True)

    def _boom(_p: Path) -> int | None:
        raise RuntimeError("matcher exploded")

    bad = ReaderPlugin(
        name="explosive",
        match=_boom,
        open=lambda _p: object(),
        source="runtime",
    )
    register_plugin(bad)
    register_plugin(_fake_plugin("safe", score=5))

    _reader, plugin, score = get_reader("/tmp/foo.xyz")
    assert plugin.name == "safe"
    assert score == 5


def test_builtin_extension_dispatch_with_real_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace each built-in's ``open`` with a sentinel and confirm dispatch
    walks the real registry to the right plugin per extension. The real
    matchers run; only ``open`` is faked so we don't need real fixture files.
    """
    sentinels = {
        "bioio": _Fake("/tmp/x.tif", "default"),
        "bioio-czi": _Fake("/tmp/x.czi", "czi"),
        "bioio-lif": _Fake("/tmp/x.lif", "lif"),
        "bioio-nd2": _Fake("/tmp/x.nd2", "nd2"),
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


# --- Registry hygiene -----------------------------------------------------


def test_register_plugin_rejects_duplicate_name() -> None:
    register_plugin(_fake_plugin("dupe"))
    with pytest.raises(ValueError, match="already registered"):
        register_plugin(_fake_plugin("dupe"))


def test_register_plugin_replace_overrides_existing() -> None:
    register_plugin(_fake_plugin("dupe", tag="first"))
    register_plugin(_fake_plugin("dupe", tag="second"), replace=True)
    reader, _winner, _ = get_reader("/tmp/whatever.xyz")
    assert reader.tag == "second"


def test_unregister_plugin_is_noop_for_unknown_name() -> None:
    unregister_plugin("never-existed")  # must not raise
