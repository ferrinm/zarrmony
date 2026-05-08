"""Direct contract tests for the ``ReaderPlugin`` registry (ADR-0001).

Each test uses fake ``ReaderPlugin`` instances built inline so the suite runs
without any bioio plugin installed. The autouse fixture resets ``_PLUGINS`` and
``_ENTRY_POINTS_LOADED`` so tests are isolated from the real registry that
``readers/__init__.py`` populates at zarrmony import time.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from zarrmony.readers import plugin as plugin_mod
from zarrmony.readers.plugin import (
    ENTRY_POINT_GROUP,
    NoMatchingPluginError,
    PluginSource,
    ReaderPlugin,
    get_reader,
    list_plugins,
    register_plugin,
    unregister_plugin,
)


def _plugin(
    name: str,
    *,
    score: int | None = 100,
    source: PluginSource = "runtime",
) -> ReaderPlugin:
    return ReaderPlugin(
        name=name,
        match=lambda _p, _s=score: _s,
        open=lambda p, _n=name: ("opened", _n, p),
        source=source,
    )


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    snapshot_plugins = dict(plugin_mod._PLUGINS)
    snapshot_loaded = plugin_mod._ENTRY_POINTS_LOADED
    plugin_mod._PLUGINS.clear()
    # Default to "loaded" so dispatch tests don't accidentally walk the real
    # entry-point group; entry-point tests flip this back to False explicitly.
    plugin_mod._ENTRY_POINTS_LOADED = True
    try:
        yield
    finally:
        plugin_mod._PLUGINS.clear()
        plugin_mod._PLUGINS.update(snapshot_plugins)
        plugin_mod._ENTRY_POINTS_LOADED = snapshot_loaded


# --- Case 1: duplicate registration --------------------------------------


def test_duplicate_registration_raises_value_error() -> None:
    register_plugin(_plugin("dupe"))
    with pytest.raises(ValueError, match="already registered"):
        register_plugin(_plugin("dupe"))


def test_duplicate_registration_with_replace_succeeds() -> None:
    register_plugin(_plugin("dupe", score=10))
    register_plugin(_plugin("dupe", score=99), replace=True)

    _reader, plugin, score = get_reader("/tmp/anything.xyz")
    assert plugin.name == "dupe"
    assert score == 99


def test_unregister_plugin_is_noop_for_unknown_name() -> None:
    unregister_plugin("never-existed")  # must not raise
    register_plugin(_plugin("present"))
    unregister_plugin("present")
    assert "present" not in plugin_mod._PLUGINS


# --- Case 2: matcher exception resilience --------------------------------


def test_matcher_exception_does_not_abort_dispatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _explode(_p: Path) -> int | None:
        raise RuntimeError("matcher exploded")

    explosive = ReaderPlugin(
        name="explosive",
        match=_explode,
        open=lambda _p: pytest.fail("explosive.open() must not run"),
    )
    register_plugin(explosive)
    register_plugin(_plugin("safe", score=5))

    with caplog.at_level(logging.WARNING, logger="zarrmony.readers.plugin"):
        _reader, plugin, score = get_reader("/tmp/foo.xyz")

    assert plugin.name == "safe"
    assert score == 5
    assert any(
        "explosive" in record.message and "no-match" in record.message for record in caplog.records
    ), caplog.text


# --- Case 3: score-tie ordering ------------------------------------------


def test_score_tie_resolves_to_first_registered_built_in() -> None:
    register_plugin(_plugin("builtin-shaped", score=50, source="builtin"))
    register_plugin(_plugin("entry-point-shaped", score=50, source="entry_point"))

    _reader, plugin, _score = get_reader("/tmp/foo.xyz")
    assert plugin.name == "builtin-shaped"
    assert plugin.source == "builtin"


# --- Case 4: entry-point loader ------------------------------------------


class _FakeEntryPoint:
    """Stand-in for ``importlib.metadata.EntryPoint`` exposing only what
    ``_ensure_entry_points_loaded`` consumes (``name`` + ``load()``).
    """

    def __init__(self, name: str, payload: Any, *, raises: BaseException | None = None) -> None:
        self.name = name
        self._payload = payload
        self._raises = raises

    def load(self) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._payload


def _install_fake_entry_points(monkeypatch: pytest.MonkeyPatch, *eps: _FakeEntryPoint) -> None:
    plugin_mod._ENTRY_POINTS_LOADED = False

    def _fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        assert group == ENTRY_POINT_GROUP
        return list(eps)

    monkeypatch.setattr(plugin_mod, "entry_points", _fake_entry_points)


def test_entry_point_loader_registers_yielded_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _plugin("from-entry-point", source="entry_point")
    _install_fake_entry_points(monkeypatch, _FakeEntryPoint(fake.name, fake))

    list_plugins()  # triggers _ensure_entry_points_loaded()

    assert "from-entry-point" in plugin_mod._PLUGINS
    assert plugin_mod._PLUGINS["from-entry-point"] is fake


def test_entry_point_loader_warns_and_continues_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    survivor = _plugin("survivor", source="entry_point")
    _install_fake_entry_points(
        monkeypatch,
        _FakeEntryPoint("broken", payload=None, raises=ImportError("missing dep")),
        _FakeEntryPoint(survivor.name, survivor),
    )

    with pytest.warns(UserWarning, match="failed to load.*broken"):
        list_plugins()

    assert "broken" not in plugin_mod._PLUGINS
    assert "survivor" in plugin_mod._PLUGINS


def test_entry_point_loader_warns_when_payload_is_not_a_reader_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_entry_points(monkeypatch, _FakeEntryPoint("bogus", payload="not-a-ReaderPlugin"))

    with pytest.warns(UserWarning, match="did not yield a ReaderPlugin"):
        list_plugins()

    assert "bogus" not in plugin_mod._PLUGINS


def test_entry_point_loader_warns_when_name_collides_with_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_plugin(_plugin("clash", source="builtin"))
    colliding = _plugin("clash", source="entry_point")
    _install_fake_entry_points(monkeypatch, _FakeEntryPoint(colliding.name, colliding))

    with pytest.warns(UserWarning, match="already registered"):
        list_plugins()

    # Pre-existing built-in is preserved; the colliding entry-point plugin is rejected.
    assert plugin_mod._PLUGINS["clash"].source == "builtin"


# --- Case 5: NoMatchingPluginError ---------------------------------------


def test_no_matching_plugin_error_names_input_path() -> None:
    register_plugin(_plugin("never-matches", score=None))

    with pytest.raises(NoMatchingPluginError, match=r"/tmp/uniquely-named-input\.xyz"):
        get_reader("/tmp/uniquely-named-input.xyz")
