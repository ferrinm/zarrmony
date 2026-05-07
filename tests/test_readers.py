"""Tests for the reader registry.

The actual format-specific ``Reader`` constructors (bioio_lif.Reader,
bioio_czi.Reader) require real LIF/CZI files. Here we monkeypatch the
factory functions to verify dispatch logic without needing real fixture files.
"""

from pathlib import Path
from typing import Any

import pytest

from zarrmony import readers as readers_pkg
from zarrmony.readers import get_reader, register_override


class _Fake:
    def __init__(self, path: str | Path, tag: str) -> None:
        self.path = str(path)
        self.tag = tag


def _fake_factory(tag: str):
    def factory(path: str | Path) -> tuple[Any, str]:
        return _Fake(path, tag), f"fake-{tag}"

    return factory


@pytest.fixture(autouse=True)
def restore_overrides() -> None:
    """Snapshot and restore the registry around each test."""
    snapshot = readers_pkg._OVERRIDES.copy()
    yield
    readers_pkg._OVERRIDES.clear()
    readers_pkg._OVERRIDES.update(snapshot)


def test_unknown_extension_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readers_pkg, "open_default_reader", _fake_factory("default"))
    reader, plugin = get_reader("/tmp/foo.tif")
    assert reader.tag == "default"
    assert plugin == "fake-default"


def test_czi_dispatches_to_czi_override() -> None:
    register_override(".czi", _fake_factory("czi"))
    reader, plugin = get_reader("/tmp/foo.czi")
    assert reader.tag == "czi"
    assert plugin == "fake-czi"


def test_lif_dispatches_to_lif_override() -> None:
    register_override(".lif", _fake_factory("lif"))
    reader, plugin = get_reader("/tmp/foo.lif")
    assert reader.tag == "lif"
    assert plugin == "fake-lif"


def test_nd2_dispatches_to_nd2_override() -> None:
    register_override(".nd2", _fake_factory("nd2"))
    reader, plugin = get_reader("/tmp/foo.nd2")
    assert reader.tag == "nd2"
    assert plugin == "fake-nd2"


def test_extension_match_is_case_insensitive() -> None:
    register_override(".czi", _fake_factory("czi"))
    reader, _ = get_reader("/tmp/foo.CZI")
    assert reader.tag == "czi"


def test_register_override_accepts_extension_without_leading_dot() -> None:
    register_override("xyz", _fake_factory("xyz"))
    reader, plugin = get_reader("/tmp/file.xyz")
    assert reader.tag == "xyz"
    assert plugin == "fake-xyz"


def test_uri_style_paths_detect_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    register_override(".czi", _fake_factory("czi"))
    # Path() handles gs:// strings as plain text and .suffix walks the last segment
    reader, _ = get_reader("gs://my-bucket/folder/sample.czi")
    assert reader.tag == "czi"


def test_register_override_replaces_existing_mapping() -> None:
    register_override(".czi", _fake_factory("first"))
    register_override(".czi", _fake_factory("second"))
    reader, _ = get_reader("/tmp/foo.czi")
    assert reader.tag == "second"
