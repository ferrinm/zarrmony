"""Tests for the ND2 vendor-SDK microscope extractor (issue #78).

Covers :mod:`zarrmony.metadata.nd2_acquisition` — reads the free-text
``capturing`` field from ``nd2.ND2File(path).text_info()`` and projects it to
``{"microscope": "Nikon <Model>"}``. bioio-nd2's OME projection omits the
``<Microscope>`` element entirely so this is the only source zarrmony has for
the microscope string on ND2 files.
"""

from __future__ import annotations

from typing import Any

import pytest

from zarrmony.metadata.nd2_acquisition import (
    _microscope_from_text_info,
    extract_nd2_acquisition,
)

# --- pure parse: _microscope_from_text_info -------------------------------


def test_capturing_with_bare_model_prepends_nikon() -> None:
    """Real-shape ``capturing`` for a Ti2 rig: model line only."""
    text_info = {"capturing": "Ti2"}
    assert _microscope_from_text_info(text_info) == "Nikon Ti2"


def test_capturing_drops_nis_elements_and_vendor_lines() -> None:
    """NIS-Elements typically writes vendor and software lines above the model."""
    text_info = {
        "capturing": "NIS-Elements AR 5.42.03\nNikon Instruments Inc.\nTi2-E",
    }
    assert _microscope_from_text_info(text_info) == "Nikon Ti2-E"


def test_capturing_with_nikon_prefix_kept_as_is() -> None:
    """A ``capturing`` line that already carries the brand isn't double-prefixed."""
    text_info = {"capturing": "Nikon A1 R"}
    assert _microscope_from_text_info(text_info) == "Nikon A1 R"


def test_capturing_split_on_semicolons() -> None:
    """Some NIS-Elements versions delimit with ``;`` instead of newlines."""
    text_info = {"capturing": "NIS-Elements 5.30; Nikon Instruments Inc.; CSU-W1"}
    assert _microscope_from_text_info(text_info) == "Nikon CSU-W1"


def test_capturing_all_vendor_lines_yields_none() -> None:
    """A file with only vendor/software lines and no microscope name → None."""
    text_info = {
        "capturing": "NIS-Elements AR 5.42.03\nNikon Instruments Inc.",
    }
    assert _microscope_from_text_info(text_info) is None


def test_missing_capturing_key_yields_none() -> None:
    assert _microscope_from_text_info({"optics": "Plan Apo 40x"}) is None


def test_empty_capturing_yields_none() -> None:
    assert _microscope_from_text_info({"capturing": ""}) is None
    assert _microscope_from_text_info({"capturing": "   "}) is None


def test_none_text_info_yields_none() -> None:
    assert _microscope_from_text_info(None) is None


def test_non_dict_text_info_yields_none() -> None:
    """The nd2 SDK's contract is a dict; anything else is treated as absent."""
    for bad in (42, "capturing: Ti2", ["capturing"]):
        assert _microscope_from_text_info(bad) is None  # type: ignore[arg-type]


# --- extract_nd2_acquisition: full reader-path wiring ---------------------


class _StubReader:
    """Minimal reader with a configurable ``_path`` — mirrors bioio-nd2."""

    def __init__(self, path: str | None = "/tmp/fake.nd2") -> None:
        if path is not None:
            self._path = path


class _FakeND2File:
    """Context-manager-shaped stand-in for ``nd2.ND2File``."""

    def __init__(self, text_info: Any) -> None:
        self._text_info = text_info
        self.closed = False

    def text_info(self) -> Any:
        return self._text_info

    def close(self) -> None:
        self.closed = True


def _install_fake_nd2(monkeypatch: pytest.MonkeyPatch, text_info: Any) -> None:
    """Patch ``nd2.ND2File`` in ``sys.modules`` so the extractor picks it up."""
    import sys
    import types

    module = types.ModuleType("nd2")

    def _ctor(_path: str) -> _FakeND2File:
        return _FakeND2File(text_info)

    module.ND2File = _ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nd2", module)


def test_extract_from_reader_dispatches_to_text_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_nd2(monkeypatch, {"capturing": "Ti2-E"})
    result = extract_nd2_acquisition(_StubReader("/tmp/fake.nd2"))
    assert result == {"microscope": "Nikon Ti2-E"}


def test_extract_returns_none_when_reader_has_no_path() -> None:
    """A reader without a discoverable ``_path`` yields None."""
    assert extract_nd2_acquisition(_StubReader(path=None)) is None


def test_extract_returns_none_when_nd2_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``nd2`` package must not crash the audit path."""
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "nd2", raising=False)
    real_import = builtins.__import__

    def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "nd2":
            raise ModuleNotFoundError("nd2 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    assert extract_nd2_acquisition(_StubReader("/tmp/fake.nd2")) is None


def test_extract_returns_none_when_text_info_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """text_info() returning a dict without ``capturing`` → None."""
    _install_fake_nd2(monkeypatch, {"optics": "Plan Apo 40x"})
    assert extract_nd2_acquisition(_StubReader()) is None


def test_extract_survives_text_info_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``text_info()`` call that raises must yield None, not propagate."""
    import sys
    import types

    class _RaisingFile:
        def text_info(self) -> Any:
            raise RuntimeError("simulated ND2 SDK failure")

        def close(self) -> None:
            pass

    module = types.ModuleType("nd2")
    module.ND2File = lambda _path: _RaisingFile()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nd2", module)
    assert extract_nd2_acquisition(_StubReader()) is None
