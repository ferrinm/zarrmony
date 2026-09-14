"""Tests for the CZI plugin's tolerance of an absent ``bioio-czi`` backend.

The matcher and the end-to-end dispatch live in ``test_readers.py``. This file
covers the two halves of issue #142. First, the lazy-import contract: zarrmony
imports, and the plugin stays registered, on a platform where the default
dependency set omits ``bioio-czi``. Second, the environment marker that
produces that platform, read back from zarrmony's own installed metadata.

The backend is simulated through ``sys.modules`` rather than uninstalled, and
the two entries below reproduce the two real failures exactly:

- ``None`` makes ``importlib.util.find_spec`` return ``None`` and the import
  raise ``ModuleNotFoundError`` — the backend is not installed.
- A stub module with a spec but no ``Reader`` makes ``find_spec`` return that
  spec and the import raise a plain ``ImportError`` — the backend is installed
  and will not load, which is how a compiled extension fails.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from zarrmony.errors import UnsupportedFormatError
from zarrmony.readers import czi as czi_mod


@pytest.fixture
def uninstalled_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``bioio-czi`` at all, as on a default Linux arm64 install."""
    monkeypatch.setitem(sys.modules, "bioio_czi", None)


@pytest.fixture
def unloadable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bioio-czi`` installed, but its compiled backend will not load."""
    stub = types.ModuleType("bioio_czi")
    stub.__spec__ = importlib.util.spec_from_loader("bioio_czi", loader=None)
    monkeypatch.setitem(sys.modules, "bioio_czi", stub)


# --- The lazy import ------------------------------------------------------


def test_plugin_is_intact_without_the_backend(uninstalled_backend: None) -> None:
    """Name, distribution and match score do not depend on the backend.

    A plugin that dropped out here would hand ``.czi`` inputs to the ``bioio``
    catch-all, whose hint names the ``bioformats`` extra.
    """
    assert czi_mod.czi_plugin.name == "bioio-czi"
    assert czi_mod.czi_plugin.distribution == "bioio-czi"
    assert czi_mod.czi_plugin.match(Path("/tmp/foo.czi")) == 100


def test_open_without_backend_names_the_czi_extra(uninstalled_backend: None) -> None:
    with pytest.raises(UnsupportedFormatError) as excinfo:
        czi_mod.czi_plugin.open(Path("/tmp/foo.czi"))

    message = str(excinfo.value)
    assert 'pip install "zarrmony[czi]"' in message
    assert "/tmp/foo.czi" in message
    # The bioformats extra is the catch-all plugin's advice, and it is the
    # wrong advice for a CZI file.
    assert "bioformats" not in message


def test_open_without_backend_chains_the_import_error(
    uninstalled_backend: None,
) -> None:
    with pytest.raises(UnsupportedFormatError) as excinfo:
        czi_mod.czi_plugin.open(Path("/tmp/foo.czi"))

    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_open_translates_an_unloadable_backend(unloadable_backend: None) -> None:
    """An installed-but-broken backend raises ``ImportError``, not its subclass.

    The CLI reports a ``ZarrmonyError`` and lets anything else through as a
    raw traceback, so catching only ``ModuleNotFoundError`` would cost the user
    the message that names the cause.
    """
    with pytest.raises(UnsupportedFormatError) as excinfo:
        czi_mod.czi_plugin.open(Path("/tmp/foo.czi"))

    cause = excinfo.value.__cause__
    assert isinstance(cause, ImportError)
    assert not isinstance(cause, ModuleNotFoundError)


def test_unloadable_backend_message_quotes_the_failure(
    unloadable_backend: None,
) -> None:
    """Naming the extra is noise here — it is installed. Say what broke."""
    with pytest.raises(UnsupportedFormatError) as excinfo:
        czi_mod.czi_plugin.open(Path("/tmp/foo.czi"))

    message = str(excinfo.value)
    assert "installed but cannot be imported" in message
    assert "cannot import name 'Reader'" in message
    assert 'pip install "zarrmony[czi]"' not in message


def test_missing_backend_message_quotes_a_compiled_dependency() -> None:
    """The realistic absence is ``aicspylibczi``, not ``bioio_czi`` itself.

    ``bioio_czi`` imports it at module scope, so an install that lost it fails
    with its name. The message must carry that name through.
    """
    exc = ModuleNotFoundError("No module named 'aicspylibczi'")
    error = czi_mod._missing_backend_error(Path("/tmp/foo.czi"), exc)

    assert "aicspylibczi" in str(error)


# --- The backend predicate ------------------------------------------------


def test_backend_predicate_is_false_when_uninstalled(
    uninstalled_backend: None,
) -> None:
    assert czi_mod._czi_backend_installed() is False


def test_backend_predicate_is_true_when_installed(unloadable_backend: None) -> None:
    # Installed is not the same as importable, and the predicate reports the
    # first — that is what picks between the two messages.
    assert czi_mod._czi_backend_installed() is True


# --- The environment marker on the core dependency -----------------------


def _czi_core_requirement() -> Requirement:
    """The ``bioio-czi`` line from zarrmony's own installed metadata."""
    requires = metadata.requires("zarrmony") or []
    for raw in requires:
        requirement = Requirement(raw)
        if requirement.name == "bioio-czi" and "extra" not in raw:
            return requirement
    raise AssertionError("bioio-czi is not a core requirement of zarrmony")


@pytest.mark.parametrize(
    ("platform_name", "environment", "included"),
    [
        # aicspylibczi ships one Linux wheel family, manylinux x86_64. Every
        # other Linux arch source-builds it, so every other Linux arch is out.
        ("linux-x86_64", {"sys_platform": "linux", "platform_machine": "x86_64"}, True),
        (
            "linux-aarch64",
            {"sys_platform": "linux", "platform_machine": "aarch64"},
            False,
        ),
        (
            "linux-armv7l",
            {"sys_platform": "linux", "platform_machine": "armv7l"},
            False,
        ),
        (
            "linux-ppc64le",
            {"sys_platform": "linux", "platform_machine": "ppc64le"},
            False,
        ),
        ("linux-s390x", {"sys_platform": "linux", "platform_machine": "s390x"}, False),
        # macOS ships arm64 and x86_64 wheels, Windows ships amd64.
        ("macos-arm64", {"sys_platform": "darwin", "platform_machine": "arm64"}, True),
        (
            "macos-x86_64",
            {"sys_platform": "darwin", "platform_machine": "x86_64"},
            True,
        ),
        ("windows-amd64", {"sys_platform": "win32", "platform_machine": "AMD64"}, True),
    ],
)
def test_czi_is_a_core_dependency_wherever_a_wheel_exists(
    platform_name: str, environment: dict[str, str], included: bool
) -> None:
    """Only the Linux arches with no ``aicspylibczi`` wheel resolve without CZI."""
    marker = _czi_core_requirement().marker
    assert marker is not None, "the marker was dropped from the core requirement"
    assert marker.evaluate(environment) is included, platform_name


def test_all_extra_pulls_the_czi_extra_back_in() -> None:
    """``czi`` carries no licensing constraint, so ``all`` must include it."""
    requires = metadata.requires("zarrmony") or []
    all_lines = [raw for raw in requires if 'extra == "all"' in raw]
    assert any("zarrmony[" in raw and "czi" in raw for raw in all_lines), all_lines


def test_dev_extra_does_not_pin_the_backend() -> None:
    """``dev`` must not re-impose the source build on an arm64 dev machine.

    The core marker already installs ``bioio-czi`` on every CI runner, so a
    ``dev`` entry would change only the environment the marker protects.
    """
    requires = metadata.requires("zarrmony") or []
    dev_lines = [raw for raw in requires if 'extra == "dev"' in raw]
    assert not any("bioio-czi" in raw for raw in dev_lines), dev_lines
