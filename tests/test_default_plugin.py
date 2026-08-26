"""Tests for the built-in ``bioio`` catch-all plugin (``readers/default.py``).

Covers the ``reader_kwargs`` passthrough (#101) and the ADR-0011
``zarrmony[bioformats]`` install hint (#102). ``BioImage`` is monkeypatched
throughout so nothing here needs a real file or a bioio backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from bioio_base.exceptions import UnsupportedFileFormatError
from click.testing import CliRunner

from zarrmony.cli import app
from zarrmony.errors import ReaderKwargError, UnsupportedFormatError
from zarrmony.readers import default as default_mod
from zarrmony.readers.default import _open_default, default_plugin


class _FakeBioImage:
    """Stand-in for ``bioio.BioImage`` that records how it was constructed."""

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.kwargs = kwargs


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Capture every ``BioImage(...)`` construction the plugin performs."""
    recorded: list[tuple[str, dict[str, Any]]] = []

    def _fake(path: str, **kwargs: Any) -> _FakeBioImage:
        recorded.append((path, kwargs))
        return _FakeBioImage(path, **kwargs)

    monkeypatch.setattr(default_mod, "BioImage", _fake)
    return recorded


# --- Passthrough (#101) ---------------------------------------------------


def test_no_kwargs_behaves_as_before(calls: list[tuple[str, dict[str, Any]]]) -> None:
    """The zero-kwarg call is still a bare ``BioImage(str(path))``."""
    _open_default(Path("/tmp/image.tiff"))

    assert calls == [("/tmp/image.tiff", {})]


def test_unknown_keys_pass_through_verbatim(
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """Keys outside the coercion table keep the documented string contract."""
    _open_default(Path("/tmp/image.tiff"), reader_mode="tile", chunk_dims="ZYX")

    assert calls == [("/tmp/image.tiff", {"reader_mode": "tile", "chunk_dims": "ZYX"})]


def test_kwargs_reach_the_plugin_through_the_registry(
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """``ReaderPlugin.open(path, **reader_kwargs)`` no longer raises TypeError."""
    default_plugin.open(Path("/tmp/image.tiff"), dask_tiles="true")

    assert calls == [("/tmp/image.tiff", {"dask_tiles": True})]


def test_unknown_kwarg_raises_backend_typeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised kwarg fails in the backend constructor, not in zarrmony."""

    def _strict(path: str, *, dask_tiles: bool = False) -> _FakeBioImage:
        return _FakeBioImage(path, dask_tiles=dask_tiles)

    monkeypatch.setattr(default_mod, "BioImage", _strict)

    with pytest.raises(TypeError, match="bogus_option"):
        _open_default(Path("/tmp/image.tiff"), bogus_option="1")


# --- Coercions (#101) -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("  TRUE  ", True),
    ],
)
def test_dask_tiles_coerced_from_string(
    calls: list[tuple[str, dict[str, Any]]], raw: str, expected: bool
) -> None:
    _open_default(Path("/tmp/slide.vsi"), dask_tiles=raw)

    assert calls[0][1] == {"dask_tiles": expected}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1024,1024", (1024, 1024)),
        ("1024, 512", (1024, 512)),
        ("1024", (1024, 1024)),
    ],
)
def test_tile_size_coerced_from_string(
    calls: list[tuple[str, dict[str, Any]]], raw: str, expected: tuple[int, int]
) -> None:
    _open_default(Path("/tmp/slide.vsi"), tile_size=raw)

    assert calls[0][1] == {"tile_size": expected}


def test_api_callers_typed_values_pass_through_untouched(
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """A Python caller already has the right types; coercion must not mangle them."""
    _open_default(Path("/tmp/slide.vsi"), dask_tiles=True, tile_size=(2048, 2048))

    assert calls[0][1] == {"dask_tiles": True, "tile_size": (2048, 2048)}


@pytest.mark.parametrize("raw", ["maybe", "", "2"])
def test_malformed_dask_tiles_raises_reader_kwarg_error(raw: str) -> None:
    with pytest.raises(ReaderKwargError, match="dask_tiles"):
        _open_default(Path("/tmp/slide.vsi"), dask_tiles=raw)


@pytest.mark.parametrize("raw", ["1024x1024", "big", "1024,1024,1024", ""])
def test_malformed_tile_size_raises_reader_kwarg_error(raw: str) -> None:
    with pytest.raises(ReaderKwargError) as excinfo:
        _open_default(Path("/tmp/slide.vsi"), tile_size=raw)

    # The message has to be actionable on its own — this is what the CLI prints
    # instead of a traceback.
    assert "tile_size" in str(excinfo.value)
    assert "1024,1024" in str(excinfo.value)


def test_cli_malformed_tile_size_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`zarrmony inspect --reader-kwarg tile_size=bogus` prints, not tracebacks."""
    monkeypatch.setattr(default_mod, "BioImage", _FakeBioImage)
    result = CliRunner().invoke(
        app, ["inspect", "/tmp/slide.vsi", "--reader-kwarg", "tile_size=bogus"]
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "tile_size" in result.output
    assert "Traceback" not in result.output


# --- ADR-0011 install hint (#102) -----------------------------------------


@pytest.fixture
def unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``BioImage`` raise bioio's own no-backend-for-this-file error."""

    def _raise(path: str, **_kwargs: Any) -> Any:
        raise UnsupportedFileFormatError("BioImage", path)

    monkeypatch.setattr(default_mod, "BioImage", _raise)


def test_hint_names_the_bioformats_extra_when_it_is_absent(
    unsupported: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: False)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        _open_default(Path("/tmp/slide.vsi"))

    message = str(excinfo.value)
    assert "/tmp/slide.vsi" in message
    assert 'pip install "zarrmony[bioformats]"' in message
    # The bioio original is chained, not swallowed.
    assert isinstance(excinfo.value.__cause__, UnsupportedFileFormatError)


def test_hint_suppressed_when_bioformats_is_installed(
    unsupported: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bio-Formats is present and still cannot read it — naming the extra is noise."""
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: True)

    with pytest.raises(UnsupportedFileFormatError):
        _open_default(Path("/tmp/slide.vsi"))


def test_bioformats_extra_stays_out_of_all_and_dev() -> None:
    """ADR-0011's GPL constraint, as a test rather than only a comment.

    ``bioio-bioformats`` is GPL-3.0 and zarrmony is Apache-2.0. Folding it into
    ``all`` (or ``dev``) would put GPL code in every "install everything" and
    every CI environment — the failure mode ADR-0011 exists to prevent, and the
    one most likely to arrive as a well-meaning tidy-up.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]

    assert any("bioio-bioformats" in dep for dep in extras["bioformats"])
    for guarded in ("all", "dev"):
        joined = " ".join(extras[guarded])
        assert "bioformats" not in joined, (
            f"the {guarded!r} extra must not pull bioio-bioformats (GPL-3.0) — "
            f"see ADR-0011"
        )


def test_bioformats_installed_probe_reflects_the_environment() -> None:
    """The probe answers about the real interpreter, not a hardcoded constant."""
    import importlib.util

    assert default_mod._bioformats_installed() == (
        importlib.util.find_spec("bioio_bioformats") is not None
    )
