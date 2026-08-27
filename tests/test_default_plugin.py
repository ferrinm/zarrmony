"""Tests for the built-in ``bioio`` catch-all plugin (``readers/default.py``).

Covers the ``reader_kwargs`` passthrough (#101), the ADR-0011
``zarrmony[bioformats]`` install hint (#102), and the post-mortem that keeps
bioio's one-size-fits-all dispatch failure from misdiagnosing an unreadable
input as a missing reader. ``BioImage`` is monkeypatched throughout so nothing
here needs a real backend.
"""

from __future__ import annotations

import errno
import logging
from pathlib import Path
from typing import Any

import pytest
from bioio_base.exceptions import UnsupportedFileFormatError
from click.testing import CliRunner

from zarrmony.cli import app
from zarrmony.errors import (
    InputAccessError,
    ReaderKwargError,
    UnsupportedFormatError,
)
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


@pytest.fixture
def readable_slide(tmp_path: Path) -> Path:
    """A real, readable file — so the access probe stays out of the way."""
    slide = tmp_path / "slide.vsi"
    slide.write_bytes(b"not really a VSI")
    return slide


def test_hint_names_the_bioformats_extra_when_it_is_absent(
    unsupported: None, readable_slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: False)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        _open_default(readable_slide)

    message = str(excinfo.value)
    assert str(readable_slide) in message
    assert 'pip install "zarrmony[bioformats]"' in message
    # The bioio original is chained, not swallowed.
    assert isinstance(excinfo.value.__cause__, UnsupportedFileFormatError)


def test_hint_suppressed_when_bioformats_is_installed(
    unsupported: None, readable_slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bio-Formats is present and still cannot read it — naming the extra is noise."""
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: True)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        _open_default(readable_slide)

    message = str(excinfo.value)
    assert 'pip install "zarrmony[bioformats]"' not in message
    assert "bioio-bioformats is installed" in message


# --- distinguishing "unreadable" from "unsupported" -------------------------


def test_unreadable_input_is_not_reported_as_a_missing_reader(
    unsupported: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this exists for: EPERM must not be advertised as a missing extra.

    bioio raises ``UnsupportedFileFormatError`` for every dispatch failure,
    including ones where no backend could have succeeded. Telling the user to
    install a reader for a file the OS will not hand over sends them down a
    dead end.
    """
    slide = tmp_path / "slide.vsi"
    slide.write_bytes(b"x")
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: False)

    def _deny(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(errno.EPERM, "Operation not permitted", str(slide))

    monkeypatch.setattr(default_mod, "open", _deny, raising=False)

    with pytest.raises(InputAccessError) as excinfo:
        _open_default(slide)

    message = str(excinfo.value)
    assert str(slide) in message
    assert "Operation not permitted" in message
    assert "zarrmony[bioformats]" not in message


def test_missing_input_says_so(
    unsupported: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: False)

    with pytest.raises(InputAccessError) as excinfo:
        _open_default(tmp_path / "absent.vsi")

    assert "does not exist" in str(excinfo.value)


def test_bioios_filenotfound_is_replaced_with_a_legible_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path a real missing input takes.

    bioio raises ``FileNotFoundError`` before dispatch and stringifies the
    fsspec protocol tuple into the message ("('file', 'local'):///x.vsi").
    The CLI takes INPUT as a raw string — remote URLs are legal — so nothing
    has checked the path before this point.
    """
    absent = tmp_path / "absent.vsi"

    def _raise(path: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError(f"('file', 'local'):///{path}")

    monkeypatch.setattr(default_mod, "BioImage", _raise)

    with pytest.raises(InputAccessError) as excinfo:
        _open_default(absent)

    assert "does not exist" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_filenotfound_about_some_other_file_is_left_alone(
    readable_slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend missing a *sibling* knows more than our probe does.

    The input is right there and readable, so rewriting the error to say it is
    missing would be a lie. Multi-file formats hit this whenever a companion
    file is absent.
    """

    def _raise(path: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError("missing companion: slide.ets")

    monkeypatch.setattr(default_mod, "BioImage", _raise)

    with pytest.raises(FileNotFoundError, match="companion"):
        _open_default(readable_slide)


def test_macos_eperm_names_the_privacy_layer(
    unsupported: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPERM on macOS is TCC, not the file mode — the fix is a different one."""
    slide = tmp_path / "slide.vsi"
    slide.write_bytes(b"x")
    monkeypatch.setattr(default_mod.sys, "platform", "darwin")

    def _deny(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(errno.EPERM, "Operation not permitted", str(slide))

    monkeypatch.setattr(default_mod, "open", _deny, raising=False)

    with pytest.raises(InputAccessError) as excinfo:
        _open_default(slide)

    assert "Full Disk Access" in str(excinfo.value)


def test_eacces_does_not_claim_the_macos_privacy_layer(
    unsupported: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain mode-bit denial is a plain mode-bit denial, even on macOS."""
    slide = tmp_path / "slide.vsi"
    slide.write_bytes(b"x")
    monkeypatch.setattr(default_mod.sys, "platform", "darwin")

    def _deny(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(errno.EACCES, "Permission denied", str(slide))

    monkeypatch.setattr(default_mod, "open", _deny, raising=False)

    with pytest.raises(InputAccessError) as excinfo:
        _open_default(slide)

    message = str(excinfo.value)
    assert "Permission denied" in message
    assert "Full Disk Access" not in message


def test_directory_inputs_skip_the_readability_probe(
    unsupported: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bf2raw bundles and Zarr stores are directories — not access failures."""
    bundle = tmp_path / "bundle.zarr"
    bundle.mkdir()
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: False)

    with pytest.raises(UnsupportedFormatError):
        _open_default(bundle)


def test_remote_urls_skip_the_readability_probe(
    unsupported: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_reader`` hands us a ``Path`` even for s3://; do not stat it."""
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: False)

    with pytest.raises(UnsupportedFormatError):
        _open_default(Path("s3://bucket/slide.vsi"))


def test_readable_file_in_an_unlistable_directory_is_diagnosed(
    unsupported: None, readable_slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VSI's failure mode: the .vsi opens, its sibling pyramid cannot be found."""
    monkeypatch.setattr(default_mod, "_parent_is_unlistable", lambda _p: True)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        _open_default(readable_slide)

    message = str(excinfo.value)
    assert str(readable_slide.parent) in message
    assert "multi-file formats" in message


def test_parent_listability_probe_reflects_the_filesystem(tmp_path: Path) -> None:
    slide = tmp_path / "slide.vsi"
    slide.write_bytes(b"x")

    assert default_mod._parent_is_unlistable(slide) is False
    assert default_mod._parent_is_unlistable(tmp_path / "gone" / "slide.vsi") is True


# --- surfacing what bioio tried and discarded -------------------------------


def test_backend_failures_bioio_logs_are_folded_into_the_message(
    readable_slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bioio logs each backend's real error, then raises a generic one.

    Without capturing the log line the user is told "install an extra" when a
    backend was in fact tried and died for an unrelated reason — the exact
    case that motivated this: a JVM ``FileNotFoundException`` on an SMB mount.
    """
    java_error = (
        f"Attempted file ({readable_slide}) load with reader: "
        f"<class 'bioio_bioformats.reader.Reader'> failed with error: "
        f"java.io.FileNotFoundException: {readable_slide} (Operation not permitted)"
    )

    def _log_then_raise(path: str, **_kwargs: Any) -> Any:
        logging.getLogger(default_mod._BIOIO_DISPATCH_LOGGER).warning(java_error)
        raise UnsupportedFileFormatError("BioImage", path)

    monkeypatch.setattr(default_mod, "BioImage", _log_then_raise)
    monkeypatch.setattr(default_mod, "_bioformats_installed", lambda: True)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        _open_default(readable_slide)

    message = str(excinfo.value)
    assert "bioio tried:" in message
    assert "java.io.FileNotFoundException" in message


def test_capture_ignores_other_files_open_concurrently(
    readable_slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bioio logger is process-wide; another file's failure is not ours."""

    def _log_then_raise(path: str, **_kwargs: Any) -> Any:
        logging.getLogger(default_mod._BIOIO_DISPATCH_LOGGER).warning(
            "Attempted file (/somewhere/else.czi) load with reader: X failed"
        )
        raise UnsupportedFileFormatError("BioImage", path)

    monkeypatch.setattr(default_mod, "BioImage", _log_then_raise)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        _open_default(readable_slide)

    assert "else.czi" not in str(excinfo.value)


def test_capture_handler_is_always_removed(
    unsupported: None, readable_slide: Path
) -> None:
    """A handler leaked per failed open would accumulate across a batch run."""
    logger = logging.getLogger(default_mod._BIOIO_DISPATCH_LOGGER)
    before = list(logger.handlers)

    with pytest.raises(UnsupportedFormatError):
        _open_default(readable_slide)

    assert logger.handlers == before


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
