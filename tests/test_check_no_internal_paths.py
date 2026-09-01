"""Tests for the pre-commit hook that keeps internal identifiers out of the repo.

The hook is the only thing standing between a paste from a lab share and a
public commit, so its patterns are worth pinning — particularly the negative
cases. A rule that fires on ordinary prose gets suppressed with
``# allow-internal-path`` until it stops protecting anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "check_no_internal_paths.py"
)
_spec = importlib.util.spec_from_file_location("check_no_internal_paths", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
sys.modules["check_no_internal_paths"] = check
_spec.loader.exec_module(check)


def scan_text(tmp_path: Path, text: str, name: str = "doc.md") -> list[str]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return check.scan(path, check.RULES)


CAUGHT = [
    pytest.param("/Volumes/microscopy-ro/slide.vsi", id="read-only-mount"),
    pytest.param("cp /data/microscopy/export .", id="cluster-share"),
    pytest.param("run_Trial#1234 finished", id="trial-number"),
    pytest.param("scenes: ('label', '20x_AAA_B_CCC_01')", id="scene-name-underscores"),
    pytest.param("scene '20x_AAA_B, CCC, DDD, EEE_01'", id="scene-name-comma-list"),
    pytest.param("40X_AAA_BBB_CCC_02.ome.zarr", id="scene-name-uppercase-mag"),
]


@pytest.mark.parametrize("line", CAUGHT)
def test_offending_lines_are_reported(tmp_path: Path, line: str) -> None:
    assert scan_text(tmp_path, line), f"expected a finding for {line!r}"


ALLOWED = [
    pytest.param(
        "The slide was imaged at 20x on a widefield scanner.", id="bare-magnification"
    ),
    pytest.param(
        "Main scene: 20×, Z=1, T=1, 10 source pyramid levels.", id="prose-magnification"
    ),
    pytest.param("Sharding was 15.2x fewer objects than the chunk grid.", id="ratio"),
    pytest.param(
        "`20x_DAPI` is two tokens and stays under the threshold.", id="two-tokens"
    ),
    pytest.param("store = f'{OUT}/slide-B/<main-scene>.ome.zarr'", id="placeholder"),
    pytest.param("Set $SRC to the reference dataset's path.", id="env-placeholder"),
    pytest.param("metadata_<dataset>.json", id="dataset-placeholder"),
]


@pytest.mark.parametrize("line", ALLOWED)
def test_ordinary_text_is_not_flagged(tmp_path: Path, line: str) -> None:
    assert scan_text(tmp_path, line) == []


def test_allow_marker_suppresses_a_line(tmp_path: Path) -> None:
    text = f"/data/microscopy/export  # {check.ALLOW_MARKER}"
    assert scan_text(tmp_path, text) == []


def test_unscanned_suffixes_are_skipped(tmp_path: Path) -> None:
    assert scan_text(tmp_path, "/data/microscopy/export", name="notes.rst") == []


def test_the_sanitized_docs_stay_clean() -> None:
    """The two files that carried a real scene name before it was replaced."""
    root = Path(__file__).resolve().parent.parent
    for rel in (
        "docs/references/vsi-acceptance-run.md",
        "docs/adr/0011-bioformats-backed-formats.md",
    ):
        assert check.scan(root / rel, check.RULES) == [], rel
