"""Tests for size_on_disk() and format_bytes() helpers."""

from pathlib import Path

import fsspec
import pytest

from zarrmony._storage import format_bytes, size_on_disk


def test_size_on_disk_single_file(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"\x00" * 1234)
    assert size_on_disk(p) == 1234


def test_size_on_disk_directory_tree_recursive(tmp_path: Path) -> None:
    root = tmp_path / "store.zarr"
    (root / "a" / "b").mkdir(parents=True)
    (root / "top.txt").write_bytes(b"x" * 10)
    (root / "a" / "mid.txt").write_bytes(b"y" * 100)
    (root / "a" / "b" / "leaf.txt").write_bytes(b"z" * 1000)
    assert size_on_disk(root) == 1110


def test_size_on_disk_missing_path_returns_zero(tmp_path: Path) -> None:
    assert size_on_disk(tmp_path / "does-not-exist") == 0


def test_size_on_disk_fsspec_memory_uri() -> None:
    fs = fsspec.filesystem("memory")
    # Clean up any leftovers from prior tests sharing the in-process memory FS.
    for path in ("/sz-test/a.bin", "/sz-test/sub/b.bin"):
        if fs.exists(path):
            fs.rm(path)
    with fs.open("/sz-test/a.bin", "wb") as f:
        f.write(b"a" * 500)
    with fs.open("/sz-test/sub/b.bin", "wb") as f:
        f.write(b"b" * 1500)
    try:
        assert size_on_disk("memory:///sz-test") == 2000
    finally:
        fs.rm("/sz-test", recursive=True)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 * 1024, "1.0 MB"),
        (int(2.3 * 1024 * 1024), "2.3 MB"),
        (1024**3, "1.0 GB"),
        (int(2.3 * 1024**3), "2.3 GB"),
        (1024**4, "1.0 TB"),
    ],
)
def test_format_bytes(n: int, expected: str) -> None:
    assert format_bytes(n) == expected
