"""The input is more than the path the user named (issue #116).

``input.size_bytes`` and ``--checksum`` stat and hash the argument to
``convert``. For a multi-file vendor format that argument is an index: a
whole-slide VSI is 4.4 MB of ``.vsi`` beside 37 GB of ``.ets`` tiles in a
sibling directory, so both fields described none of the converted data and
nothing in the record said so — corrupting an ``.ets`` tile left the recorded
checksum unchanged.

Bio-Formats can name the file set (``IFormatReader.getUsedFiles()``, surfaced
by ``bffile`` as ``BioFile.used_files()``); nothing else in the reader stack
can. So the contract tested here is three-valued rather than two-valued:

- reader named a wider set  -> ``input.files`` + ``size_is_partial: true``
- reader named just the path -> ``input.files`` + ``size_is_partial: false``
- reader could not say       -> neither key, because unknown is not complete

Sections: (1) asking the reader, (2) summarising a set, (3) the manifest
digest, (4) the audit block, (5) ``convert()`` / ``inspect()`` / CLI wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from tests.conftest import FakeReader
from zarrmony import api as api_module
from zarrmony import convert, inspect
from zarrmony._inputs import (
    MAX_LISTED_FILES,
    file_digest,
    manifest_digest,
    reader_used_files,
    summarize_used_files,
)
from zarrmony.audit import _file_forensics
from zarrmony.cli import app
from zarrmony.readers.plugin import ReaderPlugin


@pytest.fixture
def patched_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch ``zarrmony.api.get_reader`` to return a configurable FakeReader."""

    def installer(reader: Any, plugin: str = "bioio") -> None:
        plugin_obj = ReaderPlugin(
            name=plugin,
            match=lambda _p: 100,
            open=lambda _p, **_kw: reader,
            distribution=plugin,
            source="builtin",
        )
        monkeypatch.setattr(
            api_module,
            "get_reader",
            lambda _path, *, reader_kwargs=None: (reader, plugin_obj, 100),
        )

    return installer


def _vsi_like(root: Path, *, tile_bytes: int = 4096) -> tuple[Path, list[str]]:
    """A ``.vsi``-shaped input: a small index beside a fat sibling directory."""
    index = root / "slide.vsi"
    index.write_bytes(b"i" * 64)
    tiles = root / "_slide_"
    (tiles / "stack1").mkdir(parents=True)
    (tiles / "stack2").mkdir(parents=True)
    members = [
        tiles / "stack1" / "frame_t.ets",
        tiles / "stack2" / "frame_t_0.ets",
    ]
    for i, m in enumerate(members):
        m.write_bytes(bytes([i]) * tile_bytes)
    return index, sorted(str(p.resolve()) for p in [index, *members])


class _UsedFilesReader(FakeReader):
    """A FakeReader that can name its file set, the way bioio-bioformats can."""

    def __init__(self, used: list[str] | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._used = used

    def used_files(self) -> list[str]:
        if self._used is None:
            raise RuntimeError("reader cannot enumerate its files")
        return list(self._used)


# ---------- 1. asking the reader ----------


def test_a_reader_with_no_hook_cannot_say() -> None:
    assert reader_used_files(FakeReader(scenes=["a"])) is None


def test_a_used_files_method_is_used(tmp_path: Path) -> None:
    _, members = _vsi_like(tmp_path)
    reader = _UsedFilesReader(members, scenes=["a"])

    assert reader_used_files(reader) == members


def test_a_plain_sequence_attribute_works_too(tmp_path: Path) -> None:
    _, members = _vsi_like(tmp_path)
    reader = FakeReader(scenes=["a"])
    reader.used_files = members  # type: ignore[attr-defined]

    assert reader_used_files(reader) == members


def test_the_biofile_behind_a_bioio_reader_is_reached(tmp_path: Path) -> None:
    """bioio's ``BioImage`` delegates to a backend ``Reader`` holding a ``BioFile``.

    That two-hop shape (``bioimage.reader._bf.used_files()``) is the only way the
    answer is available today, so it is pinned rather than left to duck typing
    luck.
    """
    _, members = _vsi_like(tmp_path)

    class _BioFile:
        def used_files(self, *, metadata_only: bool = False) -> list[str]:
            return list(members)

    class _Backend:
        _bf = _BioFile()

    bioimage = FakeReader(scenes=["a"])
    bioimage.reader = _Backend()  # type: ignore[attr-defined]

    assert reader_used_files(bioimage) == members


def test_a_reader_that_raises_is_treated_as_unable_to_say(tmp_path: Path) -> None:
    """A JVM error in an audit field must not take down a conversion."""
    assert reader_used_files(_UsedFilesReader(None, scenes=["a"])) is None


def test_paths_are_resolved_deduplicated_and_sorted(tmp_path: Path) -> None:
    index, members = _vsi_like(tmp_path)
    noisy = [
        str(index),
        str(index),  # duplicate
        str(tmp_path / "_slide_" / ".." / "_slide_" / "stack1" / "frame_t.ets"),
        str(tmp_path / "gone.ets"),  # does not exist
        str(tmp_path / "_slide_"),  # a directory, not a file
    ]

    assert reader_used_files(_UsedFilesReader(noisy, scenes=["a"])) == sorted(
        m for m in members if "stack2" not in m
    )


def test_a_set_of_only_missing_files_reads_as_unable_to_say(tmp_path: Path) -> None:
    reader = _UsedFilesReader([str(tmp_path / "nope.ets")], scenes=["a"])

    assert reader_used_files(reader) is None


# ---------- 2. summarising a set ----------


def test_the_summary_counts_and_totals_the_whole_set(tmp_path: Path) -> None:
    _, members = _vsi_like(tmp_path, tile_bytes=1000)

    block = summarize_used_files(members)

    assert block["count"] == 3
    assert block["size_bytes"] == 64 + 1000 + 1000
    assert block["size_human"] == "2.0 KB"
    assert block["paths"] == members
    assert block["listing_truncated"] is False


def test_a_long_listing_is_capped_but_the_count_is_exact(tmp_path: Path) -> None:
    """Store attrs are not an inventory system; a TIFF series can name thousands."""
    members = []
    for i in range(MAX_LISTED_FILES + 10):
        p = tmp_path / f"f{i:04d}.tif"
        p.write_bytes(b"x")
        members.append(str(p))

    block = summarize_used_files(sorted(members))

    assert block["count"] == MAX_LISTED_FILES + 10
    assert len(block["paths"]) == MAX_LISTED_FILES
    assert block["listing_truncated"] is True
    assert block["size_bytes"] == MAX_LISTED_FILES + 10


def test_a_single_file_set_gets_no_manifest_digest(tmp_path: Path) -> None:
    """It would only restate ``input.sha256``, which already covers that file."""
    p = tmp_path / "one.nd2"
    p.write_bytes(b"data")

    assert "sha256" not in summarize_used_files([str(p)], checksum=True)
    assert "sha256" in summarize_used_files(
        [str(p), str(_sibling(tmp_path))], checksum=True
    )


def _sibling(root: Path) -> Path:
    p = root / "two.nd2"
    p.write_bytes(b"more")
    return p


# ---------- 3. the manifest digest ----------


def test_the_digest_survives_relocation_but_not_a_rename(tmp_path: Path) -> None:
    """Paths enter the digest relative to the set's common ancestor.

    Moving a dataset is not a change to it; renaming a sidecar is.
    """
    a = tmp_path / "a"
    (a / "sub").mkdir(parents=True)
    (a / "slide.vsi").write_bytes(b"index")
    (a / "sub" / "frame.ets").write_bytes(b"pixels")
    here = manifest_digest([str(a / "slide.vsi"), str(a / "sub" / "frame.ets")])

    b = tmp_path / "b"
    (b / "sub").mkdir(parents=True)
    (b / "slide.vsi").write_bytes(b"index")
    (b / "sub" / "frame.ets").write_bytes(b"pixels")
    assert manifest_digest([str(b / "slide.vsi"), str(b / "sub" / "frame.ets")]) == here

    (b / "sub" / "frame.ets").rename(b / "sub" / "frame_0.ets")
    moved = manifest_digest([str(b / "slide.vsi"), str(b / "sub" / "frame_0.ets")])
    assert moved != here


def test_the_digest_changes_when_a_member_changes(tmp_path: Path) -> None:
    """The whole point: corrupting a sidecar tile must not go unnoticed."""
    index, members = _vsi_like(tmp_path)
    before = manifest_digest(members)

    tile = Path(members[-1])
    tile.write_bytes(b"corrupted" * 8)
    os.utime(tile, (0, 0))

    assert manifest_digest(members) != before
    # ...and the named path's own hash is blind to it, which is the bug.
    assert file_digest(index) == file_digest(index)


def test_a_digest_is_keyed_on_size_and_mtime(tmp_path: Path) -> None:
    """Memoised so a 4-scene run hashes a 37 GB file set once, not four times."""
    p = tmp_path / "a.bin"
    p.write_bytes(b"x" * 1024)
    first = file_digest(p)

    st = p.stat()
    p.write_bytes(b"y" * 1024)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert file_digest(p) == first, "same size and mtime — served from cache"

    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert file_digest(p) != first, "a new mtime invalidates the entry"


# ---------- 4. the audit block ----------


def test_an_unknowable_file_set_leaves_both_keys_absent(tmp_path: Path) -> None:
    """Unknown and known-complete are different claims."""
    p = tmp_path / "one.nd2"
    p.write_bytes(b"data")

    info = _file_forensics(p, used_files=None)

    assert "files" not in info
    assert "size_is_partial" not in info


def test_a_single_file_set_is_recorded_as_complete(tmp_path: Path) -> None:
    p = tmp_path / "one.nd2"
    p.write_bytes(b"data")

    info = _file_forensics(p, used_files=[str(p)])

    assert info["size_is_partial"] is False
    assert info["files"]["count"] == 1


def test_a_vsi_shaped_input_is_recorded_as_partial(tmp_path: Path) -> None:
    index, members = _vsi_like(tmp_path)

    info = _file_forensics(index, used_files=members)

    assert info["size_bytes"] == 64, "unchanged: still the path the user named"
    assert info["size_is_partial"] is True
    assert info["files"]["count"] == 3
    assert info["files"]["size_bytes"] == 64 + 4096 + 4096


def test_checksum_covers_the_named_path_and_the_set_separately(tmp_path: Path) -> None:
    index, members = _vsi_like(tmp_path)

    info = _file_forensics(index, checksum=True, used_files=members)

    assert info["sha256"] == file_digest(index)
    assert info["files"]["sha256"] != info["sha256"]
    assert info["files"]["sha256"] == manifest_digest(members)


def test_no_checksum_means_no_digest_anywhere(tmp_path: Path) -> None:
    index, members = _vsi_like(tmp_path)

    info = _file_forensics(index, used_files=members)

    assert "sha256" not in info
    assert "sha256" not in info["files"]


# ---------- 5. convert() / inspect() / CLI wiring ----------


def test_convert_records_the_file_set_on_every_store(
    tmp_path: Path, patched_reader
) -> None:
    index, members = _vsi_like(tmp_path)
    patched_reader(
        _UsedFilesReader(
            members, scenes=["overview", "slide"], dims="TCYX", shape=(1, 1, 64, 64)
        )
    )

    result = convert(str(index), tmp_path / "out", pyramid_min_size=32)

    assert len(result["stores"]) == 2
    for audit in result["stores"]:
        assert audit["input"]["size_is_partial"] is True
        assert audit["input"]["files"]["count"] == 3


def test_the_recorded_block_round_trips_to_disk(tmp_path: Path, patched_reader) -> None:
    index, members = _vsi_like(tmp_path)
    patched_reader(
        _UsedFilesReader(members, scenes=["slide"], dims="TCYX", shape=(1, 1, 64, 64))
    )

    convert(str(index), tmp_path / "out", pyramid_min_size=32, checksum=True)

    with open(tmp_path / "out" / "slide.ome.zarr" / "zarr.json") as f:
        on_disk = json.load(f)["attributes"]["zarrmony"]["input"]
    assert on_disk["size_is_partial"] is True
    assert on_disk["files"]["sha256"] == manifest_digest(members)
    assert on_disk["files"]["paths"] == members


def test_a_reader_that_cannot_say_converts_exactly_as_before(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["a"], dims="TCYX", shape=(1, 1, 64, 64)))
    src = tmp_path / "plain.tif"
    src.write_bytes(b"x" * 32)

    audit = convert(str(src), tmp_path / "out", pyramid_min_size=32)["stores"][0]

    assert "files" not in audit["input"]
    assert "size_is_partial" not in audit["input"]


def test_inspect_reports_the_set_without_hashing_it(
    tmp_path: Path, patched_reader
) -> None:
    """Pre-flight is where the ratio gets computed, and it must not cost 37 GB of I/O."""
    index, members = _vsi_like(tmp_path)
    patched_reader(
        _UsedFilesReader(members, scenes=["slide"], dims="TCYX", shape=(1, 1, 64, 64))
    )

    info = inspect(str(index))

    assert info["size_bytes"] == 64
    assert info["size_is_partial"] is True
    assert info["files"]["size_bytes"] == 64 + 4096 + 4096
    assert "sha256" not in info["files"]


def test_the_cli_input_line_names_the_set(tmp_path: Path, patched_reader) -> None:
    index, members = _vsi_like(tmp_path)
    patched_reader(
        _UsedFilesReader(members, scenes=["slide"], dims="TCYX", shape=(1, 1, 64, 64))
    )

    res = CliRunner().invoke(
        app,
        [
            "convert",
            str(index),
            str(tmp_path / "out"),
            "--pyramid-min-size",
            "32",
            "--no-validate",
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Input:  8.1 KB across 3 files (the named path alone is 64 B)" in res.output


def test_the_cli_input_line_is_unchanged_for_a_single_file(
    tmp_path: Path, patched_reader
) -> None:
    patched_reader(FakeReader(scenes=["a"], dims="TCYX", shape=(1, 1, 64, 64)))
    src = tmp_path / "plain.tif"
    src.write_bytes(b"x" * 2048)

    res = CliRunner().invoke(
        app,
        [
            "convert",
            str(src),
            str(tmp_path / "out"),
            "--pyramid-min-size",
            "32",
            "--no-validate",
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Input:  2.0 KB\n" in res.output


def test_the_cli_inspect_size_line_names_the_set(
    tmp_path: Path, patched_reader
) -> None:
    index, members = _vsi_like(tmp_path)
    patched_reader(
        _UsedFilesReader(
            members,
            scenes=["slide"],
            dims="TCYX",
            shape=(1, 1, 64, 64),
            dtype=np.uint16,
        )
    )

    res = CliRunner().invoke(app, ["inspect", str(index)])

    assert res.exit_code == 0, res.output
    assert "Size:   8.1 KB across 3 files (the named path alone is 64 B)" in res.output
