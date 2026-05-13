"""End-to-end test: write multiple scenes + bf2raw wrapper, then verify the
structure on disk matches the bioformats2raw.layout spec.
"""

import json
from pathlib import Path

import zarr
from ome_types import from_xml
from ome_types.model import Image, Pixels, PixelType

from tests.conftest import FakeReader
from zarrmony.writers.bf2raw import write_bf2raw_wrapper
from zarrmony.writers.ome_xml import build_combined_ome_xml
from zarrmony.writers.scene import write_scene


def _ome_image_for_scene(scene_index: int, name: str) -> Image:
    return Image(
        id=f"Image:{scene_index}",
        name=name,
        pixels=Pixels(
            id=f"Pixels:{scene_index}",
            size_x=32,
            size_y=32,
            size_c=1,
            size_z=1,
            size_t=1,
            dimension_order="XYZCT",
            type=PixelType.UINT16,
        ),
    )


def test_bf2raw_layout_full_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "multiscene.ome.zarr"
    reader = FakeReader(
        scenes=["scene_a", "scene_b"], dims="TCYX", shape=(1, 1, 32, 32)
    )

    series_paths: list[str] = []
    images = []
    for i, name in enumerate(reader.scenes):
        scene_dir = out / str(i)
        write_scene(
            reader, scene_index=i, store_path=str(scene_dir), pyramid_min_size=8
        )
        series_paths.append(str(i))
        images.append(_ome_image_for_scene(i, name))

    ome_xml = build_combined_ome_xml(images)
    write_bf2raw_wrapper(
        out,
        series_paths=series_paths,
        ome_xml=ome_xml,
        source_xml="<root>fake source xml</root>",
        source_xml_filename="raw.fake.xml",
    )

    # Top-level zarr.json has the bioformats2raw.layout key under ome
    with open(out / "zarr.json") as f:
        root_zj = json.load(f)
    ome_block = root_zj["attributes"]["ome"]
    assert ome_block["bioformats2raw.layout"] == 3
    assert ome_block["version"] == "0.5"

    # OME/zarr.json has the series attribute
    with open(out / "OME" / "zarr.json") as f:
        ome_zj = json.load(f)
    ome_attrs = ome_zj["attributes"]["ome"]
    assert ome_attrs["series"] == ["0", "1"]
    assert ome_attrs["version"] == "0.5"

    # OME/METADATA.ome.xml exists and parses to 2 images
    xml = (out / "OME" / "METADATA.ome.xml").read_text()
    parsed = from_xml(xml)
    assert len(parsed.images) == 2
    assert [img.name for img in parsed.images] == ["scene_a", "scene_b"]

    # OME/source/raw.fake.xml exists
    src = (out / "OME" / "source" / "raw.fake.xml").read_text()
    assert "fake source xml" in src

    # Per-scene images are still openable as NGFF
    g0 = zarr.open_group(str(out / "0"), mode="r")
    assert "0" in g0  # level 0 array
    assert g0["0"].shape == (1, 1, 32, 32)


def test_bf2raw_wrapper_without_source_xml(tmp_path: Path) -> None:
    out = tmp_path / "no_source.ome.zarr"
    reader = FakeReader(scenes=["only"], dims="YX", shape=(64, 64))
    write_scene(reader, scene_index=0, store_path=str(out / "0"), pyramid_min_size=128)

    images = [_ome_image_for_scene(0, "only")]
    write_bf2raw_wrapper(
        out,
        series_paths=["0"],
        ome_xml=build_combined_ome_xml(images),
    )

    assert (out / "OME" / "METADATA.ome.xml").exists()
    assert not (out / "OME" / "source").exists()


def test_bf2raw_wrapper_rejects_source_xml_without_filename(tmp_path: Path) -> None:
    import pytest

    out = tmp_path / "bad.ome.zarr"
    reader = FakeReader(scenes=["s"], dims="YX", shape=(64, 64))
    write_scene(reader, scene_index=0, store_path=str(out / "0"), pyramid_min_size=128)

    images = [_ome_image_for_scene(0, "s")]
    with pytest.raises(ValueError, match="source_xml_filename"):
        write_bf2raw_wrapper(
            out,
            series_paths=["0"],
            ome_xml=build_combined_ome_xml(images),
            source_xml="<x/>",
        )
