from ome_types import OME, from_xml
from ome_types.model import Image, Pixels, PixelType, TiffData

from zarrmony.writers.ome_xml import (
    build_combined_ome_xml,
    normalize_image_for_metadata_only,
)


def _make_image(idx: int) -> Image:
    return Image(
        id=f"Image:{idx}",
        name=f"scene_{idx}",
        pixels=Pixels(
            id=f"Pixels:{idx}",
            size_x=10,
            size_y=10,
            size_c=1,
            size_z=1,
            size_t=1,
            dimension_order="XYZCT",
            type=PixelType.UINT16,
        ),
    )


def test_normalize_strips_binary_refs() -> None:
    img = _make_image(0)
    img.pixels.tiff_data_blocks = [TiffData()]
    normalize_image_for_metadata_only(img)
    assert img.pixels.tiff_data_blocks == []
    assert img.pixels.bin_data_blocks == []
    assert img.pixels.metadata_only is not None


def test_combined_xml_has_n_images() -> None:
    images = [_make_image(i) for i in range(3)]
    xml = build_combined_ome_xml(images)
    parsed = from_xml(xml)
    assert isinstance(parsed, OME)
    assert len(parsed.images) == 3
    assert [img.name for img in parsed.images] == ["scene_0", "scene_1", "scene_2"]


def test_combined_xml_uses_metadata_only() -> None:
    images = [_make_image(0)]
    xml = build_combined_ome_xml(images)
    assert "<MetadataOnly/>" in xml
    assert "<TiffData" not in xml
    assert "<BinData" not in xml


def test_image_order_preserved() -> None:
    # Order matters: bf2raw spec requires OME/series order matches XML order
    images = [_make_image(i) for i in [2, 0, 1]]
    xml = build_combined_ome_xml(images)
    parsed = from_xml(xml)
    assert [img.id for img in parsed.images] == ["Image:2", "Image:0", "Image:1"]
