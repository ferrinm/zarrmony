from ome_types import OME, from_xml
from ome_types.model import Image, Pixels, PixelType, TiffData

from zarrmony.writers.ome_xml import (
    attach_stage_position_plane,
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


# --- ADR-0005: per-tile <Plane> stage-position stamping ---------------------


def test_attach_stage_position_plane_emits_single_plane_in_um() -> None:
    img = _make_image(0)
    normalize_image_for_metadata_only(img)
    attach_stage_position_plane(
        img, position_x_um=40500.0, position_y_um=17000.0, position_z_um=11700.0
    )
    xml = build_combined_ome_xml([img])
    parsed = from_xml(xml)
    planes = parsed.images[0].pixels.planes
    assert len(planes) == 1
    p = planes[0]
    assert (p.the_c, p.the_z, p.the_t) == (0, 0, 0)
    assert p.position_x == 40500.0
    assert p.position_y == 17000.0
    assert p.position_z == 11700.0
    # Units are explicitly µm (OME convention; LIF stores meters, the caller
    # converts before stamping).
    assert p.position_x_unit.value == "µm"
    assert p.position_y_unit.value == "µm"
    assert p.position_z_unit.value == "µm"


def test_attach_stage_position_plane_allows_none_coordinates() -> None:
    """The LIF extractor may return None for a tile's PosZ — the Plane is still
    emitted with the present axes filled and the missing axis omitted."""
    img = _make_image(0)
    normalize_image_for_metadata_only(img)
    attach_stage_position_plane(
        img, position_x_um=10.0, position_y_um=20.0, position_z_um=None
    )
    parsed = from_xml(build_combined_ome_xml([img]))
    p = parsed.images[0].pixels.planes[0]
    assert p.position_x == 10.0
    assert p.position_y == 20.0
    assert p.position_z is None
