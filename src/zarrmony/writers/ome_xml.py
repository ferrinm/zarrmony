"""OME-XML construction for OME/METADATA.ome.xml.

In per-scene mode each store carries a single-Image OME-XML document; in
bf2raw mode the wrapper carries one combined OME-XML describing every scene.
Per the OME-Zarr spec, each Image's Pixels MUST use ``<MetadataOnly/>`` (not
BinData / TiffData / BinaryOnly) because the pixel data lives in the sibling
Zarr arrays.
"""

from collections.abc import Iterable

from ome_types import OME
from ome_types.model import Image, MetadataOnly, Plane, UnitsLength


def normalize_image_for_metadata_only(image: Image) -> Image:
    """Force ``image.pixels`` into MetadataOnly form, dropping any binary refs.

    Mutates the input Image in place and returns it for convenience.
    """
    image.pixels.bin_data_blocks = []
    image.pixels.tiff_data_blocks = []
    image.pixels.metadata_only = MetadataOnly()
    return image


def attach_stage_position_plane(
    image: Image,
    *,
    position_x_um: float | None,
    position_y_um: float | None,
    position_z_um: float | None,
) -> Image:
    """Stamp a single ``<Plane TheC=0 TheZ=0 TheT=0 PositionX/Y/Z .../>`` on ``image``.

    Used by the per-tile LIF mosaic writer (ADR-0005) to record each tile's
    stage origin. The OME spec scopes ``<Plane>`` to a (TheC, TheZ, TheT)
    triple; for the per-tile case the stage position is per-tile, not per-plane,
    so we stamp one Plane at ``(0,0,0)`` — downstream stitchers (ASHLAR,
    m2stitch, BigStitcher) read PositionX/Y from this single Plane element to
    register the tile. Units are explicitly ``micrometer`` per OME convention
    (LIF stores meters; the caller is responsible for the unit conversion).

    Returns ``image`` for chaining; mutates in place.
    """
    plane = Plane(
        the_c=0,
        the_z=0,
        the_t=0,
        position_x=position_x_um,
        position_y=position_y_um,
        position_z=position_z_um,
        position_x_unit=UnitsLength.MICROMETER,
        position_y_unit=UnitsLength.MICROMETER,
        position_z_unit=UnitsLength.MICROMETER,
    )
    image.pixels.planes = [plane]
    return image


def build_combined_ome_xml(images: Iterable[Image]) -> str:
    """Combine per-scene Image elements into a single OME-XML document.

    The order of images in the returned XML matches the iteration order; that
    same order MUST be reflected in the bf2raw ``OME/series`` attribute.
    """
    images_list = [normalize_image_for_metadata_only(img) for img in images]
    ome = OME(images=images_list)
    return ome.to_xml()


def build_ome_xml_for_scene(image: Image) -> str:
    """Build a single-Image OME-XML document for one per-scene store."""
    return build_combined_ome_xml([image])
