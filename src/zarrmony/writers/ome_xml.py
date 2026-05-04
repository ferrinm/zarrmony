"""OME-XML construction for OME/METADATA.ome.xml.

The bioformats2raw.layout spec requires a combined OME-XML document at
``OME/METADATA.ome.xml`` describing every image in the collection. Per spec, each
Image's Pixels MUST use ``<MetadataOnly/>`` (not BinData / TiffData / BinaryOnly)
because the pixel data lives in the sibling Zarr arrays.
"""

from collections.abc import Iterable

from ome_types import OME
from ome_types.model import Image, MetadataOnly


def normalize_image_for_metadata_only(image: Image) -> Image:
    """Force ``image.pixels`` into MetadataOnly form, dropping any binary refs.

    Mutates the input Image in place and returns it for convenience.
    """
    image.pixels.bin_data_blocks = []
    image.pixels.tiff_data_blocks = []
    image.pixels.metadata_only = MetadataOnly()
    return image


def build_combined_ome_xml(images: Iterable[Image]) -> str:
    """Combine per-scene Image elements into a single OME-XML document.

    The order of images in the returned XML matches the iteration order; that
    same order MUST be reflected in the bf2raw ``OME/series`` attribute.
    """
    images_list = [normalize_image_for_metadata_only(img) for img in images]
    ome = OME(images=images_list)
    return ome.to_xml()
