"""OME-XML construction for OME/METADATA.ome.xml.

In per-scene mode each store carries a single-Image OME-XML document; in
bf2raw mode the wrapper carries one combined OME-XML describing every scene.
Per the OME-Zarr spec, each Image's Pixels MUST use ``<MetadataOnly/>`` (not
BinData / TiffData / BinaryOnly) because the pixel data lives in the sibling
Zarr arrays.

The ``build_*`` helpers also carry an optional ``instruments`` list — the
top-level ``<Instrument>`` element(s) each ``<Image>`` may reference via its
``<InstrumentRef/>`` + ``<ObjectiveSettings/>`` children. See
:func:`build_instrument_from_objective` for the LIF-shaped projection consumed
by the per-scene writer (issue #52).
"""

from collections.abc import Iterable

from ome_types import OME
from ome_types.model import (
    Image,
    Instrument,
    InstrumentRef,
    MetadataOnly,
    Objective,
    ObjectiveSettings,
    Plane,
    UnitsLength,
)


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


def build_combined_ome_xml(
    images: Iterable[Image],
    instruments: Iterable[Instrument] = (),
) -> str:
    """Combine per-scene Image elements into a single OME-XML document.

    The order of images in the returned XML matches the iteration order; that
    same order MUST be reflected in the bf2raw ``OME/series`` attribute.

    ``instruments`` — optional top-level ``<Instrument>`` elements referenced
    by the images via their ``<InstrumentRef/>`` + ``<ObjectiveSettings/>``.
    Emitted before ``<Image>`` per the OME schema's declared order. Default
    empty tuple preserves the pre-#52 no-instrument shape.
    """
    images_list = [normalize_image_for_metadata_only(img) for img in images]
    ome = OME(instruments=list(instruments), images=images_list)
    return ome.to_xml()


def build_ome_xml_for_scene(image: Image, instrument: Instrument | None = None) -> str:
    """Build a single-Image OME-XML document for one per-scene store.

    ``instrument`` — optional top-level ``<Instrument>`` referenced by
    ``image.instrument_ref`` / ``image.objective_settings``. When provided it
    is emitted alongside the image; when ``None`` the output shape is
    unchanged (pre-#52 behaviour).
    """
    return build_combined_ome_xml(
        [image], [instrument] if instrument is not None else []
    )


def build_instrument_from_objective(
    objective: dict,
    *,
    instrument_id: str = "Instrument:0",
    objective_id: str = "Objective:0",
) -> Instrument:
    """Project a LIF-extracted objective dict into an OME ``<Instrument>``.

    Consumes the shape :func:`zarrmony.metadata.objective.extract_objective`
    returns — one dict with any subset of ``nominal_magnification``,
    ``numerical_aperture``, ``immersion``, ``model``, ``working_distance_um``.
    Only present keys are set on the OME ``<Objective>`` element; the OME
    element itself always carries its ``ID`` attribute (required by the schema
    for cross-references from ``<ObjectiveSettings/>``).

    Working distance is emitted with an explicit ``µm`` unit; the caller must
    already have converted whatever LIF's raw value was into micrometers.
    """
    fields: dict = {"id": objective_id}
    if "model" in objective:
        fields["model"] = objective["model"]
    if "nominal_magnification" in objective:
        fields["nominal_magnification"] = float(objective["nominal_magnification"])
    if "numerical_aperture" in objective:
        fields["lens_na"] = float(objective["numerical_aperture"])
    if "immersion" in objective:
        fields["immersion"] = objective["immersion"]
    if "working_distance_um" in objective:
        fields["working_distance"] = float(objective["working_distance_um"])
        fields["working_distance_unit"] = UnitsLength.MICROMETER
    return Instrument(id=instrument_id, objectives=[Objective(**fields)])


def attach_objective_to_image(
    image: Image,
    *,
    instrument: Instrument,
) -> Image:
    """Stamp ``<InstrumentRef/>`` + ``<ObjectiveSettings/>`` onto ``image``.

    The image gains references to ``instrument`` (top-level) and to its first
    (only) objective. Both refs are required for the OME-XML to round-trip:
    ``<InstrumentRef ID="Instrument:..." />`` inside the ``<Image>`` element,
    and ``<ObjectiveSettings ID="Objective:..." />`` (same ID as the objective
    inside the instrument) so consumers know *which* objective on that
    instrument was actually used. Returns ``image`` for chaining; mutates in
    place.
    """
    image.instrument_ref = InstrumentRef(id=instrument.id)
    image.objective_settings = ObjectiveSettings(id=instrument.objectives[0].id)
    return image
