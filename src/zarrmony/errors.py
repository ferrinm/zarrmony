"""Error and warning types for zarrmony."""


class ZarrmonyError(Exception):
    """Base class for all zarrmony exceptions."""


class MetadataValidationError(ZarrmonyError):
    """User-supplied metadata failed the compliance gate.

    Raised by ``convert(...)`` when the gate is enabled (default) and required
    fields are missing or invalid. Pass ``permissive=True`` to bypass.
    """


class OutputExistsError(ZarrmonyError):
    """Refused to overwrite an existing output store. Pass ``force=True`` to overwrite."""


class PlateLayoutError(ZarrmonyError):
    """A reader's :class:`PlateLayout` failed writer-side validation.

    Raised by ``writers.plate`` before any pixels are written when the layout
    is internally inconsistent (a field references a row/column not in the
    plate, two fields land on the same well/path, more than one acquisition
    in v1, etc.). Fail fast — a partially-written plate is worse than no plate.
    """


class LayoutMismatchError(ZarrmonyError):
    """Explicit ``layout='plate'`` was passed against a non-plate-shaped reader.

    Raised by ``convert()`` when the user forces plate output but the reader's
    ``layout_hint`` is not ``"plate"`` (typically ``"flat"``). The flat→plate
    direction has no source of plate structure to invent, so the request is
    rejected. Use ``layout='auto'`` (the default) to let zarrmony pick the
    matching writer for the reader's shape.
    """


class LayoutDowngradeWarning(UserWarning):
    """Plate-shaped metadata is being dropped from the output.

    Emitted by ``convert()`` when an explicit ``layout='per-scene'`` or
    ``layout='bf2raw'`` is passed against a plate-shaped reader (the writer
    runs but plate-level metadata — rows, columns, wells, acquisitions — is
    discarded). Also emitted by ``writers.plate`` when ``reader.scenes``
    contains scenes not referenced by any ``PlateField`` (those scenes are
    not written to the plate store).
    """


class ExtractorWarning(UserWarning):
    """A bioio metadata extractor failed during conversion.

    Conversion proceeds without that field and the failure is recorded in the
    audit attrs (``attrs.zarrmony.metadata_warnings``). Surfaced both as a
    Python warning and as a stderr message at conversion time.
    """


class MosaicStitchingWarning(UserWarning):
    """A reader plugin auto-stitched a mosaic with a known-imprecise stitcher.

    Currently emitted by the ``bioio-lif`` plugin: ``bioio_lif.Reader``'s
    stitcher hardcodes a 1-pixel inter-tile overlap and ignores the actual
    stage XY positions stored in the LIF metadata. For acquisitions with
    non-trivial overlap (the typical 5–15% used for content-aware stitching),
    the output has double-coverage stripes at every tile seam and is unfit
    for quantitative analysis at tile boundaries. Prefer a vendor-stitched
    sibling scene (Leica's ``*_Merged``) when present, or an external
    stitcher (ASHLAR, m2stitch, BigStitcher) otherwise.
    """
