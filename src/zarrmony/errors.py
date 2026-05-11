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


class ExtractorWarning(UserWarning):
    """A bioio metadata extractor failed during conversion.

    Conversion proceeds without that field and the failure is recorded in the
    audit attrs (``attrs.zarrmony.metadata_warnings``). Surfaced both as a
    Python warning and as a stderr message at conversion time.
    """
