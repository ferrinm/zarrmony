"""Error and warning types for zarrmony."""


class ZarrmonyError(Exception):
    """Base class for all zarrmony exceptions."""


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


class PlateSelectionError(ZarrmonyError):
    """A multi-plate LIF requires an explicit ``--plate NAME`` (or ``plate=``) selector.

    Raised by ``convert()`` when the input LIF carries more than one plate
    template and the user did not pass a ``--plate`` selector, OR by the
    LIF reader plugin when ``--plate NAME`` (either on multi-plate or
    single-plate LIF) names a plate that isn't present in the file. The
    message enumerates the available plate names so the user can re-invoke
    with the right selector. Per ADR-0009, one ``convert()`` call still
    produces one plate.zarr — the user runs ``convert`` once per plate.
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
    sibling scene (Leica's ``*_Merged``) when present, an external stitcher
    (ASHLAR, m2stitch, BigStitcher), or re-run with ``lif_mosaic="per-tile"``
    to write each tile as its own OME-Zarr with stage positions in the
    OME-XML ``<Plane>`` (ADR-0005).
    """


class ValidationWarning(UserWarning):
    """The OME-NGFF post-conversion validator flagged a problem with the output.

    Emitted by ``convert()`` when ``validate=True`` (the default if the
    ``zarrmony[validate]`` extra is installed) and ``ome-zarr-models`` reports
    a validation error against the v0.5 spec for the written store. Recorded
    in the audit attrs (``attrs.zarrmony.validation_warnings``). Conversion
    is *not* failed — the validator has known gaps (the ``omero`` block and
    the per-series subgroups of a ``bioformats2raw.layout`` bundle are not
    validated) so deleting the output on a false positive would be worse
    than letting the user re-inspect.
    """


class MosaicPlacementWarning(UserWarning):
    """Stage-based tile placement diverges from the LIF-declared intended overlap.

    Emitted by ``convert(..., lif_mosaic="stage-stitch")`` when the observed
    tile-to-tile overlap — computed from stage ``PosX``/``PosY`` positions and
    the scene's physical pixel size — differs from the LIF metadata's
    ``StitchingSettings/OverlapPercentageX/Y`` by more than 20% on either axis.
    Placement proceeds; the warning names the discrepancy so users can catch
    pixel-size / unit-conversion bugs without failing the conversion.
    """


class MosaicMergedSiblingWarning(UserWarning):
    """A mosaic scene was skipped because a vendor ``_Merged`` sibling exists.

    Emitted by ``convert()`` when the reader signals (via ``skip_reason``)
    that the current scene's pixels are already available, pre-stitched, on
    a sibling scene. Currently triggered by the ``bioio-lif`` plugin when a
    LIF mosaic scene named ``X`` has a sibling scene named ``X_Merged`` —
    the merged sibling is written instead, so we avoid the bioio-lif
    stitcher's 1-pixel-overlap assumption (see
    :class:`MosaicStitchingWarning`).
    """


class ChannelColorCollisionWarning(UserWarning):
    """Two or more channels resolved to the same display color.

    Emitted by ``zarrmony.metadata.channel_colors.assign_colors`` when the
    emission-band scheme (ADR-0007) lands two channels in the same colorblind
    slot — for example, two far-red dyes both mapping to white. The first
    channel in acquisition order keeps its natural band color; later channels
    round-robin through ``UNKNOWN_PALETTE`` skipping already-assigned colors
    and the warning names both the reassigned channel and the color it would
    have taken. Users who want deterministic per-channel colors pass
    ``convert(..., channel_colors={<channel>: "<hex6>"})`` to override.
    """
