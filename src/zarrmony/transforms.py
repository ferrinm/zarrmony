"""Axes normalization to OME-Zarr v0.5 canonical order [t?, c?, z?, y, x].

Per the OME-Zarr 0.5 spec, axes must be ordered by type — time first (if present),
then channel/custom, then space in zyx order. bioio readers do not always return
dims in this order; this module rotates them and records what was done so the
audit trail can show users why their dim order changed.

Colour images arrive with a sixth axis. Bio-Formats models an RGB plane as one
*channel* of three interleaved *samples* — ``C=1, S=3`` — but NGFF has no
samples axis, and its convention for colour is three channels carrying the
primaries. :func:`fold_samples_axis` performs that conversion before the order
check, so whole-slide inputs (which almost always carry an RGB ``label`` and
``macro`` scene beside the fluorescence scan) do not fail as "unsupported
axes".
"""

import xarray as xr

from zarrmony.metadata.channel_colors import sample_axis_channels

NGFF_AXIS_ORDER: tuple[str, ...] = ("T", "C", "Z", "Y", "X")

#: Bio-Formats' interleaved-samples axis. Not an NGFF axis; folded into ``C``.
SAMPLES_AXIS: str = "S"

NGFF_AXIS_TYPE: dict[str, str] = {
    "T": "time",
    "C": "channel",
    "Z": "space",
    "Y": "space",
    "X": "space",
}

NGFF_AXIS_UNIT: dict[str, str | None] = {
    "T": "second",
    "C": None,
    "Z": "micrometer",
    "Y": "micrometer",
    "X": "micrometer",
}


class UnsupportedAxesError(ValueError):
    """Raised when input dims contain axes outside the v0.1 supported set."""


def fold_samples_axis(xarr: xr.DataArray) -> tuple[xr.DataArray, bool]:
    """Turn an interleaved samples axis into NGFF channels.

    Returns ``(xarr, folded)``. ``folded`` is True only when ``S`` actually
    became ``C``; a degenerate ``S=1`` is dropped instead, since a one-sample
    "colour" image is just greyscale and folding it would leave a channel axis
    the caller then has to explain.

    The fold needs ``C`` to be absent or singleton. Bio-Formats guarantees that
    for interleaved data — samples live *inside* one channel — so a scene with
    both ``C>1`` and ``S>1`` is not something we can flatten without inventing
    an ordering between them, and it raises rather than guessing.
    """
    if SAMPLES_AXIS not in xarr.dims:
        return xarr, False

    n_samples = int(xarr.sizes[SAMPLES_AXIS])
    if n_samples == 1:
        return xarr.squeeze(SAMPLES_AXIS, drop=True), False

    if "C" in xarr.dims:
        n_channels = int(xarr.sizes["C"])
        if n_channels != 1:
            raise UnsupportedAxesError(
                f"cannot fold a samples axis of size {n_samples} into a channel "
                f"axis of size {n_channels}: interleaved samples belong to a "
                f"single channel, so with both above 1 there is no way to order "
                f"the {n_channels * n_samples} resulting channels without "
                f"guessing. Dims were {list(xarr.dims)}."
            )
        xarr = xarr.squeeze("C", drop=True)

    # Rename rather than reshape: the samples axis is already the fastest-
    # varying one, so this is metadata-only and costs no dask rechunk. The
    # coord makes the resulting store self-describing — without it the folded
    # channels would be named "0"/"1"/"2".
    folded = xarr.rename({SAMPLES_AXIS: "C"})
    labels, _colors = sample_axis_channels(n_samples)
    return folded.assign_coords(C=labels), True


def normalize_axes(xarr: xr.DataArray) -> tuple[xr.DataArray, dict]:
    """Reorder dims to NGFF canonical [t?, c?, z?, y, x].

    Returns the canonical xarray plus an audit-trail record describing the
    transpose. Raises UnsupportedAxesError if dims contain anything outside
    {T, C, Z, Y, X} once the samples axis has been folded.
    """
    input_dims = list(xarr.dims)

    xarr, rgb_samples_folded = fold_samples_axis(xarr)
    dims = list(xarr.dims)

    unsupported = [d for d in dims if d not in NGFF_AXIS_ORDER]
    if unsupported:
        raise UnsupportedAxesError(
            f"zarrmony v0.1 supports axes {NGFF_AXIS_ORDER}; "
            f"got unsupported axes: {unsupported}. "
            f"For CZI mosaic data, ensure scenes are stitched (mosaic reconstructed)."
        )

    if len(set(dims)) != len(dims):
        raise UnsupportedAxesError(f"input dims contain duplicates: {dims}")

    output_dims = [d for d in NGFF_AXIS_ORDER if d in dims]
    was_transposed = dims != output_dims
    canonical = xarr.transpose(*output_dims) if was_transposed else xarr

    return canonical, {
        # Pre-fold, so the record still shows the reader's own dims — the "S"
        # is the only evidence the input was a colour image.
        "input_dims": input_dims,
        "output_dims": output_dims,
        "was_transposed": was_transposed,
        "rgb_samples_folded": rgb_samples_folded,
    }
