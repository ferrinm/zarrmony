"""Axes normalization to OME-Zarr v0.5 canonical order [t?, c?, z?, y, x].

Per the OME-Zarr 0.5 spec, axes must be ordered by type — time first (if present),
then channel/custom, then space in zyx order. bioio readers do not always return
dims in this order; this module rotates them and records what was done so the
audit trail can show users why their dim order changed.
"""

import xarray as xr

NGFF_AXIS_ORDER: tuple[str, ...] = ("T", "C", "Z", "Y", "X")

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


def normalize_axes(xarr: xr.DataArray) -> tuple[xr.DataArray, dict]:
    """Reorder dims to NGFF canonical [t?, c?, z?, y, x].

    Returns the canonical xarray plus an audit-trail record describing the
    transpose. Raises UnsupportedAxesError if dims contain anything outside
    {T, C, Z, Y, X}.
    """
    input_dims = list(xarr.dims)

    unsupported = [d for d in input_dims if d not in NGFF_AXIS_ORDER]
    if unsupported:
        raise UnsupportedAxesError(
            f"zarrmony v0.1 supports axes {NGFF_AXIS_ORDER}; "
            f"got unsupported axes: {unsupported}. "
            f"For CZI mosaic data, ensure scenes are stitched (mosaic reconstructed)."
        )

    if len(set(input_dims)) != len(input_dims):
        raise UnsupportedAxesError(f"input dims contain duplicates: {input_dims}")

    output_dims = [d for d in NGFF_AXIS_ORDER if d in input_dims]
    was_transposed = input_dims != output_dims
    canonical = xarr.transpose(*output_dims) if was_transposed else xarr

    return canonical, {
        "input_dims": input_dims,
        "output_dims": output_dims,
        "was_transposed": was_transposed,
    }
