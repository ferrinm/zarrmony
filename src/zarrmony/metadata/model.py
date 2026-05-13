"""User-supplied metadata model for zarrmony conversions.

⚠️ The field set defined here is PROVISIONAL. Per the Aperture proposal, the
required-vs-optional split is an open cross-team design item. v0.1 ships a
draft to define the *shape* of the gate (and the JSON Schema export consumed by
Austin's eventual web form); the field set will tighten before pilot completion.

Callers that need to bypass the gate during prototyping should pass
``permissive=True`` to ``zarrmony.convert(...)``.
"""

from pydantic import BaseModel, ConfigDict, Field


class UserMetadata(BaseModel):
    """User-supplied metadata accompanying a conversion.

    The gate (``convert(...)`` with ``permissive=False``) requires the fields
    marked Required below to be present and non-empty. Optional fields may be
    omitted. Additional fields beyond this set are accepted (``extra="allow"``)
    so callers can experiment ahead of schema finalization.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    # Required (Shared tier — duplicated to Tenaya for cross-modal discovery)
    microscope: str = Field(
        ...,
        min_length=1,
        description="Instrument name, e.g. 'Axioscan', 'Thunder'.",
    )
    modality: str = Field(
        ...,
        min_length=1,
        description="Imaging modality, e.g. 'fluorescence', 'brightfield', 'multiplex'.",
    )

    # Optional (Shared tier)
    objective: str | None = Field(
        None,
        description="Objective lens, e.g. '20x', '63x oil'.",
    )

    # Optional (acquisition context — destined for Tenaya cross-references)
    study: str | None = Field(None, description="Study identifier.")
    project_code: str | None = Field(None, description="Calico project code.")
    researcher: str | None = Field(
        None, description="Researcher who acquired the data."
    )

    # Optional (Zarr-only — deep acquisition parameters)
    detector_gain: float | None = None
    laser_power: dict[str, float] | None = Field(
        None,
        description="Map channel name → laser power.",
    )
    exposure_times: dict[str, float] | None = Field(
        None,
        description="Map channel name → exposure (ms).",
    )
