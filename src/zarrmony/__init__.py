"""Convert any bioimage file to OME-Zarr v0.5, preserving metadata."""

__version__ = "0.3.3"

from zarrmony.api import convert, inspect
from zarrmony.metadata.model import UserMetadata

__all__ = ["__version__", "convert", "inspect", "UserMetadata"]
