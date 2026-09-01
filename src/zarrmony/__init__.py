"""Convert any bioimage file to OME-Zarr v0.5, preserving metadata."""

from importlib.metadata import version

__version__ = version("zarrmony")

from zarrmony.api import convert, inspect, rechunk
from zarrmony.geometry import Geometry

__all__ = ["Geometry", "__version__", "convert", "inspect", "rechunk"]
