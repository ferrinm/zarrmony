"""Convert any bioimage file to OME-Zarr v0.5, preserving metadata."""

from importlib.metadata import version

__version__ = version("zarrmony")

from zarrmony.api import convert, inspect

__all__ = ["__version__", "convert", "inspect"]
