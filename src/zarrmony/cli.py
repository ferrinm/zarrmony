"""Command-line interface for zarrmony."""

import click

from zarrmony import __version__


@click.group(name="zarrmony")
@click.version_option(__version__, prog_name="zarrmony")
def app() -> None:
    """Convert any bioimage file to OME-Zarr v0.5, preserving metadata."""


@app.command()
def convert() -> None:
    """Convert a bioimage file to OME-Zarr v0.5. (Not implemented yet.)"""
    raise click.ClickException("Not implemented yet — scaffolding only.")


@app.command()
def inspect() -> None:
    """Print scenes, dims, channels, and pixel sizes for an input file. (Not implemented yet.)"""
    raise click.ClickException("Not implemented yet — scaffolding only.")


@app.group()
def schema() -> None:
    """Inspect or export the user-metadata schema."""


@schema.command(name="dump")
def schema_dump() -> None:
    """Emit the JSON Schema for the user-supplied metadata model. (Not implemented yet.)"""
    raise click.ClickException("Not implemented yet — scaffolding only.")
