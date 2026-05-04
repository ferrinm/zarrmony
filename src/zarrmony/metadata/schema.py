"""JSON Schema export for the user-metadata Pydantic model.

Consumed by ``zarrmony schema dump`` so Austin's eventual web form can render a
matching UI from one source of truth.
"""

import json
from typing import Any

from .model import UserMetadata


def export_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``UserMetadata`` as a dict."""
    return UserMetadata.model_json_schema()


def export_schema_json(indent: int = 2) -> str:
    """Return the JSON Schema as a formatted JSON string."""
    return json.dumps(export_schema(), indent=indent)
