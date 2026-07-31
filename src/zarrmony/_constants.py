"""Shared constants used across writers, readers, and audit.

Kept as a leaf module so both the audit surface and every writer path can
import ``NGFF_VERSION`` from one canonical location — the OME-NGFF version
must not drift between what the writer stamps on-disk and what the audit
declares under ``attrs.zarrmony.output.ome_ngff_version``.
"""

from __future__ import annotations

NGFF_VERSION = "0.5"
