"""Reader registry: register zarrmony's built-in :class:`ReaderPlugin`
instances at import time.

Dispatch lives in ``readers/plugin.py``: ``convert()`` and ``inspect()`` call
``plugin.get_reader(path)``, which walks all registered plugins (the built-ins
below plus any external entry-point readers) and picks the highest-scoring
match. Order matters here — built-ins register before entry points are walked,
so equal-score ties resolve to built-ins (per ADR-0001).
"""

from .czi import czi_plugin
from .default import default_plugin
from .lif import lif_plugin
from .nd2 import nd2_plugin
from .plugin import register_plugin

register_plugin(default_plugin)
register_plugin(czi_plugin)
register_plugin(lif_plugin)
register_plugin(nd2_plugin)
