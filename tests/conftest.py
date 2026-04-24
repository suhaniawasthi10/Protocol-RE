"""Pytest config: ensures `server` and `tests` packages are importable.

The editable install of openenv-protocol_one_env already adds the package root
to sys.path via a .pth file, so `import server.spec` works. This conftest is
defensive: when pytest is invoked from elsewhere we still want the imports to
resolve.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
