"""Import the scripts under tools/, which are files rather than a package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_tool(name: str):
    """Import ``tools/<name>.py`` once and return the module."""
    key = f"_mlxlearn_tools_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses can resolve the module by name.
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
