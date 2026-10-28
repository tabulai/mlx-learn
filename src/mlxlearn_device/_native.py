"""NumPy-free access to the existing native Metal command loop.

The implementation generator is shared with the sklearn path. Loading the
small source generator directly avoids importing ``mlxlearn``, whose package
initializer intentionally imports NumPy. The extension still owns all external
command-buffer lifetimes through its native owner registry.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / "mlxlearn"
_SOLVERS = _ROOT / "_solvers"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_source_generator():
    """Compile just the dependency-free native Metal-source generator."""
    path = _SOLVERS / "native_command.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    required = {"_kernel_source", "_kernel", "_source"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in nodes} != required:
        raise RuntimeError("The installed native source generator is incompatible")
    scope = {"Path": Path, "functools": functools, "ast": ast, "__file__": str(path)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["_source"]


prepare_centered = _load(
    "mlxlearn_device_input_kernels", _ROOT / "_kernels" / "logistic_input.py"
).prepare_centered
source_for = _load_source_generator()
# The launch geometry must match the sklearn path exactly: the same Metal
# source is specialized by both. The module has no imports, so load it by path.
speculative_geometry = _load(
    "mlxlearn_device_native_geometry", _SOLVERS / "native_geometry.py"
).speculative_geometry
_extension_path = _SOLVERS / "_metal_loop.abi3.so"
if not _extension_path.is_file():
    metal_loop = None
else:
    try:
        metal_loop = _load("_metal_loop", _extension_path)
    except (ImportError, OSError):
        metal_loop = None
