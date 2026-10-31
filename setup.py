"""Build the optional Apple Silicon Metal and exact CPU search extensions."""

from __future__ import annotations

import os
import platform
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class MetalBuildExt(build_ext):
    """Teach distutils to compile Objective-C++ sources with clang."""

    def build_extensions(self):
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        super().build_extensions()


extensions = []
options = {}
if sys.platform == "darwin" and platform.machine() == "arm64":
    # The framework Python may advertise a universal2 / macOS 10.13 target
    # even when building on Apple silicon. The native backend targets arm64,
    # whose minimum macOS version is 11; do not emit a misleading wheel tag.
    os.environ.setdefault("ARCHFLAGS", "-arch arm64")
    deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
    if not deployment_target or tuple(map(int, deployment_target.split("."))) < (11, 0):
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
    # Universal Python distributions report a universal2 platform even when
    # ARCHFLAGS selects only arm64. Keep the wheel tag aligned with the actual
    # native binary, rather than advertising an unavailable x86_64 extension.
    os.environ.setdefault(
        "_PYTHON_HOST_PLATFORM", f"macosx-{os.environ['MACOSX_DEPLOYMENT_TARGET']}-arm64"
    )
    extensions.append(
        Extension(
            "mlxlearn._solvers._metal_loop",
            ["src/mlxlearn/_solvers/_metal_loop.mm"],
            language="c++",
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            py_limited_api=True,
            extra_compile_args=["-std=c++17", "-fobjc-arc"],
            extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
            optional=True,
        )
    )
    extensions.append(
        Extension(
            "mlxlearn._kernels._neighbors_cpu_native",
            ["src/mlxlearn/_kernels/_neighbors_cpu_native.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"],
            optional=True,
        )
    )
    # The CPU module uses Py_buffer, including on Python 3.10. Its wheel must
    # carry the actual CPython ABI tag, even though the Metal module is abi3.

setup(ext_modules=extensions, cmdclass={"build_ext": MetalBuildExt}, options=options)
