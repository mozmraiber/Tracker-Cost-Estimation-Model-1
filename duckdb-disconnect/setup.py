"""Build the `disconnect` Python module with pip or setuptools.

    pip install ./duckdb-disconnect

`make module` builds the same module in tree, without installing it; see the
Makefile for the flags, which this file mirrors.
"""

import re
import pathlib

from setuptools import Extension, setup

HERE = pathlib.Path(__file__).parent

# The version lives in the Makefile, which stamps it into the extension binary too.
VERSION = re.search(r"^EXTENSION_VERSION := (\S+)$", (HERE / "Makefile").read_text(), re.M).group(1)

setup(
    version=VERSION,
    ext_modules=[
        Extension(
            "disconnect",
            # The calling layer, then the list model, matcher, URL parser and
            # embedded list shared with the DuckDB extension.
            sources=[
                "src/python_module.cpp",
                "src/disconnect_list.cpp",
                "src/mini_json.cpp",
                "src/url_host.cpp",
                "src/generated/disconnect_data.cpp",
            ],
            include_dirs=["src/include"],
            # Builds the shared sources against the standard library rather than
            # duckdb.hpp, so no DuckDB headers or libraries are needed.
            define_macros=[("DISCONNECT_NO_DUCKDB", "1"), ("DISCONNECT_VERSION", '"%s"' % VERSION)],
            # Hidden visibility keeps the shared C++ symbols out of the dynamic
            # symbol table, so importing the module alongside a process that has
            # loaded the DuckDB extension cannot make the two share a list.
            extra_compile_args=["-std=c++17", "-fvisibility=hidden"],
            language="c++",
        )
    ],
)
