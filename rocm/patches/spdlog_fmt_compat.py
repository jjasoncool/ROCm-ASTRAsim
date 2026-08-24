#!/usr/bin/env python3
r"""Add the fmt include spdlog_setup needs so astra-sim's ns-3 frontend builds.

Replaces the sed chain from rocm/dockerfile. That chain's two regex passes were
a no-op on the pinned 28f18ea (all 43 call sites are already fmt::format) and
were dropped: \bformat\( also matches .format( and ->format(.

Usage: python3 spdlog_fmt_compat.py <astra-sim-root>
"""
import sys

FILES = [
    "extern/helper/spdlog_setup/conf.h",
    "extern/helper/spdlog_setup/details/conf_impl.h",
    "extern/helper/spdlog_setup/details/template_impl.h",
]
INCLUDE = "#include <spdlog/fmt/fmt.h>"

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/astra-sim"

for rel in FILES:
    path = f"{root}/{rel}"
    try:
        # newline="" keeps the file's own line endings; the default would rewrite them.
        with open(path, newline="") as f:
            src = f.read()
    except OSError as e:
        sys.exit(f"[spdlog_fmt_compat] ERROR: {e}")
    if src.startswith(INCLUDE):
        print(f"[spdlog_fmt_compat] already patched: {rel}")
        continue
    nl = "\r\n" if "\r\n" in src else "\n"
    with open(path, "w", newline="") as f:
        f.write(INCLUDE + nl + src)
    print(f"[spdlog_fmt_compat] patched: {rel}")
