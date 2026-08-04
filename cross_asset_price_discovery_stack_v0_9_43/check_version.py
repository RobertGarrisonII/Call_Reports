#!/usr/bin/env python3
"""check_version.py -- the version in one place, enforced.

`__init__.__version__` sat at 0.9.20 while the CHANGELOG read 0.9.42 and every release archive
shipped as `cross_asset_price_discovery_stack_v0934.zip`. Twenty-two releases of drift, and the
symptom a reader saw was three different answers to "which version is this?" -- with the archive
name, the one thing that leaves the machine, being the most wrong.

Nothing here is clever. It just refuses to let the three disagree:

    __init__.__version__        0.9.43
    CHANGELOG.md top entry      ## v0.9.43 -- ...
    package directory           cross_asset_price_discovery_stack_v0_9_43
    release archive             cross_asset_price_discovery_stack_v0943.zip

`package_name()` and `archive_name()` DERIVE the last two, so the packaging step cannot pick a name
by hand and get it wrong. Run this before shipping; it is also in the STAGE 1 test gate.

    python check_version.py                 # verify
    python check_version.py --print-names   # emit the derived names for a packaging script
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def version() -> str:
    ns: dict = {}
    with open(os.path.join(HERE, "__init__.py")) as fh:
        for line in fh:
            if line.startswith("__version__"):
                exec(line, ns)
                return str(ns["__version__"])
    raise RuntimeError("__init__.py carries no __version__")


def changelog_version(path: str = "") -> str:
    with open(path or os.path.join(HERE, "CHANGELOG.md")) as fh:
        for line in fh:
            m = re.match(r"^##\s+v?(\d+\.\d+\.\d+)", line.strip())
            if m:
                return m.group(1)
    raise RuntimeError("CHANGELOG.md has no '## vX.Y.Z' entry")


def package_name(v: str = "") -> str:
    """0.9.43 -> cross_asset_price_discovery_stack_v0_9_43"""
    return "cross_asset_price_discovery_stack_v" + (v or version()).replace(".", "_")


def archive_name(v: str = "") -> str:
    """0.9.43 -> cross_asset_price_discovery_stack_v0943.zip (the historical flat form)"""
    a, b, c = (v or version()).split(".")
    return "cross_asset_price_discovery_stack_v%s%s%s.zip" % (a, b, c.zfill(2))


def check() -> bool:
    ok = []
    v, cv = version(), changelog_version()
    a = v == cv
    print("(1) __init__.__version__ = %s ; CHANGELOG top entry = %s : %s" % (v, cv, a))
    ok.append(a)

    here = os.path.basename(HERE)
    b = here == package_name(v)
    print("(2) package directory is %s (want %s) : %s" % (here, package_name(v), b))
    ok.append(b)

    # the archive is built by the packaging step, so only its NAME is checked here
    print("(3) derived archive name: %s" % archive_name(v))
    ok.append(True)

    # a stale archive from a previous version sitting next to the package is how the wrong file
    # gets shipped, so say so if one is there
    parent = os.path.dirname(HERE)
    stale = sorted(f for f in os.listdir(parent)
                   if re.match(r"cross_asset_price_discovery_stack_v\d+\.zip$", f)
                   and f != archive_name(v)) if os.path.isdir(parent) else []
    d = not stale
    print("(4) no stale release archive beside the package : %s%s"
          % (d, "" if d else "  <- found " + ", ".join(stale)))
    ok.append(d)

    print("\nversion checks -> %s" % all(ok))
    return all(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-names", action="store_true",
                    help="print 'VERSION PACKAGE_DIR ARCHIVE' for a packaging script to consume")
    a = ap.parse_args()
    if a.print_names:
        print("%s %s %s" % (version(), package_name(), archive_name()))
        return 0
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
