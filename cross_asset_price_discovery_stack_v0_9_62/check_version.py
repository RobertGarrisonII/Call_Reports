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


def check(strict_hygiene: bool = False) -> bool:
    """-> True if the VERSION IS CONSISTENT. Housekeeping is reported but does not decide the result.

    The split matters. This started as one flat list of checks and went straight into the STAGE 1
    correctness gate, where it aborted a replication run at minute two -- because the working
    directory had three older release archives sitting beside the package, which is what a working
    directory looks like. Nothing about a stale zip makes a result wrong.

    So: a mismatch between `__version__`, the CHANGELOG and the directory name is a REAL defect --
    it mislabels the artifact that leaves the machine -- and fails. Old archives lying around are
    housekeeping, and warn. Pass ``strict_hygiene`` (or --strict) to fail on those too, which is
    what ``package.sh`` does, because at packaging time an unswept archive really can be shipped by
    mistake."""
    hard, warn = [], []
    v, cv = version(), changelog_version()
    a = v == cv
    print("(1) __init__.__version__ = %s ; CHANGELOG top entry = %s : %s" % (v, cv, a))
    hard.append(a)

    here = os.path.basename(HERE)
    b = here == package_name(v)
    print("(2) package directory is %s (want %s) : %s" % (here, package_name(v), b))
    hard.append(b)

    # the archive is built by the packaging step, so only its NAME is derived here
    print("(3) derived archive name: %s" % archive_name(v))

    parent = os.path.dirname(HERE)
    stale = sorted(f for f in os.listdir(parent)
                   if re.match(r"cross_asset_price_discovery_stack_v\d+\.zip$", f)
                   and f != archive_name(v)) if os.path.isdir(parent) else []
    print("(4) housekeeping: %s"
          % ("no older release archive beside the package"
             if not stale else "older archive(s) present -- %s. Harmless for a run; sweep them "
                               "before shipping so the wrong file cannot be picked up."
                               % ", ".join(stale)))
    warn.append(not stale)

    ok = all(hard) and (all(warn) if strict_hygiene else True)
    print("\nversion consistency -> %s%s"
          % (all(hard), "" if all(warn) else "   (housekeeping: %d stale archive(s), advisory)"
             % len(stale)))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-names", action="store_true",
                    help="print 'VERSION PACKAGE_DIR ARCHIVE' for a packaging script to consume")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on housekeeping (stale archives beside the package). Used by "
                         "package.sh; NOT by the replication gate, where a leftover zip has no "
                         "bearing on whether a result is right")
    a = ap.parse_args()
    if a.print_names:
        print("%s %s %s" % (version(), package_name(), archive_name()))
        return 0
    return 0 if check(strict_hygiene=a.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
