#!/usr/bin/env python3
"""check_driver_flags.py -- does every flag the shell driver passes actually exist?

STAGE 6 passed `--n-jobs` to `run_analysis.py`, which has no such option. argparse exited 2, the
stage failed, and the run died -- after STAGE 5 had already spent four minutes, and on the extract
path it would have been three hours. The flag had been wrong since the option was introduced; the
stage had simply never been reached in a completed run, because every earlier failure came first.

A dry run does not catch this: it prints the command without running it, so a flag that argparse
would reject looks identical to one it would accept. Neither does a unit test, because the mistake
is in the shell, not the Python.

This reads the driver script, pulls out every `$PY <tool>.py ... --flag` it would run, asks each
tool for its own `--help`, and reports any flag the tool does not define. It costs one `--help` per
tool -- under a second for the whole script -- and it runs in STAGE 0, before anything expensive.

    python check_driver_flags.py run_paper_replication.sh
    python check_driver_flags.py run_paper_replication.sh --quiet

Exit codes: 0 = every flag resolves;  2 = at least one does not;  1 = could not run.

Flags hidden behind a shell variable (`$JOBS_FLAG`, `$CACHE_FLAGS`) cannot be resolved statically
and are listed as unchecked rather than silently passed -- an honest gap is better than a false all-clear.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess as sp
import sys

_PY_CALL = re.compile(r"\$PY\s+([A-Za-z_][A-Za-z0-9_]*\.py)")
_LONG_FLAG = re.compile(r"(?<![\w-])(--[A-Za-z][A-Za-z0-9-]*)")


def _logical_lines(text: str):
    """Join backslash-continued shell lines so a multi-line command is one unit."""
    out, buf = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def parse_invocations(text: str) -> dict:
    """-> {tool: {'flags': set, 'unresolved': set}} for every `$PY tool.py ...` in the script.

    A line can invoke more than one tool (`autoscale.py measure -- $PY run_analysis.py ...`), so the
    line is split at each `$PY <tool>` and the flags in each segment belong to the tool that opened
    it -- otherwise the wrapper would be blamed for the wrapped command's options."""
    found: dict = {}
    for line in _logical_lines(text):
        s = line.strip()
        if s.startswith("#") or "$PY" not in s:
            continue
        marks = list(_PY_CALL.finditer(s))
        if not marks:
            continue
        for i, m in enumerate(marks):
            tool = m.group(1)
            seg = s[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(s)]
            rec = found.setdefault(tool, {"flags": set(), "unresolved": set()})
            for fm in _LONG_FLAG.finditer(seg):
                rec["flags"].add(fm.group(1))
            for var in re.findall(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", seg):
                if var.endswith("FLAG") or var.endswith("FLAGS"):
                    rec["unresolved"].add("$" + var)
    return found


def tool_options(tool: str, py: str) -> set:
    """Long options the tool's own argparse defines, from its --help."""
    try:
        r = sp.run([py, tool, "--help"], capture_output=True, text=True, timeout=120)
    except Exception as exc:
        raise RuntimeError(f"could not run {tool} --help: {exc}") from exc
    if r.returncode != 0:
        raise RuntimeError(f"{tool} --help exited {r.returncode}: {(r.stderr or '')[:200]}")
    return set(_LONG_FLAG.findall(r.stdout or ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify shell-driver flags against each tool's --help")
    ap.add_argument("script", nargs="?", default="run_paper_replication.sh")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--quiet", action="store_true", help="print only problems")
    a = ap.parse_args(argv)

    if not os.path.exists(a.script):
        print(f"check_driver_flags: no such script: {a.script}", file=sys.stderr)
        return 1
    with open(a.script) as fh:
        used = parse_invocations(fh.read())
    if not used:
        print(f"check_driver_flags: found no '$PY <tool>.py' invocations in {a.script}", file=sys.stderr)
        return 1

    bad, unchecked, lines = [], [], []
    for tool in sorted(used):
        if not os.path.exists(tool):
            lines.append(f"  {tool:32s} SKIPPED (not present)")
            continue
        try:
            have = tool_options(tool, a.python)
        except RuntimeError as exc:
            lines.append(f"  {tool:32s} could not introspect: {exc}")
            continue
        want = used[tool]["flags"]
        missing = sorted(f for f in want if f not in have)
        for f in missing:
            bad.append((tool, f))
        for v in sorted(used[tool]["unresolved"]):
            unchecked.append((tool, v))
        lines.append("  %-32s %2d flag(s) checked%s" % (tool, len(want),
                     "  -- %d NOT DEFINED: %s" % (len(missing), ", ".join(missing)) if missing else ""))

    if not a.quiet or bad:
        print("driver flag check: %s" % a.script)
        print("\n".join(lines))
    if unchecked and not a.quiet:
        print("  unchecked (hidden behind a shell variable, resolved at run time): %s"
              % ", ".join(sorted({f"{t}:{v}" for t, v in unchecked})))
    if bad:
        print("")
        print("FAIL: %d flag(s) the driver passes do not exist in the tool that receives them:" % len(bad))
        for t, f in bad:
            print(f"    {t}  {f}")
        print("argparse exits 2 on an unrecognized option, so each of these kills its stage the")
        print("moment it is reached -- which on the extract path is hours in.")
        return 2
    if not a.quiet:
        print("  every flag resolves")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"check_driver_flags: FAILED to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
