#!/usr/bin/env python3
"""test_driver_flags.py -- the shell driver must not pass a flag the tool rejects.

STAGE 6 passed `--n-jobs` to `run_analysis.py`, which has no such option. argparse exits 2, the
stage dies, the run dies. The flag had been wrong since it was written; the stage had simply never
been reached, because on every previous run something earlier failed first.

Neither existing safety net covers this. A dry run prints the command without parsing it, so a
rejected flag looks exactly like an accepted one. A unit test exercises the Python, and the mistake
is in the shell. The cost of finding it at run time is the whole pipeline up to that stage -- four
minutes on demo, three hours on extract.

Run: python test_driver_flags.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import check_driver_flags as cdf

SCRIPT = "run_paper_replication.sh"


def check_real_script_is_clean():
    if not os.path.exists(SCRIPT):
        print("(1) %s not present -- skipped" % SCRIPT)
        return True
    rc = cdf.main([SCRIPT, "--quiet"])
    ok = rc == 0
    print("(1) every flag %s passes exists in the tool that receives it (exit %d, want 0) : %s"
          % (SCRIPT, rc, ok))
    return ok


def check_catches_a_bad_flag():
    """The regression itself: inject the exact mistake and require a non-zero exit."""
    if not os.path.exists(SCRIPT):
        print("(2) skipped")
        return True
    with open(SCRIPT) as fh:
        text = fh.read()
    bad = text.replace("run $PY run_analysis.py --source demo",
                       "run $PY run_analysis.py --no-such-option --source demo", 1)
    injected = bad != text
    fd, path = tempfile.mkstemp(suffix=".sh"); os.close(fd)
    try:
        with open(path, "w") as fh:
            fh.write(bad)
        rc = cdf.main([path, "--quiet"])
    finally:
        os.unlink(path)
    ok = injected and rc == 2
    print("(2) a flag the tool does not define is caught (injected=%s, exit %d, want 2) : %s"
          % (injected, rc, ok))
    return ok


def check_parser_attributes_wrapped_commands():
    """`autoscale.py measure -- $PY run_analysis.py --flag` -- the flag belongs to run_analysis."""
    text = ('run_show $PY autoscale.py measure -- $PY run_analysis.py --source extract \\\n'
            '    --qc-action warn --output-dir "$OUT"\n')
    found = cdf.parse_invocations(text)
    auto = found.get("autoscale.py", {}).get("flags", set())
    ra = found.get("run_analysis.py", {}).get("flags", set())
    ok = (not auto) and {"--source", "--qc-action", "--output-dir"} <= ra
    print("(3) a wrapped command's flags are attributed to the WRAPPED tool, not the wrapper "
          "(autoscale=%s, run_analysis=%d flags) : %s" % (sorted(auto), len(ra), ok))
    return ok


def check_unresolved_flags_are_declared():
    """A flag hidden in a shell variable cannot be checked -- say so rather than imply it passed."""
    text = 'run $PY run_analysis.py --source extract $CACHE_FLAGS --output-dir "$OUT"\n'
    rec = cdf.parse_invocations(text).get("run_analysis.py", {})
    ok = "$CACHE_FLAGS" in rec.get("unresolved", set())
    print("(4) a flag behind a shell variable is reported as UNCHECKED, not as passing : %s" % ok)
    return ok


def check_line_continuations():
    """Multi-line commands are the norm in this driver; the parser must join them."""
    text = ('  run_rc $PY debug_crossing.py --date "$ymd" --product SPY \\\n'
            '         --clock receipt --ab-ordering --out "$OUT/x.txt"\n')
    rec = cdf.parse_invocations(text).get("debug_crossing.py", {})
    ok = {"--date", "--product", "--clock", "--ab-ordering", "--out"} <= rec.get("flags", set())
    print("(5) flags after a backslash continuation are seen (%d found) : %s"
          % (len(rec.get("flags", set())), ok))
    return ok


def main():
    checks = [check_real_script_is_clean, check_catches_a_bad_flag,
              check_parser_attributes_wrapped_commands, check_unresolved_flags_are_declared,
              check_line_continuations]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception as exc:
            import traceback; traceback.print_exc()
            res.append(False)
    ok = all(res)
    print("\ndriver-flag checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
