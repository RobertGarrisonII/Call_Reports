#!/usr/bin/env python3
"""extraction_status.py -- what is every extraction worker doing RIGHT NOW?

Parallel extraction suppresses console heartbeats (process workers cannot stream them cleanly),
which made a multi-hour fine-grid pull look hung: the 2026-08-06 run's STAGE 2b went silent from
its first line. Every worker now writes a per-session JSON heartbeat at each phase transition and
every ~2M replayed events (``<cache_dir>/progress/<date>_<interval>.json``); this tool tabulates
them.

    python extraction_status.py --cache-dir output/extract_cache
    watch -n 30 python extraction_status.py --cache-dir output/extract_cache

Columns: session, grid, phase (the replay phase includes % events and % grid), elapsed inside the
session, and the age of the last heartbeat. A live worker updates at least every couple of
minutes; ``STALLED?`` flags anything quiet longer than ``--stall`` seconds (default 900) that has
not reported DONE/FAILED -- the one signal that distinguishes "slow but moving" from actually
stuck. Exit codes: 0 = progress dir found (even if empty); 1 = no progress dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


def read_status(progress_dir: str) -> list:
    rows = []
    for f in sorted(glob.glob(os.path.join(progress_dir, "*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        d["_age_s"] = time.time() - float(d.get("updated_epoch", 0))
        rows.append(d)
    return rows


def describe(rows, stall_s: float) -> str:
    if not rows:
        return ("no heartbeats yet -- either no extraction is running, it predates v0.9.61, or "
                "it runs without --extract-cache (the heartbeat dir lives under the cache dir)")
    out = ["%-12s %-6s %-46s %9s %9s %s"
           % ("session", "grid", "phase", "elapsed", "beat-age", "state")]
    n_done = n_fail = n_stall = 0
    for d in sorted(rows, key=lambda r: (r.get("interval", ""), r.get("session", ""))):
        phase = str(d.get("phase", "?"))
        age = d["_age_s"]
        done = phase.startswith("DONE")
        fail = phase.startswith("FAILED")
        stalled = (not done and not fail and age > stall_s)
        state = "done" if done else ("FAILED" if fail else ("STALLED?" if stalled else "running"))
        n_done += done; n_fail += fail; n_stall += stalled
        el = float(d.get("elapsed_s", 0))
        out.append("%-12s %-6s %-46s %8.0fs %8.0fs %s"
                   % (d.get("session", "?"), d.get("interval", "?"), phase[:46], el, age, state))
    run = len(rows) - n_done - n_fail
    out.append("")
    out.append("%d session file(s): %d done, %d running, %d failed%s"
               % (len(rows), n_done, run - n_stall, n_fail,
                  (", %d STALLED? (no beat for >%.0fs -- check the worker's memory/swap first: an "
                   "over-subscribed node crawls before it dies)" % (n_stall, stall_s)) if n_stall else ""))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="output/extract_cache",
                    help="extraction cache dir (heartbeats live in <cache-dir>/progress)")
    ap.add_argument("--progress-dir", default="", help="explicit heartbeat dir (overrides --cache-dir)")
    ap.add_argument("--stall", type=float, default=900.0,
                    help="seconds of heartbeat silence before a running session is flagged")
    a = ap.parse_args(argv)
    pdir = a.progress_dir or os.path.join(a.cache_dir, "progress")
    if not os.path.isdir(pdir):
        print("no progress directory at %s" % pdir)
        return 1
    print(describe(read_status(pdir), a.stall))
    return 0


if __name__ == "__main__":
    sys.exit(main())
