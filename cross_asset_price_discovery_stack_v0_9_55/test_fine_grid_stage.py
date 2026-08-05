#!/usr/bin/env python3
"""test_fine_grid_stage.py -- the 10ms second grid is wired in, curated, and additive.

The driver half-expected a fine grid for four releases: FINE_INTERVAL was declared, STAGE 5 looked
for 10ms frames and printed "re-run STAGE 2 with --interval 10ms" when they were absent -- which is
exactly what the 2026-08-04 run did -- but nothing ever EXTRACTED them, and no stage list was
curated for them. v0.9.55 closes the loop: STAGE 2b extracts FINE_INTERVAL frames (--with-fine,
subset via --fine-dates), STAGE 3 QCs them warn-only, STAGE 5 picks them up for the Table 9 Epps
pair, and STAGE 6b runs ONLY the exhibits 1s cannot resolve (IS bounds, ecm_sde IS(S), co-jump
lead-lag, staleness/noise diagnostics). The 1s mains stay primary -- at 10ms upwards of 99% of
snapshots are stale repeats of ES's coarse tick, so panel/DCC/cross-impact would get worse there.

Pinned here:

  (1) dry-run with --with-fine prints the whole two-grid flow: the 2b extraction command at
      FINE_INTERVAL, the fine QC in STAGE 3, the fine Table 9 pair in STAGE 5, and STAGE 6b with
      the curated --only list -- and WITHOUT --with-fine none of it appears (default unchanged)
  (2) --fine-dates runs the subset (staged rollout: the per-worker peak is unmeasured at 10ms)
  (3) the curated stage names in the driver all exist in run_analysis.STAGES -- the shell would
      otherwise only discover a rename at hour three of a real run
  (4) the load-path fallback cannot self-match: a pickle named without the interval token must
      report "no fine frames", not silently hand the 1s frames to the fine stages
  (5) the curated list actually RUNS on 10ms frames: 4/4 stages ok on synthetic sub-second
      sessions, with the sub-second path engaged (auto lag rescale, Lee-Mykland, fleeting filter)

Run: python test_fine_grid_stage.py
"""
from __future__ import annotations

import os
import pickle
import re
import subprocess as sp
import sys
import tempfile

import numpy as np
import pandas as pd

TZ = "America/New_York"
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_paper_replication.sh")
CURATED = ["information_shares", "ecm_sde", "jumps", "microstructure"]


def _dry(*args):
    with tempfile.TemporaryDirectory(prefix="fine_dry_") as td:
        res = sp.run(["bash", DRIVER, "--dry-run", "--out-dir", td, *args],
                     capture_output=True, text=True, timeout=300)
        return res.returncode, res.stdout + res.stderr


def check_dry_run_prints_the_two_grid_flow():
    rc, out = _dry("--source", "extract", "--with-fine", "--stages", "2,3,5,6")
    want = ["STAGE 2b", "--interval 10ms", "qc_frames_10ms", "frames_10ms.pkl", "STAGE 6b",
            "--only " + ",".join(CURATED)]
    missing = [w for w in want if w not in out]
    ok = rc == 0 and not missing
    print("(1a) dry-run --with-fine prints 2b extraction, fine QC, fine Table 9 and curated 6b "
          "(missing: %s) : %s" % (missing or "none", ok))
    return ok


def check_default_run_is_unchanged():
    rc, out = _dry("--source", "extract", "--stages", "2,3,5,6")
    ok = rc == 0 and "STAGE 2b" not in out and "STAGE 6b" not in out
    print("(1b) without --with-fine neither 2b nor 6b appears -- default flow unchanged : %s" % ok)
    return ok


def check_fine_dates_runs_the_subset():
    sub = "2020-03-09,2020-03-12,2022-03-24"
    rc, out = _dry("--source", "extract", "--fine-dates", sub, "--stages", "2")
    m = re.search(r"--interval 10ms.*", out)
    twob = out[out.find("STAGE 2b"):]
    cmd = re.search(r"--dates (\S+).*--interval 10ms", twob)
    ok = (rc == 0 and "3 session(s) at 10ms" in out
          and cmd is not None and cmd.group(1) == sub and m is not None)
    print("(2) --fine-dates: STAGE 2b extracts exactly the 3-session subset : %s" % ok)
    return ok


def check_curated_stages_exist_in_the_registry():
    import run_analysis as ra
    names = {n for n, _ in ra.STAGES}
    missing = [c for c in CURATED if c not in names]
    with open(DRIVER) as fh:
        drv = fh.read()
    wired = ("--only " + ",".join(CURATED)) in drv
    ok = not missing and wired
    print("(3) curated stages %s all exist in run_analysis.STAGES (missing: %s) and the driver "
          "passes exactly that list (%s) : %s" % (CURATED, missing or "none", wired, ok))
    return ok


def check_load_fallback_cannot_self_match():
    with tempfile.TemporaryDirectory(prefix="fine_load_") as td:
        pkl = os.path.join(td, "frames.pkl")          # no interval token in the name
        open(pkl, "w").close()
        rc, out = _dry("--source", "load", "--pickle", pkl, "--with-fine", "--stages", "2,6")
    ok = (rc == 0 and "no 10ms frames beside the 1s pickle" in out
          and "skipped: no 10ms frames" in out)
    print("(4) load path: an interval-less pickle name reports 'no fine frames' instead of "
          "silently reusing the 1s frames : %s" % ok)
    return ok


def _fine_frames(path, n=3000):
    rng = np.random.default_rng(3)
    sessions = []
    for day, reg in [("2024-07-24", "benchmark"), ("2024-08-05", "volatile")]:
        idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="10ms", tz=TZ)
        eff = np.cumsum(rng.normal(0, 3e-6, n))
        spy = 550.0 * np.exp(eff + rng.normal(0, 8e-7, n))
        es = 5500.0 * np.exp(eff + rng.normal(0, 8e-7, n))
        df = pd.DataFrame(index=idx)
        for i in range(1, 11):
            df[f"SPY_bidprice_{i}"] = spy - 0.005 * i
            df[f"SPY_askprice_{i}"] = spy + 0.005 * i
            df[f"SPY_bidquantity_{i}"] = rng.integers(50, 150, n).astype(float)
            df[f"SPY_askquantity_{i}"] = rng.integers(50, 150, n).astype(float)
            df[f"ES_bidprice_{i}"] = es - 0.25 * i
            df[f"ES_askprice_{i}"] = es + 0.25 * i
            df[f"ES_bidquantity_{i}"] = rng.integers(20, 80, n).astype(float)
            df[f"ES_askquantity_{i}"] = rng.integers(20, 80, n).astype(float)
        df.attrs["halt_windows_SPY"] = []; df.attrs["halt_windows_ES"] = []
        sessions.append((day, reg, df))
    with open(path, "wb") as fh:
        pickle.dump(sessions, fh)


def check_curated_list_runs_on_10ms_frames():
    with tempfile.TemporaryDirectory(prefix="fine_run_") as td:
        pkl = os.path.join(td, "frames_10ms.pkl")
        _fine_frames(pkl)
        res = sp.run([sys.executable, "run_analysis.py", "--source", "load", "--pickle", pkl,
                      "--volatile", "2024-08-05", "--benchmark", "2024-07-24",
                      "--interval", "10ms", "--only", ",".join(CURATED),
                      "--output-dir", td, "--quick"],
                     capture_output=True, text=True, timeout=540,
                     cwd=os.path.dirname(DRIVER))
        out = res.stdout + res.stderr
    ran = "4/4 stages ok" in out
    sub = '"path": "sub-second"' in out and '"jump_method": "lee_mykland"' in out
    lag = '"n_lags": 60' in out                     # frequency rescale, not the 1s number
    ok = res.returncode == 0 and ran and sub and lag
    print("(5) curated list on 10ms frames: 4/4 stages ok (%s), sub-second path engaged with "
          "Lee-Mykland (%s), lag auto-rescaled to 60 (%s) : %s" % (ran, sub, lag, ok))
    return ok


def main():
    checks = [check_dry_run_prints_the_two_grid_flow,
              check_default_run_is_unchanged,
              check_fine_dates_runs_the_subset,
              check_curated_stages_exist_in_the_registry,
              check_load_fallback_cannot_self_match,
              check_curated_list_runs_on_10ms_frames]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("fine-grid stage checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
