#!/usr/bin/env python3
"""run_horizon_profile.py -- the propagation-horizon ladder (memo item E1).

Derives an interval ladder from the FINE (10ms) frames (pull-once; no new
extraction) and, per rung, runs the estimators for which the grid is the
question: cross-impact lambda by direction, information shares / CS / kappa,
the error-correction half-life, co-jump lead shares, and the staleness
companion. Prints and writes the per-day long table, the per-interval summary
with the normalized cross-impact profile, and the HALF-IMPACT HORIZON -- the
log-interpolated interval at which the SPY<-ES impact ratio crosses one half
of its coarsest-rung value: how fast tandem futures flow becomes equity price.

Usage
    python run_horizon_profile.py --source demo
    python run_horizon_profile.py --source load --pickle output/.../frames_10ms.pkl \
        --volatile 2020-03-09,... --out-dir output/horizon --tag full
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import horizon_profile as hp


def _load(args):
    if args.source == "demo":
        return hp.propagation_demo_sessions(n_days=args.n_demo, n_bars=args.n_demo_bars,
                                            delay_ms=args.demo_delay_ms)
    paths = sorted(glob.glob(args.pickle))
    if not paths:
        raise SystemExit(f"no pickle matched {args.pickle!r}")
    raw = []
    for p in paths:
        with open(p, "rb") as fh:
            raw.extend(pickle.load(fh))
    vol = {d.strip() for d in (args.volatile or "").split(",") if d.strip()}
    out = []
    for rec in raw:
        if len(rec) == 3:
            date, regime, df = rec
        else:
            date, df = rec
            regime = "volatile" if str(date)[:10] in vol else "benchmark"
        out.append((date, regime, df))
    if getattr(args, "halt_mask", True):
        import market_halts as mh
        out = [(d, r, mh.mask_frame(df)[0]) for d, r, df in out]
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="propagation-horizon ladder: estimators vs sampling grid")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for the FINE-grid List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--fine-interval", default="10ms",
                    help="the interval the loaded frames are on (default 10ms)")
    ap.add_argument("--intervals", default=",".join(hp.DEFAULT_INTERVALS),
                    help="comma-separated ladder, finest first; every rung must be an "
                         "integer multiple of --fine-interval")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--hac-lags", type=int, default=10)
    ap.add_argument("--tag", default="", help="stem tag for the output files")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--halt-mask", dest="halt_mask", action="store_true", default=True)
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false")
    ap.add_argument("--n-demo", type=int, default=2)
    ap.add_argument("--n-demo-bars", type=int, default=30000)
    ap.add_argument("--demo-delay-ms", type=int, default=200)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.simplefilter("ignore")
    sessions = _load(a)
    intervals = [s.strip() for s in a.intervals.split(",") if s.strip()]
    print("sessions: %d  fine grid: %s  ladder: %s"
          % (len(sessions), a.fine_interval, " -> ".join(intervals)))
    if a.source == "demo":
        print("NOTE: --source demo plants a %dms uniform propagation kernel; the recovered"
              % a.demo_delay_ms)
        print("      half-impact horizon should sit near half that. Real inference needs "
              "--source load.")
    n_levels = a.n_levels if a.source != "demo" else min(a.n_levels, 3)
    per_day, summary, meta = hp.profile_ladder(sessions, intervals=intervals,
                                               fine_interval=a.fine_interval,
                                               n_levels=n_levels, hac_lags=a.hac_lags)
    if summary.empty:
        print("no rung produced estimates (frames too short for the ladder?)")
        return 1
    pd.set_option("display.width", 240)
    show = [c for c in ("interval", "dt_s", "n_days", "lambda_spy_from_es",
                        "lambda_spy_from_es_se", "lambda_spy_from_es_ratio",
                        "lambda_es_from_spy", "lambda_es_from_spy_ratio", "IS_mid_ES",
                        "CS_ES", "half_life_s", "lead_share", "zero_ret_frac_SPY",
                        "zero_ret_frac_ES") if c in summary.columns]
    print(summary[show].round(4).to_string(index=False))
    for sk, days in (meta.get("skipped") or {}).items():
        print("  NOTE: %s skipped %d session(s): %s" % (sk, len(days), sorted(days)[:3]))
    print()
    hi = meta.get("half_impact_spy_from_es_s")
    hi2 = meta.get("half_impact_es_from_spy_s")
    print("HALF-IMPACT HORIZON (ratio of cross-impact to its %s value crossing 0.5):"
          % meta.get("reference_interval"))
    print("  SPY <- ES : %s" % ("%.3fs" % hi if np.isfinite(hi) else
                                "not crossed on this ladder"))
    print("  ES <- SPY : %s" % ("%.3fs" % hi2 if np.isfinite(hi2) else
                                "not crossed on this ladder"))
    print("  (read each rung against its zero-return fractions -- staleness falls with dt "
          "and is the standing confounder for any grid trend)")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        tag = (a.tag + "_") if a.tag else ""
        p1 = os.path.join(a.out_dir, f"horizon_profile_per_day_{tag}{len(intervals)}rungs")
        p2 = os.path.join(a.out_dir, f"horizon_profile_summary_{tag}{len(intervals)}rungs")
        per_day.to_csv(p1 + ".csv", index=False)
        summary.to_csv(p2 + ".csv", index=False)
        print("wrote %s.csv / %s.csv" % (p1, p2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
