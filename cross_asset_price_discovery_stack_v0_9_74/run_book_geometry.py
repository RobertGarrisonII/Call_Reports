#!/usr/bin/env python3
"""run_book_geometry.py -- the ||L|| (arc-length) book-geometry check, per session.

Computes, for every session and both assets: tau (normalized index-square arc
length of the depth profile), the Herfindahl of depth shares, the depth centroid
(= the exact full-sweep VWAP concession) per side, and the bid-ask centroid
asymmetry -- then renders the VERDICT the measure exists for: whether tau is
rank-equivalent to the Herfindahl (Spearman rho) and whether it adds any
incremental R^2 over the covering set {total depth, centroid, HHI} in explaining
short-horizon |mid returns|. See book_geometry.py for the derivation of why the
prediction is "redundant"; a failing verdict on real data is the interesting
outcome, and would be the reason to promote tau from diagnostic to regressor.

Usage
    python run_book_geometry.py --source demo
    python run_book_geometry.py --source load --pickle output/.../frames_1s.pkl \
        --volatile 2020-03-09,... --out-dir output/book_geometry --tag 1s
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import book_geometry as bg

TICKS = {"SPY": 0.01, "ES": 0.25}


def _demo_sessions(n_days, n_bars, seed=0):
    """Random-walk mids with gamma level depth (10 levels, both assets) -- depth
    VARIES across snapshots so the redundancy regression has something to rank."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_days):
        idx = pd.date_range("2024-07-24 09:30:00", periods=n_bars, freq="s",
                            tz="America/New_York")
        cols = {}
        for a, base, tick in (("SPY", 550.0, 0.01), ("ES", 5500.0, 0.25)):
            mid = base * np.exp(np.cumsum(rng.standard_normal(n_bars)) * 2e-5)
            scale = np.exp(rng.standard_normal(n_bars) * 0.4)      # common depth factor
            for i in range(1, 11):
                cols[f"{a}_bidprice_{i}"] = mid - tick * i
                cols[f"{a}_askprice_{i}"] = mid + tick * i
                cols[f"{a}_bidquantity_{i}"] = rng.gamma(2.0, 150.0, n_bars) * scale
                cols[f"{a}_askquantity_{i}"] = rng.gamma(2.0, 150.0, n_bars) * scale
        regime = "benchmark" if d < n_days // 2 else "volatile"
        out.append((f"d{d}", regime, pd.DataFrame(cols, index=idx)))
    return out


def _load(args):
    if args.source == "demo":
        return _demo_sessions(args.n_demo, args.n_demo_bars)
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
        masked = []
        for date, regime, df in out:
            mdf, _rep = mh.mask_frame(df)
            masked.append((date, regime, mdf))
        out = masked
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="book geometry: tau / Herfindahl / centroid + redundancy verdict")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=60,
                    help="forward steps for the |mid return| target in the redundancy check")
    ap.add_argument("--tag", default="", help="stem tag, e.g. 1s")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--halt-mask", dest="halt_mask", action="store_true", default=True)
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false")
    ap.add_argument("--n-demo", type=int, default=4)
    ap.add_argument("--n-demo-bars", type=int, default=2000)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.simplefilter("ignore")
    sessions = _load(a)
    print("sessions: %d  levels: %d  horizon: %d steps" % (len(sessions), a.n_levels, a.horizon))
    per_day, verdict = bg.table_book_geometry(sessions, n_levels=a.n_levels,
                                              tick_sizes=TICKS, horizon=a.horizon)
    if per_day.empty:
        print("no session produced geometry (missing book columns?)")
        return 1
    pd.set_option("display.width", 240)
    print(per_day.round(4).to_string(index=False))
    print()
    print("VERDICT: mean Spearman(tau, HHI) = %.3f ; mean incremental R^2 of tau over"
          % verdict["mean_spearman_tau_hhi"])
    print("         {spread, depth, centroid, HHI} = %+.4f  (n = %d session-assets)"
          % (verdict["mean_tau_increment"], verdict["n_session_assets"]))
    print("         reading: %s" % verdict["reading"])
    q0 = verdict.get("dwc_decay_scale", {})
    print("DWC:     fixed Q0 = %s (benchmark median inside size x5, held constant);"
          % (", ".join("%s=%.0f" % (a, v) for a, v in q0.items() if v is not None) or "n/a"))
    print("         mean incremental R^2 of the decay-weighted cost over the same set "
          "(which contains both of its Q0 limits) = %+.4f" % verdict["mean_dwc_increment"])
    print("         reading: %s" % verdict["reading_dwc"])
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        tag = (a.tag + "_") if a.tag else ""
        stem = os.path.join(a.out_dir, f"book_geometry_per_day_{tag}L{a.n_levels}")
        per_day.to_csv(stem + ".csv", index=False)
        print("wrote %s.csv" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
