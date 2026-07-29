#!/usr/bin/env python3
"""run_table9_both_ways.py -- report Table 9 on both dependent variables.

Eq. (5)'s dependent variable is the first difference of a Pearson correlation of returns
sampled on a fixed grid, at one second AND at ten milliseconds. That estimator is attenuated
by asynchronous quote updating (Epps): SPY and ES do not tick at the same instants, so in a
short bar one leg's mid is stale, the grid records a zero return for it, and the measured
correlation is pulled toward zero.

For this paper the problem is not precision, it is the DIRECTION of the bias. The severity of
the attenuation depends on how often each leg updates -- trading intensity -- and volume,
message traffic and liquidity demand are regressors in the same system. A response can be the
measurement error moving with its own regressor, and the naive estimator cannot separate that
from a market mechanism.

This driver estimates Table 9 twice on the identical sessions, identical window, identical
identification -- changing only the correlation estimator -- and prints them side by side with
the difference:

    stable across both columns    -> mechanism, safe to interpret
    shrinks toward zero under HY  -> partly the Epps channel
    significant only under HY     -> the naive estimator was masking it

Run it at BOTH aggregations. Attenuation grows as the bar shrinks, so the ten-millisecond
specification is where the gap should be largest.

Usage
    python run_table9_both_ways.py --source demo
    python run_table9_both_ways.py --source load --pickle output/1s_aggregated_*.pkl \
        --volatile 2020-03-09,2020-03-12,2020-03-16,2020-03-18
    python run_table9_both_ways.py --source load --pickle out.pkl --spec informational \
        --corr-window 100 --n-boot 499 --out-dir output/table9
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import paper_tables as pt


def _load(args):
    """-> List[(date, regime, df)]. Accepts the same pickle shapes run_analysis.load_sessions does."""
    if args.source == "demo":
        import test_hy_correlation as th          # the staleness DGP, so the gap is visible
        return [(f"d{i}", "benchmark" if i < args.n_demo // 2 else "volatile",
                 th._frame(n=args.n_demo_bars, refresh=args.demo_refresh, seed=100 + i)[0])
                for i in range(args.n_demo)]
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
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Table 9 on Pearson vs Hayashi-Yoshida d-correlation")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--spec", default="informational", help="standard | weighted | informational")
    ap.add_argument("--corr-window", type=int, default=100,
                    help="bars in the correlation window (paper: 100)")
    ap.add_argument("--n-lags", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--ident", default="cholesky", choices=["cholesky", "identity"])
    ap.add_argument("--cumulative", action="store_true")
    ap.add_argument("--n-boot", type=int, default=499, help="0 = point estimates only (fast)")
    ap.add_argument("--min-obs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n-demo", type=int, default=8)
    ap.add_argument("--n-demo-bars", type=int, default=6000)
    ap.add_argument("--demo-refresh", type=float, default=0.3,
                    help="demo only: per-bar quote refresh probability (1.0 = synchronous)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")

    sessions = _load(a)
    regimes = sorted({r for _d, r, _f in sessions})
    print("sessions: %d  regimes: %s  bars/session: %s"
          % (len(sessions), ", ".join(regimes), ", ".join(str(len(f)) for _d, _r, f in sessions[:6])))
    if a.source == "demo":
        print("NOTE: --source demo uses a SYNTHETIC staleness DGP. The numbers below demonstrate the")
        print("      format and that the machinery runs; they say nothing about SPY/ES.")
    print()

    tbl = pt.table_correlation_irf_both_ways(
        sessions, spec=a.spec, ident=a.ident, cumulative=a.cumulative, n_boot=a.n_boot,
        n_lags=a.n_lags, horizon=a.horizon, corr_window=a.corr_window, min_obs=a.min_obs,
        seed=a.seed, n_jobs=a.n_jobs)
    if tbl.df.empty:
        print(tbl.notes or "no output")
        return 1
    pd.set_option("display.width", 240)
    print(tbl.to_string())

    # the headline: how much of each response is measurement artifact
    if isinstance(tbl.df.columns, pd.MultiIndex) and "Delta (HY-Pearson)" in tbl.df.columns.get_level_values(0):
        print("\nLargest |HY - Pearson| gaps (candidate Epps artifacts):")
        d = tbl.df["Delta (HY-Pearson)"].apply(pd.to_numeric, errors="coerce")
        flat = d.stack().abs().sort_values(ascending=False)
        for (shock, regime), v in list(flat.items())[:6]:
            print("    %-16s %-12s  %+.4f" % (shock, regime, d.loc[shock, regime]))

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        stem = os.path.join(a.out_dir, f"table9_both_ways_{a.spec}_w{a.corr_window}")
        tbl.df.to_csv(stem + ".csv")
        with open(stem + ".md", "w") as fh:
            fh.write(tbl.to_markdown() + "\n")
        with open(stem + ".tex", "w") as fh:
            fh.write(tbl.to_latex(label=f"tab:table9_both_ways_w{a.corr_window}") + "\n")
        print(f"\nwrote {stem}.csv / .md / .tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
