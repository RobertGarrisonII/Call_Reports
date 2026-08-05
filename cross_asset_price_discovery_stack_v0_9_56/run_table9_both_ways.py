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


def build_parser():
    """Public so the default surface is testable -- v0.9.56 flipped --with-dcc to DEFAULT ON, and
    a silent regression of that default would resurrect the unreachable-escape-hatch bug."""
    ap = argparse.ArgumentParser(description="Table 9 on Pearson, Hayashi-Yoshida and DCC d-correlation")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--spec", default="informational", help="standard | weighted | informational")
    ap.add_argument("--corr-window", type=int, default=100,
                    help="bars in the correlation window (paper: 100)")
    # DEFAULT ON since v0.9.56. It was opt-in, auto-triggered by STAGE 4c only when the selected
    # lag EQUALLED corr_window -- a condition the default configuration makes unreachable (a
    # search capped at pmax=12 can never land on 100), so the remedy for an artifact that fires
    # on every real run was gated behind a trigger that never could. The column costs one DCC fit
    # per estimator block; --no-dcc opts out.
    ap.add_argument("--with-dcc", dest="with_dcc", action="store_true", default=True,
                    help="add a third column using the DCC conditional correlation (DEFAULT). "
                         "Pearson and HY differ in how they treat asynchronicity but BOTH "
                         "difference a fixed corr-window box, which puts an MA term at exactly "
                         "lag W and makes the selected lag order track the window rather than "
                         "the data (test_svar_lag_artifact.py). DCC is recursive, has no box, "
                         "and is the column to quote when the lag caution fires.")
    ap.add_argument("--no-dcc", dest="with_dcc", action="store_false",
                    help="skip the DCC column (saves one DCC fit per estimator block; the lag "
                         "caution then has no lag-robust column to point at)")
    ap.add_argument("--n-lags", default="6",
                    help="fixed integer, or an information criterion: bic | aic | hq. "
                         "A criterion is resolved ONCE on the pooled SVAR frame and the chosen "
                         "order is printed and used for BOTH estimator blocks")
    ap.add_argument("--pmax", type=int, default=12,
                    help="largest lag considered when --n-lags is a criterion")
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
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.simplefilter("ignore")

    sessions = _load(a)
    regimes = sorted({r for _d, r, _f in sessions})
    print("sessions: %d  regimes: %s  bars/session: %s"
          % (len(sessions), ", ".join(regimes), ", ".join(str(len(f)) for _d, _r, f in sessions[:6])))
    if a.source == "demo":
        print("NOTE: --source demo uses a SYNTHETIC staleness DGP. The numbers below demonstrate the")
        print("      format and that the machinery runs; they say nothing about SPY/ES.")
    print()

    # Resolve the lag before anything else so it can be REPORTED, not just used. The paper's
    # footnote 17 records AIC pointing at 60 lags and 6 being used because the full model would
    # not run there -- exactly the kind of choice that should be visible in the output.
    import correlation_svar as cs
    n_lags, crit, ic = cs.resolve_n_lags(sessions, a.n_lags, pmax=a.pmax,
                                         spec=a.spec, corr_window=a.corr_window)
    if crit is not None:
        if n_lags is None:
            print("could not select a lag (no session had > pmax+5 usable rows); "
                  "pass --n-lags <int>", file=sys.stderr)
            return 1
        print("lag order: p=%d chosen by %s over p<=%d (pooled SVAR frame)" % (n_lags, crit.upper(), a.pmax))
        show = ic[["aic", "bic", "hqic"]].round(3)
        print(show.to_string())
        if n_lags >= a.pmax:
            print("  WARNING: the criterion selected p = pmax = %d, i.e. it is still improving at the"
                  " edge of the search. The chosen order is a BOUND, not an optimum -- re-run with a"
                  " larger --pmax before reporting it." % a.pmax)
        n_nan = int(ic[["aic", "bic", "hqic"]].isna().all(axis=1).sum())
        if n_nan:
            print("  NOTE: %d of %d candidate orders could not be scored (singular design at that p);"
                  " the selection is the minimum over the ones that could." % (n_nan, len(ic)))
        print("  (AIC is not consistent for lag order and on ~23k-bar samples runs away -- the "
              "paper's own footnote 17 reports it choosing 60; BIC's log(T) penalty is what keeps "
              "this finite.)")
        print()

    tbl = pt.table_correlation_irf_both_ways(
        sessions, spec=a.spec, ident=a.ident, cumulative=a.cumulative, n_boot=a.n_boot,
        n_lags=n_lags, horizon=a.horizon, corr_window=a.corr_window, min_obs=a.min_obs,
        seed=a.seed, n_jobs=a.n_jobs, with_dcc=a.with_dcc)
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
        stem = os.path.join(a.out_dir, f"table9_both_ways_{a.spec}_w{a.corr_window}_p{n_lags}")
        tbl.df.to_csv(stem + ".csv")
        with open(stem + ".md", "w") as fh:
            fh.write(tbl.to_markdown() + "\n")
        with open(stem + ".tex", "w") as fh:
            fh.write(tbl.to_latex(label=f"tab:table9_both_ways_w{a.corr_window}") + "\n")
        print(f"\nwrote {stem}.csv / .md / .tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
