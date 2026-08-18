#!/usr/bin/env python3
"""run_flow_correlation.py -- tandem order flow as a time series: three exhibits.

The paper's title phenomenon is co-moving ORDER FLOW, measured until now only statically
(one OFI-innovation correlation per regime). This driver puts the flow correlation itself
on a non-overlapping bar clock (Fisher-z of the per-bar correlation of the two legs'
AR-prefiltered OFI innovations) and writes three tables:

  flow_corr_tier1_*        the a-priori-regime description: regime means with
                           day-clustered SEs, the volatile-vs-benchmark permutation p,
                           and the within-day relation to the book state
  flow_corr_mediation_*    does tandem flow MEDIATE the volatility -> return-correlation
                           link? FE panel with day-clustered SEs; total-vs-direct RV_ES
                           effect; indirect share with a day-level cluster-bootstrap CI
  flow_corr_ms_regimes_*   DATA-DRIVEN regimes: per-day 2-state Markov switching on the
                           z_flow series with a parametric-bootstrap LR against the
                           1-state AR(1) null (no arbitrary ranges), and the alignment
                           of the data's own regimes with the a-priori day labels

Usage
    python run_flow_correlation.py --source demo
    python run_flow_correlation.py --source load --pickle output/.../frames_1s.pkl \
        --volatile 2024-12-18,... --mwcb 2020-03-09,... --out-dir output/flow_corr
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import flow_correlation as fc
import paper_tables as pt


def _load(args):
    """-> List[(date, regime, df)], halt-masked unless --no-halt-mask. Same pickle shapes
    as run_table9_both_ways; --mwcb days are labeled 'mwcb' (their own Tier 1 row) even
    when the pickle carries only 2-tuples."""
    if args.source == "demo":
        import test_hy_correlation as th
        out = []
        for i in range(args.n_demo):
            rho = 0.3 if i < args.n_demo // 2 else 0.85
            regime = "benchmark" if i < args.n_demo // 2 else "volatile"
            df, _ = th._frame(n=args.n_demo_bars, rho=rho, refresh=0.6, seed=100 + i)
            out.append((f"d{i}", regime, df))
        return out
    paths = sorted(glob.glob(args.pickle))
    if not paths:
        raise SystemExit(f"no pickle matched {args.pickle!r}")
    raw = []
    for p in paths:
        with open(p, "rb") as fh:
            raw.extend(pickle.load(fh))
    vol = {d.strip() for d in (args.volatile or "").split(",") if d.strip()}
    mwcb = {d.strip() for d in (args.mwcb or "").split(",") if d.strip()}
    out = []
    for rec in raw:
        if len(rec) == 3:
            date, regime, df = rec
        else:
            date, df = rec
            regime = None
        d10 = str(date)[:10]
        if d10 in mwcb:
            regime = "mwcb"
        elif d10 in vol:
            regime = "volatile"
        elif regime is None:
            regime = "benchmark"
        out.append((date, regime, df))
    if getattr(args, "halt_mask", True):
        import market_halts as mh
        masked, n_rows = [], 0
        for date, regime, df in out:
            mdf, rep = mh.mask_frame(df)
            masked.append((date, regime, mdf))
            n_rows += sum(rep.values())
        if n_rows:
            print(f"halt mask: NaN'd {n_rows} leg-rows across {len(masked)} sessions "
                  f"(--no-halt-mask to skip)")
        out = masked
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="Tandem order flow as a time series: Tier 1 "
                                             "regimes, Tier 2 mediation, MS regimes")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--mwcb", default="", help="comma-separated YYYY-MM-DD labeled 'mwcb' "
                                               "(their own Tier 1 row)")
    ap.add_argument("--bar-seconds", type=int, default=60,
                    help="non-overlapping bar length for the flow/return correlations")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--ar-order", type=int, default=5,
                    help="AR prefilter order for the OFI innovations")
    ap.add_argument("--min-rest-steps", type=int, default=-1,
                    help="fleeting-quote filter width in grid steps for the OFI input; "
                         "-1 (default) resolves from the grid via frequency_defaults "
                         "(inert at 1s, engaged at sub-second grids)")
    ap.add_argument("--n-lags", type=int, default=3,
                    help="own-lag depth in the mediation panel (bar clock; fixed ex ante, "
                         "no criterion -- the bar DV has no window to chase)")
    ap.add_argument("--n-boot", type=int, default=499,
                    help="day-level cluster-bootstrap draws for the mediation CI")
    ap.add_argument("--controls-standardize", choices=["pooled", "day"], default="pooled",
                    help="mediation-panel control scaling: 'pooled' (legacy default; one "
                         "SD across the whole panel) or 'day' (within-day; makes the "
                         "panel invariant to a per-day level+scale break in the "
                         "spread/state controls, e.g. the 2025-11-03 tick regime)")
    ap.add_argument("--ms-boot", type=int, default=99,
                    help="parametric-bootstrap LR draws per day PER STAGE for the "
                         "sequential MS regime-count test")
    ap.add_argument("--k-max", type=int, default=4,
                    help="largest regime count the sequential LR may select per day")
    ap.add_argument("--window-minutes", type=int, default=30,
                    help="within-day window length for the price-discovery link panel")
    ap.add_argument("--pd-lags", type=int, default=-1,
                    help="VECM lag order for the per-window CS/IS estimates; -1 (default) "
                         "resolves from the grid via frequency_defaults so the model keeps "
                         "~5s of wall-clock memory at any interval (5 lags at 1s, 60 at "
                         "10ms) -- a fixed lag COUNT means a different memory span on "
                         "every grid")
    ap.add_argument("--n-perm", type=int, default=20000,
                    help="day-level permutation draws for the Tier 1 contrast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false", default=True,
                    help="skip the halt mask (Table-9-style caution: MWCB windows would "
                         "contribute spliced pseudo-flows to exactly the stressed sessions)")
    ap.add_argument("--out-dir", default="", help="write csv/md/tex per table here")
    ap.add_argument("--tag", default="", help="grid tag folded into output stems (e.g. 1s, "
                                              "10ms) so a fine-grid pass does not overwrite "
                                              "the coarse one")
    ap.add_argument("--n-demo", type=int, default=10)
    ap.add_argument("--n-demo-bars", type=int, default=6000)
    return ap


def _emit(tbl, stem, out_dir):
    print()
    print(tbl.to_string())
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, stem)
        tbl.df.to_csv(path + ".csv")
        with open(path + ".md", "w") as fh:
            fh.write(tbl.to_markdown() + "\n")
        with open(path + ".tex", "w") as fh:
            fh.write(tbl.to_latex(label=stem) + "\n")
        print(f"wrote {path}.csv / .md / .tex")


def main(argv=None):
    warnings.simplefilter("ignore")
    a = build_parser().parse_args(argv)
    sessions = _load(a)
    print(f"{len(sessions)} sessions; bar={a.bar_seconds}s, ar_order={a.ar_order}")
    if a.min_rest_steps < 0 or a.pd_lags < 0:
        import cross_asset_pd_liquidity as ca
        fd = ca.frequency_defaults(sessions[0][2])
        if a.min_rest_steps < 0:
            a.min_rest_steps = fd["min_rest_steps"]
            print(f"min_rest_steps resolved from grid: {a.min_rest_steps}")
        if a.pd_lags < 0:
            a.pd_lags = fd["n_lags"]
            print(f"pd-lags resolved from grid: {a.pd_lags} "
                  f"(~{fd['n_lags'] * fd['dt']:.1f}s of memory)")

    tag = (a.tag + "_") if a.tag else ""
    print("[flow] building per-day bars ...", flush=True)
    bars = fc.per_day_bars(sessions, a.bar_seconds, a.n_levels, a.min_rest_steps,
                           a.ar_order, verbose=True)
    print(f"[flow] {len(bars)} usable days", flush=True)

    df1, n1 = fc.table_flow_corr_regimes(sessions, a.bar_seconds, a.n_levels,
                                         a.min_rest_steps, a.ar_order,
                                         n_perm=a.n_perm, seed=a.seed, bars=bars)
    _emit(pt.Table("Tandem flow by a-priori regime (Tier 1)", df1.round(4), n1),
          f"flow_corr_tier1_{tag}b{a.bar_seconds}s", a.out_dir)

    # Tier 1 per-day rows with the ES top-of-book staleness covariate (es_stale_frac,
    # halt/early-close rows out of the denominator) -- the cross-era control lives in
    # the same CSV as the per-day estimands.
    per_day1 = fc.tier1_per_day(sessions, bars=bars)
    if a.out_dir and not per_day1.empty:
        os.makedirs(a.out_dir, exist_ok=True)
        pd_path = os.path.join(a.out_dir, f"flow_corr_tier1_per_day_{tag}b{a.bar_seconds}s.csv")
        per_day1.to_csv(pd_path, index=False)
        print(f"wrote {pd_path} ({len(per_day1)} day rows, es_stale_frac included)")

    print("[flow] downside/upside semicorrelation asymmetry ...", flush=True)
    dfa, na = fc.table_flow_corr_asymmetry(sessions, a.n_levels, a.min_rest_steps,
                                           a.ar_order, n_flip=a.n_perm, seed=a.seed)
    _emit(pt.Table("Tandem-flow asymmetry: joint selling vs joint buying", dfa.round(4), na),
          f"flow_corr_asymmetry_{tag}", a.out_dir)

    print("[flow] mediation panel ...", flush=True)
    try:
        df2, n2 = fc.table_flow_corr_mediation(sessions, a.bar_seconds, a.n_levels,
                                               a.min_rest_steps, a.ar_order,
                                               n_lags=a.n_lags, n_boot=a.n_boot,
                                               seed=a.seed, bars=bars,
                                               controls_standardize=a.controls_standardize)
        # non-default scaling gets its own stem so a 'day' pass never overwrites the
        # legacy pooled table
        med_stem = f"flow_corr_mediation_{tag}b{a.bar_seconds}s_p{a.n_lags}" + \
            ("" if a.controls_standardize == "pooled" else "_ctlday")
        _emit(pt.Table("Does tandem flow mediate the RV -> return-correlation link? (Tier 2)",
                       df2, n2), med_stem, a.out_dir)
    except Exception as e:
        print(f"[flow] mediation FAILED: {type(e).__name__}: {e}")

    print(f"[flow] Markov-switching regimes (sequential K, {a.ms_boot} LR draws/stage, "
          f"k_max={a.k_max}) ...", flush=True)
    recs, smap = fc.ms_flow_regimes(sessions, a.bar_seconds, a.n_levels, a.min_rest_steps,
                                    a.ar_order, B=a.ms_boot, seed=a.seed, k_max=a.k_max,
                                    bars=bars, verbose=True, return_series=True)
    df3, n3 = fc.table_flow_corr_ms_regimes(sessions, a.bar_seconds, a.n_levels,
                                            a.min_rest_steps, a.ar_order,
                                            B=a.ms_boot, seed=a.seed, k_max=a.k_max,
                                            bars=bars, recs=recs)
    _emit(pt.Table("Data-driven tandem-flow regimes (Markov switching, bootstrap LR)",
                   df3, n3),
          f"flow_corr_ms_regimes_{tag}b{a.bar_seconds}s", a.out_dir)

    print(f"[flow] price-discovery link ({a.window_minutes}m windows) ...", flush=True)
    try:
        df4, n4 = fc.table_flow_pd_link(sessions, a.bar_seconds, a.n_levels,
                                        a.min_rest_steps, a.ar_order,
                                        window_minutes=a.window_minutes, n_lags=a.pd_lags,
                                        B=a.ms_boot, seed=a.seed, k_max=a.k_max,
                                        bars=bars, recs=recs, series_map=smap)
        _emit(pt.Table("Tandem flow and price discovery (within-day window panel)",
                       df4, n4),
              f"flow_corr_pd_link_{tag}b{a.bar_seconds}s_w{a.window_minutes}m", a.out_dir)
    except Exception as e:
        print(f"[flow] pd link FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
