#!/usr/bin/env python3
"""run_copula.py -- tail dependence for SPY/ES: three copula exhibits.

A Gaussian DCC summarizes dependence by a linear rho_t whose tail-dependence coefficient
is ZERO for any rho < 1 -- it asserts SPY and ES become independent far enough into the
tail, backwards for a crash paper. This driver fits the copula menu (gaussian, frank, t,
clayton, gumbel, joe, survival rotations, bb1, sjc) on GARCH-margin pseudo-observations
and writes three tables:

  copula_regimes_*     return-tail dependence by a-priori regime: BIC-selected family per
                       day, lambda_L / lambda_U, and the BB1 lower-minus-upper contrast
                       (joint crashes vs joint rallies) with day-level sign-flip inference
  copula_liquidity_*   crash-tail dependence in thin- vs deep-book rows within each day --
                       the liquidity-contagion direction (lambda_L rising as books empty)
  copula_flows_*       the copula on OFI INNOVATIONS: tandem-selling tail dependence, the
                       parametric twin of the semicorrelation exhibit

Fit at the 1-second grid ONLY. At 10 ms, 80-89% of snapshots are stale tick repeats: the
rank PIT degenerates into ties and the copula measures staleness, not dependence. (The
driver runs this stage on the 1s frames and deliberately skips the fine grid.)

Usage
    python run_copula.py --source demo
    python run_copula.py --source load --pickle output/.../frames_1s.pkl \
        --volatile 2024-12-18,... --mwcb 2020-03-09,... --out-dir output/copula
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np

import copula_garch as cg
import paper_tables as pt


def _load(args):
    """Same pickle shapes and halt-mask policy as run_flow_correlation."""
    if args.source == "demo":
        import test_hy_correlation as th
        out = []
        for i in range(args.n_demo):
            rho = 0.3 if i < args.n_demo // 2 else 0.85
            regime = "benchmark" if i < args.n_demo // 2 else "volatile"
            df, _ = th._frame(n=args.n_demo_bars, rho=rho, refresh=0.6, seed=300 + i)
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
    ap = argparse.ArgumentParser(description="Copula tail dependence: regimes, liquidity "
                                             "split, and flow innovations")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--mwcb", default="", help="comma-separated YYYY-MM-DD labeled 'mwcb'")
    ap.add_argument("--families", default=",".join(cg._FAMILIES),
                    help="comma-separated copula menu for the BIC selection")
    ap.add_argument("--min-obs", type=int, default=1000,
                    help="minimum finite observations per day (returns or flow pairs)")
    ap.add_argument("--trim-min", type=float, default=15.0,
                    help="minutes trimmed at the open and close (auction desync)")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--ar-order", type=int, default=5,
                    help="AR prefilter order for the flow innovations")
    ap.add_argument("--min-rest-steps", type=int, default=-1,
                    help="fleeting-quote filter for the OFI input; -1 resolves from the "
                         "grid (inert at 1s)")
    ap.add_argument("--n-flip", type=int, default=20000,
                    help="day-level sign-flip draws for the asymmetry tests")
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false", default=True)
    ap.add_argument("--out-dir", default="", help="write csv/md/tex per table here")
    ap.add_argument("--tag", default="", help="output-stem tag (the driver passes the grid)")
    ap.add_argument("--n-demo", type=int, default=8)
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
    fams = tuple(f.strip() for f in a.families.split(",") if f.strip())
    tag = (a.tag + "_") if a.tag else ""
    print(f"{len(sessions)} sessions; families: {', '.join(fams)}")
    if a.min_rest_steps < 0:
        import cross_asset_pd_liquidity as ca
        a.min_rest_steps = ca.frequency_defaults(sessions[0][2])["min_rest_steps"]

    print("[copula] return copulas by day ...", flush=True)
    recs = cg.day_copula_records(sessions, fams, a.min_obs, a.trim_min, verbose=True)
    df1, n1 = cg.table_copula_regimes(sessions, fams, a.min_obs, a.trim_min,
                                      n_flip=a.n_flip, recs=recs)
    _emit(pt.Table("Return-tail dependence by regime (copula menu)", df1, n1),
          f"copula_regimes_{tag}t{int(a.trim_min)}m", a.out_dir)

    print("[copula] thin-vs-deep book split ...", flush=True)
    df2, n2 = cg.table_copula_liquidity(sessions, a.min_obs, a.trim_min, a.n_levels,
                                        n_flip=a.n_flip, verbose=True)
    _emit(pt.Table("Crash-tail dependence vs book depth (within-day)", df2, n2),
          f"copula_liquidity_{tag}t{int(a.trim_min)}m", a.out_dir)

    print("[copula] flow-innovation copulas ...", flush=True)
    df3, n3 = cg.table_copula_flows(sessions, fams, a.min_obs, a.trim_min, a.n_levels,
                                    a.min_rest_steps, a.ar_order, n_flip=a.n_flip,
                                    verbose=True)
    _emit(pt.Table("Tandem-flow tail dependence (OFI-innovation copula)", df3, n3),
          f"copula_flows_{tag}t{int(a.trim_min)}m", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
