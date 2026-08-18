#!/usr/bin/env python3
"""run_table5_inference.py -- day-clustered inference for the Table 5 corner log OR.

Companion to STAGE 4: that stage computes the pooled panels and prints the Woolf
z with a caveat that it overstates significance (the 2026-08-17 run printed
z = 159-214 by treating serially dependent bars as IID multinomial draws). This
runner re-scores the SAME pooled point estimates with days as the inference
unit: per-day 3x3 count matrices, day bootstrap (jackknife below 6 days) per
panel, and the era-robust within-pair sign-flip contrast of volatile vs matched
benchmark days. See table5_inference.py for the method reasoning.

Usage
    python run_table5_inference.py --pickle "output/.../frames_1s*.pkl" \
        --mwcb 2020-03-09,... --volatile D,D,... --benchmark D,D,... \
        --out output/nulls --n-boot 2000
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import table5_inference as t5i


def _load(gl, volatile):
    """STAGE 4's loader: pickles of (date, regime, df) 3-tuples; legacy 2-tuples are
    tagged from --volatile membership."""
    vol = t5i._dates(volatile)
    raw = []
    for f in sorted(glob.glob(gl or "")):
        with open(f, "rb") as fh:
            raw.extend(pickle.load(fh))
    out = []
    for r in raw:
        if len(r) == 3:
            out.append(r)
        else:
            date, df = r
            out.append((date, "volatile" if str(date)[:10] in vol else "benchmark", df))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pickle", required=True, help="glob of session-frame pickles")
    ap.add_argument("--mwcb", default="", help="MWCB dates, comma-separated")
    ap.add_argument("--volatile", default="", help="volatile dates (pair order matters)")
    ap.add_argument("--benchmark", default="", help="matched benchmark dates (pair order)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import mstbook_loader as ml
    import market_halts as mh

    sess = _load(args.pickle, args.volatile)
    if not sess:
        print("no sessions matched %r" % args.pickle)
        return 1
    if not ml.has_trade_flow(sess):
        print("frames carry no trade columns (attach_flow missing) -- nothing to infer on")
        return 1
    # halt-mask on load, exactly as STAGE 4 does: unmasked frames would count LULD
    # residual prints inside halt windows as tandem order flow in the MWCB panel.
    sess = [(d, r, mh.mask_frame(f)[0]) for d, r, f in sess]

    per_day = t5i.per_day_matrices(sess, ml.counts_from_frame, args.mwcb,
                                   ssr_fn=ml.session_is_ssr)
    if len(per_day) == 0:
        print("no session yielded a count matrix (skipped: %s)"
              % ", ".join(per_day.attrs.get("skipped", [])))
        return 1
    # STAGE 4's exSSR rebuild: the MWCB panel re-reported without Rule 201 sessions
    # (a dummy is unidentifiable at one restricted day; exclusion is the treatment).
    if "ssr" in per_day.columns:
        mw_ssr = per_day[(per_day["panel"] == "C MWCB") & per_day["ssr"]]
        if len(mw_ssr):
            ex = per_day[(per_day["panel"] == "C MWCB") & ~per_day["ssr"]].copy()
            if len(ex):
                ex["panel"] = "C MWCB exSSR"
                per_day = pd.concat([per_day, ex])

    os.makedirs(args.out, exist_ok=True)
    per_day_path = os.path.join(args.out, "table5_per_day.csv")
    per_day.to_csv(per_day_path)

    rows = []
    for panel in sorted(per_day["panel"].unique()):
        dep = t5i.panel_dependence(per_day, panel)
        boot = t5i.day_bootstrap(per_day, panel, n_boot=args.n_boot, seed=args.seed)
        rows.append({"panel": panel, "n_days": boot["n_days"],
                     "n_bars": int(per_day.loc[per_day["panel"] == panel, "n_bars"].sum()),
                     "log_OR": boot["log_OR"],
                     "log_OR_z_woolf": dep.get("log_OR_z", np.nan),
                     "ci_lo_day": boot["ci_lo"], "ci_hi_day": boot["ci_hi"],
                     "se_day": boot["se_day"], "z_day": boot["z_day"],
                     "method": boot.get("method", ""), "note": boot.get("note", "")})
    contrast = t5i.pair_contrast(per_day, args.volatile, args.benchmark,
                                 n_perm=args.n_perm, seed=args.seed)
    rows.append({"panel": "B-A within-pair contrast", "n_days": contrast["n_pairs"],
                 "log_OR": contrast["mean_pair_diff"],
                 "method": "sign-flip",
                 "note": "log_OR col = mean pair diff; p_within=%.4g p_free=%.4g n_pairs=%d"
                         % (contrast["p_within_pair"], contrast["p_free"],
                            contrast["n_pairs"])})
    inf = pd.DataFrame(rows).set_index("panel")
    inf_path = os.path.join(args.out, "table5_inference.csv")
    inf.to_csv(inf_path)

    print("\nTable 5 -- day-clustered inference for the corner log OR")
    print("  days are the inference unit: bootstrap over days (jackknife when n_days < %d),"
          % t5i.SMALL_PANEL_DAYS)
    print("  Woolf z kept as the scale reference STAGE 4 prints -- NOT as significance")
    with pd.option_context("display.width", 200):
        print(inf.round(4).to_string())
    print("\n  within-pair contrast: mean per-pair d(log_OR) = %.4f, p(sign-flip) = %.4g,"
          % (contrast["mean_pair_diff"], contrast["p_within_pair"]))
    print("  p(free permutation) = %.4g over %d pairs -- free-significant but within-pair-not"
          % (contrast["p_free"], contrast["n_pairs"]))
    print("  is the era-confounding signature, agreement is the clean outcome")
    skipped = per_day.attrs.get("skipped", [])
    if skipped:
        print("  skipped (no trade columns): %s" % ", ".join(skipped))
    print("  wrote %s and %s" % (per_day_path, inf_path))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    sys.exit(main())
