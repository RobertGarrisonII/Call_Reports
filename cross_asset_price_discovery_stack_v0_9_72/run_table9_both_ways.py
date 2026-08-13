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
    # Frames are cached PRE-mask since v0.9.57 (the halt policy is applied at analysis time so
    # the frame on disk stays a faithful record). Every other consumer masks on load; this
    # driver estimates the paper's HEADLINE table, so silently skipping the mask here means
    # Table 9 alone would be estimated halt-INCLUDED -- LULD/MWCB windows contributing spliced
    # pseudo-returns to exactly the volatile sessions the table is about.
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
    # v0.9.63: the RealBar column and the panel estimation are DEFAULT ON. RealBar is the
    # recommended dependent variable -- non-overlapping per-bar realized correlation (Fisher-z),
    # structurally immune to the window-induced MA artifact; the panel estimation builds lags
    # strictly within-day with day fixed effects instead of stacking sessions across overnight
    # seams around one intercept.
    ap.add_argument("--with-bar", dest="with_bar", action="store_true", default=True,
                    help="add the RealBar column (DEFAULT): d Fisher-z of the non-overlapping "
                         "per-bar realized correlation")
    ap.add_argument("--no-bar", dest="with_bar", action="store_false",
                    help="skip the RealBar column")
    ap.add_argument("--bar-seconds", type=int, default=60,
                    help="RealBar bar length in seconds (default 60; on a 10ms frame each bar "
                         "holds 6,000 sub-returns)")
    ap.add_argument("--panel", choices=["fe", "stack"], default="fe",
                    help="'fe' (DEFAULT): within-day lags + day fixed effects; 'stack' reproduces "
                         "the pre-v0.9.63 stacked estimation (seam-crossing lags, one intercept)")
    ap.add_argument("--mean-group", dest="mean_group", action="store_true", default=True,
                    help="append the mean-group (per-day) estimate with cross-day dispersion "
                         "(DEFAULT) -- the slope-heterogeneity check on the pooled column")
    ap.add_argument("--no-mean-group", dest="mean_group", action="store_false")
    # v0.9.72: a criterion is resolved on the RealBar BAR frame (window-free) whenever the
    # RealBar column is on. Resolving it on the Pearson frame scored the one design whose
    # selected order provably tracks the corr_window box rather than the data (footnote 17's
    # "AIC points at 60", the 2026-08-04 run's BIC still falling at pmax): the selection kept
    # landing on the SEARCH BOUND because the frame it scored carries an MA spike the search
    # can never reach. The bar frame has no window to chase, so its argmin can be dynamics.
    # The Pearson-frame path is still computed and PRINTED as the footnote-17 diagnostic.
    ap.add_argument("--n-lags", default="6",
                    help="fixed integer, or an information criterion: bic | aic | hq. "
                         "A criterion is resolved ONCE -- on the window-free RealBar bar frame "
                         "when --with-bar (the default), else on the pooled Pearson frame with a "
                         "loud caution -- and the chosen order is used for ALL estimator blocks")
    ap.add_argument("--pmax", type=int, default=12,
                    help="largest lag considered when --n-lags is a criterion")
    ap.add_argument("--flat-tol", type=float, default=2.0,
                    help="IC-unit half-width of the reported flatness set (candidate orders "
                         "whose total criterion sits within this of the minimum)")
    ap.add_argument("--lag-band", default="",
                    help="lo:hi -- refit the WHOLE table at every lag order in the band and "
                         "report per-cell sign/star stability (band-stable = one sign at every "
                         "depth and the same 10%%-significance verdict at >=90%% of depths). "
                         "Writes table9_lag_band_*.csv next to the main table.")
    ap.add_argument("--band-boot", type=int, default=199,
                    help="bootstrap draws per depth inside --lag-band (smaller than --n-boot "
                         "because the sweep multiplies cost by the band width; 0 = points only, "
                         "which reduces the verdict to sign stability)")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--ident", default="cholesky", choices=["cholesky", "identity"])
    ap.add_argument("--cumulative", action="store_true")
    ap.add_argument("--n-boot", type=int, default=499, help="0 = point estimates only (fast)")
    ap.add_argument("--min-obs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--out-dir", default="")
    # v0.9.64: loaded frames are cached PRE-mask (v0.9.57 policy); this driver must therefore
    # apply the halt mask itself, exactly as run_analysis does. Before this flag existed the
    # headline table was the ONE estimator running halt-included.
    ap.add_argument("--halt-mask", dest="halt_mask", action="store_true", default=True,
                    help="NaN each leg's market columns inside that leg's halt windows on load "
                         "(DEFAULT; the finite-row design mask then excludes halt, seam, and "
                         "touching lag windows)")
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false",
                    help="estimate on the raw frames, halts included (A/B use only)")
    ap.add_argument("--n-demo", type=int, default=8)
    ap.add_argument("--n-demo-bars", type=int, default=6000)
    ap.add_argument("--demo-refresh", type=float, default=0.3,
                    help="demo only: per-bar quote refresh probability (1.0 = synchronous)")
    return ap


def parse_lag_band(s):
    """'lo:hi' -> (lo, hi) with 1 <= lo <= hi. Raises ValueError on anything else, so a typo
    fails before the expensive sweep rather than after it."""
    lo, sep, hi = str(s).partition(":")
    if not sep:
        raise ValueError(f"--lag-band expects lo:hi, got {s!r}")
    lo, hi = int(lo), int(hi)
    if lo < 1 or hi < lo:
        raise ValueError(f"--lag-band needs 1 <= lo <= hi, got {s!r}")
    return lo, hi


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
    #
    # v0.9.72: the criterion is scored on the RealBar BAR frame when the RealBar column is on.
    # The Pearson frame's dCorr differences a corr_window-bar rolling box, which plants an MA
    # spike at exactly lag W: a criterion scored there tracks the window (or climbs to pmax when
    # the search cannot reach W) no matter what the data do -- selecting the common order on that
    # frame meant every column inherited a lag chosen by the estimator artifact. The bar frame's
    # non-overlapping bars share no data, so its argmin can be dynamics. The Pearson path is
    # still computed and printed below as the footnote-17 diagnostic; it decides nothing.
    import correlation_svar as cs
    sel_method = "bar" if a.with_bar else "rolling"
    n_lags, crit, ic = cs.resolve_n_lags(sessions, a.n_lags, pmax=a.pmax,
                                         spec=a.spec, corr_window=a.corr_window,
                                         bar_seconds=a.bar_seconds, panel=a.panel,
                                         corr_method=sel_method)
    if crit is not None:
        if n_lags is None:
            print("could not select a lag (no session had > pmax+5 usable rows on the %s frame); "
                  "pass --n-lags <int>" % sel_method, file=sys.stderr)
            return 1
        frame_name = ("RealBar bar frame, window-free" if sel_method == "bar"
                      else "pooled Pearson SVAR frame")
        print("lag order: p=%d chosen by %s over p<=%d (%s)" % (n_lags, crit.upper(), a.pmax, frame_name))
        show = ic[["aic", "bic", "hqic"]].round(3)
        print(show.to_string())
        if n_lags == 0:
            # A criterion CAN return 0 -- the data carry no VAR dynamics at this bar size. A
            # VAR(0) has no impulse response to compute, so floor at 1 and say so: p*=0 is
            # information (the criterion found nothing), not a value to fit.
            print("  NOTE: the criterion selected p=0 -- no dynamics at all. A VAR(0) has no "
                  "impulse response, so p=1 is fitted and the responses should be read as "
                  "impact-only.")
            n_lags = 1
        if sel_method == "rolling":
            print("  WARNING: --no-bar left the criterion nothing but the WINDOW-BEARING Pearson "
                  "frame to score. d(rolling correlation) carries an MA spike at exactly lag "
                  "corr_window=%d, so this selection tracks the window (or its own search bound), "
                  "not the data -- the paper's footnote-17 failure. Restore --with-bar for a "
                  "window-free selection." % a.corr_window)
            diag = cs.lag_diagnosis(n_lags, corr_window=a.corr_window, pmax=a.pmax,
                                    corr_method="rolling")
            if not diag["ok"]:
                print("  " + diag["text"])
        elif n_lags >= a.pmax:
            print("  WARNING: the criterion selected p = pmax = %d, i.e. it is still improving at the"
                  " edge of the search. The chosen order is a BOUND, not an optimum -- re-run with a"
                  " larger --pmax before reporting it. (On the bar frame there is no window spike to"
                  " chase, so unlike the Pearson frame a larger --pmax CAN converge here.)" % a.pmax)
        n_nan = int(ic[["aic", "bic", "hqic"]].isna().all(axis=1).sum())
        if n_nan:
            print("  NOTE: %d of %d candidate orders could not be scored (singular design at that p);"
                  " the selection is the minimum over the ones that could." % (n_nan, len(ic)))
        # Selection-robustness block: agreement across criteria, flatness of the criterion,
        # and the per-day vote -- the three ways one clean-looking argmin can be hollow.
        try:
            rob = cs.lag_robustness(sessions, pmax=a.pmax, criterion=crit, flat_tol=a.flat_tol,
                                    spec=a.spec, corr_method=sel_method, corr_window=a.corr_window,
                                    bar_seconds=a.bar_seconds, panel=a.panel)
        except Exception as e:
            rob = None
            print("  (robustness diagnostics unavailable: %s)" % e)
        if rob is not None and rob["p"] is not None:
            pk = rob["picks"]
            print("  criterion agreement: AIC->%s  BIC->%s  HQ->%s  (%s)"
                  % (pk.get("aic"), pk.get("bic"), pk.get("hqic"),
                     "all agree" if rob["agree"] else "DISAGREE -- the choice of criterion is "
                     "doing part of the choosing; BIC is the consistent one for lag order"))
            fs = rob["flat_set"]
            if len(fs) > 1:
                print("  flatness: %d orders within %.1f IC units of the minimum: %s -- the argmin "
                      "is one member of a near-tie, so any finding must survive the whole set "
                      "(--lag-band %d:%d checks exactly that)"
                      % (len(fs), rob["flat_tol"], fs, min(fs), max(fs)))
            else:
                print("  flatness: the minimum is isolated (no other order within %.1f IC units)"
                      % rob["flat_tol"])
            if rob["modal"] is not None and len(rob["per_day"]) > 1:
                votes = pd.Series(list(rob["per_day"].values())).value_counts().sort_index()
                print("  per-day selection: %s  -> modal p=%d (%d%% of days)%s"
                      % (", ".join("p=%d x%d" % (p_, c_) for p_, c_ in votes.items()),
                         rob["modal"], round(100 * rob["modal_share"]),
                         "" if rob["modal"] == rob["p"] else
                         " -- NOTE the pooled choice differs from the typical day's: the pooled "
                         "sample size, not any one session's dynamics, is deciding"))
        print("  (AIC is not consistent for lag order and on ~23k-bar samples runs away -- the "
              "paper's own footnote 17 reports it choosing 60; BIC's log(T) penalty is what keeps "
              "this finite.)")
        # The footnote-17 diagnostic: what the same criterion says on the window-bearing Pearson
        # frame. Printed for comparison with the paper -- it decides nothing above.
        if sel_method == "bar":
            try:
                p_roll, _tab = cs.select_svar_lag(sessions, spec=a.spec, corr_method="rolling",
                                                  corr_window=a.corr_window, criterion=crit,
                                                  pmax=a.pmax, panel=a.panel)
                if p_roll is not None:
                    d_roll = cs.lag_diagnosis(p_roll, corr_window=a.corr_window, pmax=a.pmax,
                                              corr_method="rolling")
                    print("  footnote-17 diagnostic (NOT used): on the window-bearing Pearson frame "
                          "the same %s selects p*=%d%s" % (crit.upper(), int(p_roll),
                          " -- " + d_roll["text"] if not d_roll["ok"] else
                          "; not at the bound and not the window here"))
            except Exception:
                pass
        print()

    tbl = pt.table_correlation_irf_both_ways(
        sessions, spec=a.spec, ident=a.ident, cumulative=a.cumulative, n_boot=a.n_boot,
        n_lags=n_lags, horizon=a.horizon, corr_window=a.corr_window, min_obs=a.min_obs,
        seed=a.seed, n_jobs=a.n_jobs, with_dcc=a.with_dcc, with_bar=a.with_bar,
        bar_seconds=a.bar_seconds, panel=a.panel, mean_group=a.mean_group)
    if tbl.df.empty:
        print(tbl.notes or "no output")
        return 1
    pd.set_option("display.width", 240)
    print(tbl.to_string())
    mg = getattr(tbl, "mean_group", None)
    if mg is not None and mg["point"].empty:
        skip = mg.get("n_skipped") or {}
        print()
        print("MEAN-GROUP: no day cleared the per-day identification bar (%s). A per-day VAR(p)"
              % (", ".join(f"{k}={v}" for k, v in skip.items() if v) or "no usable days"))
        print("on k variables needs > 1 + k*p + 10 usable bars; before v0.9.64 such days were")
        print("fitted anyway and lstsq's minimum-norm pseudo-responses were averaged in. Lower")
        print("--n-lags, use a coarser --bar-seconds, or run more/longer sessions.")
    if mg is not None and not mg["point"].empty:
        print()
        print("MEAN-GROUP (per-day SVAR, cross-day mean; the slope-heterogeneity check on the")
        print("pooled column -- Pesaran-Smith: agreement validates pooling, divergence IS the")
        print("heterogeneity finding). VAR(%d) per day; response x100; day-level t in brackets;"
              % mg["n_lags"])
        print("n_days per cell in the last block.")
        disp = mg["point"].round(3).astype(str)
        t = mg["tstat"]
        for c in disp.columns:
            disp[c] = disp[c] + " [" + t[c].round(2).astype(str) + "]"
        print(disp.to_string())
        wm = mg.get("point_weighted")
        if wm is not None and not wm.empty:
            print("effective-sample-weighted mean (weight = usable rows/day; agreement with the")
            print("unweighted MG above says the mean is not driven by its shortest days):")
            print(wm.round(3).to_string())
        print("n_days:")
        print(mg["n_days"].to_string())
        skip = mg.get("n_skipped") or {}
        if any(skip.values()):
            print("days excluded: " + ", ".join(f"{k}={v}" for k, v in skip.items() if v)
                  + "  (underdetermined = fewer effective rows than VAR parameters/eq + 10)")

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
        if crit is not None and not ic.empty:
            ic.assign(frame=sel_method).to_csv(
                os.path.join(a.out_dir, f"table9_lag_ic_{a.spec}_w{a.corr_window}.csv"))

    # ── lag-band sweep: the finding must not depend on the lag choice at all ──────────────
    # The table above is fitted at ONE order. The selection block reports how defensible that
    # order is; this sweep removes the question entirely by refitting the whole table at every
    # order in the band and keeping, per cell, only what survives all of them. A cell that is
    # band-stable cannot be an artifact of the lag choice, because no lag choice remains.
    if a.lag_band:
        lo, hi = parse_lag_band(a.lag_band)
        print("\n[lag-band] refitting the table at every p in %d..%d (n_boot=%d per depth) ..."
              % (lo, hi, int(a.band_boot or 0)), flush=True)
        tabs = {}
        for p in range(lo, hi + 1):
            print("[lag-band] p=%d" % p, flush=True)
            t = pt.table_correlation_irf_both_ways(
                sessions, spec=a.spec, ident=a.ident, cumulative=a.cumulative,
                n_boot=a.band_boot, n_lags=p, horizon=a.horizon, corr_window=a.corr_window,
                min_obs=a.min_obs, seed=a.seed, n_jobs=a.n_jobs, with_dcc=a.with_dcc,
                with_bar=a.with_bar, bar_seconds=a.bar_seconds, panel=a.panel,
                mean_group=False)
            if not t.df.empty:
                tabs[p] = t.df
        if not tabs:
            print("[lag-band] no depth produced a table; nothing to report")
        else:
            stab = cs.band_stability(tabs)
            n_stable = int(stab["band_stable"].sum())
            print("\nLAG-BAND VERDICT over p=%d..%d: %d of %d cells band-stable "
                  "(one sign at every depth, same 10%%-significance verdict at >=90%% of depths)."
                  % (lo, hi, n_stable, len(stab)))
            if a.band_boot == 0:
                print("  (n_boot=0 in the sweep: no stars were computed, so the verdict reduces "
                      "to sign stability alone)")
            unstable = stab[~stab["band_stable"]]
            if not unstable.empty:
                print("  NOT band-stable (read these cells as lag-dependent, whatever the main "
                      "table says):")
                for _i, r in unstable.iterrows():
                    why = []
                    if not r["sign_consistent"]:
                        why.append("zero at table precision" if r["est_min"] == 0 == r["est_max"]
                                   else "sign flips" if r["n_finite"] == r["n_depths"]
                                   else "not estimable at every depth")
                    if r["star_agreement"] < 0.9:
                        why.append("stars at only %d%% of depths" % round(100 * r["starred_share"]))
                    print("    %-14s %-12s %-16s [%+.3f, %+.3f]  (%s)"
                          % (r["estimator"], r["regime"], r["shock"],
                             r["est_min"], r["est_max"], "; ".join(why) or "borderline"))
            if a.out_dir:
                bstem = os.path.join(a.out_dir,
                                     f"table9_lag_band_{a.spec}_w{a.corr_window}_p{lo}-{hi}")
                stab.to_csv(bstem + ".csv", index=False)
                print("wrote %s.csv" % bstem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
