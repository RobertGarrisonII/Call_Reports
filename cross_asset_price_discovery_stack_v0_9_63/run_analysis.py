"""
run_analysis.py
===============
Single-entry DRIVER for the entire cross-asset price-discovery stack. One command
loads (or extracts) the per-session book frames and runs every analysis stage,
writing tables + a summary report to ./output/.

Data sources
------------
  --source demo       generate synthetic sessions (runs anywhere; for smoke tests)
  --source load --pickle PATH
                      load a previously-extracted List[(date, df)] (or [(date, regime, df)])
  --source extract    pull via the MayStreet CLI (mstbook-query + mstwx-lakequery) through
                      mstbook_loader -- book + trade tape, no SQL/Athena; needs an explicit date
                      universe (--dates / --date-range / --volatile / --benchmark)
                      are --dates, else the --volatile/--benchmark union, else a built-in list

Stages (toggle with --only / --skip)
  liquidity_curves, information_shares, liquidity_conditional, panel, cross_impact,
  dcc, irf, robustness, microstructure

Examples
  python run_analysis.py --source demo --quick
  python run_analysis.py --source load --pickle output/1s_aggregated_*.pkl --volatile 2024-08-05,2024-10-31
  python run_analysis.py --source extract --interval 1s --n-levels 10 \
      --volatile 2025-04-07,2025-04-08,2025-04-09   # these ARE the dates extracted (and labeled volatile)
  python run_analysis.py --source extract --dates 2025-04-07,2025-04-08 --benchmark 2024-09-03
"""
from __future__ import annotations
import argparse, os, sys, json, pickle, logging, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd

import liquidity_curve_metrics as lcm
import legacy_tables as legacy
import price_discovery_shares as pds
import cross_asset_pd_liquidity as ca
import cross_impact as cimp
import dcc_garch as dg
import irf as irfm
import robustness as rb
import noise_robust_cov as nrc
import microstructure_diagnostics as mdx
import functional_liquidity as fl
import jump_robust as jr
import robust_prices as rp
import ecm_sde as es

LOG = logging.getLogger("driver")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
EPS = 1e-12


# ── frequency / aggregation control ───────────────────────────────────────────
def interval_seconds(interval):
    """Parse a grid string ('1s', '100ms', '10ms', '500ms', '2s', ...) to seconds."""
    return float(pd.Timedelta(interval).total_seconds())

def frequency_config(args):
    """Build the estimation-path configuration implied by --interval. Returns the
    grid dt and the frequency-scaled VECM lags, IRF horizons, bootstrap block, and
    fleeting-quote-filter width (via cross_asset_pd_liquidity.frequency_defaults),
    plus the sub-second flag and the jump-classifier choice. An explicit --n-lags
    overrides the scaled value; --jump-method overrides the auto choice."""
    dt = interval_seconds(args.interval)
    fd = ca.frequency_defaults(dt=dt)
    n_lags = args.n_lags if args.n_lags is not None else fd["n_lags"]
    if args.jump_method == "auto":
        jump_method = "lee_mykland" if dt <= 0.2 else "truncation"   # local-vol classifier sub-second
    else:
        jump_method = args.jump_method
    return {"dt": dt, "n_lags": n_lags, "horizons": fd["horizons"], "block": fd["block"],
            "min_rest_steps": fd["min_rest_steps"], "subsecond": dt < 0.999,
            "jump_method": jump_method, "n_lags_auto": args.n_lags is None}

def _resample_to_interval(sessions, interval):
    """Aggregate loaded frames to the target grid (last book snapshot per bin). Only
    coarsens: a target finer than the native grid can't be upsampled, so it's kept and
    a warning is logged. Keeps prices and book columns aligned (one frame-wide resample)."""
    target = interval_seconds(interval); out = []
    for d, r, df in sessions:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            out.append((d, r, df)); continue
        native = ca.infer_dt(df)
        if target <= native * 1.01:
            if target < native * 0.99:
                LOG.warning("session %s native grid ~%.3gs is coarser than target %s; cannot upsample, keeping native",
                            str(d), native, interval)
            out.append((d, r, df)); continue
        rs = df.resample(pd.Timedelta(interval)).last().dropna(how="all")
        LOG.info("session %s resampled %.3gs -> %s (%d -> %d rows)", str(d), native, interval, len(df), len(rs))
        out.append((d, r, rs))
    return out


# ── data loading ──────────────────────────────────────────────────────────────
def _demo_sessions(n_sessions, n, n_levels, seed=7, interval="1s"):
    rng = np.random.default_rng(seed)
    def gen(label, regime, gain):
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.99 * s[t - 1] + rng.normal(0, 0.12)
        f = np.cumsum(rng.normal(0, 0.0008, n))
        a_spy = np.clip(0.05 * np.exp(+gain * s), 0.002, 0.45)
        a_es = np.clip(0.03 * np.exp(-gain * s), 0.002, 0.45)
        L = np.linalg.cholesky([[1, .3], [.3, 1]]); e = rng.normal(0, 0.01, (n, 2)) @ L.T
        lp1 = np.zeros(n); lp2 = np.zeros(n)
        for t in range(1, n):
            z = lp1[t - 1] - lp2[t - 1]
            lp1[t] = lp1[t - 1] + (f[t] - f[t - 1]) - a_spy[t] * z + e[t, 0]
            lp2[t] = lp2[t - 1] + (f[t] - f[t - 1]) + a_es[t] * z + e[t, 1]
        ms, me = 500 * np.exp(lp1), 5000 * np.exp(lp2)
        lev = np.arange(1, n_levels + 1); dw = np.exp(-0.3 * (lev - 1)); cols = {}
        for a, mid, tick in (("SPY", ms, 0.01), ("ES", me, 0.25)):
            depth = 300.0 * np.exp((0.5 if a == "ES" else -0.5) * s)
            for i, l in enumerate(lev):
                q = np.maximum(depth * dw[i] * (1 + rng.normal(0, 0.05, n)), 1.0)
                cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
                cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.02, n))
        idx = pd.date_range("2024-08-05 09:30", periods=n, freq=pd.Timedelta(interval), tz="America/New_York")
        return (label, regime, pd.DataFrame(cols, index=idx))
    out = []
    for i in range(n_sessions):
        regime = "volatile" if i % 2 == 0 else "benchmark"
        out.append(gen(f"2024-08-{i + 1:02d}", regime, 0.6 if regime == "volatile" else 0.3))
    return out

def _attach_regimes(raw, volatile):
    """raw: List[(date, df)] or List[(date, regime, df)]. Mark dates in `volatile`
    as 'volatile', the rest 'benchmark' (unless regimes already present)."""
    vol = set(volatile or [])
    out = []
    for item in raw:
        if len(item) == 3:
            out.append(tuple(item)); continue
        date, df = item
        out.append((date, "volatile" if str(date) in vol else "benchmark", df))
    return out

def _expand_date_range(spec):
    """Expand 'START:END' or 'START..END' to business-day date strings. Holidays that are not
    trading days simply return no Athena data downstream and are skipped, so a plain
    business-day calendar is fine here."""
    sep = ".." if ".." in spec else ":"
    a, b = (x.strip() for x in spec.split(sep, 1))
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    if tb < ta:                                          # reversed range -> empty universe; fail loudly
        raise ValueError(f"--date-range end ({b}) precedes start ({a}); expected START:END in "
                         f"chronological order (e.g. 2025-03-24:2025-04-08). Check for a year typo.")
    out = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(ta, tb)]
    if not out:
        raise ValueError(f"--date-range {spec!r} expands to no business days.")
    return out


def _extraction_universe(args):
    """Dates to extract in --source extract mode: explicit --dates and/or --date-range if
    given, else the union of --volatile and --benchmark, else None (use market_analysis_fixed's
    built-in list). --volatile only LABELS regimes; without an explicit universe, extract would
    ignore it and pull the built-in default dates."""
    explicit = list(args.dates) if args.dates else []
    if getattr(args, "date_range", None):
        explicit += _expand_date_range(args.date_range)
    if explicit:
        return sorted(dict.fromkeys(explicit))
    if args.volatile or args.benchmark:
        return sorted({*(args.volatile or []), *(args.benchmark or [])})
    return None


def _mask_halt_rows(sessions, enabled=True, assets=("SPY", "ES")):
    """NaN each leg's market columns inside its halt windows, AFTER alignment (ffill would refill
    them before it). See market_halts.mask_frame for the full reasoning; the short version is that
    the QC gate has excluded halt snapshots since v0.9.26 and the estimators never did -- every
    stage of the 2026-08-05 run fit the four MWCB days halt-included. The stages need no changes:
    their design masks drop non-finite rows, which after this is exactly the halt, the reopen seam,
    and any lag window touching either."""
    if not enabled:
        LOG.warning("halt masking DISABLED (--no-halt-mask): halted snapshots -- where neither leg "
                    "has a valid midpoint -- will enter every estimator. Diagnosis only.")
        return sessions
    import market_halts as mh
    out = []
    for d, r, df in sessions:
        masked, rep = mh.mask_frame(df, assets=assets)
        if any(rep.values()):
            LOG.info("session %s: halt-masked %s row(s) -- excluded from every estimator (the halt "
                     "has no valid midpoint; the QC gate already excluded these, the estimators "
                     "now do too)", str(d),
                     ", ".join("%s=%d" % (a, n) for a, n in rep.items() if n))
        out.append((d, r, masked))
    return out


def _align_books(sessions, assets=("SPY", "ES"), enabled=True):
    """Forward-fill each session onto its grid so a row where one asset did not update carries
    its last quote rather than NaN. SPY and ES are joined on non-aligned quote times, so without
    this each asset's mid/OFI is NaN wherever the *other* updated -- which silently NaN-poisons
    the OFI-based stages (cross-impact, structural IRF) and the tick/noise diagnostics. Leading
    warmup rows before both assets have a quote are dropped. No look-ahead (ffill only).

    Independently of `enabled`, object-dtype book columns (e.g. Decimal/None coming back from
    Athena) are coerced to numeric float, so the order-flow / weighted-spread / cross-flow paths
    don't hit object-vs-float64 arithmetic (which raises a UFuncTypeError in the table suite)."""
    out = []
    for d, r, df in sessions:
        obj = [c for c in df.columns if df[c].dtype == object]
        if obj:                                              # Athena often returns sizes/prices as object
            df = df.copy()
            df[obj] = df[obj].apply(pd.to_numeric, errors="coerce")
        if not enabled or not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            out.append((d, r, df)); continue
        before = int(df.isna().sum().sum())
        f = df.sort_index().ffill()
        keycols = [f"{a}_bidprice_1" for a in assets if f"{a}_bidprice_1" in f.columns]
        if keycols:
            f = f.loc[f[keycols].notna().all(axis=1)]
        filled = before - int(f.isna().sum().sum()); trimmed = len(df) - len(f)
        if len(f) == 0:
            # Alignment keeps only rows where EVERY asset has a quote, so one leg that is entirely
            # absent empties the session -- silently, at INFO, while the session count downstream
            # still says it is there. That is how four MWCB days reached a "24 usable session(s)"
            # summary and a dataset containing 20. Name the leg that did it, at ERROR.
            dead = [a for a in assets if f"{a}_bidprice_1" in df.columns
                    and not pd.to_numeric(df[f"{a}_bidprice_1"], errors="coerce").notna().any()]
            LOG.error("session %s is EMPTY after alignment (%d -> 0 rows): %s never quotes, and "
                      "alignment keeps only rows where every asset has a quote. This session "
                      "contributes NOTHING to any table -- the sample is smaller than the session "
                      "count says.", str(d), len(df),
                      " and ".join(dead) if dead else "one of the assets")
        elif filled or trimmed:
            LOG.info("session %s aligned: carried %d NaN book cells, trimmed %d leading warmup row(s) (%d -> %d rows)",
                     str(d), filled, trimmed, len(df), len(f))
        out.append((d, r, f))
    n_empty = sum(1 for _d, _r, f in out if len(f) == 0)
    if n_empty:
        LOG.error("ALIGNMENT SUMMARY: %d of %d session(s) are empty and contribute no rows: %s",
                  n_empty, len(out), ", ".join(str(d) for d, _r, f in out if len(f) == 0))
    return out


def _write_extract_report(rep: dict, args) -> str:
    """Persist which sessions failed / were degraded, next to the run's other output.

    The sample is a result. A run whose universe quietly shrank from 24 days to 22 -- or kept two
    days whose book crosses on 4% of snapshots -- produces tables that look exactly like a clean
    run's, so the provenance has to survive outside the log."""
    path = os.path.join(getattr(args, "output_dir", ".") or ".", "extract_report.txt")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write("requested %d session(s): %s\n" % (len(rep.get("requested", [])),
                                                        ", ".join(map(str, rep.get("requested", [])))))
            fh.write("usable    %d: %s\n" % (len(rep.get("ok", [])), ", ".join(map(str, rep.get("ok", [])))))
            if rep.get("failed"):
                fh.write("\nFAILED (dropped -- extraction raised):\n")
                for d, m in rep["failed"]:
                    fh.write(f"  {d}: {m}\n")
            if rep.get("degraded"):
                fh.write("\nDEGRADED (book invariant violated; qc_action=%s):\n"
                         % getattr(args, "qc_action", "warn"))
                for d, reasons in rep["degraded"]:
                    fh.write(f"  {d}: {' | '.join(reasons)}\n")
        LOG.warning("extraction was not clean -- see %s", path)
    except OSError as exc:
        LOG.warning("could not write extract report: %s", exc)
    return path


def load_sessions(args):
    if args.source == "demo":
        LOG.info("Generating %d synthetic demo session(s) of %d rows at %s",
                 args.n_sessions, args.session_len, args.interval)
        return _demo_sessions(args.n_sessions, args.session_len, args.n_levels, interval=args.interval)
    if args.source == "load":
        if not args.pickle or not os.path.exists(args.pickle):
            raise FileNotFoundError(f"--pickle not found: {args.pickle}")
        with open(args.pickle, "rb") as fh:
            raw = pickle.load(fh)
        LOG.info("Loaded %d session(s) from %s", len(raw), args.pickle)
        sessions = _attach_regimes(raw, args.volatile)
        sessions = _resample_to_interval(sessions, args.interval)        # honor --interval on loaded frames
        aligned = _align_books(sessions, enabled=not args.no_align_books)
        # Keep the PRE-MASK frames for --save-frames. Saving the masked list (v0.9.51-56 behaviour)
        # baked the NaN into the artifact: a later --no-halt-mask diagnosis on such a pickle was a
        # silent no-op, and the halt-mask A/B (ab_halt_mask.py) became impossible from the very
        # file a run hands its successor. Masking is cheap and re-applied on every load; the frames
        # at rest should be the data, not one estimation policy applied to it.
        args._premask_sessions = aligned
        return _mask_halt_rows(aligned, enabled=not getattr(args, "no_halt_mask", False))
    if args.source == "extract":
        import mstbook_loader as ml          # CLI path: mstbook-query + mstwx-lakequery (no SQL/Athena)
        extract_dates = _extraction_universe(args)
        if not extract_dates:
            raise ValueError("--source extract (CLI) needs an explicit date universe: pass --dates, "
                             "--date-range, --volatile and/or --benchmark (the CLI path has no built-in list).")
        vol = set(args.volatile or [])
        specs = [(d, "volatile" if d in vol else "benchmark") for d in extract_dates]
        LOG.info("CLI extract universe (%d date(s)) via mstwx-lakequery: %s",
                 len(extract_dates), ",".join(extract_dates))
        _prog = {"auto": None, "on": True, "off": False}.get(getattr(args, "progress", "auto"), None)
        rep: dict = {}
        sessions = ml.extract_sessions(specs, es_symbol="ES", levels=args.n_levels,
                                       interval=args.interval, with_flow=True,
                                       classify=getattr(args, "classify", "aggressor"),
                                       book_source=getattr(args, "book_source", "reconstruct"),
                                       es_book_source=getattr(args, "es_book_source", "aggregated"),
                                       round_lot=getattr(args, "round_lot", 100),
                                       odd_lot_inclusive=not getattr(args, "round_lot_only", False),
                                       clock=getattr(args, "clock", "receipt"), progress=_prog,
                                       max_workers=getattr(args, "max_workers", 1),
                                       backend=getattr(args, "extract_backend", None),
                                       cache_dir=getattr(args, "extract_cache", "") or "",
                                       resume=not getattr(args, "no_resume", False),
                                       qc_action=getattr(args, "qc_action", "warn"),
                                       crossed_tol=getattr(args, "crossed_tol", 0.005),
                                       report=rep)
        LOG.info("Extracted %d session(s) via mstbook_loader (book + trade flow)", len(sessions))
        # An incomplete universe changes every pooled number downstream, so it is recorded next to
        # the results rather than left in the scrollback of a multi-hour log.
        if rep.get("failed") or rep.get("degraded"):
            _write_extract_report(rep, args)
        aligned = _align_books(sessions, enabled=not args.no_align_books)
        args._premask_sessions = aligned                    # pre-mask frames for --save-frames
        return _mask_halt_rows(aligned, enabled=not getattr(args, "no_halt_mask", False))
    raise ValueError(args.source)


# ── analysis stages (each takes (sessions, args) -> result dict) ───────────────
def run_liquidity_curves(sessions, args):
    date, regime, df = sessions[0]
    rows = []
    for a in ("SPY", "ES"):
        m = lcm.side_curve_metrics(df, a, "bid", args.n_levels)
        rows.append({"asset": a,
                     "entropy_norm": float(np.nanmean(m[f"{a}_bid_entropy_norm"])),
                     "depth_slope": float(np.nanmean(m[f"{a}_bid_depth_slope"])),
                     "total_depth": float(np.nanmean(m[f"{a}_bid_total_depth"]))})
    return {"sample_session": str(date), "bid_curve_means": pd.DataFrame(rows)}

def run_information_shares(sessions, args):
    mids = rb._mid_sessions(sessions)
    per_day = pds.estimate_sample(mids, n_lags=args.n_lags)
    _v = per_day["ec_valid"] if "ec_valid" in per_day.columns else pd.Series(True, index=per_day.index)
    res = {"per_day": per_day, "mean_CS_ES": float(per_day.CS_ES.mean()),
           # the mean over days whose error correction actually points at the equilibrium; on
           # ec_valid=False days CS is a quotient of same-signed alphas, not a share
           "mean_CS_ES_ec_valid": float(per_day.loc[_v, "CS_ES"].mean()) if _v.any() else float("nan"),
           "n_ec_invalid": int((~_v).sum()),
           "mean_IS_mid_ES": float(per_day.IS_mid_ES.mean())}
    if len({r for _, r, _ in sessions}) >= 2:
        try: res["regime_test"] = pds.compare_regimes(per_day, metric="CS_ES")
        except Exception as e: res["regime_test_error"] = str(e)
    try: res["panel_vecm"] = pds.panel_vecm(mids, n_lags=args.n_lags)
    except Exception as e: res["panel_vecm_error"] = str(e)
    return res

def run_liquidity_conditional(sessions, args):
    out = {}
    for tag, statef in (("depth", lambda df: ca.relative_depth_state(df, args.n_levels)),
                        ("fpca", lambda df: fl.relative_fpc_state(df, n_levels=args.n_levels))):
        rows = []
        for date, regime, df in sessions:
            lc = ca.liquidity_conditional_vecm(ca._mid(df, "SPY"), ca._mid(df, "ES"),
                                               statef(df), n_lags=args.n_lags)
            sb = lc["shares_by_state"]
            rows.append({"date": date, "regime": regime,
                         "CS_ES_q10": sb[0.1]["CS_ES"], "CS_ES_q50": sb[0.5]["CS_ES"],
                         "CS_ES_q90": sb[0.9]["CS_ES"], "t_delta_es": lc["t_delta_es"]})
        out[f"state_{tag}"] = pd.DataFrame(rows)
    return out

def run_panel(sessions, args):
    panel = ca.build_window_panel(sessions, window=args.window, n_levels=args.n_levels,
                                  n_lags=args.n_lags, min_rest_steps=args._freq["min_rest_steps"])
    out = {"n_windows": len(panel), "panel": panel}
    if len(panel) >= 5:
        try: out["regression_IS_on_depth"] = ca.panel_regression(panel, y="IS_mid_ES", X_cols=["rel_logdepth"])
        except Exception as e: out["regression_error"] = str(e)
        try: out["state_split"] = ca.state_split_shares(panel, state="rel_logdepth")
        except Exception as e: out["state_split_error"] = str(e)
    else:
        out["note"] = "panel too small for regression (need >=5 windows)"
    return out

def run_cross_impact(sessions, args):
    panel, summary = cimp.cross_impact_panel(sessions, n_levels=args.n_levels,
                                             min_rest_steps=args._freq["min_rest_steps"])
    return {"panel": panel, "regime_summary": summary}

def run_dcc(sessions, args):
    # ALL sessions, not sessions[0]. The 2026-08-05 headline (mean_rho 0.357 against a realized
    # correlation of 0.82, persistence pinned at the boundary) was a single-session fit -- and the
    # single session was 2020-03-09, a circuit-breaker day estimated halt-included at the time.
    # Returns are differenced PER SESSION before stacking, so no overnight pseudo-return crosses a
    # day boundary; the realized correlation of the same stacked returns is reported beside
    # mean_rho so a 0.357-vs-0.82 divergence is visible in the output rather than in a post-mortem.
    rs = []
    for _d, _r, df in sessions:
        r = np.column_stack([np.diff(np.log(ca._mid(df, "SPY"))), np.diff(np.log(ca._mid(df, "ES")))])
        rs.append(r[np.all(np.isfinite(r), axis=1)])   # halt-masked rows drop; no splice (diff first)
    R = np.vstack([x for x in rs if len(x)]) if rs else np.zeros((0, 2))
    fit = dg.dcc_garch_x(R)
    realized = float(np.corrcoef(R[:, 0], R[:, 1])[0, 1]) if len(R) > 10 else float("nan")
    out = {"dcc_a": float(fit["dcc_a"]), "dcc_b": float(fit["dcc_b"]),
           "persistence": float(fit["dcc_a"] + fit["dcc_b"]), "mean_rho": float(np.mean(fit["rho"])),
           "realized_corr": realized, "n_sessions": len(rs)}
    if np.isfinite(realized) and abs(out["mean_rho"] - realized) > 0.2:
        out["warning"] = ("mean_rho differs from the realized correlation of the SAME returns by "
                          "%.2f with persistence %.4f -- near-integrated recursion, path likely "
                          "initialization-dominated; do not use rho_t as the comovement exhibit "
                          "before investigating" % (abs(out["mean_rho"] - realized), out["persistence"]))
    return out

def run_irf(sessions, args):
    # sessions[0] is kept for the LP exhibit but NAMED now: the 2026-08-05 headline FEVD (ret_ES
    # 30.6% "from SPY flow") was silently this single session, and sorted-first is 2020-03-09 -- a
    # circuit-breaker day, halt-included at the time, whose cross-impact matrix (which identifies
    # B; this FEVD is NOT Cholesky) carried the reopen seam as a leverage point. The headline FEVD
    # is now the per-regime MEDIAN across sessions; the single-session matrix stays, labelled.
    d0, _, df = sessions[0]
    fc = args._freq
    H = range(0, 6) if args.quick else fc["horizons"]            # frequency-scaled (quick caps for speed)
    nb = 60 if args.quick else 300
    lp = irfm.local_projection_irf(df, impulse="ES", response="SPY",
                                   state=ca.relative_depth_state(df, args.n_levels),
                                   horizons=H, n_lags=args.n_lags, n_levels=args.n_levels,
                                   n_boot=nb, block=fc["block"],
                                   min_rest_steps=fc["min_rest_steps"])
    sv = irfm.structural_vecm_irf(df, horizons=H, n_lags=args.n_lags, n_levels=args.n_levels,
                                  min_rest_steps=fc["min_rest_steps"])
    fevd0 = irfm.fevd_from_irf(sv["return_irf"], H=max(H))
    per, ofi_corr = {}, {}
    for _dd, rr, dfx in sessions:
        try:
            svx = irfm.structural_vecm_irf(dfx, horizons=H, n_lags=min(int(args.n_lags), 12),
                                           n_levels=args.n_levels,
                                           min_rest_steps=fc["min_rest_steps"])
            per.setdefault(rr, []).append(irfm.fevd_from_irf(svx["return_irf"], H=max(H)))
            # The FEVD partitions variance only under UNCORRELATED shocks; the tandem-flow
            # result says OFI_SPY/OFI_ES are not. Report the correlation the shares ignore.
            o1 = np.asarray(ca.order_flow_imbalance(dfx, "SPY", args.n_levels,
                                                    fc["min_rest_steps"]), float)[1:]
            o2 = np.asarray(ca.order_flow_imbalance(dfx, "ES", args.n_levels,
                                                    fc["min_rest_steps"]), float)[1:]
            m = np.isfinite(o1) & np.isfinite(o2)
            if m.sum() > 30:
                ofi_corr.setdefault(rr, []).append(float(np.corrcoef(o1[m], o2[m])[0, 1]))
        except Exception:
            continue
    by_regime = {}
    for rr, mats in per.items():
        M = np.median(np.stack(mats), axis=0)
        M = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)   # rows re-sum to 1 after median
        by_regime[rr] = pd.DataFrame(M, index=["ret_SPY", "ret_ES"], columns=["OFI_SPY", "OFI_ES"])
    f0 = pd.DataFrame(fevd0, index=["ret_SPY", "ret_ES"], columns=["OFI_SPY", "OFI_ES"])
    return {"local_projection": lp,
            "fevd": by_regime.get("benchmark", f0),
            "fevd_by_regime": by_regime,
            "ofi_innovation_corr": {rr: float(np.median(v)) for rr, v in ofi_corr.items()},
            "fevd_note": ("shares assume UNCORRELATED OFI shocks (see irf.fevd_from_irf); at the "
                          "measured ofi_innovation_corr they are a diagonal approximation and the "
                          "SPY-flow/ES-flow split is partly common tandem flow"),
            "fevd_session0": f0, "fevd_session0_date": str(d0)}

def run_robustness(sessions, args):
    out = {}
    out["lag_sensitivity"] = rb.lag_sensitivity(sessions, lags=(3, 5) if args.quick else (3, 5, 10, 20))
    out["beta_fixed_vs_estimated"] = rb.beta_fixed_vs_estimated(sessions, n_lags=args.n_lags)
    out["window_size"] = rb.window_size_sensitivity(
        sessions, windows=("10min", "20min") if args.quick else ("10min", "20min", "30min"),
        n_levels=args.n_levels, n_lags=args.n_lags)
    out["alt_state"] = rb.alt_state_variable(sessions, n_levels=args.n_levels, n_lags=args.n_lags)
    cs = pds.estimate_sample(rb._mid_sessions(sessions), n_lags=args.n_lags).CS_ES.to_numpy()
    out["bootstrap_CS_ES"] = rb.stationary_bootstrap_ci(cs, n_boot=1000 if args.quick else 5000, mean_block=2)
    return out

def run_microstructure(sessions, args):
    _, _, df = sessions[0]
    dt = ca.infer_dt(df); fc = args._freq
    out = {"grid_dt_seconds": dt, "target_interval": args.interval,
           "staleness_report": mdx.staleness_report(df)}
    if fc["subsecond"]:
        out["note"] = ("sub-second grid: naive realized variance/correlation are biased -- source "
                       "second moments from noise_robust_cov (two-scale / realized kernel), and the "
                       "fleeting-quote filter is engaged on OFI inputs (min_rest_steps=%d)."
                       % fc["min_rest_steps"])
    else:
        out["note"] = ("grid is >=1s: noise-robust realized covariance is unnecessary and the "
                       "fleeting-quote filter is inert; sub-second use needs raw irregular quote times.")
    return out

def run_jumps(sessions, args):
    """Continuous vs jump price discovery: per-session information-share split (built on
    price_discovery_shares) + a Lee-Mykland co-jump lead-lag read on the first session.
    The split method (truncation vs Lee-Mykland) follows the sampling grid: Lee-Mykland's
    local-volatility classifier is used sub-second, where it deflates the jump fraction."""
    method = args._freq["jump_method"]
    mids = rb._mid_sessions(sessions)
    split = jr.estimate_sample_jump_split(mids, n_lags=args.n_lags, method=method)
    # ec_valid filter (v0.9.64): on a non-correcting day (kappa <= 0) psi ~ alpha_perp makes
    # every IS number a quotient of same-signed alphas -- noise, not a share. The IS means skip
    # those days (the jump-FRACTION means do not: jump fractions are QV ratios, kappa-free).
    _v = split["ec_valid"] if "ec_valid" in split.columns else pd.Series(True, index=split.index)
    out = {"split_method": method, "per_day": split,
           "mean_ISc_ES": float(split.ISc_ES[_v].mean()) if _v.any() else float("nan"),
           "mean_ISj_ES": float(split.ISj_ES[_v].mean()) if _v.any() else float("nan"),
           "n_ec_invalid": int((~_v).sum()),
           "mean_jump_frac_cf": float(split.jump_frac_cf.mean()),
           "mean_jump_frac_cf_lm": float(split.jump_frac_cf_lm.mean()),
           "frac_days_cf_jump_5pct": float((split.bns_cf_p < 0.05).mean())}
    _, _, df0 = sessions[0]
    try:
        out["cojump_session0"] = jr.cojump_from_mids(ca._mid(df0, "SPY"), ca._mid(df0, "ES"), max_lag=2)
    except Exception as e:
        out["cojump_error"] = str(e)
    return out

def run_ecm_sde(sessions, args):
    """Liquidity-conditional price discovery as a state-dependent ECM-SDE: the pooled
    Euler-Maruyama interacted estimator. The loading gradient a1 (coef on z*S) with its
    day-clustered t-stat is the continuous-time form of the thesis test. Reported on the
    raw mid and on a noise-robust microprice observable (robust_prices)."""
    state = lambda df: ca.relative_depth_state(df, args.n_levels)
    fit = es.estimate_sample(sessions, state_fn=state, degree=1, n_lags=0)
    out = {"a1_SPY": fit["a1_SPY"], "t_a1_SPY": fit["t_a1_SPY"],
           "a1_ES": fit["a1_ES"], "t_a1_ES": fit["t_a1_ES"],
           "a0_SPY": fit["a0_SPY"], "a0_ES": fit["a0_ES"], "curve": fit["curve"],
           "IS_ES_med_mid": float(fit["curve"]["IS_ES"].median())}
    try:
        rbf = es.estimate_sample(sessions, state_fn=rp.state_fn("rel_logdepth", args.n_levels),
                                 price_fn=rp.price_fn("microprice", args.n_levels), degree=1, n_lags=0)
        out["a1_SPY_microprice"] = rbf["a1_SPY"]; out["t_a1_SPY_microprice"] = rbf["t_a1_SPY"]
        out["IS_ES_med_microprice"] = float(rbf["curve"]["IS_ES"].median())
        out["curve_microprice"] = rbf["curve"]
    except Exception as e:
        out["microprice_error"] = str(e)
    return out

def run_legacy(sessions, args):
    """Before/after: paper-style (legacy) estimators next to the upgraded ones. Gated by --legacy."""
    return legacy.build_legacy_comparison(sessions, args)


def _daily_vol_state(sessions):
    """Ex-ante daily stress proxy for event-study control matching: the PRIOR session's realized vol
    of the SPY mid (sqrt of summed squared log-returns). Genuinely ex-ante — uses t-1, never t's own
    outcome — so control matching does not condition on the dependent variable. Replace with a
    prior-close VIX series on real data if available."""
    rows = {}
    for date, _regime, df in sessions:
        m = ca._mid(df, "SPY"); lr = np.diff(np.log(np.maximum(m, 1e-12)))
        rows[pd.Timestamp(date).normalize()] = float(np.sqrt(np.nansum(lr ** 2))) if lr.size else np.nan
    s = pd.Series(rows).sort_index(); ex = s.shift(1)
    mu = s.mean()
    return ex.fillna(mu if np.isfinite(mu) else 0.0)


def run_events(sessions, args):
    """Event-clock study for selected shock CATEGORIES/CLASSES (gated by --events / --event-class).
    Resolves each event's release time (the event clock), matches controls on the ex-ante stress
    proxy, and runs the event-study profile / DiD / RD-in-time / cross-market (ES->SPY) spillover.
    Requires the extracted sessions to include BOTH event days AND candidate control days (use
    --date-range over a window, or --volatile event days with --benchmark control days)."""
    import event_study_driver as esd
    cats = tuple(args.events) if args.events else None
    classes = tuple(args.event_class) if args.event_class else None
    sessions_by_date = {pd.Timestamp(d): df for d, _r, df in sessions}
    state = _daily_vol_state(sessions)
    res = esd.run_event_study(state, sessions_by_date, categories=cats, classes=classes,
                              n_controls=2, buffer_days=5, asset="SPY", outcome="cost_to_fill",
                              source="ES", target="SPY", pre_min=5, post_min=5,
                              target_qty=None, n_levels=args.n_levels)
    out = {"categories": cats, "classes": classes, "n_treated": res.get("n_treated"),
           "n_control": res.get("n_control"), "regimes": res.get("regimes"),
           "mixed_regime": res.get("mixed_regime"), "dropped": len(res.get("dropped", []))}
    if res.get("warning"):
        out["warning"] = res["warning"]
    if isinstance(res.get("did"), dict):
        out["did_coef_bps"] = res["did"].get("did_coef"); out["did_t"] = res["did"].get("did_t")
    if isinstance(res.get("rd"), dict):
        out["rd_jump_bps"] = res["rd"].get("jump"); out["rd_jump_t"] = res["rd"].get("jump_t")
    if isinstance(res.get("spillover"), dict):
        spc = res["spillover"].get("spill_post_treated", {})
        out["spillover_ES_to_SPY_coef"] = spc.get("coef"); out["spillover_ES_to_SPY_t"] = spc.get("t")
    if isinstance(res.get("profile"), pd.DataFrame):
        out["event_time_profile"] = res["profile"]
    return out


STAGES = [
    ("liquidity_curves", run_liquidity_curves),
    ("information_shares", run_information_shares),
    ("liquidity_conditional", run_liquidity_conditional),
    ("ecm_sde", run_ecm_sde),
    ("panel", run_panel),
    ("cross_impact", run_cross_impact),
    ("dcc", run_dcc),
    ("irf", run_irf),
    ("jumps", run_jumps),
    ("robustness", run_robustness),
    ("microstructure", run_microstructure),
    ("events", run_events),
    ("legacy", run_legacy),
]


# ── reporting ─────────────────────────────────────────────────────────────────
def _scalar_summary(results):
    s = {}
    def g(path, default=None):
        cur = results
        try:
            for k in path: cur = cur[k]
            return cur
        except Exception:
            return default
    s["mean_CS_ES"] = g(["information_shares", "mean_CS_ES"])
    s["mean_IS_mid_ES"] = g(["information_shares", "mean_IS_mid_ES"])
    lc_d = g(["liquidity_conditional", "state_depth"])
    lc_f = g(["liquidity_conditional", "state_fpca"])
    if isinstance(lc_d, pd.DataFrame):
        s["liq_cond_depth_mean_t_delta_es"] = float(lc_d.t_delta_es.mean())
        s["liq_cond_depth_mean_CS_spread"] = float((lc_d.CS_ES_q90 - lc_d.CS_ES_q10).mean())
    if isinstance(lc_f, pd.DataFrame):
        s["liq_cond_fpca_mean_t_delta_es"] = float(lc_f.t_delta_es.mean())
    reg = g(["panel", "regression_IS_on_depth"])
    if isinstance(reg, pd.DataFrame) and "rel_logdepth" in reg.index:
        s["panel_IS_on_depth_coef"] = float(reg.loc["rel_logdepth", "coef"])
        s["panel_IS_on_depth_t"] = float(reg.loc["rel_logdepth", "t"])
    s["dcc_a"] = g(["dcc", "dcc_a"]); s["dcc_b"] = g(["dcc", "dcc_b"]); s["mean_rho"] = g(["dcc", "mean_rho"])
    fevd = g(["irf", "fevd"])
    if isinstance(fevd, pd.DataFrame):
        s["fevd_SPYret_from_ESofi"] = float(fevd.loc["ret_SPY", "OFI_ES"])
    boot = g(["robustness", "bootstrap_CS_ES"])
    if isinstance(boot, dict):
        s["CS_ES_ci"] = [boot.get("ci_lo"), boot.get("ci_hi")]
    s["mean_ISj_ES"] = g(["jumps", "mean_ISj_ES"])
    s["mean_jump_frac_cf"] = g(["jumps", "mean_jump_frac_cf"])
    s["mean_jump_frac_cf_lm"] = g(["jumps", "mean_jump_frac_cf_lm"])
    s["ecm_sde_dAlphaSPY_dS_t"] = g(["ecm_sde", "t_a1_SPY"])
    s["ecm_sde_t_a1_SPY_microprice"] = g(["ecm_sde", "t_a1_SPY_microprice"])
    s["ecm_sde_IS_ES_mid_vs_microprice"] = (g(["ecm_sde", "IS_ES_med_mid"]), g(["ecm_sde", "IS_ES_med_microprice"]))
    return {k: v for k, v in s.items() if v is not None}

def write_report(results, status, summary, path, args, artifacts=None):
    L = ["# Cross-Asset Price-Discovery — Analysis Run", "",
         f"- generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
         f"- source: `{args.source}`  | interval: `{args.interval}` (dt={args._freq['dt']:.4g}s, "
         f"{'sub-second' if args._freq['subsecond'] else '>=1s'})  | levels: {args.n_levels}  | "
         f"lags: {args.n_lags}{' (auto)' if args._freq['n_lags_auto'] else ''}  | "
         f"jumps: `{args._freq['jump_method']}`  | fleeting min_rest: {args._freq['min_rest_steps']}",
         f"- stages: " + ", ".join(f"{n} ({'ok' if status[n]=='ok' else 'FAIL'})" for n in status), ""]
    if artifacts:
        L += ["## Artifacts", *[f"- {k}: `{v}`" for k, v in artifacts.items()], ""]
    L += ["## Headline numbers", "```", json.dumps(summary, indent=2, default=str), "```", ""]
    for name, _ in STAGES:
        if name not in results: continue
        L += [f"## {name}"]
        res = results[name]
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, pd.DataFrame):
                    L += [f"**{k}**", "```", v.to_string(index=True), "```"]
                elif isinstance(v, (pd.Series,)):
                    L += [f"**{k}**", "```", v.to_string(), "```"]
                else:
                    L += [f"- {k}: `{v}`"]
        L += [""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


# ── dataset & table export ────────────────────────────────────────────────────
def _parquet_engine():
    import importlib.util as u
    if u.find_spec("pyarrow"):     return "pyarrow"
    if u.find_spec("fastparquet"): return "fastparquet"
    return None

def consolidate_dataset(sessions):
    """Stack the per-session frames into one tidy long-format dataset with leading
    session_date / regime / timestamp columns — the single 'final dataset'."""
    frames = []
    for date, regime, df in sessions:
        f = df.copy(); f.index.name = "timestamp"; f = f.reset_index()
        f.insert(0, "session_date", str(date)); f.insert(1, "regime", regime)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)

def write_dataset(sessions, run_dir, fmt="auto"):
    """Write the consolidated dataset as parquet (if an engine is available) or a
    single gzip-compressed CSV — one file, not per-day."""
    ds = consolidate_dataset(sessions)
    eng = _parquet_engine()
    use_parquet = fmt == "parquet" or (fmt == "auto" and eng is not None)
    if use_parquet and eng is None:
        LOG.warning("parquet requested but no engine installed; writing csv.gz instead")
        use_parquet = False
    if use_parquet:
        path = os.path.join(run_dir, "final_dataset.parquet"); ds.to_parquet(path, engine=eng, index=False)
    else:
        path = os.path.join(run_dir, "final_dataset.csv.gz"); ds.to_csv(path, index=False, compression="gzip")
    return path, ds.shape

def export_tables(results, tables_dir):
    """Write every resulting table (DataFrame / Series, plus scalar summaries) as CSV."""
    os.makedirs(tables_dir, exist_ok=True)
    written = []
    for stage, res in results.items():
        if not isinstance(res, dict):
            continue
        scalars = {}
        for key, val in res.items():
            p = os.path.join(tables_dir, f"{stage}__{key}.csv")
            if isinstance(val, pd.DataFrame):
                val.to_csv(p); written.append(p)
            elif isinstance(val, pd.Series):
                val.to_frame(name=key).to_csv(p); written.append(p)
            elif isinstance(val, dict) and val and all(np.isscalar(v) or v is None for v in val.values()):
                pd.DataFrame([val]).to_csv(p, index=False); written.append(p)
            elif np.isscalar(val) or val is None:
                scalars[key] = val
        if scalars:
            p = os.path.join(tables_dir, f"{stage}.csv")
            pd.DataFrame([scalars]).to_csv(p, index=False); written.append(p)
    return written


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Driver for the cross-asset price-discovery stack.")
    p.add_argument("--source", choices=["demo", "load", "extract"], default="demo")
    p.add_argument("--pickle", default=None, help="path to extracted frames (for --source load)")
    p.add_argument("--volatile", type=lambda s: s.split(","), default=None,
                   help="comma-separated dates to label 'volatile'. For --source extract this ALSO "
                        "defines the extraction universe when --dates is not given (so these dates are "
                        "the ones pulled, not just labeled). TAXONOMY NOTE: the replication driver "
                        "passes its MWCB dates in this list, so every two-regime stage here reads "
                        "'volatile' as volatile-INCLUDING-MWCB; Table 5 keeps MWCB as its own panel. "
                        "Comparisons across those exhibits must say which composition they use")
    p.add_argument("--benchmark", type=lambda s: s.split(","), default=None,
                   help="comma-separated dates to treat as benchmark; for --source extract they join the "
                        "extraction universe (when --dates is absent)")
    p.add_argument("--dates", type=lambda s: s.split(","), default=None,
                   help="explicit extraction universe (comma-separated) for --source extract; overrides the "
                        "--volatile/--benchmark union and the built-in default list")
    p.add_argument("--date-range", default=None,
                   help="extraction universe as START:END business days (e.g. 2020-02-20:2020-03-20); "
                        "combines with --dates")
    p.add_argument("--interval", default="1s",
                   help="aggregation grid ('1s', '100ms', '10ms', ...); sets sampling AND the "
                        "frequency-scaled estimation path (lags/horizons/block/fleeting filter/jump method)")
    p.add_argument("--n-levels", type=int, default=10)
    p.add_argument("--n-lags", type=int, default=None,
                   help="VECM lags; default None = auto-scaled from --interval (override to force)")
    p.add_argument("--jump-method", choices=["auto", "truncation", "lee_mykland"], default="auto",
                   help="jump-split classifier; auto = lee_mykland sub-second (<=100ms), truncation otherwise")
    p.add_argument("--window", default="30min")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--extract-cache", default="",
                   help="directory to memoize each successfully extracted session in. A day costs "
                        "10-25 min of vendor I/O, so a re-run after a partial failure should only "
                        "pay for the days that are missing. Sessions that fail the book invariant "
                        "are NOT cached (they are worth retrying, not memoizing)")
    p.add_argument("--no-resume", action="store_true",
                   help="with --extract-cache, ignore what is already cached and re-fetch every day")
    p.add_argument("--qc-action", choices=["warn", "drop", "raise"], default="warn",
                   help="what to do with an extracted session whose top of book crosses (or whose "
                        "leg is missing entirely): warn = keep and say so loudly (default), drop = "
                        "exclude it from the dataset, raise = stop the run")
    p.add_argument("--crossed-tol", type=float, default=0.005,
                   help="fraction of snapshots allowed to cross before a session is called degraded; "
                        "consolidated multi-venue books do cross briefly, so this is not zero")
    p.add_argument("--only", type=lambda s: s.split(","), default=None)
    p.add_argument("--skip", type=lambda s: s.split(","), default=None)
    p.add_argument("--es-book-source", default="aggregated", choices=["aggregated", "replay"],
                   help="ES leg from CME's own 10-level ladder (mt_aggregated_price_update, the "
                        "default: one venue, one mechanism on every date in the panel, and the "
                        "source validate_aggregated.py already benchmarks the replay against) or "
                        "from the message replay ('replay': order-level detail, but the message "
                        "family differs by era and must be selected at run time)")
    p.add_argument("--no-halt-mask", action="store_true",
                   help="do NOT exclude halt snapshots from the estimators. A halted market has no "
                        "valid midpoint, so this is for diagnosis only -- with it, the four MWCB "
                        "days carry 900 mechanical snapshots into every VECM, IS, OFI and jump "
                        "estimate, and the reopen gap books as a jump")
    p.add_argument("--no-align-books", action="store_true",
                   help="do not forward-fill each book onto the grid; alignment fixes unaligned SPY/ES "
                        "rows that otherwise NaN-poison the OFI-based stages (cross-impact, structural IRF)")
    p.add_argument("--quick", action="store_true", help="smaller bootstrap/horizons for a fast run")
    p.add_argument("--n-sessions", type=int, default=4, help="demo: number of sessions")
    p.add_argument("--session-len", type=int, default=6000, help="demo: rows per session")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--dataset-format", choices=["auto", "parquet", "csv"], default="auto",
                   help="consolidated final-dataset format (auto: parquet if available, else csv.gz)")
    p.add_argument("--save-dataset", action="store_true", help="also write the dataset for --source demo")
    p.add_argument("--save-frames", action="store_true", default=True,
                   help="write frames_<interval>.pkl -- the List[(date, regime, df)] every "
                        "downstream driver loads with --source load (on by default)")
    p.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    p.add_argument("--no-dataset", action="store_true", help="do not write the consolidated dataset")
    p.add_argument("--save-objects", action="store_true", help="also dump full Python objects as a pickle")
    p.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False,
                   help="also compute the paper's original (legacy) estimators next to the upgraded "
                        "ones for a before/after comparison (Pearson vs HY, iid vs clustered SE, "
                        "spread vs book-stress, Cholesky vs Rigobon). Default off here; the bundled "
                        "run_analysis.sh launcher turns it on by default.")
    p.add_argument("--events", type=lambda s: s.split(","), default=None,
                   help="run the event-clock study for these shock CATEGORIES (e.g. TARIFF,GEOPOLITICAL). "
                        "Sessions must include both event days and candidate control days.")
    p.add_argument("--event-class", type=lambda s: s.split(","), default=None,
                   help="like --events but selects coarse event CLASSES (e.g. TRADE_POLICY,GEOPOLITICAL).")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    t0 = time.time()

    # --interval drives both aggregation and the estimation path
    fc = frequency_config(args)
    args._freq = fc
    args.n_lags = fc["n_lags"]                                # resolve auto/override to a concrete int
    LOG.info("Grid %s (dt=%.4gs, %s) | n_lags=%d%s | IRF horizons<=%d, block=%d | fleeting min_rest=%d | jumps=%s",
             args.interval, fc["dt"], "sub-second" if fc["subsecond"] else ">=1s", fc["n_lags"],
             " (auto)" if fc["n_lags_auto"] else " (set)", max(fc["horizons"]), fc["block"],
             fc["min_rest_steps"], fc["jump_method"])

    sessions = load_sessions(args)
    return run_stages(sessions, args, ts=ts, t0=t0)


def run_stages(sessions, args, ts=None, t0=None):
    """Run the selected stages on an ALREADY-LOADED sessions list; write report/tables/summary and
    return (results, status, summary). Split out of main() so another driver (e.g. run_contagion) can
    run the upgraded analysis IN-PROCESS on the same in-memory sessions — no pickle round-trip, no
    second load. Computes the frequency config itself if the caller did not run main() first."""
    if getattr(args, "_freq", None) is None:
        _fc = frequency_config(args); args._freq = _fc; args.n_lags = _fc["n_lags"]
    fc = args._freq
    if ts is None: ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if t0 is None: t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    regimes = sorted({r for _, r, _ in sessions})
    LOG.info("Sessions: %d | regimes: %s", len(sessions), regimes)

    chosen = [n for n, _ in STAGES]
    if args.only:  chosen = [n for n in chosen if n in args.only]
    if args.skip:  chosen = [n for n in chosen if n not in args.skip]
    if not args.legacy:  chosen = [n for n in chosen if n != "legacy"]
    if not (args.events or args.event_class):  chosen = [n for n in chosen if n != "events"]

    results, status = {}, {}
    for name, fn in STAGES:
        if name not in chosen: continue
        st = time.time()
        try:
            results[name] = fn(sessions, args)
            status[name] = "ok"
            LOG.info("[ok]   %-22s (%.1fs)", name, time.time() - st)
        except Exception as e:
            status[name] = f"FAIL: {e}"
            LOG.error("[FAIL] %-22s %s", name, e, exc_info=False)

    summary = _scalar_summary(results)
    summary = {"grid_interval": args.interval, "grid_dt_seconds": round(fc["dt"], 6),
               "path": "sub-second" if fc["subsecond"] else ">=1s", "n_lags": fc["n_lags"],
               "jump_method": fc["jump_method"], "fleeting_min_rest_steps": fc["min_rest_steps"],
               **summary}
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    tables_dir = os.path.join(run_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # the final dataset — one consolidated file (parquet or csv.gz), unless suppressed
    ds_path = None
    if (not args.no_dataset) and (args.save_dataset or args.source != "demo"):
        ds_path, ds_shape = write_dataset(sessions, run_dir, fmt=args.dataset_format)
        LOG.info("Final dataset -> %s  (rows=%d, cols=%d)", ds_path, ds_shape[0], ds_shape[1])

    # The SESSION FRAMES, as List[(date, regime, DataFrame)] -- the shape every downstream driver
    # loads with --source load. The flat dataset is not a substitute: it discards the per-session
    # split, and rebuilding it by grouping on a date column is exactly the kind of re-derivation
    # that goes wrong quietly. An extract run that wrote only the parquet left the replication
    # driver's STAGE 3 pointing at a pickle that was never created, and the run died there after
    # 85 minutes of extraction with nothing downstream able to start.
    if args.save_frames and sessions:
        fpath = os.path.join(run_dir, f"frames_{args.interval}.pkl")
        # PRE-MASK frames when available: the artifact must be the data, not this run's masking
        # policy baked in as NaN (which made --no-halt-mask a silent no-op on reloaded frames).
        to_save = getattr(args, "_premask_sessions", None) or sessions
        _variant = "pre-mask" if getattr(args, "_premask_sessions", None) else \
            "as-estimated (no pre-mask list on args -- masked-at-rest if masking ran)"
        with open(fpath, "wb") as fh:
            pickle.dump([(str(d), r, f) for d, r, f in to_save], fh, protocol=4)
        LOG.info("Session frames -> %s  (%d session(s), %s, the List[(date, regime, df)] "
                 "shape --source load expects)", fpath, len(to_save), _variant)

    # the resulting tables — one CSV per analysis output
    table_files = export_tables(results, tables_dir)

    # readable summary + report (no raw-data pickle / per-day CSV clutter)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump({"summary": summary, "status": status}, fh, indent=2, default=str)
    artifacts = {"final_dataset": os.path.basename(ds_path) if ds_path else "(not written)",
                 "tables": f"{len(table_files)} CSVs in tables/", "summary": "summary.json"}
    write_report(results, status, summary, os.path.join(run_dir, "report.md"), args, artifacts)

    # optional: full Python objects for programmatic reload (off by default)
    if args.save_objects:
        with open(os.path.join(run_dir, "analysis_objects.pkl"), "wb") as fh:
            pickle.dump({"results": results, "status": status, "summary": summary}, fh)

    ok = sum(1 for v in status.values() if v == "ok")
    LOG.info("Done in %.1fs | %d/%d stages ok", time.time() - t0, ok, len(status))
    LOG.info("Run directory: %s", run_dir)
    LOG.info("  %s | %d table CSVs | report.md | summary.json%s",
             os.path.basename(ds_path) if ds_path else "(no dataset)",
             len(table_files), " | analysis_objects.pkl" if args.save_objects else "")
    print("\nHEADLINE:", json.dumps(summary, indent=2, default=str))
    return results, status, summary

if __name__ == "__main__":
    main()
