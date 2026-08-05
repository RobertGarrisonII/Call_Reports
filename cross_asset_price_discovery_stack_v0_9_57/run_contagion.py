"""
run_contagion.py
===============
The top-level driver: one call sequences the whole liquidity-contagion pipeline on a real
session set and emits the manuscript table file.

Pipeline
--------
  1. SELECTION   (stress_index + market_shocks): regime labels / onsets from the daily vol,
                 the annotated shock registry, and the level-matched-vs-ln(sigma) control
                 balance -- the design that replaces hand-picked volatile days.
  2. EVENT STUDY (event_study_driver): registry -> ex-ante level-matched controls -> release
                 boundary -> build_panel -> DiD / RD-in-time / cross-market spillover, for the
                 requested identification regime. The assembled treated/control design is then
                 FED INTO the table suite, so the MWCB tables use the same matched design.
  3. TABLES      (paper_tables): the full 16-table reproduction+revamp suite (Tables 5/9/11/13,
                 the spread/chi IRF with bootstrap+Romano-Wolf, the volume-clock comparison,
                 sigma_s welfare, regime IS, Hawkes contagion, tick-corrected IS, ...), built
                 on the real sessions and the event-study design.
  4. EMIT        a combined Markdown+LaTeX manuscript table file and a run summary.

Each stage is isolated (a failure is recorded, the pipeline continues). Returns a structured
results bundle. Designed for a real MIDAS session set, with a synthetic self-test that
exercises the whole chain. Pure numpy / pandas (plus the stack modules).
"""
import os
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

import paper_tables as pt
import event_study_driver as esd
import market_shocks as mk
import stress_index as si

_CALM = {"calm", "baseline", "quiet", "normal"}
_STRESS = {"stress", "volatile", "crisis", "turmoil"}
EPS = 1e-12


def _derive_sessions_by_date(sessions):
    """{date -> df} from a (date, regime, df) panel (last session wins on a date collision)."""
    out = {}
    for date, _regime, df in sessions:
        out[pd.Timestamp(date).date()] = df
    return out


def _pick_regime(sessions, which):
    """First session df whose regime is in `which`; fall back to first/last session."""
    for date, regime, df in sessions:
        if str(regime).lower() in which:
            return df
    return sessions[0][2] if which is _CALM else sessions[-1][2]


def run_contagion(sessions, daily_vol, sessions_by_date=None, daily_state=None, counts_fn=None,
                  calm=None, stress=None, event_categories=("MWCB",), event_classes=None,
                  n_controls=2, buffer_days=3, window_days=20, caliper=None, default_release_et="09:30",
                  target_qty=None, n_levels=10, outcome="cost_to_fill", source="ES", target="SPY",
                  pre_min=5, post_min=5, n_boot=0, n_jobs=None, out_dir=None, write=True, verbose=True,
                  tz="America/New_York"):
    """Run the full pipeline. `sessions` is a (date, regime, df) panel; `daily_vol` a daily
    vol Series (also the ex-ante matching state unless `daily_state` is given). Optional:
    `sessions_by_date` ({date->df} for the event study; derived from `sessions` if omitted),
    `counts_fn` (Table 5 order-flow counts), `calm`/`stress` session frames (auto-picked by
    regime otherwise). `event_categories`/`event_classes` select the event-study regime.
    Returns a dict with `selection`, `event_study`, `tables`, `report` paths, and `stages`."""
    results = {"stages": OrderedDict(),
               "config": {"event_categories": event_categories, "event_classes": event_classes,
                          "outcome": outcome, "n_controls": n_controls, "buffer_days": buffer_days,
                          "window_days": window_days, "n_boot": n_boot}}
    log = (lambda m: print(m)) if verbose else (lambda m: None)

    if sessions_by_date is None:
        sessions_by_date = _derive_sessions_by_date(sessions)
    if daily_state is None:
        daily_state = daily_vol
    ds_idx = pd.DatetimeIndex(daily_state.index)             # registry dates are tz-aware; match the state to them
    if ds_idx.tz is None:
        daily_state = pd.Series(np.asarray(daily_state, float), index=ds_idx.tz_localize(tz))
    if calm is None:
        calm = _pick_regime(sessions, _CALM)
    if stress is None:
        stress = _pick_regime(sessions, _STRESS)

    # ── Stage 1: selection ────────────────────────────────────────────────────
    log("[1/4] selection (stress_index + market_shocks)")
    try:
        sel = {}
        days = si.select_days(daily_vol)
        if "label" in days.columns:
            sel["regime_counts"] = {str(k): int(v) for k, v in days["label"].value_counts().items()}
        if "onset" in days.columns:
            sel["n_onsets"] = int(days["onset"].sum())
        bal = mk.compare_control_rules(daily_vol)
        sel["control_balance"] = {"smd_matched": float(bal["smd_matched"]), "smd_naive": float(bal["smd_naive"])}
        sel["n_shocks"] = int(len(mk.shocks_frame()))
        results["selection"] = sel
        results["stages"]["selection"] = "ok"
    except Exception as e:
        results["selection"] = {"error": repr(e)}
        results["stages"]["selection"] = "ERROR: " + repr(e)

    # ── Stage 2: event study (registry-driven) ────────────────────────────────
    log("[2/4] event study (event_study_driver, regime=%s)" % (event_classes or event_categories,))
    mwcb_treated, mwcb_control, release_by_date, used_fallback = [], [], {}, True
    try:
        es = esd.run_event_study(daily_state, sessions_by_date, categories=event_categories,
                                 classes=event_classes, n_controls=n_controls, buffer_days=buffer_days,
                                 window_days=window_days, caliper=caliper, default_release_et=default_release_et, tz=tz,
                                 asset=target, outcome=outcome, source=source, target=target,
                                 pre_min=pre_min, post_min=post_min, target_qty=target_qty, n_levels=n_levels)
        ev = {k: es.get(k) for k in ("regimes", "mixed_regime", "n_treated", "n_control", "dropped")}
        for k in ("did", "rd", "spillover"):
            if es.get(k) is not None:
                ev[k] = es[k]
        results["event_study"] = ev
        if es["n_treated"] >= 2 and es["n_control"] >= 2:
            mwcb_treated, mwcb_control, release_by_date = es["treated"], es["control"], es["release_by_date"]
            used_fallback = False
            results["stages"]["event_study"] = "ok (registry-driven, %d treated / %d control)" % (
                es["n_treated"], es["n_control"])
        else:
            results["stages"]["event_study"] = "ok (insufficient registry coverage -> tables use synthetic MWCB books)"
    except Exception as e:
        results["event_study"] = {"error": repr(e)}
        results["stages"]["event_study"] = "ERROR: " + repr(e)

    if used_fallback:                                       # keep the MWCB tables renderable
        rel = pd.Timestamp("2020-03-09 13:00", tz=tz)
        mwcb_treated = [(f"T{i}", pt._synth_release_book(f"2020-03-{9+i:02d}", rel, True, 100 + i)) for i in range(4)]
        mwcb_control = [(f"C{i}", pt._synth_release_book(f"2020-02-{10+i:02d}", rel, False, 200 + i)) for i in range(4)]
        release_by_date = {d: rel for d, _ in mwcb_treated + mwcb_control}
        results.setdefault("event_study", {})["fallback_tables"] = True

    # ── Stage 3: full table suite (fed the event-study design) ─────────────────
    log("[3/4] tables (paper_tables.build_all_tables, n_boot=%d, boot-workers=%s)"
        % (n_boot, "all" if n_jobs is None else n_jobs))
    tables = OrderedDict()
    try:
        tables = pt.build_all_tables(sessions, counts_fn, daily_vol, mwcb_treated, mwcb_control,
                                     release_by_date, calm, stress, n_boot=n_boot, verbose=verbose,
                                     n_jobs=n_jobs)
        n_err = sum(1 for t in tables.values() if "BUILD ERROR" in (t.notes or ""))
        results["tables"] = tables
        results["n_tables"] = len(tables)
        results["n_table_errors"] = n_err
        results["stages"]["tables"] = "%d/%d ok" % (len(tables) - n_err, len(tables))
    except Exception as e:
        results["tables"] = tables
        results["stages"]["tables"] = "ERROR: " + repr(e)

    # ── Stage 4: emit manuscript artifacts ─────────────────────────────────────
    log("[4/4] emit manuscript tables + summary")
    if write and tables:
        out_dir = out_dir or os.environ.get("OUT_DIR", "/mnt/user-data/outputs")
        md, tex = pt.write_report(tables,
                                  os.path.join(out_dir, "contagion_manuscript_tables.md"),
                                  os.path.join(out_dir, "contagion_manuscript_tables.tex"))
        summ = _write_summary(results, os.path.join(out_dir, "contagion_run_summary.md"))
        results["report"] = {"markdown": md, "latex": tex, "summary": summ}
    return results


def _write_summary(results, path):
    """A concise run summary: stage statuses + headline numbers + the table inventory."""
    L = ["# Liquidity-Contagion Pipeline — Run Summary", "",
         "End-to-end run of `run_contagion` (Garrison, Jain & Paddrik 2026 revamp).", "",
         "## Stages", ""]
    for k, v in results["stages"].items():
        L.append(f"- **{k}**: {v}")
    sel = results.get("selection", {})
    if "control_balance" in sel:
        cb = sel["control_balance"]
        L += ["", "## Selection", "",
              f"- shock registry events: {sel.get('n_shocks', 'NA')}",
              f"- regime day counts: {sel.get('regime_counts', 'NA')}; onsets: {sel.get('n_onsets', 'NA')}",
              f"- control balance (standardized mean diff): level-matched |SMD|={abs(cb['smd_matched']):.3f} "
              f"vs ln(σ)-rule |SMD|={abs(cb['smd_naive']):.3f} (lower is better-balanced)"]
    ev = results.get("event_study", {})
    if "did" in ev:
        d, r = ev["did"], ev.get("rd", {})
        sp = (ev.get("spillover") or {}).get("spill_post_treated", {})
        L += ["", "## Event study (%s)" % (", ".join(ev.get("regimes", [])) or "NA"), "",
              f"- treated sessions: {ev.get('n_treated')}, control sessions: {ev.get('n_control')}"
              + ("  (synthetic-book fallback for tables)" if ev.get("fallback_tables") else ""),
              f"- DiD (treated×post) = {d.get('did_coef'):.4f} (t={d.get('did_t'):.2f})",
              f"- RD-in-time jump at the boundary = {r.get('jump'):.4f} (t={r.get('jump_t'):.2f})"]
        if sp:
            L.append(f"- {ev.get('source','ES')}→{ev.get('target','SPY')} spillover (post×treated) "
                     f"= {sp.get('coef'):.4f} (t={sp.get('t'):.2f})")
    tables = results.get("tables", {})
    if tables:
        L += ["", "## Tables (%d; %d build errors)" % (results.get("n_tables", len(tables)),
                                                       results.get("n_table_errors", 0)), ""]
        for name, t in tables.items():
            flag = "  ⚠ BUILD ERROR" if "BUILD ERROR" in (t.notes or "") else ""
            L.append(f"- `{name}` — {t.title}{flag}")
        L += ["", "Full tables in `contagion_manuscript_tables.md` / `.tex`."]
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


# ── descriptive (regime) analysis ─────────────────────────────────────────────
def _ofi_proxy_counts(df, total_per_bar=10.0):
    """Book-OFI-implied order-flow buy/sell counts for Table 5 when no trade tape is on hand:
    per-bar buy share = 0.5 + 0.4*tanh(standardized OFI), times a constant total. A PROXY for
    the message-tape counts -- supply a real counts_fn for the exact Table 5."""
    import cross_asset_pd_liquidity as ca
    res = {}
    for a in ("SPY", "ES"):
        ofi = np.nan_to_num(np.asarray(ca.order_flow_imbalance(df, a, 10), float))
        sd = float(ofi.std()) or 1.0
        pbuy = 0.5 + 0.4 * np.tanh(ofi / (2.0 * sd))
        mid = np.asarray(ca._mid(df, a), float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = np.concatenate([[np.nan], np.diff(np.log(mid)) * 1e4])
        res[a] = (total_per_bar * pbuy, total_per_bar * (1.0 - pbuy), ret)
    return (res["SPY"][0], res["SPY"][1], res["ES"][0], res["ES"][1], res["SPY"][2], res["ES"][2])


def _compute_daily_vol(sessions, asset="SPY", ann=252 * 6.5 * 3600):
    """Per-session annualized realized vol from the (secondly) mid returns, date-indexed."""
    import cross_asset_pd_liquidity as ca
    rows = {}
    for d, _r, df in sessions:
        lp = np.log(np.asarray(ca._mid(df, asset), float)); r = np.diff(lp); r = r[np.isfinite(r)]
        if len(r) >= 2:
            rows[pd.Timestamp(d)] = float(np.sqrt(np.mean(r ** 2) * ann))
    return pd.Series(rows).sort_index()


def _load_or_compute_vol(sessions, vol_csv=None, vol_pickle=None):
    if vol_csv:
        d = pd.read_csv(vol_csv); low = {c.lower(): c for c in d.columns}
        dc = low.get("date", d.columns[0]); vc = low.get("vol", low.get("volatility", d.columns[1]))
        return pd.Series(d[vc].to_numpy(float), index=pd.to_datetime(d[dc])).sort_index()
    if vol_pickle:
        return pd.read_pickle(vol_pickle)
    return _compute_daily_vol(sessions)


def run_descriptive(sessions, daily_state=None, n_lags=5, vol_win=300, out_dir=None,
                    write=True, verbose=True, criterion=None, pmax=12):
    """Descriptive (regime) analysis -- ES information share as a continuous function of the
    EX-ANTE volatility level, the design appropriate when treatment is selected on vol. Two
    reads: (1) per-session liquidity-state-conditional VECM on a trailing realized-vol state
    (CS_ES at low/median/high vol + the slope t-stat t_delta_es); (2) a cross-day OLS of
    per-day CS_ES on the ex-ante daily vol level. Both condition on a predetermined vol, never
    the contemporaneous realized vol the outcome also responds to."""
    import cross_asset_pd_liquidity as ca, price_discovery_shares as pds, robustness as rb
    log = (lambda m: print(m)) if verbose else (lambda m: None)
    log("[descriptive] per-session vol-conditioned VECM (%d sessions)" % len(sessions))
    rows = []
    for d, regime, df in sessions:
        lp = np.log(np.asarray(ca._mid(df, "SPY"), float)); r = np.diff(lp, prepend=lp[0])
        rv = pd.Series(r ** 2).rolling(vol_win).sum().shift(1).to_numpy()   # trailing, predetermined
        state = np.nan_to_num(np.sqrt(np.maximum(rv, 0.0)))
        try:
            lc = ca.liquidity_conditional_vecm(ca._mid(df, "SPY"), ca._mid(df, "ES"), state, n_lags=n_lags)
            sb = lc["shares_by_state"]
            rows.append({"date": str(d), "regime": regime,
                         "CS_ES_lo_vol": round(sb[0.1]["CS_ES"], 4), "CS_ES_med_vol": round(sb[0.5]["CS_ES"], 4),
                         "CS_ES_hi_vol": round(sb[0.9]["CS_ES"], 4), "t_delta_es": round(lc["t_delta_es"], 3)})
        except Exception as e:
            rows.append({"date": str(d), "regime": regime, "error": repr(e)})
    per_session = pd.DataFrame(rows)

    log("[descriptive] cross-day CS_ES-on-vol slope")
    slope = {}
    try:
        per_day = pds.estimate_sample(rb._mid_sessions(sessions), n_lags=n_lags,
                                      criterion=criterion, pmax=pmax, lag_selection="pooled")
        y = per_day["CS_ES"].to_numpy()
        if daily_state is not None:
            dv = {str(pd.Timestamp(i).date()): float(x)
                  for i, x in zip(pd.DatetimeIndex(daily_state.index), np.asarray(daily_state, float))}
            v = np.array([dv.get(str(pd.Timestamp(i).date()), np.nan) for i in per_day.index])
            X = np.column_stack([np.ones(len(v)), v]); ok = np.all(np.isfinite(np.column_stack([X, y])), axis=1)
            if ok.sum() >= 3:
                b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
                u = y[ok] - X[ok] @ b; n, k = X[ok].shape
                V = (u @ u / max(n - k, 1)) * np.linalg.inv(X[ok].T @ X[ok])
                slope = {"intercept": float(b[0]), "slope_vol": float(b[1]),
                         "t_slope": float(b[1] / (np.sqrt(V[1, 1]) + EPS)), "n_days": int(ok.sum())}
            else:
                slope = {"note": "need >=3 days with matched vol for the cross-day slope"}
        else:
            slope = {"note": "no daily_state provided"}
    except Exception as e:
        slope = {"error": repr(e)}

    out = {"per_session_vol_conditioning": per_session, "cross_day_slope": slope}
    if write:
        out_dir = out_dir or os.environ.get("OUT_DIR", "/mnt/user-data/outputs")
        os.makedirs(out_dir, exist_ok=True)
        per_session.to_csv(os.path.join(out_dir, "descriptive_vol_conditioning.csv"), index=False)
        with open(os.path.join(out_dir, "descriptive_vol_conditioning.md"), "w") as fh:
            fh.write("# Descriptive: ES information share vs ex-ante volatility\n\n"
                     "ES information share (CS_ES) as a function of the predetermined vol level. "
                     "This is the regime read appropriate when treatment is selected on volatility "
                     "(a slope/level relationship, not a causal event contrast).\n\n"
                     "## Per-session: CS_ES at low / median / high trailing vol\n\n")
            fh.write(per_session.to_string(index=False))
            fh.write("\n\n## Cross-day: per-day CS_ES regressed on the ex-ante daily vol level\n\n")
            fh.write(str(slope) + "\n")
        out["artifacts"] = {"csv": "descriptive_vol_conditioning.csv", "md": "descriptive_vol_conditioning.md"}
        log("[descriptive] wrote descriptive_vol_conditioning.{csv,md}")
    return out


def _auto_regime_labels(sessions, daily_vol, q=0.67, halflife=5, verbose=True):
    """Sense volatile/benchmark labels from the EX-ANTE daily vol rather than hand-picking via
    --volatile: a session is 'volatile' if its lagged-EWMA vol is at/above the q-quantile of
    that level *across the loaded sessions*, else 'benchmark'. A within-sample quantile split
    (top 1-q fraction) avoids both hand-picking and the degenerate all-one-regime labels that
    an absolute threshold gives on a crisis-heavy window. Returns (relabeled_sessions, map)."""
    ds = pd.Series(np.asarray(daily_vol, float), index=pd.DatetimeIndex(daily_vol.index)).sort_index()
    lvl = ds.shift(1).ewm(halflife=halflife, min_periods=1).mean()            # ex-ante level
    by_date = {str(pd.Timestamp(i).date()): float(v) for i, v in lvl.items()}
    sess_dates = [str(pd.Timestamp(d).date()) for d, _r, _df in sessions]
    vals = np.array([by_date.get(k, np.nan) for k in sess_dates], float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 3:
        if verbose:
            print("[auto-regime] too few finite ex-ante vols; keeping existing labels")
        return sessions, {}
    thr = float(np.quantile(finite, q))
    lab = {k: ("volatile" if (np.isfinite(by_date.get(k, np.nan)) and by_date[k] >= thr) else "benchmark")
           for k in sess_dates}
    out = [(d, lab[str(pd.Timestamp(d).date())], df) for (d, _r, df) in sessions]
    if verbose:
        from collections import Counter
        print("[auto-regime] ex-ante vol q=%.2f threshold=%.4g -> %s" % (q, thr, dict(Counter(lab.values()))))
    return out, lab


# ── command-line interface ────────────────────────────────────────────────────
def main_cli(argv=None):
    """CLI: load sessions (via run_analysis), get/derive the daily vol, and run the causal
    and/or descriptive contagion models, writing the manuscript tables + summaries."""
    import argparse, logging
    import run_analysis as ra
    p = argparse.ArgumentParser(description="Liquidity-contagion pipeline (causal and/or descriptive) from the CLI.")
    # data loading (reuses run_analysis.load_sessions: same date-selection + book alignment)
    p.add_argument("--source", choices=["demo", "load", "extract"], default="demo")
    p.add_argument("--pickle", default=None, help="extracted frames for --source load")
    p.add_argument("--save-pickle", default=None,
                   help="after loading/extraction (and --auto-regime), dump the (date,regime,df) "
                        "sessions to this pickle so a downstream `run_analysis --source load --pickle` "
                        "can run the upgraded analysis on the SAME frames without re-extracting.")
    p.add_argument("--upgraded", action="store_true",
                   help="after the contagion pipeline, run the re-floored v0.6-0.8 analysis "
                        "(run_analysis: day-clustered/wild-cluster inference, Hayashi-Yoshida + Rigobon "
                        "leadership, the two-axis liquidity-stress conditioner, the event-clock study, "
                        "and the LEGACY before/after tables) IN-PROCESS on the same in-memory sessions — "
                        "no pickle, no second extraction/load.")
    p.add_argument("--upgraded-skip", type=lambda s: s.split(","), default=None,
                   help="stages to skip inside the --upgraded run (e.g. 'dcc' for the 100ms lens)")
    p.add_argument("--volatile", type=lambda s: s.split(","), default=None,
                   help="dates to label 'volatile' (and, for extract, the extraction universe)")
    p.add_argument("--benchmark", type=lambda s: s.split(","), default=None)
    p.add_argument("--dates", type=lambda s: s.split(","), default=None, help="explicit extraction universe")
    p.add_argument("--date-range", default=None,
                   help="extraction universe as START:END business days, e.g. 2020-02-20:2020-03-20")
    p.add_argument("--auto-regime", action="store_true",
                   help="sense volatile/benchmark labels from the ex-ante daily vol (stress_index) "
                        "instead of hand-picking with --volatile; affects the regime-split tables")
    p.add_argument("--interval", default="1s")
    p.add_argument("--n-levels", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=4,
                   help="parallelize the extract session-loop across this many processes (each day is "
                        "one I/O-heavy unit; peak RAM ~ this x one full day, so set to the memory-safe "
                        "count, <= #sessions). 1 = serial (keeps the per-fetch heartbeat). Only the "
                        "--source extract path uses it.")
    p.add_argument("--extract-backend", choices=["process", "threading", "sequential"], default=None,
                   help="extract parallelism engine: 'process' (default when --max-workers>1; joblib loky, "
                        "else a stdlib process pool), 'threading' (shared-memory), or 'sequential' "
                        "(force serial regardless of --max-workers).")
    p.add_argument("--progress", choices=["auto", "on", "off"], default="auto",
                   help="extract: per-session progress bar (tqdm) with ETA + per-fetch heartbeat. "
                        "'auto' shows a live bar on a TTY and falls back to INFO logging when redirected.")
    p.add_argument("--auction", action="store_true",
                   help="extract: also compute opening/closing-cross imbalance features (mt_order_imbalance) "
                        "and the SPY opening-imbalance -> ES move linkage; writes auction_imbalance.{csv,md} "
                        "and auction_linkage.csv (auction_imbalance module).")
    p.add_argument("--copula", action="store_true",
                   help="also fit copula-GARCH dependence: constant-copula selection with tail dependence, "
                        "a t-copula-DCC dynamic baseline, and liquidity-conditional lower-tail dependence; "
                        "writes copula_garch_summary.md and copula_selection.csv (copula_garch module).")
    p.add_argument("--no-align-books", action="store_true")
    p.add_argument("--classify", choices=["aggressor", "tick", "side"], default="aggressor",
                   help="extract: trade-direction rule for the real trade tape -- 'aggressor' (CME "
                        "tag 5797; equity tick-rule fallback), 'tick' (Lee-Ready everywhere), or "
                        "'side' (trust the message side field)")
    p.add_argument("--book-source", choices=["snapshot", "reconstruct"], default="reconstruct",
                   help="extract: both legs are reconstructed from mstwx-lakequery messages on one "
                        "capture clock (lob_reconstruct) -- SPY consolidated multi-venue, ES single-venue "
                        "CME (order-based). 'snapshot' is retired (mstbook-query is on a different clock "
                        "and being sunset); it now warns and reconstructs anyway.")
    p.add_argument("--round-lot", type=int, default=100,
                   help="reconstruct: shares for round-lot NBBO eligibility (SPY=100)")
    p.add_argument("--round-lot-only", action="store_true",
                   help="reconstruct: build a round-lot-only ladder (default ladder is odd-lot-inclusive)")
    p.add_argument("--clock", choices=["receipt", "exchange"], default="receipt",
                   help="reconstruct: event/grid clock -- 'receipt' (GPS source-capture time, one "
                        "uniform UTC methodology across venues; the default) or 'exchange' (per-venue "
                        "engine time, a robustness lens). Under capture co-located at the venues the "
                        "two agree to engine/GPS jitter, so this should barely move results.")
    p.add_argument("--n-sessions", type=int, default=6, help="demo: number of sessions")
    p.add_argument("--session-len", type=int, default=4000, help="demo: rows per session")
    # daily vol (selection / matching state)
    p.add_argument("--vol-csv", default=None, help="CSV with columns date,vol (annualized); else computed per session")
    p.add_argument("--vol-pickle", default=None, help="pickle of a daily vol Series")
    # what to run
    p.add_argument("--mode", choices=["causal", "descriptive", "both", "mean_variance"], default="both")
    # causal (event study)
    p.add_argument("--event-categories", type=lambda s: tuple(s.split(",")), default=("MWCB",),
                   help="registry categories for treatment (one identification regime), e.g. MWCB")
    p.add_argument("--event-classes", type=lambda s: tuple(s.split(",")), default=None,
                   help="registry classes, e.g. TRADE_POLICY or GEOPOLITICAL (use instead of categories)")
    p.add_argument("--outcome", default="cost_to_fill",
                   choices=["cost_to_fill", "inside_depth", "composite", "auc_decay", "curve_length"],
                   help="liquidity object that spills over. auc_decay = decay-weighted cost-to-demand (bps), "
                        "no fixed fill size (--target-qty sets the decay scale Q0 if given); curve_length = "
                        "dimensionless ||L|| convexity of the impact curve (ignores --target-qty)")
    p.add_argument("--spillover-source", default="ES")
    p.add_argument("--spillover-target", default="SPY")
    p.add_argument("--target-qty", type=float, default=None)
    p.add_argument("--pre-min", type=int, default=5)
    p.add_argument("--post-min", type=int, default=5)
    p.add_argument("--n-controls", type=int, default=2)
    p.add_argument("--buffer-days", type=int, default=5)
    p.add_argument("--window-days", type=int, default=60)
    p.add_argument("--caliper", type=float, default=None, help="max standardized vol distance for a matched control")
    p.add_argument("--default-release-et", default="09:30")
    p.add_argument("--n-boot", type=int, default=0)
    p.add_argument("--boot-workers", type=int, default=None,
                   help="threads for the day-cluster/IRF bootstrap (default: all cores). The bootstrap "
                        "runs after extraction with data resident, so it is not memory-bound -- unlike "
                        "--max-workers (extraction), this can safely use every vCPU.")
    p.add_argument("--no-counts", action="store_true", help="skip the Table-5 contingency (else a book-OFI proxy is used)")
    # descriptive
    p.add_argument("--n-lags", type=int, default=5)
    p.add_argument("--lag-criterion", choices=["fixed", "bic", "aic", "hqic"], default="fixed",
                   help="VECM/SVAR lag order: 'fixed' uses --n-lags; otherwise the order is chosen by "
                        "the information criterion over [0, --max-lags]. BIC is the consistent (parsimonious) "
                        "choice and is preferred for information-share estimation. Selection is POOLED "
                        "across sessions (one order minimising the summed criterion) for comparability.")
    p.add_argument("--max-lags", type=int, default=12, help="max lag order searched when --lag-criterion != fixed")
    p.add_argument("--price", choices=["mid", "wmid", "wmid_orth", "depth_wmid", "mid_depth"], default="mid",
                   help="mean_variance config: mid = ln(simple mid) LHS + microprice-anchored curves; "
                        "wmid = ln(microprice) LHS + simple-mid-anchored curves; wmid_orth = microprice on "
                        "both, covariates orthogonalized; depth_wmid = ln(depth-weighted mid) LHS + simple-mid "
                        "curves; mid_depth = simple-mid LHS + depth-weighted-mid-anchored curves")
    p.add_argument("--vol-win", type=int, default=300, help="descriptive: trailing-vol state window (bars)")
    p.add_argument("--output-dir", default=os.environ.get("OUT_DIR", "/mnt/user-data/outputs"))
    p.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if args.selftest:
        _selftest(); return None

    sessions = ra.load_sessions(args)                       # reuses the loader (date universe + alignment)
    daily_vol = _load_or_compute_vol(sessions, args.vol_csv, args.vol_pickle)
    if args.auto_regime:
        sessions, _ = _auto_regime_labels(sessions, daily_vol)
    if getattr(args, "save_pickle", None):
        import pickle as _pkl
        with open(args.save_pickle, "wb") as _fh:
            _pkl.dump(sessions, _fh)
        LOG.info("saved %d sessions -> %s  (reuse: run_analysis --source load --pickle %s)",
                 len(sessions), args.save_pickle, args.save_pickle)
    sbd = {pd.Timestamp(d).date(): df for d, _r, df in sessions}
    os.makedirs(args.output_dir, exist_ok=True)
    counts = None
    if not args.no_counts:
        try:
            import mstbook_loader as ml
            if ml.has_trade_flow(sessions):                 # CLI extract attached the real trade tape
                counts = ml.counts_from_frame
                LOG.info("Table 5 counts: using real trade-tape counts from the CLI extract")
            else:
                counts = _ofi_proxy_counts
        except Exception:
            counts = _ofi_proxy_counts
        if counts is _ofi_proxy_counts:
            LOG.info("Table 5 counts: using the book-OFI proxy (no trade tape attached)")
    results = {}

    if args.mode in ("causal", "both"):
        results["causal"] = run_contagion(
            sessions, daily_vol, sessions_by_date=sbd, counts_fn=counts,
            event_categories=args.event_categories, event_classes=args.event_classes,
            n_controls=args.n_controls, buffer_days=args.buffer_days, window_days=args.window_days,
            caliper=args.caliper, default_release_et=args.default_release_et,
            target_qty=args.target_qty, n_levels=args.n_levels, outcome=args.outcome,
            source=args.spillover_source, target=args.spillover_target,
            pre_min=args.pre_min, post_min=args.post_min, n_boot=args.n_boot, n_jobs=args.boot_workers,
            out_dir=args.output_dir, write=True, verbose=True)
        ev = results["causal"].get("event_study", {})
        print("\nCAUSAL stages:", dict(results["causal"]["stages"]))
        if "did" in ev:
            print("  DiD=%.4f (t=%.2f) | RD jump=%.4f (t=%.2f) | treated=%s control=%s"
                  % (ev["did"]["did_coef"], ev["did"]["did_t"], ev["rd"]["jump"], ev["rd"]["jump_t"],
                     ev.get("n_treated"), ev.get("n_control")))

    if args.mode in ("descriptive", "both"):
        results["descriptive"] = run_descriptive(
            sessions, daily_state=daily_vol, n_lags=args.n_lags, vol_win=args.vol_win,
            out_dir=args.output_dir, write=True, verbose=True,
            criterion=(None if args.lag_criterion == "fixed" else args.lag_criterion), pmax=args.max_lags)
        print("\nDESCRIPTIVE cross-day CS_ES-on-vol slope:", results["descriptive"]["cross_day_slope"])

    if args.mode == "mean_variance":
        import mean_variance as mvm
        results["mean_variance"] = mvm.run_mean_variance(
            sessions, n_lags=args.n_lags, n_levels=args.n_levels, decay_scale=args.target_qty,
            price=args.price, out_dir=args.output_dir, write=True, verbose=True,
            criterion=(None if args.lag_criterion == "fixed" else args.lag_criterion), pmax=args.max_lags)
        r = results["mean_variance"]
        print("\nMEAN-VARIANCE: pooled CS_ES=%.3f | rho mean=%.3f calm=%.3f stress=%.3f (stress-calm=%.3f)"
              % (r.get("info_shares_pooled", {}).get("CS_ES_mean", float("nan")), r.get("rho_mean", float("nan")),
                 r.get("rho_calm", float("nan")), r.get("rho_stress", float("nan")),
                 r.get("rho_diff_stress_minus_calm", float("nan"))))
        print("  gamma liquidity->vol (SPY):", r.get("gamma_liquidity_to_vol", {}).get("SPY"))

    # optional: opening/closing-cross imbalance features + SPY-open -> ES-move linkage
    if getattr(args, "auction", False):
        try:
            import auction_imbalance as ai
            ar = ai.run_auction_analysis(sessions, out_dir=args.output_dir, write=True,
                                         clock=getattr(args, "clock", "receipt"))
            s = ar["summary"].get("overall", {})
            np_ = len(ar["auction_panel"])
            print("\nAUCTION: %d (session x auction) feature rows | SPY opening imbalance -> ES post-open "
                  "return: corr=%.3f slope=%.1f (n=%d)"
                  % (np_, s.get("corr", float("nan")), s.get("slope", float("nan")), s.get("n", 0)))
            if np_ == 0:
                print("  (no auction imbalance returned — needs --source extract with the message binary)")
        except Exception as exc:
            print("\nAUCTION: skipped (%s)" % exc)

    # optional: copula-GARCH dependence (tail dependence + liquidity-conditional joint-crash risk)
    if getattr(args, "copula", False):
        try:
            import copula_garch as cg
            cr = cg.run_copula_analysis(sessions=sessions, liq_state="auto",
                                        out_dir=args.output_dir, write=True)
            sel, dyn, lc = cr["selection"], cr["dynamic"], cr["liquidity_conditional"]
            print("\nCOPULA: best=%s (BIC) | t-copula nu=%.1f, mean lambda_t=%.3f (Gaussian DCC tail=0); "
                  "mean rho_t=%.3f" % (sel["best"], dyn["nu"], dyn["lambda_mean"], dyn["rho_mean"]))
            if lc is not None:
                print("  liquidity-conditional lambda_L: thin=%.3f deep=%.3f (delta=%.3f), block corr=%.2f"
                      % (lc["lambda_L_thin"], lc["lambda_L_deep"], lc["lambda_L_delta"],
                         lc["block_corr_lambdaL_liq"]))
        except Exception as exc:
            print("\nCOPULA: skipped (%s)" % exc)

    # optional: the upgraded (re-floored v0.6-0.8) analysis, IN-PROCESS on the same in-memory
    # sessions -> no pickle round-trip, no second extraction/load. Reuses run_analysis.run_stages.
    if getattr(args, "upgraded", False):
        try:
            a = ra.parse_args([])                              # fully-populated run_analysis defaults
            a.n_levels = args.n_levels
            a.interval = getattr(args, "interval", "1s")
            a.legacy = True
            a.events = None
            ec = getattr(args, "event_classes", None)
            if isinstance(ec, str):
                ec = [x for x in ec.split(",") if x]
            a.event_class = list(ec) if ec else None
            if getattr(args, "upgraded_skip", None):
                a.skip = list(args.upgraded_skip)
            a.output_dir = os.path.join(args.output_dir or ".", "upgraded")
            # carry the PRE-MASK frames through: load_sessions stashed them on THIS driver's args,
            # and the fresh namespace would otherwise fall back to saving the masked in-memory
            # list -- a masked-at-rest frames_<interval>.pkl under upgraded/, logged as pre-mask
            a._premask_sessions = getattr(args, "_premask_sessions", None)
            LOG.info("UPGRADED: re-floored analysis in-process on %d sessions -> %s",
                     len(sessions), a.output_dir)
            up = ra.run_stages(sessions, a)
            results["upgraded"] = up[0]
            print("\nUPGRADED: re-floored analysis + legacy before/after written to %s" % a.output_dir)
        except Exception as exc:
            print("\nUPGRADED: skipped (%s)" % exc)

    # run-completion manifest: list what was actually written, with sizes
    try:
        import glob
        files = sorted(f for f in glob.glob(os.path.join(args.output_dir, "*"))
                       if os.path.isfile(f) and not f.endswith(".gitkeep"))
        print("\nArtifacts written to %s (%d file(s)):" % (args.output_dir, len(files)))
        for f in files:
            kb = os.path.getsize(f) / 1024.0
            print("  %8.1f KB  %s" % (kb, os.path.basename(f)))
        print("Sessions analyzed: %d" % len(sessions))
    except Exception as exc:                             # never let a manifest error mask a good run
        print("\nArtifacts in", args.output_dir, "(manifest unavailable: %s)" % exc)
    return results


# ── self-test ─────────────────────────────────────────────────────────────────
def _selftest():
    print("run_contagion self-test\n")
    tz = "America/New_York"

    # (a) a (date, regime, df) panel for the correlation / regime / tick / Hawkes / pricing tables
    sessions = []
    for i in range(8):
        regime = "calm" if i < 4 else "stress"
        d = f"2024-08-{i+1:02d}"
        sessions.append((d, regime, pt._synth_session(d, regime, T=1200, seed=i)))
    daily_vol = pt._synth_vol()                              # spans 2019-09 .. 2020-08 (incl. the MWCB days)

    # (b) date-keyed sessions on the real March-2020 MWCB days + clean neighbours (10 levels, for the event study + MWCB tables)
    days = pd.bdate_range("2020-02-18", "2020-04-09", tz=tz)
    rby = {k.date(): v for k, v in mk.release_times(categories=("MWCB",), tz=tz).items()}
    sessions_by_date = {}
    for i, dd in enumerate(days):
        reopen = rby.get(dd.date())
        sessions_by_date[dd] = esd._day_book(dd.date().isoformat(), reopen, reopen is not None,
                                             seed=300 + i, n_levels=10)

    res = run_contagion(sessions, daily_vol, sessions_by_date=sessions_by_date, counts_fn=pt._counts_fn,
                        event_categories=("MWCB",), n_controls=2, buffer_days=3, window_days=15,
                        target_qty=100, n_levels=10, n_boot=20,
                        out_dir=os.environ.get("OUT_DIR", "/mnt/user-data/outputs"), write=True, verbose=True)

    ev = res.get("event_study", {})
    rep = res.get("report", {})
    n_tab = res.get("n_tables", 0); n_err = res.get("n_table_errors", 99)
    sizes_ok = all(os.path.getsize(rep[k]) > 800 for k in ("markdown", "latex", "summary")) if rep else False
    print("\nstages:")
    for k, v in res["stages"].items():
        print("   %-12s %s" % (k, v))
    print("\nheadline: DiD=%.4f (t=%.2f), RD jump=%.4f (t=%.2f); treated=%s control=%s; fallback=%s"
          % (ev.get("did", {}).get("did_coef", float("nan")), ev.get("did", {}).get("did_t", float("nan")),
             ev.get("rd", {}).get("jump", float("nan")), ev.get("rd", {}).get("jump_t", float("nan")),
             ev.get("n_treated"), ev.get("n_control"), ev.get("fallback_tables", False)))
    print("tables: %d (%d errors); artifacts written: %s" % (n_tab, n_err, bool(rep)))
    print("\nchecks:",
          res["stages"].get("selection") == "ok"
          and ev.get("n_treated", 0) == 4 and not ev.get("fallback_tables", False)   # registry-driven path, not fallback
          and ev.get("did", {}).get("did_coef", -1) > 0 and ev.get("did", {}).get("did_t", 0) > 2
          and n_tab == 16 and n_err == 0 and sizes_ok)

    # descriptive model on the same panel
    dr = run_descriptive(sessions, daily_state=daily_vol, n_lags=5, vol_win=200,
                         out_dir=os.environ.get("OUT_DIR", "/mnt/user-data/outputs"), write=True, verbose=False)
    ps = dr["per_session_vol_conditioning"]
    print("descriptive checks:",
          len(ps) == len(sessions) and "CS_ES_hi_vol" in ps.columns
          and ("slope_vol" in dr["cross_day_slope"] or "note" in dr["cross_day_slope"]))


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main_cli()
