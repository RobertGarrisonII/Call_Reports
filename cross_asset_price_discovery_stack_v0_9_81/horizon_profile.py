"""
horizon_profile.py
==================
The propagation-horizon ladder (memo item E1): the same estimators run at several
sampling grids derived from ONE set of fine frames, so the grid itself becomes the
experimental variable.

The 2026-08-14 run bracketed the SPY<-ES cross-impact between ~0 at 10ms and ~0.4
at 1s: futures flow is impounded into the equity price at some horizon between
those grids, and two points cannot locate it. This module derives an interval
ladder from the 10ms frames (pull-once: `derive_coarse_frame`, no new extraction)
and, per rung, runs the estimators for which the grid IS the question:

  * cross-impact lambda by direction (the headline; cross_impact_panel, with the
    fleeting-quote filter width scaled per grid via frequency_defaults)
  * information shares / CS / kappa (the resolution curve of price discovery --
    the 10ms->1s IS_ES decline of the findings memo, traced through the interior)
  * the error-correction half-life ln2/kappa x dt (memo E4's grid-dependence
    experiment: smooth convergence implicates lag truncation, a staleness-shaped
    profile implicates alpha attenuation)
  * co-jump lead shares (0% resolvable leads at 1s vs 4.6:1 at 10ms -- the interior
    shows where the lead becomes visible)
  * the staleness companion (zero-return fraction per asset -- the confounder every
    number on the ladder must be read against)

The summary statistic is the HALF-IMPACT HORIZON: with the cross-impact profile
normalized by its value at the coarsest rung, the (log-interpolated) interval at
which the ratio crosses 0.5 -- the single number that says how fast tandem flow
becomes price. Per-bar lambda units are not comparable across grids (bar returns
and bar OFI both scale with the bar), which is exactly why the profile is defined
as a RATIO to the coarsest rung: cumulative impact captured within dt, as a
fraction of the impact captured at the reference horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import cross_impact as cimp
import derive_frames as dfr
import jump_robust as jr
import microstructure_diagnostics as md
import price_discovery_shares as pds
import robustness as rb

EPS = 1e-12
DEFAULT_INTERVALS = ("10ms", "50ms", "100ms", "250ms", "500ms", "1s")


def _sessions_at(fine_sessions, interval, fine_interval):
    """The ladder rung: the fine frames themselves at the fine interval, a derived
    coarse frame per session otherwise. Sessions whose frame cannot be derived
    (too short, misaligned) are skipped with the reason recorded."""
    if interval == fine_interval:
        return list(fine_sessions), {}
    out, skipped = [], {}
    for date, regime, df in fine_sessions:
        try:
            out.append((date, regime, dfr.derive_coarse_frame(df, interval, fine_interval)))
        except (ValueError, KeyError) as exc:
            skipped[str(date)] = str(exc)
    return out, skipped


def profile_ladder(fine_sessions, intervals=DEFAULT_INTERVALS, fine_interval="10ms",
                   n_levels=10, cojump_max_lag=2, hac_lags=10):
    """Run the ladder. fine_sessions = List[(date, regime, df)] on the fine grid.

    -> (per_day long DataFrame, per-interval summary DataFrame, meta dict).
    per_day rows: one per (interval, date) with lambda_cross both directions,
    IS_mid_ES, CS_ES, kappa, half_life_s, cojump counts/lead_share, and the
    zero-return fractions. summary: day-means per interval with the normalized
    cross-impact ratio; meta carries the half-impact horizons and any skips."""
    rows = []
    skips = {}
    for iv in intervals:
        sess, skipped = _sessions_at(fine_sessions, iv, fine_interval)
        if skipped:
            skips[iv] = skipped
        if not sess:
            continue
        dt = dfr._interval_seconds(iv)
        fc = ca.frequency_defaults(dt=dt)
        panel, _sum = cimp.cross_impact_panel(sess, n_levels=n_levels, hac_lags=hac_lags,
                                              min_rest_steps=fc["min_rest_steps"])
        lam = {}
        for _i, r in panel.iterrows():
            lam.setdefault(str(r["date"]), {})[f"{r['return']}<-{r['ofi']}"] = float(r["lambda"])
        mids = rb._mid_sessions(sess)
        isd = pds.estimate_sample(mids, n_lags=fc["n_lags"])
        try:
            cj = jr.cojump_by_day(mids, max_lag=cojump_max_lag)
        except Exception:
            cj = pd.DataFrame()
        for date, regime, df in sess:
            d = str(date)
            row = {"interval": iv, "dt_s": dt, "date": d, "regime": regime,
                   "n_lags": fc["n_lags"], "min_rest_steps": fc["min_rest_steps"]}
            row["lambda_spy_from_es"] = lam.get(d, {}).get("SPY<-ES", float("nan"))
            row["lambda_es_from_spy"] = lam.get(d, {}).get("ES<-SPY", float("nan"))
            if d in isd.index:
                r = isd.loc[d]
                row["IS_mid_ES"] = float(r["IS_mid_ES"])
                row["CS_ES"] = float(r["CS_ES"])
                row["kappa"] = float(r["kappa"])
                row["half_life_s"] = (float(np.log(2.0) / r["kappa"] * dt)
                                      if np.isfinite(r["kappa"]) and r["kappa"] > EPS
                                      else float("nan"))
            if len(cj) and d in cj.index:
                c = cj.loc[d]
                row["n_cojump"] = int(c.get("n_cojump", 0))
                row["lead_share"] = float(c.get("lead_share", float("nan")))
            try:
                st = md.staleness_report(df)
                for _j, s in st.iterrows():
                    row[f"zero_ret_frac_{s['asset']}"] = float(s["zero_return_frac"])
            except Exception:
                pass
            rows.append(row)
    per_day = pd.DataFrame(rows)
    if per_day.empty:
        return per_day, pd.DataFrame(), {"skipped": skips}
    agg = {c: "mean" for c in
           ("lambda_spy_from_es", "lambda_es_from_spy", "IS_mid_ES", "CS_ES",
            "half_life_s", "lead_share", "zero_ret_frac_SPY", "zero_ret_frac_ES")
           if c in per_day.columns}
    summary = per_day.groupby(["interval", "dt_s"], as_index=False).agg(agg)
    summary = summary.sort_values("dt_s").reset_index(drop=True)
    n_days = per_day.groupby("interval")["date"].nunique()
    summary["n_days"] = summary["interval"].map(n_days)
    # day-clustered SE of the headline direction, so the profile plots with a band
    se = (per_day.groupby("interval")["lambda_spy_from_es"].std()
          / np.sqrt(per_day.groupby("interval")["lambda_spy_from_es"].count().clip(lower=1)))
    summary["lambda_spy_from_es_se"] = summary["interval"].map(se)
    # the normalized profile: cumulative impact captured within dt as a fraction of
    # the coarsest rung's (per-bar lambdas are not unit-comparable across grids)
    meta = {"skipped": skips, "reference_interval": str(summary["interval"].iloc[-1])}
    for col, name in (("lambda_spy_from_es", "half_impact_spy_from_es_s"),
                      ("lambda_es_from_spy", "half_impact_es_from_spy_s")):
        ref = float(summary[col].iloc[-1])
        ratio = summary[col] / ref if np.isfinite(ref) and abs(ref) > EPS else summary[col] * np.nan
        summary[col + "_ratio"] = ratio
        meta[name] = half_impact_horizon(summary["dt_s"].to_numpy(float), ratio.to_numpy(float))
    return per_day, summary, meta


def half_impact_horizon(dt_s, ratio, level=0.5):
    """First (log-interpolated) interval at which the normalized impact profile
    crosses `level`. Rungs with non-finite ratios are dropped; a profile already
    at/above the level on its finest rung returns that rung (the horizon is at or
    below the resolution floor); one that never crosses returns NaN."""
    dt_s = np.asarray(dt_s, float); ratio = np.asarray(ratio, float)
    m = np.isfinite(dt_s) & np.isfinite(ratio)
    dt_s, ratio = dt_s[m], ratio[m]
    if len(dt_s) == 0:
        return float("nan")
    order = np.argsort(dt_s)
    dt_s, ratio = dt_s[order], ratio[order]
    if ratio[0] >= level:
        return float(dt_s[0])
    for i in range(1, len(dt_s)):
        if ratio[i] >= level:
            r0, r1 = ratio[i - 1], ratio[i]
            if r1 - r0 <= EPS:
                return float(dt_s[i])
            w = (level - r0) / (r1 - r0)
            return float(np.exp(np.log(dt_s[i - 1]) + w * (np.log(dt_s[i]) - np.log(dt_s[i - 1]))))
    return float("nan")


# ── planted-propagation DGP (shared by the gate and the demo runner) ──────────
def propagation_demo_sessions(n_days=2, n_bars=30000, delay_ms=200, seed=0,
                              instantaneous=False, n_levels=3):
    """Fine-grid (10ms) sessions where ES order flow moves the SPY mid over the
    following `delay_ms` (uniform distributed lag), or within the same 10ms bar
    when `instantaneous`. ES OFI is injected exactly as level-1 bid-quantity
    cumsum (the flow-correlation gate's construction: OFI == the injected series);
    both mids otherwise follow random walks. The recovered half-impact horizon
    must sit near delay_ms/2 for the delayed DGP and at the resolution floor for
    the instantaneous one -- that contrast is the gate."""
    rng = np.random.default_rng(seed)
    L = max(1, int(round(delay_ms / 10.0)))
    out = []
    for d in range(n_days):
        idx = pd.date_range("2024-07-24 09:30:00", periods=n_bars, freq="10ms",
                            tz="America/New_York")
        f = rng.normal(0.0, 40.0, n_bars)                  # ES flow innovations
        if instantaneous:
            impact = 2e-5 * f / 40.0
        else:
            w = np.ones(L) / L                             # uniform kernel over delay_ms
            impact = 2e-5 * np.convolve(f, w, mode="full")[:n_bars] / 40.0
        r_spy = impact + rng.normal(0, 4e-6, n_bars)
        r_es = 0.5e-5 * f / 40.0 + rng.normal(0, 4e-6, n_bars)
        mid = {"SPY": 550.0 * np.exp(np.cumsum(r_spy)),
               "ES": 5500.0 * np.exp(np.cumsum(r_es))}
        cols = {}
        for a, tick in (("SPY", 0.01), ("ES", 0.25)):
            for i in range(1, n_levels + 1):
                cols[f"{a}_bidprice_{i}"] = mid[a] - tick * i
                cols[f"{a}_askprice_{i}"] = mid[a] + tick * i
                base = 5000.0 + np.zeros(n_bars)
                if a == "ES" and i == 1:
                    base = 5000.0 + np.cumsum(f)           # OFI == f exactly
                cols[f"{a}_bidquantity_{i}"] = base
                cols[f"{a}_askquantity_{i}"] = np.full(n_bars, 5000.0)
        out.append((f"d{d}", "volatile" if d % 2 else "benchmark",
                    pd.DataFrame(cols, index=idx)))
    return out
