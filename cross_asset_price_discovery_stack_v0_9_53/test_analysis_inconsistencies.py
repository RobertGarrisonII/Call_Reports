#!/usr/bin/env python3
"""test_analysis_inconsistencies.py -- the 2026-08-05 report's contradictions, pinned at the code.

Reading the first full analysis run turned up three internal contradictions (report §4). Chasing
them found that two of them were largely ONE defect -- `run_irf` and `run_dcc` estimated their
headline objects on ``sessions[0]`` alone, and sorted-first is 2020-03-09: a circuit-breaker day,
estimated halt-included at the time. The third suspect (a 2020 cross-impact reversal) is exposed to
a single halt-reopen seam observation carrying the coefficient. And a pattern sweep for the
jump_robust splice defect (removed in v0.9.51) found five surviving sites that compressed NaN out
of prices BEFORE differencing -- re-splicing the halt seam into exactly the estimators the halt
masking was built for.

Pinned here:

  (1) NO SPLICE, BY ARITHMETIC. `estimate_day`, `panel_vecm` and `liquidity_conditional_vecm` on a
      halt-masked session must drop the halt diffs AND the contaminated lag windows -- the row
      counts are exact. The old compression produced a different (larger) count with a 200-bp
      pseudo-return inside.
  (2) `select_lag` must not see the seam: its innovation logdet on a masked day matches a no-halt
      control; a spliced 200-bp return would inflate it by orders of magnitude.
  (3) `ec_valid`: a non-correcting day (kappa <= 0) is flagged, a correcting day is not, and the
      flag equals sign(kappa) by definition.
  (4) `impact_regression`'s drop-one diagnostic: one injected seam co-movement manufactures a
      cross-impact coefficient from a true zero; the diagnostic exposes it and stays quiet on
      clean data.
  (5) `run_dcc` pools ALL sessions, reports the realized correlation of the same stacked returns,
      and its divergence warning fires exactly when |mean_rho - realized| > 0.2.
  (6) `run_irf` returns the per-regime MEDIAN FEVD and labels the retained single-session matrix
      with its date instead of presenting it as the headline.

Run: python test_analysis_inconsistencies.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import cross_impact as cimp
import market_halts as mh
import price_discovery_shares as pds
from test_halt_masked_estimation import _pair_session, _mid

TZ = "America/New_York"


def _masked_mids(**kw):
    df, n_halt = _pair_session(**kw)
    masked, _ = mh.mask_frame(df)
    return _mid(masked, "SPY"), _mid(masked, "ES"), n_halt, df


def check_estimate_day_row_accounting():
    """Masked day, n_lags=10: rows = (n-1-p) - (n_halt+1) - p. The old compression gave
    (n-n_halt-1-p) -- 11 more rows, one of which was the 200-bp seam return."""
    n, p = 7000, 10
    ms, me, n_halt, _df = _masked_mids(n=n)
    out = pds.estimate_day(ms, me, n_lags=p)
    expect = (n - 1 - p) - (n_halt + 1) - p
    spliced = (n - n_halt - 1) - p
    right = out["n_obs"] == expect
    finite = all(np.isfinite(out[k]) for k in ("alpha_spy", "alpha_es", "CS_ES", "IS_mid_ES"))
    print("(1a) estimate_day on a masked day: n_obs=%d, expected %d (spliced code would say %d) "
          "(%s); shares finite (%s) : %s" % (out["n_obs"], expect, spliced, right, finite,
                                             right and finite))
    return right and finite


def check_panel_vecm_row_accounting():
    """One masked MWCB day + one clean day, pooled: the stacked design must count
    [(n-1-p) - (n_halt+1) - p] + [(n-1-p)] rows exactly."""
    n, p = 6000, 10
    ms, me, n_halt, _ = _masked_mids(n=n)
    dfc, _ = _pair_session(n=n, seed=11, day="2024-07-24", gap_bps=0.0)
    dfc.attrs["halt_windows_SPY"] = []; dfc.attrs["halt_windows_ES"] = []
    mc_s, mc_e = _mid(dfc, "SPY"), _mid(dfc, "ES")
    res = pds.panel_vecm([("2020-03-09", "volatile", ms, me),
                          ("2024-07-24", "benchmark", mc_s, mc_e)], n_lags=p)
    expect = ((n - 1 - p) - (n_halt + 1) - p) + (n - 1 - p)
    right = res["n_obs"] == expect
    finite = (np.isfinite(res["alpha_benchmark"]).all() and np.isfinite(res["alpha_volatile"]).all()
              and res["n_clusters"] == 2)
    print("(1b) panel_vecm pools %d rows, expected %d (%s); alphas finite over 2 clusters (%s) : %s"
          % (res["n_obs"], expect, right, finite, right and finite))
    return right and finite


def check_liquidity_vecm_row_accounting():
    n, p = 6000, 8
    ms, me, n_halt, df = _masked_mids(n=n)
    rng = np.random.default_rng(7)
    state = rng.normal(0, 1, n)
    res = ca.liquidity_conditional_vecm(ms, me, state, n_lags=p)
    expect = (n - 1 - p) - (n_halt + 1) - p
    right = res["n_obs"] == expect
    shares = res["shares_by_state"]
    finite = (np.isfinite(res["alpha"]).all() and np.isfinite(res["delta"]).all()
              and all(np.isfinite(v["CS_ES"]) for v in shares.values()))
    print("(1c) liquidity_conditional_vecm: n_obs=%d, expected %d (%s); alpha/delta and state-"
          "conditional shares finite (%s) : %s" % (res["n_obs"], expect, right, finite,
                                                   right and finite))
    return right and finite


def check_select_lag_sees_no_seam():
    """The candidate fits inside select_lag ran on COMPRESSED prices before v0.9.53, so every IC
    row carried the reopen gap as one return. The innovation logdet on the masked day must match a
    no-halt control built from the same DGP -- a spliced 200-bp return in ~7000 obs inflates the
    innovation variance by well over an order of magnitude."""
    n = 7000
    ms, me, _, _ = _masked_mids(n=n)
    dfc, _ = _pair_session(n=n, seed=3, gap_bps=0.0)     # same seed, no gap, never halted
    dfc.attrs["halt_windows_SPY"] = []; dfc.attrs["halt_windows_ES"] = []
    p_m, tab_m = pds.select_lag(np.log(ms), np.log(me), pmax=5)
    p_c, tab_c = pds.select_lag(np.log(_mid(dfc, "SPY")), np.log(_mid(dfc, "ES")), pmax=5)
    gap = float(abs(tab_m.loc[0, "logdet"] - tab_c.loc[0, "logdet"]))
    close = gap < 0.5
    print("(2) select_lag innovation logdet, masked vs no-halt control: |gap| = %.3f (< 0.5: %s); "
          "chosen p %d vs %d" % (gap, close, p_m, p_c))
    return close


def check_ec_valid_flags_non_correcting_days():
    """A correcting pair (SPY adjusts toward ES) is ec_valid; a mildly EXPLOSIVE pair -- the basis
    feeds back positively, both legs drifting away from equilibrium, as on 2023-08-07 /
    2023-12-20 -- has kappa < 0 and is flagged. estimate_sample's frame carries the flag so the
    driver can average CS over days where it is a share, not a quotient of noise."""
    rng = np.random.default_rng(5)
    n = 4000
    # correcting: z = p1 - p2 mean-reverts through the SPY leg (alpha_spy < 0, alpha_es ~ 0)
    p2 = np.cumsum(rng.normal(0, 3e-5, n)) + np.log(2800.0)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + rng.normal(0, 1e-5)
    p1 = p2 - np.log(10.0) + u
    good = pds.estimate_day(np.exp(p1), np.exp(p2), n_lags=5)
    # non-correcting: dp1 loads POSITIVELY on the basis, dp2 not at all -> kappa < 0, z explosive
    q1 = np.zeros(n); q2 = np.zeros(n)
    q1[0], q2[0] = np.log(280.0), np.log(2800.0)
    e1 = rng.normal(0, 5e-5, n); e2 = rng.normal(0, 5e-5, n)
    zoff = q1[0] - q2[0]
    for t in range(1, n):
        q1[t] = q1[t - 1] + 0.002 * ((q1[t - 1] - q2[t - 1]) - zoff) + e1[t]
        q2[t] = q2[t - 1] + e2[t]
    bad = pds.estimate_day(np.exp(q1), np.exp(q2), n_lags=5)
    g_ok = good["ec_valid"] and good["kappa"] > 0
    b_ok = (not bad["ec_valid"]) and bad["kappa"] <= 0
    defn = (good["ec_valid"] == (good["kappa"] > 0)) and (bad["ec_valid"] == (bad["kappa"] > 0))
    per_day = pds.estimate_sample([("good", "benchmark", np.exp(p1), np.exp(p2)),
                                   ("bad", "benchmark", np.exp(q1), np.exp(q2))], n_lags=5)
    col = "ec_valid" in per_day.columns and bool(per_day.loc["good", "ec_valid"]) \
        and not bool(per_day.loc["bad", "ec_valid"])
    ok = g_ok and b_ok and defn and col
    print("(3) ec_valid: correcting day kappa=%+.4f flagged valid (%s); non-correcting day "
          "kappa=%+.4f flagged invalid (%s); flag == sign(kappa) (%s); column in per-day frame "
          "(%s) : %s" % (good["kappa"], g_ok, bad["kappa"], b_ok, defn, col, ok))
    return ok


def check_drop_one_exposes_seam_leverage():
    """The 2020-days cross-impact reversal (ES<-SPY ~0.3, t up to 7.7) is the motivating case: one
    halt-reopen co-movement can carry a coefficient on its own. Clean data: flag quiet. One
    injected seam observation: a large lambda(ES<-SPY) appears from a true zero, the flag fires,
    and the drop-one refit restores ~0."""
    rng = np.random.default_rng(2)
    T = 5000
    O = rng.normal(0, 1, (T, 2))
    R = np.column_stack([0.5 * O[:, 0] + rng.normal(0, 0.1, T),      # SPY <- own flow only
                         0.4 * O[:, 1] + rng.normal(0, 0.1, T)])     # ES  <- own flow only
    clean = cimp.impact_regression(R, O)
    Od, Rd = O.copy(), R.copy()
    Od[1000, 0] = 60.0; Rd[1000, 1] = 25.0                           # the seam: SPY flow x ES return
    dirty = cimp.impact_regression(Rd, Od)
    lam = float(dirty["Lambda"][1, 0]); lam1 = float(dirty["Lambda_drop1"][1, 0])
    manufactured = abs(lam) > 0.1
    restored = abs(lam1) < 0.05
    flags = (not clean["drop1_flag"]) and dirty["drop1_flag"]
    ok = manufactured and restored and flags
    print("(4) drop-one leverage: injected seam manufactures lambda(ES<-SPY)=%.3f from a true zero "
          "(%s); drop-one refit restores %.3f (%s); flag clean=%s dirty=%s : %s"
          % (lam, manufactured, lam1, restored, clean["drop1_flag"], dirty["drop1_flag"], ok))
    return ok


def _book_sessions(n=1500, n_sessions=3, seed=13):
    """Small level-1 frames rich enough for the driver stages (mids, quantities, OFI)."""
    rng = np.random.default_rng(seed)
    out = []
    days = ["2020-03-09", "2024-07-24", "2025-04-04"]
    regimes = ["volatile", "benchmark", "benchmark"]
    for k in range(n_sessions):
        idx = pd.date_range(f"{days[k]} 09:30:00", periods=n, freq="s", tz=TZ)
        eff = np.cumsum(rng.normal(0, 3e-5, n))
        spy = 280.0 * np.exp(eff + rng.normal(0, 5e-6, n))
        es = 2800.0 * np.exp(eff + rng.normal(0, 5e-6, n))
        df = pd.DataFrame({"SPY_bidprice_1": spy - 0.005, "SPY_askprice_1": spy + 0.005,
                           "SPY_bidquantity_1": rng.integers(50, 150, n).astype(float),
                           "SPY_askquantity_1": rng.integers(50, 150, n).astype(float),
                           "ES_bidprice_1": es - 0.125, "ES_askprice_1": es + 0.125,
                           "ES_bidquantity_1": rng.integers(20, 80, n).astype(float),
                           "ES_askquantity_1": rng.integers(20, 80, n).astype(float)}, index=idx)
        df.attrs["halt_windows_SPY"] = []; df.attrs["halt_windows_ES"] = []
        out.append((days[k], regimes[k], df))
    return out


def check_run_dcc_pools_all_sessions():
    import run_analysis as ra
    sessions = _book_sessions()
    out = ra.run_dcc(sessions, SimpleNamespace())
    pooled = out.get("n_sessions") == 3
    realized = out.get("realized_corr", float("nan"))
    real_ok = np.isfinite(realized) and realized > 0.8       # common-factor DGP: near-1 realized
    div = abs(out["mean_rho"] - realized)
    warn_consistent = ("warning" in out) == (div > 0.2)
    ok = pooled and real_ok and warn_consistent
    print("(5) run_dcc: pools %s sessions (%s); realized_corr=%.3f of the SAME stacked returns "
          "(%s); mean_rho=%.3f, divergence %.3f, warning present==divergent (%s) : %s"
          % (out.get("n_sessions"), pooled, realized, real_ok, out["mean_rho"], div,
             warn_consistent, ok))
    return ok


def check_run_irf_medians_and_labels():
    import run_analysis as ra
    sessions = _book_sessions()
    args = SimpleNamespace(quick=True, n_lags=5, n_levels=1,
                           _freq={"dt": 1.0, "horizons": range(0, 11), "block": 10,
                                  "min_rest_steps": 0})
    out = ra.run_irf(sessions, args)
    has_regimes = set(out["fevd_by_regime"]) == {"volatile", "benchmark"}
    labelled = out["fevd_session0_date"] == "2020-03-09"
    headline = out["fevd"]
    rows_sum = np.allclose(headline.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    is_median = headline.equals(out["fevd_by_regime"]["benchmark"])
    ok = has_regimes and labelled and rows_sum and is_median
    print("(6) run_irf: per-regime median FEVD present for %s (%s); single-session matrix labelled "
          "'%s' (%s); headline is the benchmark median with rows summing to 1 (%s/%s) : %s"
          % (sorted(out["fevd_by_regime"]), has_regimes, out["fevd_session0_date"], labelled,
             is_median, rows_sum, ok))
    return ok


def main():
    checks = [check_estimate_day_row_accounting,
              check_panel_vecm_row_accounting,
              check_liquidity_vecm_row_accounting,
              check_select_lag_sees_no_seam,
              check_ec_valid_flags_non_correcting_days,
              check_drop_one_exposes_seam_leverage,
              check_run_dcc_pools_all_sessions,
              check_run_irf_medians_and_labels]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("analysis-inconsistency checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
