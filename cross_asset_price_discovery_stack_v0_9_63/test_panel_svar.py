#!/usr/bin/env python3
"""test_panel_svar.py -- the Table 9 redesign: panel-correct estimation and a window-free LHS.

What the pre-v0.9.63 Table 9 actually estimated was neither a panel SVAR nor a fixed-effects
SVAR: sessions were np.vstack'ed into one long series, so the first p bars of every day borrowed
the PREVIOUS day's close as their "lags" (an overnight seam treated as one grid step), and all
days shared a single intercept -- conflating within-day dynamics with between-day level shifts
(a 2020 MWCB day and a 2023 baseline day differ in RV/spread LEVELS by orders of magnitude).
And the dependent variable, d(rolling correlation), carried an MA term at exactly the window
length by construction, which is why the selected lag order marched toward corr_window.

v0.9.63 rebuilds both halves; pinned here:

  (1) SEAM ROW ACCOUNTING: the panel design has exactly sum(T_i - p) rows -- the p seam rows per
      later session the stacked design manufactured are gone -- and no design row of one session
      carries another session's values
  (2) DAY FE KILLS BETWEEN-DAY BIAS: on a DGP with zero within-day cross-dynamics but strongly
      co-moving day-level means, the stacked estimation manufactures a cross-coefficient and the
      FE panel does not
  (3) MEAN-GROUP: identical days -> near-zero cross-day dispersion; sign-flipped days -> the
      dispersion IS the heterogeneity, and n_days books correctly
  (4) THE HEADLINE: on constant-correlation null data the rolling LHS still drives BIC to the
      window (the artifact, kept as the contrast) while corr_method='bar' -- non-overlapping
      per-bar realized correlation, Fisher-z -- selects a SMALL order on the same data
  (5) Fisher-z is finite at |rho|=1 and monotone
  (6) panel='stack' reproduces the pre-v0.9.63 estimation exactly (vstack + var_ols), so the
      old numbers remain computable for comparison
  (7) the CLI renders the RealBar column and the MEAN-GROUP block

Run: python test_panel_svar.py
"""
from __future__ import annotations

import os
import subprocess as sp
import sys

import numpy as np
import pandas as pd

import correlation_svar as cs
from test_svar_lag_artifact import _no_dynamics_frame

TZ = "America/New_York"


def check_seam_rows_and_purity():
    rng = np.random.default_rng(0)
    X1 = rng.normal(0, 1, (200, 2))                  # session 1: values ~ N(0,1)
    X2 = rng.normal(100, 1, (150, 2))                # session 2: values ~ N(100,1)
    p = 3
    Y, Z, g = cs._panel_var_design([X1, X2], p, fe=False)
    n_panel = Y.shape[0]
    n_expect = (200 - p) + (150 - p)
    n_stacked = 200 + 150 - p                        # what vstack-then-lag produced: p more rows
    counts = n_panel == n_expect and n_stacked - n_panel == p
    z2 = Z[g == 1][:, 1:]                            # session 2's lag columns (skip intercept)
    purity = float(z2.min()) > 50.0                  # no session-1 value leaked into its lags
    ok = counts and purity
    print("(1) panel design rows = %d (expected %d; the stacked design had %d, i.e. %d seam "
          "rows) (%s); session-2 lags carry only session-2 values (min %.1f) (%s) : %s"
          % (n_panel, n_expect, n_stacked, p, counts, z2.min(), purity, ok))
    return ok


def check_fe_kills_between_day_bias():
    """Zero true cross-dynamics within day; day-level means of the two variables co-move
    strongly across days. One intercept over the stack lets that between-day covariation load
    onto the lag coefficients; day FE removes it."""
    rng = np.random.default_rng(1)
    n_days, T = 20, 400
    Xs = []
    for _ in range(n_days):
        m = rng.multivariate_normal([0, 0], [[1, 0.95], [0.95, 1]]) * 5.0   # co-moving day means
        e = rng.normal(0, 1, (T, 2))                                        # independent within-day
        Xs.append(m + e)
    p = 2
    # stacked (pre-v0.9.63): vstack + single-intercept VAR
    A_st, _, _ = cs.var_ols(np.vstack(Xs), p)
    # FE panel
    A_fe, _, _, _ = cs.panel_var_ols(Xs, p, fe=True)
    cross_st = abs(A_st[0][0, 1])
    cross_fe = abs(A_fe[0][0, 1])
    ok = cross_fe < 0.02 and cross_st > 5 * cross_fe
    print("(2) spurious cross-lag coefficient: stacked=%.4f vs FE panel=%.4f (FE ~0 and stacked "
          ">5x larger) : %s" % (cross_st, cross_fe, ok))
    return ok


def _book_frame(n=3000, seed=5, flip_ofi=False, day="2024-07-24"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=TZ)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n) * 1e-4
    mid = {"SPY": 550 * np.exp(np.cumsum(z[:, 0])), "ES": 5500 * np.exp(np.cumsum(z[:, 1]))}
    df = pd.DataFrame(index=idx)
    for a in ("SPY", "ES"):
        half = 0.01 if a == "SPY" else 0.25
        for i in range(1, 4):
            bq = rng.integers(50, 500, n).astype(float)
            aq = rng.integers(50, 500, n).astype(float)
            if flip_ofi:
                bq, aq = aq, bq
            df[f"{a}_bidprice_{i}"] = mid[a] - half * i
            df[f"{a}_askprice_{i}"] = mid[a] + half * i
            df[f"{a}_bidquantity_{i}"] = bq
            df[f"{a}_askquantity_{i}"] = aq
    return df


def check_mean_group_dispersion():
    same = [("d1", "volatile", _book_frame(seed=5)), ("d2", "volatile", _book_frame(seed=5)),
            ("d3", "benchmark", _book_frame(seed=6)), ("d4", "benchmark", _book_frame(seed=6, flip_ofi=True))]
    mg = cs.correlation_irf_mean_group(same, spec="informational", n_levels=3, n_lags=2,
                                       corr_method="bar", bar_seconds=30)
    ok_shape = (not mg["point"].empty and set(mg["point"].columns) >= {"volatile", "benchmark"}
                and int(mg["n_days"].iloc[0]["volatile"]) == 2)
    se = mg["se"]
    ident_tight = float(np.nanmax(se["volatile"].to_numpy(float))) < 1e-9   # identical days
    het_wide = float(np.nanmax(se["benchmark"].to_numpy(float))) > 1e-6    # flipped-book days
    ok = ok_shape and ident_tight and het_wide
    print("(3) mean-group: n_days books (%s); identical days -> zero dispersion (max se %.2g, %s); "
          "flipped-book days -> dispersion is the heterogeneity (max se %.2g, %s) : %s"
          % (ok_shape, np.nanmax(se["volatile"].to_numpy(float)), ident_tight,
             np.nanmax(se["benchmark"].to_numpy(float)), het_wide, ok))
    return ok


def check_bar_breaks_the_window_artifact():
    df = _no_dynamics_frame(n=12000)
    W = 25
    p_roll, _ = cs.select_svar_lag(df, spec="standard", n_levels=3, corr_method="rolling",
                                   corr_window=W, criterion="bic", pmax=50)
    p_bar, _ = cs.select_svar_lag(df, spec="standard", n_levels=3, corr_method="bar",
                                  bar_seconds=20, criterion="bic", pmax=15)
    artifact = p_roll is not None and abs(int(p_roll) - W) <= 1
    freed = p_bar is not None and int(p_bar) <= 5 and int(p_bar) < W // 2
    ok = artifact and freed
    print("(4) constant-correlation null: rolling LHS drives BIC to the window (p*=%s vs W=%d, "
          "the artifact) while the bar LHS selects p*=%s on the SAME data : %s"
          % (p_roll, W, p_bar, ok))
    return ok


def check_fisher_z():
    z1 = cs._fisher_z(1.0); zm = cs._fisher_z(-1.0)
    vals = cs._fisher_z(np.array([-0.5, 0.0, 0.5, 0.92]))
    ok = (np.isfinite(z1) and np.isfinite(zm) and z1 > 9 and zm < -9
          and np.all(np.diff(vals) > 0) and abs(vals[1]) < 1e-12)
    print("(5) Fisher-z: finite at |rho|=1 (%.1f), monotone, zero at zero : %s" % (z1, ok))
    return ok


def check_stack_reproduces_old_estimation():
    sessions = [("d1", "volatile", _book_frame(seed=11)), ("d2", "volatile", _book_frame(seed=12))]
    p = 3
    got = cs.correlation_irf(sessions, spec="standard", n_levels=3, n_lags=p,
                             corr_method="rolling", corr_window=40, min_obs=100, panel="stack")
    # the pre-v0.9.63 computation, reproduced by hand: vstack + single-intercept VAR + IRF
    Xs = []
    for _d, _r, df in sessions:
        X, names, ci = cs.build_svar_frame(df, "standard", 3, None, "rolling", 40, "cost_to_fill")
        Xs.append(X)
    A, Sigma, _ = cs.var_ols(np.vstack(Xs), p)
    Theta = cs.orthogonalized_irf(A, Sigma, 10, "cholesky")
    expect = Theta[0][ci, :] * 100.0
    exp_rows = np.delete(expect, ci)
    same = np.allclose(got["volatile"].to_numpy(float), exp_rows, rtol=0, atol=1e-12)
    fe = cs.correlation_irf(sessions, spec="standard", n_levels=3, n_lags=p,
                            corr_method="rolling", corr_window=40, min_obs=100, panel="fe")
    differs = not np.allclose(got["volatile"].to_numpy(float),
                              fe["volatile"].to_numpy(float), atol=1e-12)
    ok = same and differs
    print("(6) panel='stack' equals the hand-computed pre-v0.9.63 estimation exactly (%s), and "
          "'fe' is a genuinely different estimator (%s) : %s" % (same, differs, ok))
    return ok


def check_cli_renders_bar_and_mean_group():
    here = os.path.dirname(os.path.abspath(__file__))
    r = sp.run([sys.executable, "run_table9_both_ways.py", "--source", "demo", "--n-demo", "4",
                "--n-demo-bars", "1200", "--n-boot", "0", "--corr-window", "60",
                "--bar-seconds", "30"],
               capture_output=True, text=True, timeout=540, cwd=here)
    out = r.stdout
    ok = (r.returncode == 0 and "RealBar" in out and "MEAN-GROUP" in out
          and "fixed-effects panel VAR" in out and "n_days:" in out)
    print("(7) CLI: RealBar column, panel note, and MEAN-GROUP block all render (rc=%d) : %s"
          % (r.returncode, ok))
    return ok


def main():
    checks = [check_seam_rows_and_purity,
              check_fe_kills_between_day_bias,
              check_mean_group_dispersion,
              check_bar_breaks_the_window_artifact,
              check_fisher_z,
              check_stack_reproduces_old_estimation,
              check_cli_renders_bar_and_mean_group]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("panel-SVAR checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
