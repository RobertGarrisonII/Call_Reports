#!/usr/bin/env python3
"""Gate: the panel local-projection Table 9 estimates what it claims, on DGPs where the
answer is known by construction.

The module's selling point is LAG-ORDER ROBUSTNESS: a VAR IRF is the fitted model iterated,
so a truncated lag order biases every step after impact, while an LP reads each horizon off
a direct projection and only needs the shock to be clean. That is a property to DEMONSTRATE
on a planted VAR(3), not to assert -- so this gate fits both estimators at a deliberately
wrong p=1 and pins the asymmetry (check 2). The rest pins the machinery the demonstration
rests on: recovery of a known Cholesky IRF within bootstrap CIs (check 1), the within-day /
halt-aware window discipline by exact row arithmetic (check 3), the day-cluster bootstrap
actually clustering (check 4), and the CLI writing the exhibit (check 5).
"""
import contextlib
import io
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

import correlation_svar as cs
import lp_table9 as lp

K = 3
CIDX = 2
NAMES = ["v0", "v1", "dCorr"]
# Planted VAR(3). Variable 0 is AR(1) in itself ONLY, so its recursively-identified shock
# is clean even when the fitted lag order is truncated to p=1 -- which makes shock v0 the
# cell where LP-vs-VAR asymmetry under truncation is a theorem, not luck. The dependent
# (index 2) loads on lag-3 of v0/v1, so truncation has something material to lose.
A1 = np.array([[0.20, 0.00, 0.00],
               [0.30, 0.20, 0.00],
               [0.00, 0.00, 0.10]])
A2 = np.zeros((3, 3))
A3 = np.array([[0.00, 0.00, 0.00],
               [0.00, 0.00, 0.00],
               [0.70, -0.35, 0.00]])
L0 = np.array([[1.00, 0.00, 0.00],
               [0.50, 0.80, 0.00],
               [0.30, -0.40, 0.90]])


def true_irf_x100(H):
    """Cholesky IRF of the planted system, x100: Theta_h = Psi_h @ L0 with unit-variance
    structural shocks -- the exact object panel_lp_irf and the SVAR both estimate."""
    Psi = cs._ma_from_var([A1, A2, A3], H)
    return np.array([100.0 * (Psi[h] @ L0)[CIDX, :] for h in range(H + 1)])   # (H+1, k)


def simulate_panel(n_days, T, seed, burn=100, day_shift_sd=1.0):
    """Independent days from the planted VAR(3), u_t = L0 eps_t, eps ~ N(0, I). Each day
    carries its own level shift on every variable (a day fixed effect) so the within-day
    demeaning is doing real work, not vacuously passing."""
    rng = np.random.default_rng(seed)
    Xs = []
    for _d in range(n_days):
        e = rng.standard_normal((T + burn, K)) @ L0.T
        Y = np.zeros((T + burn, K))
        for t in range(3, T + burn):
            Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + A3 @ Y[t - 3] + e[t]
        Xs.append(Y[burn:] + rng.normal(0, day_shift_sd, K)[None, :])
    return Xs


def check_recovers_true_irf():
    """WHY: the LP is only a usable Table 9 counterpart if, at an adequate control lag
    order, it recovers a KNOWN recursive IRF -- point estimates near truth and the
    day-cluster bootstrap CI covering it on >= 80% of (shock, horizon) cells."""
    H = 5
    Xs = simulate_panel(n_days=16, T=240, seed=7)
    tab = lp.panel_lp_irf(Xs, NAMES, CIDX, n_lags=3, horizon=H, n_boot=199, seed=1)
    truth = true_irf_x100(H)
    n_cover = n_cells = 0
    for j, nm in ((0, "v0"), (1, "v1")):
        for h in range(H + 1):
            r = tab[(tab["shock"] == nm) & (tab["horizon"] == h)].iloc[0]
            n_cells += 1
            n_cover += int(r["ci_lo_x100"] <= truth[h, j] <= r["ci_hi_x100"])
    t0 = tab[(tab["shock"] == "v0")].sort_values("horizon")
    print("    shock v0: truth  " + " ".join("%+7.2f" % v for v in truth[:, 0]))
    print("              LP     " + " ".join("%+7.2f" % v for v in t0["theta_x100"]))
    ok = n_cover >= int(np.ceil(0.8 * n_cells))
    print("(1) planted VAR(3), 16 days x 240 bars: truth inside the 95%% day-cluster CI on "
          "%d/%d (shock, horizon) cells (need >= %d) : %s"
          % (n_cover, n_cells, int(np.ceil(0.8 * n_cells)), ok))
    return bool(ok)


def check_lag_truncation_robustness():
    """WHY: the selling point, pinned. Fit BOTH estimators at a deliberately truncated p=1
    on the VAR(3) DGP. The iterated VAR IRF must be biased where the truth lives at lag 3
    (h >= 2 for shock v0: |SVAR - truth| large, outside the LP's own CI), while the LP at
    the same wrong p stays on the truth (truth inside its CI, error a fraction of the
    SVAR's). If this ever fails, the module's reason to exist has regressed."""
    H = 5
    Xs = simulate_panel(n_days=16, T=240, seed=11)
    tab = lp.panel_lp_irf(Xs, NAMES, CIDX, n_lags=1, horizon=H, n_boot=199, seed=2)
    A, Sigma, _, _ = cs.panel_var_ols(Xs, 1, fe=True)
    Theta = cs.orthogonalized_irf(A, Sigma, H, "cholesky")
    svar = np.array([100.0 * Theta[h][CIDX, :] for h in range(H + 1)])
    truth = true_irf_x100(H)
    sub = tab[tab["shock"] == "v0"].set_index("horizon")
    lp_err = np.array([abs(sub.loc[h, "theta_x100"] - truth[h, 0]) for h in range(H + 1)])
    sv_err = np.abs(svar[:, 0] - truth[:, 0])
    print("    shock v0, p=1 for BOTH:  h     truth      LP  SVAR(1)")
    for h in range(H + 1):
        print("                             %d  %+8.2f %+8.2f %+8.2f"
              % (h, truth[h, 0], sub.loc[h, "theta_x100"], svar[h, 0]))
    # (a) LP still covers the truth at the horizons where truncation bites (h=3, 4)
    cover = all(sub.loc[h, "ci_lo_x100"] <= truth[h, 0] <= sub.loc[h, "ci_hi_x100"]
                for h in (3, 4))
    # (b) at h=3 (truth ~ +52 by construction) the truncated SVAR is far off -- error > 20
    #     x100 and OUTSIDE the LP's CI -- while the LP error stays under 10
    far = sv_err[3] > 20.0 and lp_err[3] < 10.0
    outside = not (sub.loc[3, "ci_lo_x100"] <= svar[3, 0] <= sub.loc[3, "ci_hi_x100"])
    # (c) aggregate over h >= 2: the truncated VAR's mean error exceeds the LP's
    agg = float(np.mean(sv_err[2:])) > float(np.mean(lp_err[2:]))
    ok = cover and far and outside and agg
    print("(2) truncated p=1 on a VAR(3): LP covers truth at h=3,4 (%s); SVAR(1) error at "
          "h=3 = %.1f > 20 with LP error %.1f < 10 (%s); SVAR(1) outside the LP CI (%s); "
          "mean|err| h>=2 SVAR %.1f > LP %.1f (%s) : %s"
          % (cover, sv_err[3], lp_err[3], far, outside,
             float(np.mean(sv_err[2:])), float(np.mean(lp_err[2:])), agg, ok))
    return bool(ok)


def check_day_boundary_discipline():
    """WHY: 'lags and leads strictly within day' and 'a window touching a halt-masked bar
    is dropped, not zero-filled' are exact combinatorial claims, so pin them by exact row
    arithmetic. Clean panel: n_obs(h) = n_days * (T - p - h) -- a seam-crossing stacked
    builder would report n_days*T - p - h instead. One NaN row planted mid-day must delete
    exactly the p + h + 1 windows [t-p, t+h] that contain it."""
    T, p, H = 60, 2, 4
    Xs = simulate_panel(n_days=2, T=T, seed=3)
    tab = lp.panel_lp_irf(Xs, NAMES, CIDX, n_lags=p, horizon=H, n_boot=0)
    ok_clean = all(int(tab[(tab["shock"] == "v0") & (tab["horizon"] == h)]["n_obs"].iloc[0])
                   == 2 * (T - p - h) for h in range(H + 1))
    Xh = [x.copy() for x in Xs]
    Xh[0][30, :] = np.nan                                    # a halt-masked bar, mid-day
    tabh = lp.panel_lp_irf(Xh, NAMES, CIDX, n_lags=p, horizon=H, n_boot=0)
    ok_halt = all(int(tabh[(tabh["shock"] == "v0") & (tabh["horizon"] == h)]["n_obs"].iloc[0])
                  == 2 * (T - p - h) - (p + h + 1) for h in range(H + 1))
    print("(3) row arithmetic, 2 days x T=%d, p=%d: clean n_obs(h) == 2*(T-p-h) for h=0..%d "
          "(%s); one planted NaN bar deletes exactly p+h+1 windows (%s) : %s"
          % (T, p, H, ok_clean, ok_halt, ok_clean and ok_halt))
    return bool(ok_clean and ok_halt)


def check_cluster_bootstrap_matters():
    """WHY: days are the independent unit for inference in this stack. On a DGP whose shock
    transmission varies BY DAY (random day-level slope), the iid OLS SE is a fiction --
    the day-cluster bootstrap SE must come out materially larger, or the 'day-cluster' in
    the module's inference claim has regressed to row resampling."""
    rng = np.random.default_rng(19)
    n_days, T = 24, 120
    Xs = []
    for _d in range(n_days):
        beta_d = 1.0 + 0.6 * rng.standard_normal()
        x = rng.standard_normal(T)
        y = beta_d * x + 0.5 * rng.standard_normal(T)
        Xs.append(np.column_stack([x, y]))
    tab = lp.panel_lp_irf(Xs, ["x", "y"], 1, n_lags=1, horizon=2, n_boot=299, seed=4)
    r0 = tab[(tab["shock"] == "x") & (tab["horizon"] == 0)].iloc[0]
    ratio = float(r0["se_x100"] / r0["se_ols_x100"])
    ok = np.isfinite(ratio) and ratio > 2.0
    print("(4) day-level slope heterogeneity: day-cluster bootstrap SE %.2f vs naive iid "
          "OLS SE %.2f (ratio %.1f > 2) : %s"
          % (r0["se_x100"], r0["se_ols_x100"], ratio, ok))
    return bool(ok)


def check_runner():
    """WHY: the exhibit is only real if the CLI runs the whole path -- demo frames, RealBar
    LP frame with NaN retention, estimation, SVAR comparison column, CSV + md written."""
    with tempfile.TemporaryDirectory() as td:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = lp.main(["--source", "demo", "--n-demo", "4", "--n-demo-bars", "1800",
                          "--spec", "standard", "--bar-seconds", "60", "--n-lags", "1",
                          "--horizon", "3", "--n-boot", "59", "--out-dir", td])
        csv_p = os.path.join(td, "table9_local_projection.csv")
        md_p = os.path.join(td, "table9_local_projection.md")
        ok_files = os.path.exists(csv_p) and os.path.exists(md_p)
        ok_cols = ok_rows = False
        if ok_files:
            t = pd.read_csv(csv_p)
            need = {"regime", "shock", "horizon", "theta_x100", "se_x100", "ci_lo_x100",
                    "ci_hi_x100", "se_ols_x100", "n_obs", "n_days", "svar_x100",
                    "lp_minus_svar_x100"}
            ok_cols = need <= set(t.columns)
            ok_rows = len(t) > 0 and set(t["regime"]) == {"benchmark", "volatile"} \
                and t["theta_x100"].notna().any()
        out = buf.getvalue()
        ok_print = "local projections" in out and "SVAR IRF at the same horizons" in out
    print("(5) runner: rc=%s, csv+md written (%s), tidy columns complete (%s), both regimes "
          "estimated (%s), tables printed (%s) : %s"
          % (rc, ok_files, ok_cols, ok_rows, ok_print,
             rc == 0 and ok_files and ok_cols and ok_rows and ok_print))
    return rc == 0 and ok_files and ok_cols and ok_rows and ok_print


def main():
    checks = [check_recovers_true_irf, check_lag_truncation_robustness,
              check_day_boundary_discipline, check_cluster_bootstrap_matters, check_runner]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("lp-table9 checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    sys.exit(main())
