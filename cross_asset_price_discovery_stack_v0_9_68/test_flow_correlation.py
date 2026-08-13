#!/usr/bin/env python3
"""test_flow_correlation.py -- the v0.9.68 flow-correlation exhibits behave as specified.

  (1) ar_innovations strips own-persistence and RECOVERS the innovation cross-correlation
      when the two legs carry different AR coefficients (raw correlation is biased there)
  (2) _bar_corr recovers a designed per-bar correlation and NaNs under-covered bars
  (3) ofi_innovations: designed quantity paths reproduce an exact injected OFI series, and
      halt-masked (NaN-price) rows propagate to NaN innovations (no fake zeros)
  (4) flow_corr_bars recovers an injected innovation correlation at the bar level
  (5) mediation_from_bars finds the planted indirect channel (share > 0, bootstrap CI
      excludes 0) and does NOT invent one under the no-mediation null (CI covers 0)
  (6) fit_ms_ar1 / ms_lr_test: a planted 2-state series is detected (bootstrap p <= 0.05,
      regime means recovered in order); a 1-state AR(1) series is NOT rejected
  (7) semicorrelation asymmetry: symmetric innovations give a ~zero down-up gap and a
      large sign-flip p; a planted sign-dependent common factor is detected
  (8) regime-dynamics asymmetry: planted sticky-high-regime + entered-on-selling days
      yield small sign-flip p for both the duration ratio and the entry direction
  (9) select_k_ms lets the data pick the regime COUNT: an AR(1) series selects K=1, a
      planted 3-state series selects K=3 with its means recovered (levels are free to be
      zero or negative -- nothing presupposes polarity)
 (10) the price-discovery link recovers planted window-panel slopes (z_flow and top-regime
      occupancy) with day-clustered inference, and builds end-to-end on synthetic sessions
 (11) the table builders run end-to-end on synthetic sessions, the Tier 1 regime
      contrast has the planted sign with a small permutation p, and the driver wires
      run_flow_correlation.py into STAGE 5b

Run: python test_flow_correlation.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NY = "America/New_York"


def _ar_series(T, phi, u):
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + u[t]
    return x


def check_ar_innovations():
    import flow_correlation as fc
    rng = np.random.default_rng(11)
    T = 40000
    rho = 0.7
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    U = rng.standard_normal((T, 2)) @ L.T
    xa = _ar_series(T, 0.8, U[:, 0])
    xb = _ar_series(T, 0.2, U[:, 1])
    raw = float(np.corrcoef(xa, xb)[0, 1])
    ua = fc.ar_innovations(xa, 5)
    ub = fc.ar_innovations(xb, 5)
    m = np.isfinite(ua) & np.isfinite(ub)
    inn = float(np.corrcoef(ua[m], ub[m])[0, 1])
    ok = abs(inn - rho) < 0.03 and (rho - raw) > 0.05 and abs(inn - rho) < abs(raw - rho)
    print(f"  raw corr {raw:.3f} (biased), innovation corr {inn:.3f} (true {rho})")
    return ok


def check_bar_corr():
    import flow_correlation as fc
    rng = np.random.default_rng(7)
    T, W = 12000, 60
    rho = 0.5
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    U = rng.standard_normal((T, 2)) @ L.T
    bar = np.repeat(np.arange(T // W), W)
    r = fc._bar_corr(U[:, 0], U[:, 1], bar, min_sub=10)
    ok1 = abs(float(np.nanmean(r)) - rho) < 0.05
    x = U[:, 0].copy()
    x[:W - 5] = np.nan                                  # first bar: 5 finite pairs < min_sub
    r2 = fc._bar_corr(x, U[:, 1], bar, min_sub=10)
    ok2 = not np.isfinite(r2.iloc[0]) and np.isfinite(r2.iloc[1])
    print(f"  mean bar corr {float(np.nanmean(r)):.3f} (true {rho}); undercovered bar -> NaN: {ok2}")
    return ok1 and ok2


def _designed_frame(fa, fb, n_levels=10, halt=None):
    """Constant-price books whose level-1 bid quantities carry cumsum(f): with prices
    constant, OFI reduces EXACTLY to the injected increment series."""
    T = len(fa)
    idx = pd.date_range("2024-08-05 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    base = 1e6
    for a, f, p0, tick in (("SPY", fa, 500.0, 0.01), ("ES", fb, 5000.0, 0.25)):
        for l in range(1, n_levels + 1):
            cols[f"{a}_bidprice_{l}"] = np.full(T, p0 - l * tick)
            cols[f"{a}_askprice_{l}"] = np.full(T, p0 + l * tick)
            q = np.full(T, base)
            if l == 1:
                q = base + np.cumsum(f)
            cols[f"{a}_bidquantity_{l}"] = q
            cols[f"{a}_askquantity_{l}"] = np.full(T, base)
    df = pd.DataFrame(cols, index=idx)
    if halt is not None:
        h0, h1 = halt
        for c in df.columns:
            if "price" in c:
                df.iloc[h0:h1, df.columns.get_loc(c)] = np.nan
    return df


def check_ofi_injection():
    import flow_correlation as fc
    import cross_asset_pd_liquidity as ca
    rng = np.random.default_rng(3)
    T = 3000
    fa = rng.standard_normal(T)
    fb = rng.standard_normal(T)
    df = _designed_frame(fa, fb)
    ofi = np.asarray(ca.order_flow_imbalance(df, "SPY", 10), float)
    ok1 = np.allclose(ofi[1:], fa[1:], atol=1e-6)
    dfh = _designed_frame(fa, fb, halt=(1000, 1200))
    u = fc.ofi_innovations(dfh, "SPY", 10, ar_order=5)
    ok2 = np.all(~np.isfinite(u[1000:1200])) and np.isfinite(u[900:990]).all()
    print(f"  injected OFI exact: {ok1}; halt rows NaN in innovations: {ok2}")
    return ok1 and ok2


def check_flow_corr_bars():
    import flow_correlation as fc
    rng = np.random.default_rng(19)
    T = 21600
    rho = 0.65
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    U = rng.standard_normal((T, 2)) @ L.T
    fa = _ar_series(T, 0.7, U[:, 0])
    fb = _ar_series(T, 0.3, U[:, 1])
    df = _designed_frame(fa, fb)
    b = fc.flow_corr_bars(df, bar_seconds=60, ar_order=5)
    z = b["z_flow"].to_numpy(float)
    zbar = float(np.nanmean(z))
    est = float(np.tanh(zbar))
    # per-bar corr from 60 obs is downward-noisy in z-mean; wide but directional tolerance
    ok = abs(est - rho) < 0.10 and np.isfinite(z).sum() > 300
    print(f"  bar-level flow corr {est:.3f} (true {rho}), {np.isfinite(z).sum()} finite bars")
    return ok


def _mediation_bars(b_med, n_days=12, T=300, seed=0):
    """Bars with a planted chain rv_es -(a)-> dz_flow -(b_med)-> dz_ret plus a direct
    rv_es -(c)-> dz_ret path. share = a*b/(c + a*b) after standardization-ish scaling."""
    rng = np.random.default_rng(seed)
    out = []
    a, c = 0.5, 0.2
    for d in range(n_days):
        rv = 1.0 + 0.5 * np.abs(rng.standard_normal(T))
        dz_f = np.zeros(T)
        dz_r = np.zeros(T)
        for t in range(1, T):
            dz_f[t] = a * (rv[t - 1] - 1.0) + 0.3 * rng.standard_normal()
            dz_r[t] = b_med * dz_f[t] + c * (rv[t - 1] - 1.0) + 0.3 * rng.standard_normal()
        idx = pd.date_range("2024-01-0%d 09:30" % (d % 7 + 1), periods=T, freq="60s", tz=NY)
        bars = pd.DataFrame({
            "z_flow": np.cumsum(dz_f), "z_ret": np.cumsum(dz_r),
            "rv_es": rv, "rv_spy": 1.0 + 0.5 * np.abs(rng.standard_normal(T)),
            "state": rng.standard_normal(T),
            "d_wspr_es": rng.standard_normal(T), "d_wspr_spy": rng.standard_normal(T),
            "n_pairs": np.full(T, 60)}, index=idx)
        out.append((f"2024-01-{d + 1:02d}", "volatile" if d % 2 else "benchmark", bars))
    return out


def check_mediation():
    import flow_correlation as fc
    res = fc.mediation_from_bars(_mediation_bars(0.6, seed=5), n_lags=2, n_boot=99, seed=1)
    lo, hi = res["ci_indirect"]
    ok1 = res["share"] > 0.3 and lo > 0
    res0 = fc.mediation_from_bars(_mediation_bars(0.0, seed=6), n_lags=2, n_boot=99, seed=2)
    lo0, hi0 = res0["ci_indirect"]
    ok2 = lo0 < 0 < hi0
    print(f"  planted: share {res['share']:.2f}, indirect CI [{lo:.4f},{hi:.4f}]; "
          f"null: CI [{lo0:.4f},{hi0:.4f}] covers 0: {ok2}")
    return ok1 and ok2


def check_ms_regimes():
    import flow_correlation as fc
    rng = np.random.default_rng(23)
    T, phi = 400, 0.5
    means = (0.30, 1.20)
    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    y = np.zeros(T)
    y[0] = means[0]
    s = 0
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        if rng.random() > P[s, s]:
            s = 1 - s
        states[t] = s
    for t in range(1, T):
        mu_t = means[states[t]] * (1 - phi)
        y[t] = mu_t + phi * y[t - 1] + 0.15 * rng.standard_normal()
    r = fc.ms_lr_test(y, B=49, seed=3)
    m_lo, m_hi = r["fit2"]["means"][0], r["fit2"]["means"][-1]
    ok1 = r["p_boot"] <= 0.05 and abs(m_lo - means[0]) < 0.2 and abs(m_hi - means[1]) < 0.2
    y0 = np.zeros(T)
    for t in range(1, T):
        y0[t] = 0.6 * (1 - phi) + phi * y0[t - 1] + 0.2 * rng.standard_normal()
    r0 = fc.ms_lr_test(y0, B=49, seed=4)
    ok2 = r0["p_boot"] > 0.05
    print(f"  planted 2-state: p={r['p_boot']:.3f}, means {m_lo:.2f}/{m_hi:.2f} "
          f"(true {means[0]}/{means[1]}); 1-state null: p={r0['p_boot']:.3f}")
    return ok1 and ok2


def check_semicorr_asymmetry():
    import flow_correlation as fc
    rng = np.random.default_rng(31)
    T = 60000
    # symmetric null: bivariate normal, rho = 0.5 -> gap ~ 0
    L = np.linalg.cholesky([[1.0, 0.5], [0.5, 1.0]])
    U = rng.standard_normal((T, 2)) @ L.T
    d_dn, d_up, *_ = fc.semicorrelations(U[:, 0], U[:, 1])
    ok1 = abs(d_dn - d_up) < 0.04
    # planted asymmetry: a common factor whose loading DOUBLES when it is negative --
    # joint-selling co-movement tighter than joint-buying by construction
    f = rng.standard_normal(T)
    lam = np.where(f < 0, 1.6, 0.6)
    x = lam * f + rng.standard_normal(T)
    y = lam * f + rng.standard_normal(T)
    a_dn, a_up, *_ = fc.semicorrelations(x, y)
    ok2 = (a_dn - a_up) > 0.10
    # day-level sign-flip: symmetric day gaps -> p large; shifted -> p small
    sym = rng.standard_normal(24) * 0.05
    shifted = sym + 0.12
    p_sym = fc.sign_flip_p(sym, n_flip=5000, seed=1)
    p_shift = fc.sign_flip_p(shifted, n_flip=5000, seed=1)
    ok3 = p_sym > 0.05 and p_shift < 0.01
    print(f"  null gap {d_dn - d_up:+.3f}; planted gap {a_dn - a_up:+.3f}; "
          f"sign-flip p sym {p_sym:.3f} / shifted {p_shift:.4f}")
    return ok1 and ok2 and ok3


def check_regime_dynamics_asymmetry():
    import flow_correlation as fc
    rng = np.random.default_rng(41)
    # planted: high-tandem regime is stickier (dur 33 vs 10 bars) and entered on selling
    days = []
    for d in range(6):
        T = 300
        states = np.zeros(T, dtype=int)
        ret = np.zeros(T)
        s = 0
        for t in range(1, T):
            p_stay = 0.97 if s == 1 else 0.90
            if rng.random() > p_stay:
                s = 1 - s
            states[t] = s
            # selling precedes entries: a negative return shock the bar BEFORE a switch up
            ret[t] = rng.standard_normal()
            if s == 1 and states[t - 1] == 0:
                ret[t - 1] -= 3.0
        z = np.where(states == 1, 1.1, 0.3) + 0.15 * rng.standard_normal(T)
        idx = pd.date_range("2024-02-0%d 09:30" % (d % 7 + 1), periods=T, freq="60s", tz=NY)
        b = pd.DataFrame({"z_flow": z, "z_ret": z, "ret_spy": ret,
                          "ret_es": ret, "rv_es": np.ones(T), "rv_spy": np.ones(T),
                          "state": rng.standard_normal(T),
                          "d_wspr_es": np.zeros(T), "d_wspr_spy": np.zeros(T),
                          "n_pairs": np.full(T, 60)}, index=idx)
        days.append((f"2024-02-{d + 1:02d}", "volatile" if d % 2 else "benchmark", b))
    df, notes = fc.table_flow_corr_ms_regimes([], B=39, k_max=2, bars=days)
    row = df.loc["all days"]
    ok = (float(row["duration asymmetry p"]) < 0.05
          and float(row["entry ret (bps, demeaned)"]) < 0
          and float(row["entry-direction p"]) < 0.05)
    print(f"  duration asym p {row['duration asymmetry p']:.4f}; entry ret "
          f"{row['entry ret (bps, demeaned)']:.2f} (p {row['entry-direction p']:.4f})")
    return ok


def check_select_k():
    import flow_correlation as fc
    rng = np.random.default_rng(41)
    T, phi = 300, 0.4
    y1 = np.zeros(T)
    for t in range(1, T):
        y1[t] = 0.5 * (1 - phi) + phi * y1[t - 1] + 0.2 * rng.standard_normal()
    s1 = fc.select_k_ms(y1, k_max=3, B=39, seed=7)
    ok1 = s1["k"] == 1
    means3 = np.array([-0.20, 0.40, 1.10])
    sds = np.array([0.10, 0.12, 0.15])
    P = np.full((3, 3), 0.03)
    np.fill_diagonal(P, 0.94)
    s = 0
    y3 = np.zeros(400)
    y3[0] = means3[0]
    cum = np.cumsum(P, axis=1)
    for t in range(1, 400):
        s = min(int(np.searchsorted(cum[s], rng.random())), 2)
        y3[t] = means3[s] * (1 - phi) + phi * y3[t - 1] + sds[s] * rng.standard_normal()
    s3 = fc.select_k_ms(y3, k_max=4, B=39, seed=8)
    m = np.sort(np.asarray(s3["fit"]["means"], float))
    ok2 = s3["k"] == 3 and np.all(np.abs(m - means3) < 0.25)
    print(f"  AR(1) -> K={s1['k']} (p_path {s1['p_path']}); planted 3-state -> K={s3['k']}, "
          f"means {np.round(m, 2).tolist()} (true {means3.tolist()})")
    return ok1 and ok2


def check_pd_link():
    import flow_correlation as fc
    import paper_tables as pt
    rng = np.random.default_rng(55)
    rows = []
    for d in range(12):
        off = 0.05 * rng.standard_normal()
        for w in range(12):
            z = rng.standard_normal()
            occ = rng.random()
            rows.append({"date": f"d{d}", "regime": "benchmark", "window": str(w),
                         "CS_ES": 0.4 + off + 0.08 * z + 0.02 * rng.standard_normal(),
                         "IS_mid_ES": 0.5 + off + 0.10 * occ + 0.02 * rng.standard_normal(),
                         "ec_valid": True, "z_flow": z, "occ_top": occ})
    panel = pd.DataFrame(rows)
    bz, sz, pz, _n, _G = fc._fe_reg_windows(panel, "CS_ES", "z_flow", standardize=True)
    bo, so, po, _n2, _G2 = fc._fe_reg_windows(panel, "IS_mid_ES", "occ_top", standardize=False)
    ok1 = abs(bz - 0.08) < 0.02 and pz < 0.01 and abs(bo - 0.10) < 0.03 and po < 0.01
    sessions = [(f"2024-08-{i + 1:02d}", "calm" if i < 4 else "stress",
                 pt._synth_session(f"2024-08-{i + 1:02d}",
                                   "calm" if i < 4 else "stress", T=2500, seed=40 + i))
                for i in range(8)]
    t4, n4 = fc.table_flow_pd_link(sessions, window_minutes=10, B=9, k_max=2, min_bars=25)
    ok2 = (not t4.empty) and "windows / days" in t4.index
    print(f"  planted slopes: z {bz:.3f} (true 0.08, p {pz:.4f}), occ {bo:.3f} "
          f"(true 0.10, p {po:.4f}); end-to-end table ok: {ok2}")
    return ok1 and ok2


def check_tables_end_to_end():
    import flow_correlation as fc
    import test_hy_correlation as th
    sessions = []
    for i in range(10):
        rho = 0.25 if i < 5 else 0.85
        regime = "benchmark" if i < 5 else "volatile"
        df, _ = th._frame(n=6000, rho=rho, refresh=0.6, seed=200 + i)
        sessions.append((f"d{i}", regime, df))
    bars = fc.per_day_bars(sessions, bar_seconds=60)
    t1, n1 = fc.table_flow_corr_regimes(sessions, n_perm=5000, bars=bars)
    zb = float(t1.loc["benchmark", "mean z"])
    zv = float(t1.loc["volatile", "mean z"])
    p = float(t1.loc["volatile - benchmark", "perm p"])
    ok1 = zv > zb and p < 0.05
    t2, n2 = fc.table_flow_corr_mediation(sessions, n_lags=2, n_boot=49, bars=bars)
    ok2 = not t2.empty and "indirect share of RV_ES effect" in t2.index
    t3, n3 = fc.table_flow_corr_ms_regimes(sessions, B=19, bars=bars)
    ok3 = not t3.empty and "all days" in t3.index and "duration asymmetry p" in t3.columns
    ta, na = fc.table_flow_corr_asymmetry(sessions, n_flip=2000)
    ok3 = ok3 and not ta.empty and "sign-flip p" in ta.columns
    drv = open(os.path.join(HERE, "run_paper_replication.sh")).read()
    ok4 = "run_flow_correlation.py" in drv
    import run_flow_correlation as rfc
    hp = rfc.build_parser().format_help()
    ok5 = all(f in hp for f in ("--bar-seconds", "--ms-boot", "--n-perm", "--out-dir"))
    # lag length must be denominated in WALL CLOCK, not lag count: the pd-link default
    # resolves from the grid (5 lags at 1s, capped 60 at 10ms), like run_analysis
    import cross_asset_pd_liquidity as ca
    ok5 = (ok5 and rfc.build_parser().parse_args([]).pd_lags == -1
           and ca.frequency_defaults(dt=1.0)["n_lags"] == 5
           and ca.frequency_defaults(dt=0.01)["n_lags"] == 60)
    print(f"  tier1 z: ben {zb:.3f} < vol {zv:.3f}, perm p {p:.4f}; mediation rows ok: {ok2}; "
          f"ms table ok: {ok3}; driver wired: {ok4}; runner flags ok: {ok5}")
    return ok1 and ok2 and ok3 and ok4 and ok5


def main():
    warnings.simplefilter("ignore")
    checks = [("ar_innovations strips persistence, keeps innovation corr", check_ar_innovations),
              ("_bar_corr recovery + coverage floor", check_bar_corr),
              ("ofi injection exact + halt rows NaN", check_ofi_injection),
              ("flow_corr_bars recovers injected corr", check_flow_corr_bars),
              ("mediation: planted channel found, null clean", check_mediation),
              ("MS regimes: planted detected, null not rejected", check_ms_regimes),
              ("semicorrelation asymmetry: null clean, planted found", check_semicorr_asymmetry),
              ("regime-dynamics asymmetry: duration + entry direction", check_regime_dynamics_asymmetry),
              ("select_k: AR(1) -> 1 regime, planted 3-state -> 3", check_select_k),
              ("pd link: planted slopes recovered + end-to-end table", check_pd_link),
              ("tables end-to-end + driver/runner wiring", check_tables_end_to_end)]
    rc = 0
    for i, (name, fn) in enumerate(checks, 1):
        try:
            ok = fn()
        except Exception as e:
            ok = False
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
        print(f"{'ok' if ok else 'FAIL':>4}  {i}. {name}")
        rc |= 0 if ok else 1
    print("test_flow_correlation:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
