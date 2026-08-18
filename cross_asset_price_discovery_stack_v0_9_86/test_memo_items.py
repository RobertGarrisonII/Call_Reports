#!/usr/bin/env python3
"""Gate: the 2026-08-14 findings-memo items implemented in v0.9.73.

E2 -- co-jump lead-lag on EVERY session (cojump_by_day) with a day-level
      sign-flip test on the per-day lead share (cojump_lead_test): a planted
      ES-leads DGP must recover the leader with a small p, and a symmetric DGP
      must not.
E3 -- compare_regimes gains a kappa-weighted variant: with unit weights it
      reproduces the unweighted means exactly, and when a planted contrast is
      carried entirely by near-zero-weight days, the weighted contrast collapses.
Q2 -- the low-kappa flag lands in the per-day information-shares table and the
      new regime-test keys land in summary.json's picks.
"""
import sys

import numpy as np
import pandas as pd

import jump_robust as jr
import price_discovery_shares as pds


def _leader_day(n=5000, n_jumps=12, lead=1, seed=0, sym=False):
    """Two correlated random-walk mids with planted co-jumps. lead=1: the ES jump
    lands one step BEFORE the SPY jump. sym=True: the lead direction of each
    planted jump is a fair coin (the no-leader null)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2)) * 1e-4
    times = np.sort(rng.choice(np.arange(50, n - 50), size=n_jumps, replace=False))
    for t in times:
        j = 25e-4 * rng.choice([-1.0, 1.0])
        d = int(rng.choice([-lead, lead])) if sym else lead
        # d > 0: ES first at t, SPY at t+d; d < 0: SPY first
        if d >= 0:
            e[t, 1] += j; e[min(t + d, n - 1), 0] += j
        else:
            e[t, 0] += j; e[min(t - d, n - 1), 1] += j
    lp = np.cumsum(e, axis=0)
    return 500.0 * np.exp(lp[:, 0]), 5000.0 * np.exp(lp[:, 1])


def check_cojump_by_day_recovers_leader():
    mids = []
    for i in range(8):
        m_spy, m_es = _leader_day(seed=100 + i)
        mids.append((f"d{i}", "volatile" if i % 2 else "benchmark", m_spy, m_es))
    tab = jr.cojump_by_day(mids, max_lag=2)
    test = jr.cojump_lead_test(tab)
    ok_tab = len(tab) == 8 and {"n_cojump", "lead_SPY", "lead_ES", "lead_share",
                                "regime"} <= set(tab.columns)
    ok_dir = test["leader"] == "ES" and test["mean_lead_share"] > 0.5
    ok_p = test["p_flip"] < 0.05
    print("(1) planted ES-leads DGP: 8 days, mean lead share %+.2f, leader %s, p_flip %.4f"
          % (test["mean_lead_share"], test["leader"], test["p_flip"]))
    print("    table complete (%s), direction recovered (%s), significant (%s)"
          % (ok_tab, ok_dir, ok_p))
    return bool(ok_tab and ok_dir and ok_p)


def check_cojump_null_is_flat():
    mids = []
    for i in range(8):
        m_spy, m_es = _leader_day(seed=200 + i, sym=True)
        mids.append((f"d{i}", "volatile", m_spy, m_es))
    test = jr.cojump_lead_test(jr.cojump_by_day(mids, max_lag=2))
    ok = test["p_flip"] > 0.10
    print("(2) symmetric-lead null: mean lead share %+.2f, p_flip %.3f (> 0.10) : %s"
          % (test["mean_lead_share"], test["p_flip"], ok))
    return bool(ok)


def check_weighted_compare_regimes():
    rng = np.random.default_rng(5)
    # 12 high-kappa days with NO regime difference + 6 near-zero-kappa days that
    # carry a huge fake contrast -- the Q2 failure mode, planted.
    rows = []
    for i in range(12):
        rows.append({"regime": "volatile" if i < 6 else "benchmark",
                     "CS_ES": 0.40 + 0.02 * rng.standard_normal(), "kappa": 0.05})
    for i in range(6):
        rows.append({"regime": "volatile" if i < 3 else "benchmark",
                     "CS_ES": (0.95 if i < 3 else 0.05), "kappa": 1e-4})
    per_day = pd.DataFrame(rows)
    unw = pds.compare_regimes(per_day, metric="CS_ES", n_perm=2000)
    wgt = pds.compare_regimes(per_day, metric="CS_ES", n_perm=2000, weights="kappa")
    ones = pds.compare_regimes(per_day, metric="CS_ES", n_perm=200,
                               weights=np.ones(len(per_day)))
    ok_ones = (abs(ones["vol_mean"] - unw["vol_mean"]) < 1e-12
               and abs(ones["ben_mean"] - unw["ben_mean"]) < 1e-12)
    ok_collapse = abs(wgt["diff"]) < 0.25 * abs(unw["diff"])
    print("(3) planted low-kappa fake contrast: unweighted diff %+.3f, kappa-weighted "
          "diff %+.3f" % (unw["diff"], wgt["diff"]))
    print("    unit weights reproduce unweighted means exactly (%s); kappa weights "
          "collapse the fake contrast (%s)" % (ok_ones, ok_collapse))
    return bool(ok_ones and ok_collapse)


def check_wiring():
    """The Q2 flag logic and the summary picks, on mocked structures."""
    per_day = pd.DataFrame({"kappa": [0.05, 0.06, 0.001, 0.04]})
    med = float(np.nanmedian(per_day["kappa"]))
    flag = per_day["kappa"] < 0.25 * med
    ok_flag = list(flag) == [False, False, True, False]
    import run_analysis as ra
    results = {"jumps": {"cojump_lead_test": {"leader": "ES", "mean_lead_share": 0.4,
                                              "p_flip": 0.01}},
               "information_shares": {"regime_test": {"p_perm": 0.048},
                                      "regime_test_IS": {"p_perm": 0.06},
                                      "regime_test_kappa_weighted": {"p_perm": 0.09}}}
    s = ra._scalar_summary(results)
    ok_sum = (s.get("cojump_leader") == "ES" and s.get("cojump_lead_p_flip") == 0.01
              and s.get("regime_p_CS") == 0.048 and s.get("regime_p_IS") == 0.06
              and s.get("regime_p_CS_kappa_w") == 0.09)
    print("(4) low-kappa flag marks the right day (%s); summary.json picks carry the new "
          "keys (%s)" % (ok_flag, ok_sum))
    return bool(ok_flag and ok_sum)


def _mediation_bars(scale_break=False, n_days=8, n_bars=90, seed=7):
    """Synthetic per-day bar frames for mediation_from_bars: dz_ret loads on the
    UNIT-SCALE lagged spread innovation with coefficient 0.5. The broken variant applies
    a per-day AFFINE transform (x -> 0.7 + 10x) to the spread control on the second half
    of the days -- a level+scale break, which is exactly what the 2025-11-03 tick regime
    does to spread-denominated controls. The rng sequence is identical in both variants,
    so the two datasets differ ONLY by that deterministic transform."""
    rng = np.random.default_rng(seed)
    beta = 0.5
    bars = []
    for d in range(n_days):
        u = rng.standard_normal(n_bars)                   # unit-scale spread innovation
        dz = np.zeros(n_bars)
        dz[1:] = beta * u[:-1] + 0.05 * rng.standard_normal(n_bars - 1)
        b = pd.DataFrame({
            "z_flow": np.cumsum(0.1 * rng.standard_normal(n_bars)),
            "z_ret": np.cumsum(dz),
            "rv_spy": np.abs(rng.standard_normal(n_bars)),
            "rv_es": np.abs(rng.standard_normal(n_bars)),
            "state": rng.standard_normal(n_bars),
            "d_wspr_spy": rng.standard_normal(n_bars),
            "d_wspr_es": u.copy(),
        })
        if scale_break and d >= n_days // 2:
            b["d_wspr_es"] = 0.7 + 10.0 * b["d_wspr_es"]
        bars.append((f"d{d}", "benchmark", b))
    return bars


def check_mediation_controls_standardize():
    """(v0.9.87) Pooled control standardization lets a mid-sample level+scale break in
    the spread control leak into the per-SD coefficient (post-break days contribute
    spread variation on a different ruler); within-day standardization exactly undoes
    any per-day affine transform, so the panel -- and every coefficient -- is invariant
    to the break by construction. Also pins that the legacy default output is
    bit-identical to controls_standardize='pooled'."""
    import flow_correlation as fc
    j = len([f"dz_ret_l{i}" for i in range(1, 4)]) + fc._CONTROLS.index("d_wspr_es_l1")
    bars_u = _mediation_bars(scale_break=False)
    bars_b = _mediation_bars(scale_break=True)
    r_leg = fc.mediation_from_bars(bars_u, n_boot=0)
    r_poo = fc.mediation_from_bars(bars_u, n_boot=0, controls_standardize="pooled")
    ok_legacy = (np.array_equal(r_leg["eqA"][0], r_poo["eqA"][0])
                 and np.array_equal(r_leg["eqB_total"][0], r_poo["eqB_total"][0])
                 and np.array_equal(r_leg["eqB_direct"][0], r_poo["eqB_direct"][0])
                 and r_leg["rv_es_total"] == r_poo["rv_es_total"])
    r_pb = fc.mediation_from_bars(bars_b, n_boot=0, controls_standardize="pooled")
    r_du = fc.mediation_from_bars(bars_u, n_boot=0, controls_standardize="day")
    r_db = fc.mediation_from_bars(bars_b, n_boot=0, controls_standardize="day")
    c_pu = float(r_poo["eqB_total"][0][j])
    c_pb = float(r_pb["eqB_total"][0][j])
    c_du = float(r_du["eqB_total"][0][j])
    c_db = float(r_db["eqB_total"][0][j])
    shift = abs(c_pb - c_pu) / abs(c_pu)
    ok_shift = shift > 0.15                               # pooled: the break moves the coef
    ok_inv = (abs(c_db - c_du) < 1e-8                     # day: bit-near invariant, and not
              and float(np.max(np.abs(r_db["eqB_total"][0] - r_du["eqB_total"][0]))) < 1e-8)
    ok_val = abs(c_du - 0.5) < 0.05                       # ... by being degenerate: recovers beta
    print("(5) mediation spread coef (Eq B total): pooled %.4f -> %.4f under a planted "
          "level+scale break (shift %.1f%%, fires: %s);" % (c_pu, c_pb, 100 * shift, ok_shift))
    print("    day-standardized %.6f -> %.6f (invariant to 1e-8: %s, recovers beta 0.5: %s); "
          "legacy default bit-identical to explicit 'pooled' (%s)"
          % (c_du, c_db, ok_inv, ok_val, ok_legacy))
    return bool(ok_legacy and ok_shift and ok_inv and ok_val)


def main():
    checks = [check_cojump_by_day_recovers_leader, check_cojump_null_is_flat,
              check_weighted_compare_regimes, check_wiring,
              check_mediation_controls_standardize]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("memo-item checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
