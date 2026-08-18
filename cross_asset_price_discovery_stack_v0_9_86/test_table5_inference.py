#!/usr/bin/env python3
"""Gate: day-clustered inference for the Table 5 corner log OR behaves as designed.

The module exists because the run scored pooled panels with the Woolf SE and
printed z = 159-214 -- a bar-count artifact, not evidence. Every check here is a
planted DGP whose answer is known by construction:

  1. sign + coverage -- planted cross-market coupling (one common directional
     factor driving both markets' buy probabilities) must give pooled log_OR > 0
     with a day-bootstrap CI excluding 0; independent factors must give a CI
     covering 0. A CI that misses these endpoints is inferentially useless.
  2. the z=214 pathology, pinned -- with serially dependent bars and day-to-day
     coupling heterogeneity, the Woolf z (IID-bar assumption) must be an order
     of magnitude LARGER than the day-bootstrap z on the same panel. If this
     ratio collapses, the module has stopped correcting the thing it was built
     to correct.
  3. within-pair sign-flip -- a planted regime difference in coupling must be
     detected (p < 0.05, free permutation agreeing); no planted difference must
     not be (p > 0.1). This is the era-robust test the stack standardises on.
  4. the inference sits on the OLD estimate -- the pooled log OR from SUMMED
     per-day count matrices must equal table5_from_sessions' pooled log_OR to
     1e-9. New inference, same point estimate; any drift means the two stages
     report different objects under one name.
  5. runner hygiene -- the CLI loads STAGE-4-style pickles, halt-masks, writes
     both CSVs, and the per-panel rows carry finite day-clustered statistics.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import mstbook_loader as ml
import table5_inference as t5i
import tandem_order_flow as tof


# ── planted DGP ───────────────────────────────────────────────────────────────
def _session_frame(T, beta, lam, seed, rho=0.0, date="2024-07-24"):
    """One synthetic session with the trade columns counts_from_frame expects.

    Each market's per-bar buy probability is a logit of its own directional factor
    x = lam*z + sqrt(1-lam^2)*u, where z is COMMON across markets and u is
    idiosyncratic; all factors are AR(1) with coefficient rho (serial dependence is
    the point of check 2). lam is therefore the planted cross-market coupling dial
    with the within-market directionality held fixed: lam=1 is pure tandem trading,
    lam=0 is the exact null the corner log OR isolates (each market still
    directional and clustered on its own, zero cross-market linkage)."""
    rng = np.random.default_rng(seed)

    def _ar1(n):
        e = rng.normal(size=n)
        if rho <= 0:
            return e
        z = np.empty(n)
        z[0] = e[0]
        c = np.sqrt(1.0 - rho * rho)
        for t in range(1, n):
            z[t] = rho * z[t - 1] + c * e[t]
        return z

    z = _ar1(T)
    w = np.sqrt(max(0.0, 1.0 - lam * lam))
    x_e = lam * z + w * _ar1(T)
    x_f = lam * z + w * _ar1(T)
    p_e = 1.0 / (1.0 + np.exp(-beta * x_e))
    p_f = 1.0 / (1.0 + np.exp(-beta * x_f))
    n_e = rng.poisson(505, T); n_f = rng.poisson(112, T)
    buy_e = rng.binomial(n_e, p_e); buy_f = rng.binomial(n_f, p_f)
    idx = pd.date_range(f"{date} 09:30:00", periods=T, freq="s", tz="America/New_York")
    px_e = 500.0 * np.exp(np.cumsum(rng.normal(0, 2e-5, T)))
    px_f = 5000.0 * np.exp(np.cumsum(rng.normal(0, 2e-5, T)))
    return pd.DataFrame({"SPY_trade_buy": buy_e, "SPY_trade_sell": n_e - buy_e,
                         "ES_trade_buy": buy_f, "ES_trade_sell": n_f - buy_f,
                         "SPY_trade_px": px_e, "ES_trade_px": px_f}, index=idx)


def _sessions(n_days, lam, seed0, beta=1.3, rho=0.0, regime="benchmark",
              start="2019-01-07", lam_jitter=0.0, T=3000):
    """n_days sessions on consecutive weekdays; lam_jitter plants day-to-day
    coupling heterogeneity (the day-level variance the Woolf SE cannot see)."""
    rng = np.random.default_rng(seed0)
    days = pd.bdate_range(start, periods=n_days)
    out = []
    for i, d in enumerate(days):
        lm = float(np.clip(lam + (rng.uniform(-lam_jitter, lam_jitter)
                                  if lam_jitter else 0.0), 0.0, 1.0))
        out.append((str(d.date()), regime,
                    _session_frame(T, beta, lm, seed0 + 100 + i, rho=rho,
                                   date=str(d.date()))))
    return out


# ── checks ────────────────────────────────────────────────────────────────────
def check_sign_and_coverage():
    """Planted coupling -> positive pooled log OR with a day-bootstrap CI excluding
    0; independent flows -> CI covering 0. The two endpoints of any usable CI."""
    dep = _sessions(8, lam=0.85, seed0=1)
    ind = _sessions(8, lam=0.0, seed0=2)
    pd_dep = t5i.per_day_matrices(dep, ml.counts_from_frame, mwcb="")
    pd_ind = t5i.per_day_matrices(ind, ml.counts_from_frame, mwcb="")
    b_dep = t5i.day_bootstrap(pd_dep, "A baseline", n_boot=2000, seed=0)
    b_ind = t5i.day_bootstrap(pd_ind, "A baseline", n_boot=2000, seed=0)
    ok_pos = b_dep["log_OR"] > 0 and b_dep["ci_lo"] > 0
    ok_null = b_ind["ci_lo"] < 0 < b_ind["ci_hi"]
    print("(1) coupled: log_OR=%.3f CI=[%.3f, %.3f] excludes 0 (%s); independent: "
          "CI=[%.3f, %.3f] covers 0 (%s)"
          % (b_dep["log_OR"], b_dep["ci_lo"], b_dep["ci_hi"], ok_pos,
             b_ind["ci_lo"], b_ind["ci_hi"], ok_null))
    return bool(ok_pos and ok_null)


def check_woolf_overstatement():
    """The run's z=159-214 pathology, pinned: serially dependent bars (AR(1) factor,
    rho=0.99) plus day-to-day coupling heterogeneity make the effective sample the
    DAYS, not the bars. The Woolf z must overstate the day-bootstrap z by an order
    of magnitude on the same panel -- that ratio IS the reason this module exists."""
    sess = _sessions(10, lam=0.65, seed0=7, rho=0.99, lam_jitter=0.3, T=8000)
    per_day = t5i.per_day_matrices(sess, ml.counts_from_frame, mwcb="")
    dep = t5i.panel_dependence(per_day, "A baseline")
    boot = t5i.day_bootstrap(per_day, "A baseline", n_boot=2000, seed=0)
    zw, zd = dep.get("log_OR_z", np.nan), boot["z_day"]
    ok = np.isfinite(zw) and np.isfinite(zd) and zd > 0 and zw >= 10.0 * zd
    print("(2) Woolf z=%.1f vs day-bootstrap z=%.2f (ratio %.1fx, need >= 10x) : %s"
          % (zw, zd, zw / zd if zd > 0 else np.nan, ok))
    return bool(ok)


def check_pair_contrast():
    """Planted regime difference (volatile coupling 1.9 vs benchmark 0.8) must be
    found by the within-pair sign-flip (p < 0.05) with the free permutation
    agreeing; identical coupling in both regimes must NOT be found (p > 0.1).
    Detecting nothing, or everything, would make the contrast decorative."""
    vol = _sessions(8, lam=0.9, seed0=11, regime="volatile", start="2020-03-02")
    ben = _sessions(8, lam=0.45, seed0=12, regime="benchmark", start="2019-03-04")
    per_day = t5i.per_day_matrices(vol + ben, ml.counts_from_frame, mwcb="")
    vdates = [d for d, _, _ in vol]; bdates = [d for d, _, _ in ben]
    hit = t5i.pair_contrast(per_day, vdates, bdates, n_perm=20000, seed=0)
    vol0 = _sessions(8, lam=0.6, seed0=13, regime="volatile", start="2020-03-02")
    ben0 = _sessions(8, lam=0.6, seed0=14, regime="benchmark", start="2019-03-04")
    per_day0 = t5i.per_day_matrices(vol0 + ben0, ml.counts_from_frame, mwcb="")
    miss = t5i.pair_contrast(per_day0, [d for d, _, _ in vol0],
                             [d for d, _, _ in ben0], n_perm=20000, seed=0)
    ok_hit = hit["p_within_pair"] < 0.05 and hit["n_pairs"] == 8 and hit["mean_pair_diff"] > 0
    ok_free = hit["p_free"] < 0.05
    ok_miss = miss["p_within_pair"] > 0.1
    print("(3) planted difference: mean d=%.3f p_within=%.4g (<0.05: %s), p_free=%.4g "
          "agrees (%s); no difference: p_within=%.3f (>0.1: %s)"
          % (hit["mean_pair_diff"], hit["p_within_pair"], ok_hit,
             hit["p_free"], ok_free, miss["p_within_pair"], ok_miss))
    return bool(ok_hit and ok_free and ok_miss)


def check_point_estimate_identity():
    """panel_log_or from summed per-day COUNT matrices must equal the pooled log_OR
    table5_from_sessions computes from concatenated bars (percent matrix), to 1e-9.
    dependence_summary is scale-invariant, so any disagreement means the new
    inference is being attached to a different point estimate than STAGE 4 reports."""
    sess = _sessions(6, lam=0.7, seed0=21)
    by = tof.table5_from_sessions(sess, ml.counts_from_frame)
    old = by["benchmark"]["dependence"]["log_OR"]
    per_day = t5i.per_day_matrices(sess, ml.counts_from_frame, mwcb="")
    new = t5i.panel_log_or(per_day, "A baseline")
    ok = np.isfinite(old) and abs(new - old) < 1e-9
    print("(4) pooled log_OR: summed per-day counts %.12f vs table5_from_sessions %.12f "
          "(|diff|=%.2e < 1e-9) : %s" % (new, old, abs(new - old), ok))
    return bool(ok)


def check_runner():
    """The CLI end-to-end on a STAGE-4-style pickle: loads 3-tuples, halt-masks,
    writes table5_per_day.csv + table5_inference.csv, per-panel rows finite, the
    small MWCB panel falls back to the jackknife with its caveat note."""
    import pickle
    import run_table5_inference as rti
    vol = _sessions(6, lam=0.8, seed0=31, regime="volatile", start="2020-02-24")
    ben = _sessions(6, lam=0.5, seed0=32, regime="benchmark", start="2019-02-25")
    mwcb = _sessions(2, lam=0.5, seed0=33, regime="volatile", start="2020-03-09")
    with tempfile.TemporaryDirectory() as td:
        pkl = os.path.join(td, "frames_1s.pkl")
        with open(pkl, "wb") as fh:
            pickle.dump(vol + ben + mwcb, fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rti.main(["--pickle", pkl,
                           "--mwcb", ",".join(d for d, _, _ in mwcb),
                           "--volatile", ",".join(d for d, _, _ in vol),
                           "--benchmark", ",".join(d for d, _, _ in ben),
                           "--out", td, "--n-boot", "500", "--n-perm", "5000"])
        per_day_ok = os.path.exists(os.path.join(td, "table5_per_day.csv"))
        inf_ok = os.path.exists(os.path.join(td, "table5_inference.csv"))
        ok_rows = ok_jack = ok_contrast = False
        if inf_ok:
            t = pd.read_csv(os.path.join(td, "table5_inference.csv")).set_index("panel")
            ok_rows = ({"A baseline", "B volatile", "C MWCB"} <= set(t.index)
                       and np.isfinite(t.loc["A baseline", "z_day"])
                       and np.isfinite(t.loc["B volatile", "z_day"]))
            ok_jack = (t.loc["C MWCB", "method"] == "jackknife"
                       and t.loc["C MWCB", "note"] == "exact/caveat")
            ok_contrast = ("B-A within-pair contrast" in t.index
                           and "p_within=" in str(t.loc["B-A within-pair contrast", "note"]))
        out = buf.getvalue()
        ok_print = "day-clustered inference" in out and "within-pair contrast" in out
    print("(5) runner: rc=%s, CSVs written (%s/%s), panel rows finite (%s), MWCB "
          "jackknife+caveat (%s), contrast row (%s), table printed (%s)"
          % (rc, per_day_ok, inf_ok, ok_rows, ok_jack, ok_contrast, ok_print))
    return rc == 0 and per_day_ok and inf_ok and ok_rows and ok_jack and ok_contrast and ok_print


def main():
    checks = [check_sign_and_coverage, check_woolf_overstatement, check_pair_contrast,
              check_point_estimate_identity, check_runner]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("table5-inference checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
