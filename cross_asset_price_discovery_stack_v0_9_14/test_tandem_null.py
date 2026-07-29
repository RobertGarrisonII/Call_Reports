"""test_tandem_null.py -- the Table 5 null must isolate CROSS-market dependence.

Table 5.I Panel A benchmarks the observed corner cells against a Binomial(n, 1/2)
independence null with n = 505 (ETF) and n = 112 (future) orders per bar. That null bundles
two separate hypotheses:

    (i)  each market's own order flow is a fair coin over n orders, and
    (ii) the two markets are independent of one another.

Only (ii) is about tandem trading, and on this data it is (i) that fails. The paper's own
Table 5.II Panel A puts the ETF in a directional state in 68.7% of one-second bars against a
2.4% binomial prediction, and the future in 83.2% against 29.0%. Within-market order flow is
clustered and autocorrelated, so the marginals are nowhere near coin flips -- and that alone
inflates all four corner cells, no cross-market linkage required.

This file pins down three consequences:

  1. A session built with ZERO cross-market dependence but realistic within-market
     clustering reproduces large corner cells anyway. Judged against the binomial null it
     looks like overwhelming tandem trading. Judged against independence-given-marginals it
     correctly shows none. This is the false positive the paper's null admits.
  2. The binomial null is a function of orders-per-bar, so a per-second calibration cannot be
     reused at a finer aggregation. Table 7 reports 10-millisecond and action-time NCMOF and
     compares them to "the theoretical probabilities of independent order flow"; at those
     frequencies the null is ~23% and ~50%, not 0.4%.
  3. The corner log odds ratio recovers true cross-market dependence and is invariant to how
     directional either market is on its own -- so it states the paper's claim in a form that
     within-market clustering cannot manufacture.
"""
import sys
import warnings

import numpy as np

import tandem_order_flow as tof


def _session(rho, T=120_000, lam_e=505, lam_f=112, persist=0.9, slope=1.1, seed=0):
    """One synthetic session of per-bar new-order buy/sell counts.

    `persist` drives WITHIN-market directional clustering (an AR(1) latent buy-pressure per
    market, as real order flow has) and `slope` how strongly that pressure produces a
    directional bar -- together they set the MARGINALS. `rho` is the CROSS-market correlation
    of those latent factors -- the only thing tandem trading should show up in. rho=0 => no
    tandem trading at all, whatever the marginals look like."""
    rng = np.random.default_rng(seed)
    ze = np.empty(T); zf = np.empty(T)
    s = np.sqrt(1 - persist ** 2)
    ze[0], zf[0] = rng.standard_normal(2)
    for t in range(1, T):                       # AR(1) latent pressure, cross-correlated rho
        e1, e2 = rng.standard_normal(2)
        e2 = rho * e1 + np.sqrt(max(0.0, 1 - rho ** 2)) * e2
        ze[t] = persist * ze[t - 1] + s * e1
        zf[t] = persist * zf[t - 1] + s * e2
    pe = 1 / (1 + np.exp(-slope * ze)); pf = 1 / (1 + np.exp(-slope * zf))
    ne = rng.poisson(lam_e, T); nf = rng.poisson(lam_f, T)
    be = rng.binomial(ne, pe); bf = rng.binomial(nf, pf)
    return be, ne - be, bf, nf - bf


def main():
    warnings.simplefilter("ignore")
    ok = True

    # ── 1. no cross-market dependence, realistic within-market clustering ─────────────
    be, se_, bf, sf_ = _session(rho=0.0, seed=1)
    t5 = tof.table5_from_series(be, se_, bf, sf_)
    raw = t5["pcmof_ncmof"]
    dep = t5["dependence"]
    binom = (tof.theoretical_null().loc["Sell", "Sell"] + tof.theoretical_null().loc["Buy", "Buy"])
    # the binomial null would call this overwhelming tandem trading ...
    looks_huge = raw["PCMOF"] > 20 * binom
    # ... while the marginal-preserving benchmark correctly finds none
    honest = abs(dep["PCMOF_ratio"] - 1.0) < 0.10 and abs(dep["NCMOF_ratio"] - 1.0) < 0.10
    a_ok = looks_huge and honest
    print("(1) NO cross-market dependence, clustered within-market flow")
    print("    raw PCMOF=%.1f%% vs binomial null %.2f%% -> looks like %.0fx tandem trading"
          % (raw["PCMOF"], binom, raw["PCMOF"] / binom))
    print("    vs independence-given-marginals: PCMOF %.2fx, NCMOF %.2fx (want ~1.00x) : %s"
          % (dep["PCMOF_ratio"], dep["NCMOF_ratio"], a_ok))
    ok &= a_ok

    # ── 2. genuine cross-market dependence is still detected ─────────────────────────
    be, se_, bf, sf_ = _session(rho=0.7, seed=2)
    t5b = tof.table5_from_series(be, se_, bf, sf_)
    depb = t5b["dependence"]
    b_ok = depb["PCMOF_ratio"] > 1.2 and depb["NCMOF_ratio"] < 0.8 and depb["log_OR"] > 0.5
    print("(2) TRUE cross-market dependence (rho=0.7)")
    print("    PCMOF %.2fx, NCMOF %.2fx, logOR=%+.2f (z=%.0f) -> detected : %s"
          % (depb["PCMOF_ratio"], depb["NCMOF_ratio"], depb["log_OR"], depb.get("log_OR_z", np.nan), b_ok))
    ok &= b_ok

    # ── 3. the log odds ratio is invariant to the MARGINALS, exactly ─────────────────
    # This is the property the cross-panel comparison needs. The paper reads tandem trading off
    # the raw corner shares and compares them across baseline / volatile / MWCB -- but those
    # three panels have very different marginals (the ETF is directional in 68.7% of baseline
    # bars and only 42.2% of MWCB bars), so part of the movement in the corner shares is the
    # marginals moving, not the cross-market association.
    #
    # Rake the SAME association to different marginals (iterative proportional fitting changes
    # row/column totals while leaving every odds ratio untouched, by construction) and check:
    # the raw corner shares move a lot, the log OR does not move at all. Note this is
    # invariance to the MARGINS of the table -- not to re-discretising an underlying continuous
    # association at different cut points, which does change the odds ratio.
    base = tof.table5_from_series(*_session(rho=0.5, seed=3))["frequency"]
    M0 = base.reindex(index=tof.DIR3, columns=tof.DIR3).to_numpy(float)
    M0 = M0 / M0.sum()
    tgt_r = np.array([0.25, 0.50, 0.25]); tgt_c = np.array([0.45, 0.10, 0.45])   # new marginals
    M1 = M0.copy()
    for _ in range(200):                                   # IPF: same odds ratios, new margins
        M1 *= (tgt_r / M1.sum(1))[:, None]
        M1 *= (tgt_c / M1.sum(0))[None, :]
    def _pc_nc_or(M):
        P = M / M.sum() * 100.0
        return (P[0, 0] + P[2, 2], P[0, 2] + P[2, 0],
                float(np.log((P[0, 0] * P[2, 2]) / (P[0, 2] * P[2, 0]))))
    pc0, nc0, or0 = _pc_nc_or(M0)
    pc1, nc1, or1 = _pc_nc_or(M1)
    c_ok = abs(pc0 - pc1) > 5.0 and abs(or0 - or1) < 1e-8
    print("(3) SAME association, marginals re-raked (what differs across the paper's 3 panels)")
    print("    raw PCMOF %.1f%% -> %.1f%% (moves %.1fpp) | logOR %+.4f -> %+.4f (moves %.1e) : %s"
          % (pc0, pc1, abs(pc0 - pc1), or0, or1, abs(or0 - or1), c_ok))
    ok &= c_ok

    # ── 4. the binomial null is frequency-dependent -- a per-second n cannot be reused ─
    per_sec = tof.independence_null_from_counts(np.random.default_rng(5).poisson(505, 200_000),
                                                np.random.default_rng(6).poisson(112, 200_000))
    per_10ms = tof.independence_null_from_counts(np.random.default_rng(7).poisson(5.05, 200_000),
                                                 np.random.default_rng(8).poisson(1.12, 200_000))
    action = tof.independence_null_from_counts(np.ones(200_000), np.ones(200_000))
    d_ok = per_sec["NCMOF"] < 1.0 and per_10ms["NCMOF"] > 15.0 and action["NCMOF"] > 40.0
    print("(4) independence null vs aggregation (Table 7 compares 10ms and action time to the")
    print("    per-second figure): NCMOF null = %.2f%% at 1s, %.1f%% at 10ms, %.1f%% in action time : %s"
          % (per_sec["NCMOF"], per_10ms["NCMOF"], action["NCMOF"], d_ok))
    ok &= d_ok

    # ── 5. permutation null agrees with the analytic one, and is calibrated ──────────
    be, se_, bf, sf_ = _session(rho=0.0, T=40_000, seed=9)
    st_e = tof.classify(tof.order_imbalance(be, se_)); st_f = tof.classify(tof.order_imbalance(bf, sf_))
    pn = tof.permutation_null(st_f, st_e, n_perm=299, seed=0)
    e_ok = pn["PCMOF"]["p_value"] > 0.05 and pn["NCMOF"]["p_value"] > 0.05
    pn2 = tof.permutation_null(*(lambda a: (tof.classify(tof.order_imbalance(a[2], a[3])),
                                            tof.classify(tof.order_imbalance(a[0], a[1]))))(
        _session(rho=0.7, T=40_000, seed=10)), n_perm=299, seed=0)
    e_ok &= pn2["PCMOF"]["p_value"] < 0.01
    print("(5) permutation null: p=%.3f/%.3f under no dependence (want >0.05), p=%.3f under rho=0.7 : %s"
          % (pn["PCMOF"]["p_value"], pn["NCMOF"]["p_value"], pn2["PCMOF"]["p_value"], e_ok))
    ok &= e_ok

    print("\ntandem-null checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
