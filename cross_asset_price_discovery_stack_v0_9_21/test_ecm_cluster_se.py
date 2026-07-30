"""test_ecm_cluster_se.py -- why the ECM-SDE gradient needs a DAY-CLUSTERED SE, not the pooled HAC.

The thesis test is a1 = d alpha_SPY/dS (the z*S loading), pooled across sessions. The conditioning
strength is not a single constant -- it is a daily regime that varies day to day (a random coefficient
g1_d across days). The pooled estimate of a1 is the AVERAGE daily gradient; its sampling variance is
driven by the BETWEEN-DAY dispersion of g1_d. A pooled Newey-West HAC, treating all days as one long
series, cannot see between-day dispersion: it reports a tiny within-series SE and so OVERSTATES
significance. The day-cluster SE (Liang-Zeger, panel_regression's convention) treats each day as a
cluster and captures exactly that dispersion. Two demonstrations + a bootstrap cross-check:

  (1) ORDER-INDEPENDENCE   a correct SE for exchangeable day-clusters cannot depend on the order the
      days are stacked. The cluster t is exactly invariant under a day permutation; the pooled HAC t
      is not (its Bartlett windows cross the overnight gaps) -- so the HAC is the wrong object here.
  (2) NULL SIZE            with a1 truly 0 (each day's slope ~ N(0, sigma_g^2)), the pooled HAC
      rejects |t|>1.96 far above the nominal 5%; the day-cluster test (t_{G-1}) is close to nominal.
"""
import sys
import numpy as np
from scipy import stats
import ecm_sde as es

def sim_day(rng, n, g1_bar, sigma_g, phi=0.99, g0=0.060, e0=0.004):
    gd = g1_bar + rng.normal(0, sigma_g)               # this DAY's gradient (random coefficient)
    S = np.zeros(n)
    for k in range(1, n):
        S[k] = phi * S[k - 1] + rng.normal(0, 0.07)
    S = (S - S.mean()) / (S.std() + 1e-12); S = np.clip(S, -3, 3)
    Sig = np.array([[6e-4, 2e-4], [2e-4, 6e-4]]); L = np.linalg.cholesky(Sig); sig_f = 4e-4
    p1 = np.zeros(n); p2 = np.zeros(n)
    for k in range(1, n):
        zk = p1[k - 1] - p2[k - 1]; df = rng.normal(0, sig_f)
        aS = -(g0 + gd * S[k - 1]); aE = e0
        e = L @ rng.normal(0, 1, 2)
        p1[k] = p1[k - 1] + df + aS * zk + e[0]
        p2[k] = p2[k - 1] + df + aE * zk + e[1]
    z = (p1 - p2); z = z - z.mean()
    r = np.column_stack([np.diff(p1), np.diff(p2)]); z = z[:-1]; S = S[:-1]
    return z, r, S

def build_panel(D, n, g1_bar, sigma_g, seed):
    rng = np.random.default_rng(seed)
    Zs, Rs, Ss, Gs = [], [], [], []
    for d in range(D):
        z, r, S = sim_day(rng, n, g1_bar, sigma_g)
        Zs.append(z); Rs.append(r); Ss.append(S); Gs.append(np.full(len(z), d))
    return (np.concatenate(Zs), np.vstack(Rs), np.concatenate(Ss) - np.concatenate(Ss).mean(),
            np.concatenate(Gs))

def t_hac(z, r, Sc):
    f = es.state_interacted_vecm(None, None, None, degree=1, center_state=False, hac_lags=20,
                                 _z=z, _ret=r, _Sc=Sc)
    return f["t_a1_SPY"], f["se_a1_SPY"]

def t_clu(z, r, Sc, g):
    f = es.state_interacted_vecm(None, None, None, degree=1, center_state=False,
                                 _z=z, _ret=r, _Sc=Sc, _groups=g)
    return f["t_a1_SPY"], f["se_a1_SPY"]

def reorder_by_day(z, r, Sc, g, perm):
    sel = np.concatenate([np.where(g == d)[0] for d in perm])
    relabel = np.concatenate([np.full(int((g == d).sum()), i) for i, d in enumerate(perm)])
    return z[sel], r[sel], Sc[sel], relabel

if __name__ == "__main__":
    D, n = 14, 2500
    z, r, Sc, g = build_panel(D, n, g1_bar=-0.020, sigma_g=0.012, seed=1)
    th, seh = t_hac(z, r, Sc); tc, sec = t_clu(z, r, Sc, g)
    print("one panel (avg d.alpha_SPY/dS = -0.020 with between-day sd 0.012; %d days x %d):" % (D, n))
    print("  pooled HAC : se_a1=%.5f  t_a1=%6.2f" % (seh, th))
    print("  day-cluster: se_a1=%.5f  t_a1=%6.2f    (cluster SE / HAC SE = %.2fx)" % (sec, tc, sec / seh))

    rng = np.random.default_rng(7); perm = rng.permutation(D)
    z2, r2, Sc2, g2 = reorder_by_day(z, r, Sc, g, perm)
    th2, _ = t_hac(z2, r2, Sc2); tc2, _ = t_clu(z2, r2, Sc2, g2)
    print("\norder-invariance (permute the day blocks):")
    print("  HAC t:     %.5f -> %.5f  (delta = %.2e, nonzero -> order-dependent)" % (th, th2, th2 - th))
    print("  cluster t: %.5f -> %.5f  (delta = %.2e -> invariant)" % (tc, tc2, tc2 - tc))
    invariant_clu = abs(tc2 - tc) < 1e-7
    hac_order_dependent = abs(th2 - th) > 1e-12
    se_larger = sec > seh

    bse = es._daycluster_boot_a1(z, r, Sc, g, degree=1, n_lags=0, names=("SPY", "ES"),
                                 n_boot=400, seed=3)["a1_SPY_se"]
    print("\nbootstrap cross-check: analytic cluster se=%.5f  pairs-bootstrap se=%.5f  (ratio %.2f)"
          % (sec, bse, bse / sec))
    boot_agrees = 0.5 < (bse / sec) < 2.0

    R = 150; crit_n = 1.96; crit_t = stats.t.ppf(0.975, df=D - 1)
    rej_h = rej_c = 0
    for s in range(R):
        zz, rr, ss, gg = build_panel(D, 1200, g1_bar=0.0, sigma_g=0.012, seed=1000 + s)
        thh, _ = t_hac(zz, rr, ss); tcc, _ = t_clu(zz, rr, ss, gg)
        rej_h += abs(thh) > crit_n; rej_c += abs(tcc) > crit_t
    print("\nnull (a1=0, daily slopes ~ N(0,0.012^2)), %d reps:" % R)
    print("  pooled-HAC rejects %.1f%%  (nominal 5%%)   |   day-cluster rejects %.1f%%"
          % (100.0 * rej_h / R, 100.0 * rej_c / R))
    hac_overrejects = (rej_h / R) > 0.15 and (rej_h > rej_c)

    ok = invariant_clu and hac_order_dependent and se_larger and boot_agrees and hac_overrejects
    print("\nchecks: cluster_order_invariant=%s hac_order_dependent=%s cluster_se_larger=%s "
          "boot_agrees=%s hac_overrejects=%s" % (invariant_clu, hac_order_dependent, se_larger,
                                                  boot_agrees, hac_overrejects))
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)
