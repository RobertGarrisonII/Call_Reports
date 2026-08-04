"""test_panel_vecm_cluster.py -- guards the day-clustered SE on price_discovery_shares.panel_vecm.

panel_vecm's volatile-dummy interaction on the EC term (alpha_volatile - alpha_benchmark) was reported
with a plain homoskedastic OLS standard error; the regime loading varies day to day (random coefficient),
so the honest SE must cluster by session. This compares the pre-fix OLS SE against the clustered SE on a
synthetic panel and runs a null-size contrast.
"""
import sys
import numpy as np
from scipy import stats
import price_discovery_shares as pds

def sim_day(rng, n, alpha_spy, a_es=0.004):
    p1 = np.zeros(n); p2 = np.zeros(n)
    for k in range(1, n):
        z = p1[k - 1] - p2[k - 1]
        p1[k] = p1[k - 1] + alpha_spy * z + rng.normal(0, 0.02)
        p2[k] = p2[k - 1] + a_es * z + rng.normal(0, 0.02)
    return np.exp(p1), np.exp(p2)                 # price levels (panel_vecm logs them)

def build(D, n, dalpha, sigma_g, seed):
    rng = np.random.default_rng(seed); sess = []
    for d in range(D):
        regime = "volatile" if d % 2 == 0 else "benchmark"
        day_re = rng.normal(0, sigma_g)
        aspy = -(0.06 + (dalpha if regime == "volatile" else 0.0) + day_re)
        ms, me = sim_day(rng, n, aspy)
        sess.append((f"d{d}", regime, ms, me))
    return sess

def ols_se_old(sessions, n_lags):
    """Mirror of panel_vecm's design matrix -> the PRE-FIX plain homoskedastic OLS SE on the z*D coef."""
    dYs, ECs, Ds, Ls = [], [], [], []
    for label, regime, ms, me in sessions:
        p1 = np.log(np.asarray(ms, float)); p2 = np.log(np.asarray(me, float))
        ok = np.isfinite(p1) & np.isfinite(p2); p1, p2 = p1[ok], p2[ok]
        z = (p1 - p2); z = z - z.mean(); dp1 = np.diff(p1); dp2 = np.diff(p2); T = len(dp1)
        dYs.append(np.column_stack([dp1[n_lags:T], dp2[n_lags:T]])); ECs.append(z[n_lags:T])
        Ds.append(np.full(T - n_lags, 1.0 if regime == "volatile" else 0.0))
        lag = [np.ones(T - n_lags)]
        for L in range(1, n_lags + 1):
            lag.append(dp1[n_lags - L:T - L]); lag.append(dp2[n_lags - L:T - L])
        Ls.append(np.column_stack(lag))
    dY = np.vstack(dYs); ec = np.concatenate(ECs); D = np.concatenate(Ds); Lall = np.vstack(Ls)
    X = np.column_stack([ec, ec * D, Lall]); B, resid = pds._ols(dY, X)
    XtXinv = np.linalg.inv(X.T @ X)
    return [np.sqrt((resid[:, k] ** 2).mean() * XtXinv[1, 1]) for k in range(2)], B[1, :]

if __name__ == "__main__":
    D, n, nl = 14, 2500, 2
    sess = build(D, n, dalpha=-0.03, sigma_g=0.012, seed=1)
    pv = pds.panel_vecm(sess, n_lags=nl)
    ols, bint = ols_se_old(sess, nl)
    se_clu = pv["se_interaction_es"]; t_clu = pv["t_interaction_es"]; t_ols = bint[1] / (ols[1] + 1e-12)
    print("panel_vecm interaction (alpha_volatile - alpha_benchmark on z), ES equation:")
    print("  pre-fix plain OLS : se=%.5f  t=%6.2f" % (ols[1], t_ols))
    print("  day-cluster (fix) : se=%.5f  t=%6.2f   se_kind=%s n_clusters=%d  (cluster/OLS=%.2fx)"
          % (se_clu, t_clu, pv["se_kind"], pv["n_clusters"], se_clu / ols[1]))
    rng = np.random.default_rng(5); perm = list(rng.permutation(D))
    pv2 = pds.panel_vecm([sess[i] for i in perm], n_lags=nl)
    inv = abs(pv2["t_interaction_es"] - t_clu) < 1e-6
    print("  clustered t order-invariant: %.4f -> %.4f (delta %.1e)"
          % (t_clu, pv2["t_interaction_es"], pv2["t_interaction_es"] - t_clu))
    R = 80; cn = 1.96; ct = stats.t.ppf(0.975, df=D - 1); rh = rc = 0
    for s in range(R):
        ss = build(D, 1200, dalpha=0.0, sigma_g=0.012, seed=500 + s)
        p = pds.panel_vecm(ss, n_lags=nl); o, bi = ols_se_old(ss, nl)
        rh += abs(bi[1] / (o[1] + 1e-12)) > cn; rc += abs(p["t_interaction_es"]) > ct
    print("  null (no regime diff), %d reps: OLS rejects %.0f%% | day-cluster %.0f%% (nominal 5%%)"
          % (R, 100 * rh / R, 100 * rc / R))
    print("  note: with a mean-reverting EC and a within-day-demeaned z, the within-day score is")
    print("        ~negatively autocorrelated, so clustering DEFLATES the SE here (the pre-fix OLS was")
    print("        conservative). Direction is structure-dependent; the over-rejection case is in")
    print("        test_ecm_cluster_se. Either way the OLS SE assumed intraday independence and is wrong.")
    changed = abs(se_clu / ols[1] - 1.0) > 0.05
    fix_active = (pv["se_kind"] == "day-cluster" and pv["n_clusters"] == D)
    ok = fix_active and changed and inv
    print("checks: fix_active=%s se_materially_changed=%s order_invariant=%s -> %s"
          % (fix_active, changed, inv, ok))
    sys.exit(0 if ok else 1)
