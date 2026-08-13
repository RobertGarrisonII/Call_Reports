"""test_wild_cluster.py -- guards inference.wild_cluster_bootstrap and inference.ibragimov_muller.

Canonical Cameron-Gelbach-Miller (2008) size experiment: a regressor with a cluster-level
component and a cluster random effect in the error. Under the null (no slope) the iid t
over-rejects massively; the wild cluster bootstrap (Webb 6-point, few clusters) and the
Ibragimov-Muller group t are near the nominal 5%; both retain power under the alternative.
"""
import sys
import numpy as np
from scipy import stats
import inference as inf

def build(G, n, beta, sigma_a, rng):
    zg = rng.normal(0, 1, G)                       # cluster component of x
    ag = rng.normal(0, sigma_a, G)                 # cluster random effect in the error
    cl = np.repeat(np.arange(G), n)
    x = zg[cl] + rng.normal(0, 1, G * n)
    y = beta * x + ag[cl] + rng.normal(0, 1, G * n)
    X = np.column_stack([np.ones(G * n), x])
    return y, X, cl

def im_per_cluster(y, X, cl):
    bg = []
    for g in np.unique(cl):
        m = cl == g; Xg = X[m]
        bg.append(np.linalg.lstsq(Xg, y[m], rcond=None)[0][1])
    return inf.ibragimov_muller(np.array(bg))

def run(G, n, beta, sigma_a, R, B, seed0):
    tcrit = stats.t.ppf(0.975, df=G - 1)
    rej = {"naive": 0, "crve": 0, "wcb": 0, "im": 0}
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        y, X, cl = build(G, n, beta, sigma_a, rng)
        b, se = inf.ols_naive_se(y, X)
        rej["naive"] += abs(b[1] / (se[1] + 1e-12)) > 1.96
        cr = inf.cluster_robust_ols(y, X, cl)
        rej["crve"] += abs(cr["t"][1]) > tcrit
        wcb = inf.wild_cluster_bootstrap(y, X, cl, test_idx=1, n_boot=B, seed=seed0 + r)
        rej["wcb"] += wcb["p"] < 0.05
        rej["im"] += im_per_cluster(y, X, cl)["p"] < 0.05
    return {k: v / R for k, v in rej.items()}

if __name__ == "__main__":
    G, n, sigma_a, B = 10, 120, 0.7, 199
    print("Wild cluster bootstrap / Ibragimov-Muller size & power  (G=%d clusters, n=%d/cluster)" % (G, n))
    nullr = run(G, n, beta=0.0, sigma_a=sigma_a, R=160, B=B, seed0=100)
    print("  NULL (beta=0), rejection at nominal 5%%:")
    print("     iid naive t      : %5.1f%%   <- treats %d ticks as independent" % (100 * nullr["naive"], G * n))
    print("     CRVE t (t_{G-1}) : %5.1f%%   <- cluster-robust but few-cluster biased" % (100 * nullr["crve"]))
    print("     wild cluster boot: %5.1f%%   (Webb 6-point)" % (100 * nullr["wcb"]))
    print("     Ibragimov-Muller : %5.1f%%" % (100 * nullr["im"]))
    powr = run(G, n, beta=0.4, sigma_a=sigma_a, R=80, B=B, seed0=900)
    print("  POWER (beta=0.4), rejection:")
    print("     wild cluster boot: %5.1f%%   |   Ibragimov-Muller: %5.1f%%" % (100 * powr["wcb"], 100 * powr["im"]))
    ok = (nullr["naive"] > 0.25                                   # iid badly over-rejects
          and 0.02 <= nullr["wcb"] <= 0.115                       # WCB ~ nominal
          and 0.01 <= nullr["im"] <= 0.12                         # IM ~ nominal (conservative)
          and nullr["wcb"] < nullr["naive"]                       # WCB strictly better than iid
          and powr["wcb"] > 0.5 and powr["im"] > 0.4)             # both retain power
    print("checks: iid_overrejects=%s wcb_nominal=%s im_nominal=%s wcb_has_power=%s -> %s"
          % (nullr["naive"] > 0.25, 0.02 <= nullr["wcb"] <= 0.115,
             0.01 <= nullr["im"] <= 0.12, powr["wcb"] > 0.5, ok))
    sys.exit(0 if ok else 1)
