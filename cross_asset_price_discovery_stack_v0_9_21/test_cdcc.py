"""test_cdcc.py -- guards dcc_garch.cdcc_fit (Aielli 2013 corrected cDCC).

Generates standardized residuals from a TRUE cDCC recursion with known (a, b, S), then checks
the estimator inverts the model: recovers a, b, and the targeting correlation S_12. Engle's DCC
is fit on the same data and reported alongside (its correlation targeting is inconsistent, the
defect cDCC corrects).
"""
import sys
import numpy as np
import dcc_garch as dg

def sim_cdcc(T, a, b, S, seed):
    """Draw Z_t ~ N(0, R_t) online from the true cDCC; return Z and the realized mean true corr."""
    rng = np.random.default_rng(seed); k = S.shape[0]
    Q = S.copy(); Z = np.zeros((T, k)); r12 = np.zeros(T)
    for t in range(T):
        d = np.sqrt(np.maximum(np.diag(Q), 1e-12))
        Rt = np.clip(Q / np.outer(d, d), -1.0, 1.0); np.fill_diagonal(Rt, 1.0)
        r12[t] = Rt[0, 1]
        Z[t] = np.linalg.cholesky(Rt) @ rng.standard_normal(k)
        zstar = d * Z[t]
        Q = (1 - a - b) * S + a * np.outer(zstar, zstar) + b * Q
    return Z, float(r12.mean())

if __name__ == "__main__":
    a_t, b_t, s12 = 0.06, 0.90, 0.5
    S = np.array([[1.0, s12], [s12, 1.0]])
    Z, true_avg = sim_cdcc(4000, a_t, b_t, S, seed=3)
    print("Aielli cDCC recovery  (true a=%.2f b=%.2f S12=%.2f, T=%d)" % (a_t, b_t, s12, len(Z)))

    fit = dg.cdcc_fit(Z)
    eng = dg.dcc_fit(Z)
    s12_hat = fit["S"][0, 1]
    cdcc_avg = float(fit["R"][:, 0, 1].mean()); eng_avg = float(eng["R"][:, 0, 1].mean())
    print("  cDCC : a=%.3f b=%.3f  S12=%.3f  avg_corr=%.3f" % (fit["a"], fit["b"], s12_hat, cdcc_avg))
    print("  Engle: a=%.3f b=%.3f             avg_corr=%.3f" % (eng["a"], eng["b"], eng_avg))
    print("  true average realized corr = %.3f" % true_avg)
    print("  |avg_corr - true|:  cDCC=%.3f  Engle=%.3f" % (abs(cdcc_avg - true_avg), abs(eng_avg - true_avg)))

    ok = (abs(fit["a"] - a_t) < 0.05 and abs(fit["b"] - b_t) < 0.07 and abs(s12_hat - s12) < 0.08)
    print("checks: a_ok=%s b_ok=%s S12_ok=%s -> %s"
          % (abs(fit["a"] - a_t) < 0.05, abs(fit["b"] - b_t) < 0.07, abs(s12_hat - s12) < 0.08, ok))
    sys.exit(0 if ok else 1)
