"""test_xi_symmetry.py -- guards cross_impact.cross_impact_symmetry (Schneider-Lillo 2019).

Two-asset cross-impact with OFI regressors. Under the symmetric (no-dynamic-arbitrage) DGP
Lambda_12 = Lambda_21 the Wald test must not reject; under a clearly asymmetric DGP it must reject.
The per-equation SEs cannot do this -- the test uses the joint cross-equation HAC covariance.
"""
import sys
import numpy as np
import cross_impact as ci

def gen(L12, L21, T, seed, L11=0.5, L22=0.5, noise=1.0):
    rng = np.random.default_rng(seed)
    O = rng.normal(0, 1, (T, 2))
    R = np.zeros((T, 2))
    R[:, 0] = L11 * O[:, 0] + L12 * O[:, 1] + rng.normal(0, noise, T)
    R[:, 1] = L21 * O[:, 0] + L22 * O[:, 1] + rng.normal(0, noise, T)
    return R, O

if __name__ == "__main__":
    T = 4000
    print("Cross-impact symmetry test (Schneider-Lillo): Lambda_12 == Lambda_21 ?")
    Rs, Os = gen(0.30, 0.30, T, seed=1)                         # symmetric
    sym = ci.cross_impact_symmetry(Rs, Os)
    Ra, Oa = gen(0.50, 0.10, T, seed=2)                         # asymmetric
    asy = ci.cross_impact_symmetry(Ra, Oa)
    print("  symmetric DGP (L12=L21=0.30): L12=%.3f L21=%.3f  asym=%+.3f z=%.2f p=%.3f  symmetric=%s"
          % (sym["lambda_12"], sym["lambda_21"], sym["asymmetry"], sym["z"], sym["p_value"], sym["symmetric"]))
    print("  asymmetric DGP (L12=0.50,L21=0.10): L12=%.3f L21=%.3f  asym=%+.3f z=%.2f p=%.4f  symmetric=%s"
          % (asy["lambda_12"], asy["lambda_21"], asy["asymmetry"], asy["z"], asy["p_value"], asy["symmetric"]))

    ok = (sym["symmetric"] is True and sym["p_value"] > 0.05
          and abs(sym["lambda_12"] - 0.30) < 0.05 and abs(sym["lambda_21"] - 0.30) < 0.05
          and asy["symmetric"] is False and asy["p_value"] < 0.01
          and abs(asy["lambda_12"] - 0.50) < 0.05 and abs(asy["lambda_21"] - 0.10) < 0.05)
    print("checks: symmetric_not_rejected=%s asymmetric_rejected=%s coefs_recovered=%s -> %s"
          % (sym["symmetric"] and sym["p_value"] > 0.05, (not asy["symmetric"]) and asy["p_value"] < 0.01,
             abs(asy["lambda_12"] - 0.50) < 0.05 and abs(asy["lambda_21"] - 0.10) < 0.05, ok))
    sys.exit(0 if ok else 1)
