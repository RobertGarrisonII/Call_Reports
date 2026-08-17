"""test_fpca_interp.py -- guards functional_liquidity.interpret_components.

Synthetic book-depth curves built from a known LEVEL factor (constant loading across price levels)
and a known SLOPE factor (linear, sign-changing loading), with the level factor carrying more
variance. FPCA must recover the two shapes; interpret_components must label them level/slope and
report variance-explained matching the injected proportions.
"""
import sys
import numpy as np
import functional_liquidity as fl

if __name__ == "__main__":
    M, T = 10, 6000
    x = np.arange(M, dtype=float)
    level = np.ones(M) / np.sqrt(M)                              # constant loading (no sign change)
    slope = (x - x.mean()); slope = slope / np.linalg.norm(slope)  # linear loading (one sign change)
    rng = np.random.default_rng(0)
    sL, sS = 2.0, 1.0                                            # level factor carries more variance
    Y = (5.0 + rng.normal(0, sL, (T, 1)) * level + rng.normal(0, sS, (T, 1)) * slope
         + rng.normal(0, 0.05, (T, M)))
    inj = sL ** 2 / (sL ** 2 + sS ** 2)                         # injected level share (ex-noise)
    print("FPCA component interpretation  (injected level share ~%.2f, slope ~%.2f)" % (inj, 1 - inj))

    res = fl.fpca(Y, n_components=3)
    interp = fl.interpret_components(res)
    for c in interp:
        print("  FPC%d: %-9s  variance_explained=%.2f  sign_changes=%d"
              % (c["component"], c["label"], c["variance_explained"], c["n_sign_changes"]))

    c1, c2 = interp[0], interp[1]
    ok = (c1["label"] == "level" and c2["label"] == "slope"
          and c1["variance_explained"] > c2["variance_explained"]
          and 0.70 <= c1["variance_explained"] <= 0.90
          and (c1["variance_explained"] + c2["variance_explained"]) > 0.9)
    print("checks: FPC1=level=%s FPC2=slope=%s level_dominates=%s share~injected=%s -> %s"
          % (c1["label"] == "level", c2["label"] == "slope",
             c1["variance_explained"] > c2["variance_explained"],
             0.70 <= c1["variance_explained"] <= 0.90, ok))
    sys.exit(0 if ok else 1)
