"""test_liquidity_stress.py -- guards the liquidity-stress conditioner + validation battery.

Synthetic TICK-PINNED book (spread always one tick) driven by two INDEPENDENT latent factors:
  stress_t -> total depth (D_t = base*exp(-0.6*stress)); high stress = thin book
  hollow_t -> near-touch shape (marginal depth ~ exp(g*level), g=0.3*hollow); high hollow = hollow inside
The outcome y is the ground-truth cost to walk the ask for a fixed size, which the depth/shape state
must explain while the pinned spread cannot. Checks: spread is dead; the two axes recover their factors
and are orthogonal; the state has large incremental content over spread+Amihud; the resiliency axis adds
beyond the level axis; FPCA recovers the two axes as distinct components; the state predicts realized impact.
"""
import sys
import numpy as np
import pandas as pd
import liquidity_stress as ls

EPS = 1e-12

def make_book(T=4000, N=10, seed=0):
    rng = np.random.default_rng(seed); tick = 0.01
    mid = 100.0 + np.cumsum(rng.normal(0, 0.002, T))          # tiny RW so returns exist
    stress = rng.normal(0, 1, T); hollow = rng.normal(0, 1, T)  # INDEPENDENT
    D = 5000.0 * np.exp(-0.6 * stress)                        # total shares; high stress -> thin
    lv = np.arange(N, dtype=float)
    shape = np.exp((0.3 * hollow)[:, None] * lv[None, :])     # g>0 -> mass shifts DEEP (hollow inside)
    shape /= shape.sum(axis=1, keepdims=True)
    q = np.maximum(np.round(D[:, None] * shape), 1.0)         # (T,N) marginal depth, symmetric sides
    cols = {}
    for i in range(1, N + 1):
        cols[f"SPY_bidprice_{i}"] = mid - tick / 2 - (i - 1) * tick
        cols[f"SPY_askprice_{i}"] = mid + tick / 2 + (i - 1) * tick
        cols[f"SPY_bidquantity_{i}"] = q[:, i - 1]; cols[f"SPY_askquantity_{i}"] = q[:, i - 1]
    df = pd.DataFrame(cols)
    # ground-truth cost (bps) to BUY a fixed size walking the ask
    Sstar = 0.4 * np.median(q.sum(axis=1)); ask_dist = tick / 2 + lv * tick
    cum = np.cumsum(q, axis=1); take = np.clip(Sstar - (cum - q), 0.0, q)
    cost_px = (take * ask_dist[None, :]).sum(axis=1) / np.maximum(take.sum(axis=1), EPS)
    y = cost_px / mid * 1e4                                   # bps
    y = y + rng.normal(0, 0.05 * y.std(), T)
    ret = np.concatenate([[0.0], np.diff(np.log(mid))])
    dvol = 5e6 * np.exp(-0.3 * stress) * np.exp(rng.normal(0, 0.2, T))
    return df, stress, hollow, y, ret, dvol

def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])

if __name__ == "__main__":
    df, stress, hollow, y, ret, dvol = make_book(seed=0)
    day = np.repeat(np.arange(10), len(y) // 10)[:len(y)]      # 10 pseudo-day clusters
    spread = ls.quoted_spread_bps(df, "SPY"); amihud = ls.amihud_illiquidity(ret, dvol)
    st = ls.stress_state(df, "SPY"); di = st["depth_illiq"].to_numpy(); ho = st["hollowness"].to_numpy()
    print("Liquidity-stress conditioner + validation battery")

    # (A) the quoted spread is dead
    sp_std = float(np.nanstd(spread)); r2_sp = corr(y, spread) ** 2
    print("  (A) spread: std=%.2e bps  R^2(y~spread)=%.3f   (pinned -> uninformative)" % (sp_std, r2_sp))

    # (C) axes recover their factors and are orthogonal
    c_di, c_ho = corr(di, stress), corr(ho, hollow)
    cross, leak1, leak2 = abs(corr(di, ho)), abs(corr(di, hollow)), abs(corr(ho, stress))
    print("  (C) corr(depth_illiq,stress)=%.2f  corr(hollowness,hollow)=%+.2f  |corr(axes)|=%.2f  leak=%.2f/%.2f"
          % (c_di, c_ho, cross, leak1, leak2))

    # (B) incremental content of the state over spread+Amihud
    incB = ls.incremental_content(y, np.column_stack([spread, amihud]), np.column_stack([di, ho]), clusters=day)
    print("  (B) state over [spread,amihud]: partial_R2=%.3f  p_F=%.1e  cluster_p=%.1e (G=%d)"
          % (incB["partial_r2"], incB["p_F"], incB["p_wald"], incB["n_clusters"]))

    # (D) resiliency axis adds beyond level+spread
    incD = ls.incremental_content(y, np.column_stack([spread, di]), ho, clusters=day)
    print("  (D) hollowness over [spread,depth_illiq]: partial_R2=%.3f  cluster_p=%.1e"
          % (incD["partial_r2"], incD["p_wald"]))

    # (E) FPCA recovers the two axes as distinct components
    al = ls.fpca_alignment(df, "SPY")
    print("  (E) FPCA: depth~FPC%d (|r|=%.2f, %s)  hollow~FPC%d (|r|=%.2f, %s)  distinct=%s  evr=%s"
          % (al["depth_FPC"], al["corr_depth"], al["labels"][al["depth_FPC"]-1],
             al["hollow_FPC"], al["corr_hollow"], al["labels"][al["hollow_FPC"]-1],
             al["distinct"], np.round(al["evr"], 2)))

    # (F) the state predicts realized impact
    rng = np.random.default_rng(1); realized = y + rng.normal(0, 0.1 * y.std(), len(y))
    lv = ls.lambda_validation(realized, di, ho, clusters=day)
    print("  (F) lambda: coef(depth_illiq)=%.3f t=%.1f  coef(hollowness)=%.3f t=%.1f  R2=%.2f"
          % (lv["coef"][1], lv["t"][1], lv["coef"][2], lv["t"][2], lv["r2"]))

    ok = (sp_std < 0.05 and r2_sp < 0.05                                   # A: spread dead
          and c_di > 0.9 and abs(c_ho) > 0.5 and cross < 0.2 and leak1 < 0.2 and leak2 < 0.2  # C: recover+orthogonal
          and incB["partial_r2"] > 0.3 and incB["p_F"] < 0.01 and incB["p_wald"] < 0.05        # B: headline
          and incD["partial_r2"] > 0.02 and incD["p_wald"] < 0.05                              # D: resiliency adds
          and al["distinct"] and al["corr_depth"] > 0.7 and al["corr_hollow"] > 0.5            # E: FPCA distinct
          and lv["t"][1] > 2.0)                                                                # F: predicts impact
    print("checks: A=%s C=%s B=%s D=%s E=%s F=%s -> %s" % (
        sp_std < 0.05 and r2_sp < 0.05,
        c_di > 0.9 and abs(c_ho) > 0.5 and cross < 0.2,
        incB["partial_r2"] > 0.3 and incB["p_wald"] < 0.05,
        incD["partial_r2"] > 0.02 and incD["p_wald"] < 0.05,
        al["distinct"] and al["corr_depth"] > 0.7 and al["corr_hollow"] > 0.5,
        lv["t"][1] > 2.0, ok))
    sys.exit(0 if ok else 1)
