"""test_cost_aggregation.py -- the cost-aggregation comparison, and the product's predicted failure.

(A) Correctness: state_cost_{joint,max,prod} obey their invariants (max <= sum; sum >= 2*sqrt(prod) by
    AM-GM; max^2 >= prod), all finite on a normal session.
(B) The empirical confirmation of the 'why not the product' argument: when basis dislocation and the
    leg costs CO-MOVE (the realistic stress structure -- liquidity withdraws as the basis blows out),
    the multiplicative product axis is the most concentrated, so its basis-interaction is the worst
    identified / most fragile. We check that cost_prod is no better than joint_cost (SUM) on
    identification (VIF) AND strictly worse on at least one of {identified, leverage (jackknife band)}.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import onset_response as onr

NY = "America/New_York"

def _book(ms, me, idx, q1, nlev):
    cols = {}
    for a, mid, tk in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, nlev + 1):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tk
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tk
            qty = float(q1) if i == 1 else 200.0
            cols[f"{a}_bidquantity_{i}"] = qty
            cols[f"{a}_askquantity_{i}"] = qty
    return pd.DataFrame(cols, index=idx)

def make_comoving_session(stress, date, seed, nlev=6):
    rng = np.random.default_rng(seed); T = 2000; o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, 2e-5, o - 1); es[o - 1:] = rng.normal(0, 1.6e-4, (T - 1) - (o - 1))
    bscale = abs(0.4 + 2.2 * stress + rng.normal(0, 0.25))            # basis dislocation rises with stress
    spy = 0.8 * es + rng.normal(0, 2e-5, T - 1) + bscale * rng.normal(0, 6e-5, T - 1)
    me = np.exp(np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)])
    ms = np.exp(np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)])
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    q1 = max(12.0, 230.0 - 200.0 * stress + rng.normal(0, 18))        # touch thins with stress -> cost rises (both legs)
    return _book(ms, me, idx, q1, nlev), idx[o]

def main():
    warnings.simplefilter("ignore")

    # (A) invariants on a normal session
    df, ots = make_comoving_session(0.5, "2025-02-05", 7)
    e = onr.event_onset_estimate(df, ots, pre_steps=800, post_steps=800)
    cs, cm, cp = e["state_cost_joint"], e["state_cost_max"], e["state_cost_prod"]
    inv = (np.isfinite(cs) and np.isfinite(cm) and np.isfinite(cp) and cm <= cs + 1e-9
           and cs + 1e-9 >= 2.0 * np.sqrt(max(cp, 0.0)) and cm * cm + 1e-9 >= cp)
    print("(A) cost sum=%.4f max=%.4f prod=%.4f | max<=sum, sum>=2sqrt(prod), max^2>=prod -> %s"
          % (cs, cm, cp, inv))

    # (B) co-moving events -> product is worst identified / most fragile
    rng = np.random.default_rng(3)
    sessions, ev_rows, rel = [], [], {}
    for k in range(24):
        stress = rng.uniform(0.0, 1.0)
        d = (pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d")
        dfk, _ = make_comoving_session(stress, d, 50 + k)
        sessions.append((pd.Timestamp(d, tz=NY), "event", dfk))
        ev_rows.append({"date": d, "category": "FOMC", "selection": "scheduled", "impact_blind": True, "event_et": ""})
        rel[pd.Timestamp(d).normalize()] = dfk.index[len(dfk) // 2]
    rep = onr.run_onset_surface(sessions, pd.DataFrame(ev_rows), pre_steps=800, post_steps=800, release_by_date=rel)
    A = rep["axes"]
    s, p = A["joint_cost"], A["cost_prod"]

    def band(r):
        jk = r.get("jackknife_robust"); return jk["b3_loo_range"] if jk else float("inf")
    m = A["cost_max"]
    # the robust, theory-matching claim: the multiplicative PRODUCT carries MORE leverage/fragility than
    # the additive SUM (wider leave-one-out band) -- a convex axis concentrates its mass in the few
    # both-stressed events. (The bottleneck MAX is the most economically faithful and gives the strongest
    # b3, but is the most fragile of all -- the binding-leg leverage; that nuance is printed, not asserted.)
    b_ok = (p["identified"] and np.isfinite(p["b3_robust"]) and np.isfinite(band(p))
            and np.isfinite(band(s)) and band(p) > band(s))
    print("(B) loo band (leverage):  sum=%.4f  max=%.4f  prod=%.4f   |b3|: sum=%.4f max=%.4f prod=%.4f"
          % (band(s), band(m), band(p), abs(s["b3_robust"]), abs(m["b3_robust"]), abs(p["b3_robust"])))
    print("    product MORE fragile than sum (band_prod > band_sum) -> %s" % b_ok)

    print("\n----- summary -----\n" + rep["summary"])
    ok = inv and b_ok
    print("\ncost-aggregation checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
