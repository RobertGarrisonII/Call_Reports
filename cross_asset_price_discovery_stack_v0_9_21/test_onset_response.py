"""test_onset_response.py -- guards the cross-event onset driver.

(A) event_onset_estimate recovers the within-event transmission: a session with a stable structural
    coefficient beta (SPY_ret = beta*ES_ret + eps) and a DIFFERENTIAL variance shift at the onset
    (ES shock variance jumps, SPY idiosyncratic constant) is identified by the Rigobon 2-regime
    solution -> recovered beta ~ true beta, var_ratio > 1, a finite predetermined state.
(B) fit_stress_response recovers f(state)'s slope on the IMPACT-BLIND benchmark, and
    excess_over_benchmark flags injected excess for the SALIENCE-CURATED tail (IM-significant).
(C) cross_event_onset wires sessions + an events frame into a panel end-to-end.
"""
import sys
import numpy as np
import pandas as pd
import onset_response as onr

def make_event_session(beta, T=4000, sig_spy=2e-5, sig_es_pre=2e-5, sig_es_post=1.6e-4,
                       date="2025-06-18", seed=0):
    rng = np.random.default_rng(seed)
    onset = T // 2
    es_ret = np.empty(T - 1)
    es_ret[:onset - 1] = rng.normal(0, sig_es_pre, onset - 1)         # calm pre
    es_ret[onset - 1:] = rng.normal(0, sig_es_post, (T - 1) - (onset - 1))  # shocked post
    spy_ret = beta * es_ret + rng.normal(0, sig_spy, T - 1)            # stable structural beta
    log_es = np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es_ret)]
    log_spy = np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy_ret)]
    mid_es, mid_spy = np.exp(log_es), np.exp(log_spy)
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz="America/New_York")
    cols = {}
    for a, mid, tick in (("SPY", mid_spy, 0.01), ("ES", mid_es, 0.25)):
        for i in range(1, 3):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tick
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tick
            cols[f"{a}_bidquantity_{i}"] = 100.0
            cols[f"{a}_askquantity_{i}"] = 100.0
    return pd.DataFrame(cols, index=idx), idx[onset]

def main():
    ok = True

    # (A) within-event transmission recovery via Rigobon onset
    beta = 0.8
    df, onset_ts = make_event_session(beta, seed=3)
    est = onr.event_onset_estimate(df, onset_ts, pre_steps=1800, post_steps=1800)
    a_trans = abs(est["transmission"] - beta) < 0.15
    a_var = est["var_ratio"] > 2.0 and np.isfinite(est["state"])
    a_id = est["id_strength"] > 0
    print("(A) recovered transmission=%.3f (true %.1f) | var_ratio=%.1f state=%.3g id_strength=%.2g -> %s"
          % (est["transmission"], beta, est["var_ratio"], est["state"], est["id_strength"], a_trans and a_var and a_id))
    ok &= a_trans and a_var and a_id

    # (B) cross-event f(state) slope + excess-over-benchmark
    rng = np.random.default_rng(1)
    b0, b1, excess = 0.5, 1.2, 0.45
    n_ib, n_cur = 24, 8
    s_ib = rng.uniform(0.5, 5.0, n_ib); t_ib = b0 + b1 * s_ib + rng.normal(0, 0.15, n_ib)
    s_cur = rng.uniform(0.5, 5.0, n_cur); t_cur = b0 + b1 * s_cur + excess + rng.normal(0, 0.15, n_cur)
    panel = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n_ib + n_cur, freq="7D"),
        "state": np.r_[s_ib, s_cur], "transmission": np.r_[t_ib, t_cur],
        "impact_blind": [True] * n_ib + [False] * n_cur,
        "selection": ["scheduled"] * n_ib + ["salience_curated"] * n_cur, "id_strength": 1.0})
    fit = onr.fit_stress_response(panel, degree=1)
    b_slope = abs(fit["slope"] - b1) < 0.2 and fit["n"] == n_ib            # benchmark uses impact-blind only
    ex = onr.excess_over_benchmark(panel, fit)
    b_excess = abs(ex["mean"] - excess) < 0.2 and ex["mean"] > 0 and ex["im"]["p"] < 0.05
    print("(B) benchmark slope=%.3f (true %.1f), n=%d | excess mean=%.3f (true %.2f) IM p=%.4f -> %s"
          % (fit["slope"], b1, fit["n"], ex["mean"], excess, ex["im"]["p"], b_slope and b_excess))
    ok &= b_slope and b_excess

    # (C) end-to-end wiring: sessions + events frame -> panel
    sessions, ev_rows = [], []
    for k, bk in enumerate([0.6, 0.9, 1.2, 0.7]):
        d = f"2025-0{k+1}-15"
        dfk, ots = make_event_session(bk, T=3000, date=d, seed=10 + k)
        sessions.append((pd.Timestamp(d, tz="America/New_York"), "event", dfk))
        ev_rows.append({"date": d, "category": "FOMC", "selection": "scheduled",
                        "impact_blind": True, "event_et": ""})
    evf = pd.DataFrame(ev_rows)
    rel = {pd.Timestamp(r["date"]).normalize(): s[2].index[len(s[2]) // 2]
           for r, s in zip(ev_rows, sessions)}
    pan = onr.cross_event_onset(sessions, evf, pre_steps=1400, post_steps=1400, release_by_date=rel)
    c_wire = len(pan) == 4 and pan["transmission"].notna().all() and (pan["impact_blind"]).all()
    c_corr = np.corrcoef(pan["transmission"], [0.6, 0.9, 1.2, 0.7])[0, 1] > 0.8
    print("(C) panel rows=%d all finite=%s | corr(recovered, true beta)=%.2f -> %s"
          % (len(pan), pan["transmission"].notna().all(), c_corr, c_wire and c_corr))
    ok &= c_wire and c_corr

    print("onset-driver checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
