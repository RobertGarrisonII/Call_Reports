"""test_onset_sensitivity.py -- the diagnostics must DETECT the failure modes, not merely run.

(A) post_window_profile DETECTS cascade drift: in a session whose post-onset transmission holds at b1
    for the first L bars (pre-feedback) then shifts to b2 (cascade, ES variance elevated throughout),
    the profiled transmission sits on a b1 plateau for post_steps<=L and drifts upward for post>>L.
(B) id_strength flags weak ID: a COMMON-SCALE onset (ES shock and SPY idiosyncratic variance both
    scale by the same factor -> Om_post ~ c*Om_pre -> eigenvalues coincide) is reported weakly
    identified (id_strength ~ 0) DESPITE a large variance ratio, while a differential-het onset is
    strongly identified; id_strength_profile / id_threshold_sweep act on this correctly.
(C) window_sweep runs end-to-end over sessions -> a grid DataFrame of finite slopes.
(D) cholesky_bracket computes the order-free coefficient next to two DIFFERING Cholesky orderings.
"""
import sys
import numpy as np
import pandas as pd
import onset_sensitivity as osn
import onset_response as onr

NY = "America/New_York"

def _book(ms, me, idx):
    cols = {}
    for a, mid, tick in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, 3):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tick
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tick
            cols[f"{a}_bidquantity_{i}"] = 100.0
            cols[f"{a}_askquantity_{i}"] = 100.0
    return pd.DataFrame(cols, index=idx)

def _mids(es, spy):
    le = np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)]
    ls = np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)]
    return np.exp(ls), np.exp(le)

def make_event_session(beta, T=4000, sig_spy=2e-5, sp=2e-5, sq=1.6e-4, date="2025-06-18", seed=0):
    rng = np.random.default_rng(seed); o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, sp, o - 1); es[o - 1:] = rng.normal(0, sq, (T - 1) - (o - 1))
    spy = beta * es + rng.normal(0, sig_spy, T - 1)
    ms, me = _mids(es, spy)
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    return _book(ms, me, idx), idx[o]

def make_cascade_session(b1=0.8, b2=1.6, T=5000, L=500, sig_spy=2e-5, sp=2e-5, sq=1.6e-4, date="2025-06-18", seed=0):
    rng = np.random.default_rng(seed); o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, sp, o - 1); es[o - 1:] = rng.normal(0, sq, (T - 1) - (o - 1))
    beta = np.empty(T - 1); beta[:o - 1] = b1; ps = o - 1; beta[ps:ps + L] = b1; beta[ps + L:] = b2
    spy = beta * es + rng.normal(0, sig_spy, T - 1)
    ms, me = _mids(es, spy)
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    return _book(ms, me, idx), idx[o]

def make_common_scale_session(beta=0.8, c=9.0, T=4000, sig=2e-5, date="2025-06-18", seed=0):
    rng = np.random.default_rng(seed); o = T // 2
    scale = np.ones(T - 1); scale[o - 1:] = np.sqrt(c)
    es = rng.normal(0, sig, T - 1) * scale
    spy = beta * es + rng.normal(0, sig, T - 1) * scale
    ms, me = _mids(es, spy)
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    return _book(ms, me, idx), idx[o]

def main():
    ok = True

    # (A) cascade contamination shows as INSTABILITY: the detector must fire on a within-post
    #     structural break and stay quiet on a clean constant-beta session (window discrimination).
    dfc, otc = make_cascade_session(b1=0.8, b2=1.6, L=500, seed=4)         # break: b1 then b2 in-post
    pc = osn.post_window_profile(dfc, otc, post_grid=(200, 500, 1500, 2400), pre_steps=2000)
    c_plat = float(pc.loc[pc.post_steps == 200, "transmission"].iloc[0])   # within the b1 plateau
    c_far = float(pc.loc[pc.post_steps == 2400, "transmission"].iloc[0])   # window has eaten the break
    cascade_fires = abs(c_plat - 0.8) < 0.2 and abs(c_far - c_plat) > 0.3
    dfk, otk = make_event_session(0.8, T=5000, seed=8)                     # clean: constant beta, no break
    pk = osn.post_window_profile(dfk, otk, post_grid=(200, 500, 1500, 2400), pre_steps=2000)
    k_plat = float(pk.loc[pk.post_steps == 200, "transmission"].iloc[0])
    k_far = float(pk.loc[pk.post_steps == 2400, "transmission"].iloc[0])
    clean_quiet = abs(k_far - k_plat) < 0.2 and abs(k_far - 0.8) < 0.2
    a_ok = cascade_fires and clean_quiet
    print("(A) cascade plateau=%.3f far=%.3f |dep|=%.3f fires=%s | clean plateau=%.3f far=%.3f quiet=%s -> %s"
          % (c_plat, c_far, abs(c_far - c_plat), cascade_fires, k_plat, k_far, clean_quiet, a_ok))
    ok &= a_ok

    # (B) weak-ID detection
    dfw, otw = make_common_scale_session(c=9.0, seed=5)
    ew = onr.event_onset_estimate(dfw, otw, pre_steps=1800, post_steps=1800)
    dfs, ots2 = make_event_session(0.8, seed=6)
    es_ = onr.event_onset_estimate(dfs, ots2, pre_steps=1800, post_steps=1800)
    b_weak = ew["id_strength"] < 1e-2 and ew["var_ratio"] > 2.0
    b_strong = es_["id_strength"] > 1e-2
    panel = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="7D"),
        "state": np.linspace(0.5, 5, 10),
        "transmission": np.linspace(0.5, 5, 10) * 1.2 + 0.5,
        "id_strength": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1e-6],
        "var_ratio": 10.0, "impact_blind": True})
    pid = osn.id_strength_profile(panel, weak_tol=1e-3)
    sw = osn.id_threshold_sweep(panel, thresholds=(0.0, 1e-3, 1e-1))
    n_lo = int(sw.loc[sw.min_id_strength == 0.0, "n"].iloc[0])
    n_hi = int(sw.loc[sw.min_id_strength == 1e-1, "n"].iloc[0])
    b_prof = abs(pid["share_weak"] - 0.4) < 1e-9
    b_sw = n_lo == 10 and n_hi == 6
    print("(B) weak id=%.2g (var_ratio=%.1f) strong id=%.2g | share_weak=%.2f sweep n:%d->%d -> %s"
          % (ew["id_strength"], ew["var_ratio"], es_["id_strength"], pid["share_weak"], n_lo, n_hi,
             b_weak and b_strong and b_prof and b_sw))
    ok &= b_weak and b_strong and b_prof and b_sw

    # (C) window_sweep end-to-end
    sessions, ev_rows, rel = [], [], {}
    for k, bk in enumerate([0.6, 0.9, 1.2, 0.7]):
        d = f"2025-0{k+1}-15"; dk, _ = make_event_session(bk, T=3000, date=d, seed=20 + k)
        sessions.append((pd.Timestamp(d, tz=NY), "event", dk))
        ev_rows.append({"date": d, "category": "FOMC", "selection": "scheduled", "impact_blind": True, "event_et": ""})
        rel[pd.Timestamp(d).normalize()] = dk.index[len(dk) // 2]
    ws = osn.window_sweep(sessions, pd.DataFrame(ev_rows), pre_grid=(600, 1200), post_grid=(300, 800), release_by_date=rel)
    c_ok = len(ws) == 4 and ws["slope"].notna().all()
    print("(C) window_sweep rows=%d finite slopes=%s -> %s" % (len(ws), ws["slope"].notna().all(), c_ok))
    ok &= c_ok

    # (D) cholesky bracket
    dfb, otb = make_event_session(0.8, seed=7)
    br = osn.cholesky_bracket(dfb, otb, pre_steps=1800, post_steps=300)
    fin = all(np.isfinite([br["rigobon"], br["chol_resp_first"], br["chol_imp_first"]]))
    diff = abs(br["chol_resp_first"] - br["chol_imp_first"]) > 1e-6
    d_ok = fin and diff and isinstance(br["within_bracket"], bool)
    print("(D) rigobon=%.3f chol[resp,imp]=[%.3f,%.3f] differ=%s within=%s -> %s"
          % (br["rigobon"], br["chol_resp_first"], br["chol_imp_first"], diff, br["within_bracket"], d_ok))
    ok &= d_ok

    print("onset-sensitivity checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
