"""test_crossing_qc.py -- known-answer guard for the reconstruction-QC columns on the onset panel.

The failure mode being guarded is NOT "does a crossed-book number get printed" but "does it get
printed for the bars the estimate is actually built from". Crossing concentrates where message rates
spike -- i.e. at the onset -- so an event can look acceptable session-wide while its own pre/post
windows are badly crossed, and a session-level rate would wave it through. Three sessions with
ANALYTICALLY KNOWN crossed counts are constructed:

  (A) clean              -- no crossed bars            -> window ~0,   session ~0,   not flagged
  (B) onset-concentrated -- crossed only around onset  -> window >> session          (the detection)
  (C) uniform            -- same COUNT spread evenly   -> window ~= session          (negative control)

(C) is what makes (B) meaningful: it proves the window/session gap is a real localization signal and
not an artifact the measure manufactures from any crossing at all. Each expected rate is computed from
the injected bar counts and the estimator's own window slice, so the test fails if the QC silently
measures the wrong rows.
"""
import sys
import warnings

import numpy as np
import pandas as pd

import onset_response as onr

NY = "America/New_York"
T = 2000                      # session bars
ONSET_POS = T // 2            # onset at the midpoint
STEPS = 200                   # pre_steps == post_steps
NCROSS = 200                  # crossed bars injected in (B) and (C) -- SAME count, different placement


def make_session(cross_rows=(), date="2025-01-06", seed=0):
    """A well-formed 6-level session; `cross_rows` are frame positions where the SPY top is forced
    CROSSED (bid1 > ask1) by swapping the touch quotes. The swap leaves the mid unchanged, so the
    estimator still runs and the QC is measured independently of the return series."""
    rng = np.random.default_rng(seed)
    es = np.empty(T - 1)
    es[:ONSET_POS - 1] = rng.normal(0, 2e-5, ONSET_POS - 1)
    es[ONSET_POS - 1:] = rng.normal(0, 1.6e-4, (T - 1) - (ONSET_POS - 1))
    spy = 0.8 * es + rng.normal(0, 2e-5, T - 1)
    me = np.exp(np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)])
    ms = np.exp(np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)])
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    for a, mid, tk in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, 7):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tk
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tk
            cols[f"{a}_bidquantity_{i}"] = 100.0
            cols[f"{a}_askquantity_{i}"] = 100.0
    df = pd.DataFrame(cols, index=idx)
    if len(cross_rows):
        r = np.asarray(list(cross_rows), dtype=int)
        b = df["SPY_bidprice_1"].to_numpy(float).copy()
        a_ = df["SPY_askprice_1"].to_numpy(float).copy()
        b[r], a_[r] = a_[r], b[r]                      # swap the touch -> bid1 > ask1 on exactly these bars
        df["SPY_bidprice_1"], df["SPY_askprice_1"] = b, a_
    return df, idx[ONSET_POS]


def window_slice():
    """Reproduce the estimator's own window slice so expected rates are computed on the same rows."""
    ridx = np.arange(T - 1)                                   # returns align to idx[1:]
    pre_all = ridx[ridx < ONSET_POS - 1 + 1]                  # ridx position j <-> frame position j+1
    pre_all = np.where(np.arange(T - 1) + 1 < ONSET_POS)[0]
    post_all = np.where(np.arange(T - 1) + 1 >= ONSET_POS)[0]
    pre_i, post_i = pre_all[-STEPS:], post_all[:STEPS]
    return slice(int(pre_i[0]), int(post_i[-1]) + 2)


def run_panel(sessions_spec):
    """Build the panel through the real driver so the plumbing (estimate -> row -> column) is exercised."""
    sessions, ev_rows, rel = [], [], {}
    for k, (name, cross_rows) in enumerate(sessions_spec):
        d = pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * k)
        ds = d.strftime("%Y-%m-%d")
        dfk, _ = make_session(cross_rows, date=ds, seed=100 + k)
        sessions.append((pd.Timestamp(ds, tz=NY), "event", dfk))
        ev_rows.append({"date": ds, "category": "FOMC", "selection": "scheduled",
                        "impact_blind": True, "event_et": ""})
        rel[pd.Timestamp(ds).normalize()] = dfk.index[ONSET_POS]
    panel = onr.cross_event_onset(sessions, pd.DataFrame(ev_rows), pre_steps=STEPS, post_steps=STEPS,
                                  release_by_date=rel)
    return panel


def main():
    warnings.simplefilter("ignore")
    ok = True
    sl = window_slice()
    win_rows = np.arange(sl.start, sl.stop)
    n_win = len(win_rows)

    # (B) onset-concentrated: a contiguous band centred on the onset, fully inside the window
    b_rows = np.arange(ONSET_POS - NCROSS // 2, ONSET_POS + NCROSS // 2)
    # (C) uniform: same COUNT, evenly spread across the whole session
    c_rows = np.linspace(0, T - 1, NCROSS, dtype=int)

    exp = {}
    for tag, rows in (("clean", np.array([], int)), ("onset", b_rows), ("uniform", c_rows)):
        in_win = np.isin(rows, win_rows).sum()
        exp[tag] = (in_win / n_win, len(rows) / T)            # (expected window rate, expected session rate)

    panel = run_panel([("clean", []), ("onset", b_rows), ("uniform", c_rows)])
    if len(panel) != 3:
        print("FAIL: expected 3 panel rows, got %d" % len(panel))
        return False
    got = {t: (float(panel["crossed_max"].iloc[i]), float(panel["crossed_resp_sess"].iloc[i]))
           for i, t in enumerate(["clean", "onset", "uniform"])}

    print("  %-9s %12s %12s %12s %12s" % ("session", "win_expect", "win_got", "sess_expect", "sess_got"))
    for t in ("clean", "onset", "uniform"):
        (we, se), (wg, sg) = exp[t], got[t]
        hit = np.isclose(wg, we, atol=0.005) and np.isclose(sg, se, atol=0.005)
        ok &= hit
        print("  %-9s %11.2f%% %11.2f%% %11.2f%% %11.2f%%  %s"
              % (t, 100 * we, 100 * wg, 100 * se, 100 * sg, "ok" if hit else "MISMATCH"))
    print("(A) window & session rates recover the injected counts -> %s" % ok)

    # (B) THE detection: onset-concentrated crossing must show window >> session, while the uniform
    # control with the IDENTICAL crossed count must not. This is the property a session-level QC lacks.
    ratio_onset = got["onset"][0] / max(got["onset"][1], 1e-12)
    ratio_unif = got["uniform"][0] / max(got["uniform"][1], 1e-12)
    detect = ratio_onset > 3.0 and ratio_unif < 1.5
    ok &= detect
    print("(B) localization: onset window/session=%.2fx (want >3), uniform=%.2fx (want <1.5) -> %s"
          % (ratio_onset, ratio_unif, detect))

    # (C) a session-level-only view would have MISSED the onset event at the contaminated threshold
    missed_by_session = got["onset"][1] <= onr._XED_SUSPECT < got["onset"][0]
    ok &= missed_by_session
    print("(C) session-level rate (%.2f%%) is under the contamination bar while the window rate (%.2f%%) "
          "is over -> session-only QC would pass it: %s"
          % (100 * got["onset"][1], 100 * got["onset"][0], missed_by_session))

    # (D) the summary block must actually flag it, and must stay quiet on a clean panel
    lines = "\n".join(onr._crossing_qc_lines(panel))
    flagged = "CONTAMINATED" in lines and "ACTION" in lines
    clean_panel = run_panel([("clean", []), ("clean2", [])])
    clean_lines = "\n".join(onr._crossing_qc_lines(clean_panel))
    quiet = "ACTION" not in clean_lines and "2/2 events clean" in clean_lines
    ok &= flagged and quiet
    print("(D) summary flags contamination=%s ; stays quiet on a clean panel=%s" % (flagged, quiet))

    # (E) absent book columns must degrade to NaN, never raise
    bare = pd.DataFrame({"x": [1.0, 2.0]})
    safe = np.isnan(onr._crossed_frac(bare, "SPY"))
    ok &= safe
    print("(E) missing bid1/ask1 columns -> NaN, no exception: %s" % safe)

    print("\ncrossing-QC checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
