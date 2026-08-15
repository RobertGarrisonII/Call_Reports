#!/usr/bin/env python3
"""run_onset_surface.py -- CLI launcher for the cross-event ONSET stress-surface analysis.

Builds the impact-blind scheduled-release backbone (FOMC by default) over the requested window, derives
the session date universe from it, loads the matching sessions (reusing run_analysis's loader and book
alignment), and runs onset_response.run_onset_surface -- the 2-D liquidity-spiral surface across the LEG
TRIPLE (leg-neutral joint round-trip cost / ES book stress / SPY book stress) on the NOISE-ROBUST
transmission. Prints the decision-legible summary and writes the onset panel CSV.

  # smoke / demo (synthetic deep-book sessions, no MIDAS feed needed):
  python run_onset_surface.py --source demo --demo-events 16
  # real FOMC backbone over a window:
  python run_onset_surface.py --source extract --date-range 2023-01-01:2025-12-31 --categories FOMC
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import run_analysis as ra
import onset_response as onr
import market_shocks as mk
import autoscale

NY = "America/New_York"


def _demo_session(basis_scale, q1, nlev, date, seed):
    """One synthetic deep-book session: SPY tracks ES (beta) plus a basis-scaled idiosyncratic term
    (so the predetermined basis state grows with basis_scale); the touch depth q1 against fixed deeper
    levels sets the hollowness. Constant beta -> no spiral (the demo exercises plumbing, not a finding)."""
    rng = np.random.default_rng(seed); T = 2000; o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, 2e-5, o - 1); es[o - 1:] = rng.normal(0, 1.6e-4, (T - 1) - (o - 1))
    spy = 0.8 * es + rng.normal(0, 2e-5, T - 1) + basis_scale * rng.normal(0, 6e-5, T - 1)
    me = np.exp(np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)])
    ms = np.exp(np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)])
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    for a, mid, tk in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, nlev + 1):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tk
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tk
            qty = float(q1) if i == 1 else 200.0
            cols[f"{a}_bidquantity_{i}"] = qty
            cols[f"{a}_askquantity_{i}"] = qty
    return pd.DataFrame(cols, index=idx), idx[o]


def _demo(n_events, n_levels):
    """Synthetic sessions on a (basis x touch-depth) plane + a matching impact-blind events frame."""
    side = max(2, int(np.ceil(np.sqrt(n_events))))
    basis_grid = np.linspace(0.3, 2.6, side)
    q_grid = np.linspace(20.0, 220.0, side)
    sessions, rows, rel = [], [], {}
    k = 0
    for bs, q1 in itertools.product(basis_grid, q_grid):
        if k >= n_events:
            break
        d = (pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d")
        df, _ = _demo_session(bs, q1, n_levels, d, 200 + k)
        sessions.append((pd.Timestamp(d, tz=NY), "event", df))
        rows.append({"date": d, "category": "FOMC", "selection": "scheduled", "impact_blind": True, "event_et": ""})
        rel[pd.Timestamp(d).normalize()] = df.index[len(df) // 2]
        k += 1
    return sessions, pd.DataFrame(rows), rel


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-event onset stress-surface launcher (leg triple, robust transmission)")
    ap.add_argument("--source", choices=["demo", "load", "extract"], default="demo")
    ap.add_argument("--date-range", default=None, help="YYYY-MM-DD:YYYY-MM-DD window for the scheduled backbone")
    ap.add_argument("--dates", default=None, help="explicit comma-separated session dates (overrides the calendar)")
    ap.add_argument("--categories", default="FOMC",
                    help="comma-separated scheduled categories. RTH-VALID (onset inside the 09:30-16:00 "
                         "extraction window): FOMC(14:00), ISM_MFG/ISM_SVC/CONF_BOARD(10:00, rule-generated), "
                         "JOLTS/NEW_HOME/UMICH(10:00, require load_release_schedule). NOT usable without "
                         "widening the extraction window: the 08:30 block (CPI,NFP,PPI,RETAIL,GDP,PCE,CLAIMS,"
                         "DURABLE,HOUSING,TRADE) and OPEX(09:30) -- their pre-window is empty, so they return "
                         "NaN and are silently dropped.")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--pre-steps", type=int, default=600)
    ap.add_argument("--post-steps", type=int, default=600)
    ap.add_argument("--notional", type=float, default=1e6)
    ap.add_argument("--min-id-strength", type=float, default=0.0)
    ap.add_argument("--demo-events", type=int, default=16)
    ap.add_argument("--max-workers", type=int, default=None,
                    help="extraction parallelism (memory-bound); default: autoscale to RAM/cores/#sessions")
    ap.add_argument("--onset-jobs", type=int, default=None,
                    help="parallel workers for the per-event onset loop (core-bound); default: autoscale to cores")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())

    if args.source == "demo":
        sessions, events, rel = _demo(args.demo_events, args.n_levels)
    else:
        s = e = None
        if args.date_range and ":" in args.date_range:
            s, e = args.date_range.split(":")
        events = mk.shocks_frame(include_scheduled=True, scheduled_categories=cats, start=s, end=e)
        if "impact_blind" in events:
            events = events[events["impact_blind"]]
        if args.dates:
            date_list = [d.strip() for d in args.dates.split(",")]
        else:
            date_list = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in events["date"]})
        a = ra.parse_args([])                                  # full run_analysis defaults
        a.source = args.source; a.dates = date_list; a.date_range = None
        a.interval = "1s"; a.n_levels = args.n_levels; a.output_dir = args.output_dir
        a.max_workers = (args.max_workers if args.max_workers is not None
                         else autoscale.extraction_workers(n_sessions=len(date_list)))
        print("[onset-surface] extraction max_workers=%d (%d sessions, memory-bound)"
              % (a.max_workers, len(date_list)))
        sessions = ra.load_sessions(a)
        rel = None                                            # onset times come from each event's event_et

    n_jobs = args.onset_jobs if args.onset_jobs is not None else autoscale.cpu_jobs()
    if n_jobs > 1 and len(sessions) >= onr._MIN_PARALLEL_EVENTS:
        print("[onset-surface] onset loop: %d threads over %d events (core-bound)" % (n_jobs, len(sessions)))
    rep = onr.run_onset_surface(sessions, events, pre_steps=args.pre_steps, post_steps=args.post_steps,
                                notional=args.notional, n_levels=args.n_levels,
                                min_id_strength=args.min_id_strength, release_by_date=rel, n_jobs=n_jobs)
    print(rep["summary"])
    panel_path = os.path.join(args.output_dir, "onset_surface_panel.csv")
    rep["panel"].to_csv(panel_path, index=False)
    print("\n[onset-surface] panel -> %s  (%d events, %d impact-blind)"
          % (panel_path, rep["n_events"], rep["n_impact_blind"]))
    return rep


if __name__ == "__main__":
    main()
