#!/usr/bin/env python3
"""qc_frames.py -- gate the SAVED session frames on the no-crossing invariant, without re-fetching.

Why this exists alongside debug_crossing.py
-------------------------------------------
``debug_crossing.py`` is a ROOT-CAUSE tool: it re-pulls one session's raw messages and works out
*why* a book crosses (was the sequencenumber column present, are the orphaned cancels inside the
window, is a Modify resurrecting a Cancelled order). That costs a full vendor fetch per session --
on a 24-day sample, a second multi-hour pass over the same data the extraction just read.

The GATE question is different and much cheaper: do the frames that were actually saved -- the ones
every downstream table will be estimated on -- satisfy the invariant? That is answerable from the
pickle in seconds, and it has a property the re-fetch gate does not: it checks the object that gets
used. A re-fetch can succeed on a session whose saved frame is broken (or vice versa), and the run
would gate on a book it then throws away.

So: run this first as the gate, and run debug_crossing only on the sessions it flags.

    python qc_frames.py --pickle output/run/1s_aggregated_*.pkl
    python qc_frames.py --pickle out.pkl --crossed-tol 0.01 --out output/run/qc_frames.txt

Exit codes:  0 = every session clean;  2 = at least one violation;  1 = could not run.
Machine-readable: each violating session prints a line beginning 'BAD ' followed by its date.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys

import numpy as np
import pandas as pd

import mstbook_loader as ml


def _load(pattern: str) -> list:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"qc_frames: no pickle matched {pattern!r}")
    out = []
    for p in paths:
        with open(p, "rb") as fh:
            raw = pickle.load(fh)
        for rec in raw:
            if isinstance(rec, (tuple, list)) and len(rec) >= 3:
                out.append((rec[0], rec[1], rec[2]))
            elif isinstance(rec, (tuple, list)) and len(rec) == 2:
                out.append((rec[0], "auto", rec[1]))
            else:
                raise SystemExit(f"qc_frames: unrecognized record shape in {p}: {type(rec)}")
    return out


def qc_sessions(sessions, assets=("SPY", "ES"), crossed_tol: float = 0.005) -> pd.DataFrame:
    """-> one row per session: rows, per-asset finite-snapshot count and crossed fraction, ok."""
    rows = []
    for date, regime, df in sessions:
        rep = ml.session_qc(df, assets=assets, crossed_tol=crossed_tol, date=date)
        row = {"date": str(date), "regime": regime, "rows": rep["n_rows"], "ok": rep["ok"],
               "halt": rep.get("halt_snapshots", 0), "reasons": " | ".join(rep["reasons"])}
        for a in assets:
            row[f"{a}_finite"] = rep[a]["n_finite"]
            row[f"{a}_crossed"] = rep[a]["crossed_frac"]
            row[f"{a}_crossed_open"] = rep[a].get("crossed_frac_ex_halt", float("nan"))
            row[f"{a}_med_bid"] = rep[a]["median_bid"]
        # Rule 201 state. Which sessions were short-sale restricted is a SAMPLE fact the paper has
        # to report: SSR constrains the sell side only, so a restricted day has mechanically
        # asymmetric signed order flow -- the exact quantity Tables 5 and 7 count.
        try:
            import market_state as _ms
            msum = df.attrs.get("market_state") or _ms.summarize(df, asset=assets[0])
            row["ssr_frac"] = msum.get("ssr_frac_on", float("nan")) if msum.get("ssr_known") else float("nan")
            row["luld_known"] = msum.get("luld_known", 0.0)
        except Exception:
            row["ssr_frac"] = float("nan"); row["luld_known"] = 0.0
        # a fetch failure recorded at reconstruction time is the usual cause; surface it here too
        ff = df.attrs.get("fetch_failed") if hasattr(df, "attrs") else None
        row["fetch_failed"] = ",".join(sorted(ff)) if ff else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gate saved session frames on the no-crossing invariant")
    ap.add_argument("--pickle", required=True, help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--assets", default="SPY,ES")
    ap.add_argument("--crossed-tol", type=float, default=0.005,
                    help="fraction of finite snapshots allowed to cross (consolidated multi-venue "
                         "books do cross briefly at sub-second scale, so this is not zero)")
    ap.add_argument("--out", default="", help="also write the table here")
    a = ap.parse_args(argv)

    assets = tuple(s.strip() for s in a.assets.split(",") if s.strip())
    sessions = _load(a.pickle)
    tbl = qc_sessions(sessions, assets=assets, crossed_tol=a.crossed_tol)

    pd.set_option("display.width", 200)
    show = tbl.drop(columns=["reasons"]).copy()
    for a_ in assets:
        for c in (f"{a_}_crossed", f"{a_}_crossed_open"):
            if c in show.columns:
                show[c] = show[c].map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.2%}")
    for c in ("ssr_frac", "luld_known"):
        if c in show.columns:
            show[c] = show[c].map(lambda v: "n/r" if not np.isfinite(v) else f"{v:.0%}")
    body = ["crossed-book gate on the SAVED frames (tolerance %.2f%%)" % (100 * a.crossed_tol),
            "%d session(s) from %s" % (len(tbl), a.pickle), "", show.to_string(), ""]
    bad = tbl[~tbl["ok"]]
    if len(bad):
        body.append("VIOLATIONS -- these frames are what the tables would be estimated on:")
        for d, r in bad.iterrows():
            body.append(f"BAD {d}: {r['reasons']}"
                        + (f"  [fetch failed: {r['fetch_failed']}]" if r["fetch_failed"] else ""))
        body += ["",
                 "OUTSIDE a halt a matching engine cannot cross, so this is a replay fault, not a",
                 "market state -- the crossed_open column already excludes MWCB halt snapshots.",
                 "Run debug_crossing.py on each date above for the root cause (it re-fetches the",
                 "raw messages, which is why it is not the gate), then re-extract or drop the day.",
                 "Do NOT estimate on a session that violates its own invariant."]
    ssr = tbl[tbl["ssr_frac"].fillna(0) > 0] if "ssr_frac" in tbl.columns else tbl.iloc[:0]
    if len(ssr):
        body += ["",
                 "SHORT SALE RESTRICTED sessions (Rule 201): %s"
                 % ", ".join("%s (%.0f%% of the session)" % (d, 100 * r["ssr_frac"])
                             for d, r in ssr.iterrows()),
                 "  Short sales cannot execute or display at or below the NBB, so the sell side is",
                 "  constrained and the buy side is not. Signed order flow on these days is",
                 "  mechanically asymmetric -- report them in the sample appendix and control for",
                 "  them before reading Tables 5 and 7 as evidence of directional tandem trading."]
    if "luld_known" in tbl.columns and float(tbl["luld_known"].fillna(0).max()) == 0.0:
        body += ["",
                 "LULD bands: not reported by this source on any session. The columns exist and are",
                 "all-NaN, so do NOT use them as a control -- an all-NaN regressor still fits."]
    halted = tbl[tbl["halt"] > 0] if "halt" in tbl.columns else tbl.iloc[:0]
    if len(halted):
        body += ["",
                 "MWCB halt sessions (%s):" % ", ".join(str(d) for d in halted.index),
                 "  Exchanges stop matching during a halt but resting orders stay, so a crossed book",
                 "  there is CORRECT -- crossed_open is the column that judges the replay. Those",
                 "  snapshots must still be excluded from every estimate: a halted market has no",
                 "  valid midpoint, and including it adds mechanical comovement, not price discovery."]
    else:
        body.append("every session satisfies the invariant")
    text = "\n".join(body)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 2 if len(bad) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:                                # a gate that cannot run must not pass
        print(f"qc_frames: FAILED to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
