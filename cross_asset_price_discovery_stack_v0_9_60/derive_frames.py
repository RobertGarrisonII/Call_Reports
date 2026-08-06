#!/usr/bin/env python3
"""derive_frames.py -- pull once, process twice: derive the coarse-grid frames from the fine ones.

The two-grid runs paid the vendor twice: STAGE 2 fetched and replayed every session's messages to
build the 1s books, then STAGE 2b fetched and replayed the SAME messages to build the 10ms books.
The pull is the dominant cost (10-25 minutes of vendor I/O per session per grid), and it is
redundant by construction: the 1s grid points are a strict subset of the 10ms grid, and a book row
is the state at-or-before its timestamp (searchsorted side='right' - 1 in both book builders), so
the coarse frame is EXACTLY derivable from the fine one -- provided the derivation is
column-aware:

  * BOOK and MARKET-STATE columns (bid/ask price+quantity per level, nbbo, mid, ssr/band columns)
    are step functions sampled at grid points: the coarse row IS the fine row at the same
    timestamp. Row selection, exact.
  * FLOW COUNT/VOLUME columns (``*_trade_buy``, ``*_trade_sell``, ``*_trade_buy_volume``,
    ``*_trade_sell_volume``) are per-bin aggregates over left-labeled bins [T, T+dt): the coarse
    bin is the SUM of its sub-bins. Exact -- except the FINAL bar (see below).
  * ``*_trade_px`` (last trade price in bin): last non-NaN sub-bin value. Exact.
  * ``*_trade_vwap``: recombined volume-weighted, with weights = buy_volume + sell_volume of each
    sub-bin. APPROXIMATE when the tape carries prints the classifier could not sign (sign = 0
    trades carry volume in the true VWAP weight but not in the frame's signed volumes); the
    verification mode measures the actual discrepancy against a directly-extracted frame.

  KNOWN BOUNDARY CASE -- the final bar. The session grid ends AT 16:00:00, so the fine frame's
  last row covers [16:00:00.000, 16:00:00.010) while a directly-extracted coarse frame's last bin
  covers [16:00:00, 16:00:01) and can include post-close prints from the rest of that second. The
  derived final bar can therefore UNDERCOUNT final-bar flow relative to a direct extraction; book
  columns are unaffected. The verification mode reports the final bar separately for this reason.

Cache semantics mirror the extractor exactly: derived frames are written into the same
interval-keyed session cache (``session_<ymd>_<interval>_L<levels>_<clock>_<src>.pkl``), only when
they pass ``session_qc``, and never overwrite an existing file -- a directly-extracted cache
always wins. ``--verify-existing`` compares the derived frame against any direct cache already
present, which on a workbench with a populated 1s cache validates the equivalence on the real
feed, all sessions, before the derivation is ever trusted for a session that has no direct frame.

    python derive_frames.py --pickle frames_10ms.pkl --target 1s \
        --cache-dir output/extract_cache --levels 10 --clock receipt --verify-existing

Exit codes: 0 = derived (and verified, where possible); 2 = grid misalignment (target not an
integer multiple of the fine grid -- nothing derived); 3 = verification found a real mismatch.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

import mstbook_loader as ml

SUM_SUFFIXES = ("_trade_buy", "_trade_sell", "_trade_buy_volume", "_trade_sell_volume")
PX_SUFFIX = "_trade_px"
VWAP_SUFFIX = "_trade_vwap"


def _interval_seconds(interval: str) -> float:
    from run_analysis import interval_seconds
    return float(interval_seconds(interval))


def derive_coarse_frame(fine: pd.DataFrame, target: str, fine_interval: str | None = None) -> pd.DataFrame:
    """One session: coarse frame derived from the fine frame. Raises ValueError on misalignment."""
    if not isinstance(fine.index, pd.DatetimeIndex) or len(fine) < 3:
        raise ValueError("fine frame needs a DatetimeIndex with >2 rows")
    dt_f = (_interval_seconds(fine_interval) if fine_interval
            else float(np.median(np.diff(fine.index.view("int64"))) / 1e9))
    dt_c = _interval_seconds(target)
    k = dt_c / dt_f
    if dt_c <= dt_f or abs(k - round(k)) > 1e-9:
        raise ValueError("target %s is not a coarser integer multiple of the fine grid (%.4gs)"
                         % (target, dt_f))
    freq = pd.tseries.frequencies.to_offset(target)
    coarse_idx = pd.date_range(fine.index[0], fine.index[-1], freq=freq)
    missing = coarse_idx.difference(fine.index)
    if len(missing):
        raise ValueError("coarse grid points absent from the fine index (misaligned grids): "
                         + str(missing[:3].tolist()))
    out = pd.DataFrame(index=coarse_idx)
    sum_cols = [c for c in fine.columns if c.endswith(SUM_SUFFIXES)]
    px_cols = [c for c in fine.columns if c.endswith(PX_SUFFIX)]
    vwap_cols = [c for c in fine.columns if c.endswith(VWAP_SUFFIX)]
    state_cols = [c for c in fine.columns if c not in set(sum_cols) | set(px_cols) | set(vwap_cols)]
    # state/book: the coarse row IS the fine row at the same timestamp
    sel = fine.loc[coarse_idx, state_cols]
    for c in state_cols:
        out[c] = sel[c].to_numpy()
    # flow: left-labeled bins -> group sub-bins by their floor onto the coarse grid
    if sum_cols or px_cols or vwap_cols:
        key = fine.index.floor(freq)
        g = fine[sum_cols + px_cols + vwap_cols].groupby(key)
        for c in sum_cols:
            out[c] = g[c].sum().reindex(coarse_idx).to_numpy()
        for c in px_cols:
            out[c] = g[c].last().reindex(coarse_idx).to_numpy()      # .last() skips NaN
        for c in vwap_cols:
            root = c[: -len(VWAP_SUFFIX)]
            bv, sv = f"{root}_trade_buy_volume", f"{root}_trade_sell_volume"
            if bv in fine.columns and sv in fine.columns:
                w = fine[bv].fillna(0.0) + fine[sv].fillna(0.0)
                num = (fine[c] * w).groupby(key).sum().reindex(coarse_idx)
                den = w.groupby(key).sum().reindex(coarse_idx)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[c] = (num / den.where(den > 0)).to_numpy()
            else:
                out[c] = g[c].last().reindex(coarse_idx).to_numpy()
    out.attrs = dict(fine.attrs)
    out.attrs["derived_from_interval"] = fine_interval or ("%gs" % dt_f)
    return out


def _verify(derived: pd.DataFrame, direct: pd.DataFrame) -> dict:
    """Compare a derived coarse frame with a directly-extracted one. Book/state columns must match
    exactly (NaN-aware); flow columns must match on all bars EXCEPT the final one (post-close
    prints -- see module docstring); vwap is reported but tolerated (signed-weight approximation)."""
    rep = {"ok": True, "book_mismatch": [], "flow_mismatch": [], "vwap_max_dev": 0.0,
           "final_bar_flow_diff": 0.0, "n_rows": int(len(derived))}
    if len(derived) != len(direct) or not derived.index.equals(direct.index):
        rep["ok"] = False
        rep["book_mismatch"].append("<index differs: %d vs %d rows>" % (len(derived), len(direct)))
        return rep
    common = [c for c in derived.columns if c in direct.columns]
    for c in common:
        a = pd.to_numeric(derived[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(direct[c], errors="coerce").to_numpy(float)
        if c.endswith(VWAP_SUFFIX):
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                rep["vwap_max_dev"] = max(rep["vwap_max_dev"], float(np.max(np.abs(a[m] - b[m]))))
            continue
        if c.endswith(SUM_SUFFIXES) or c.endswith(PX_SUFFIX):
            body = slice(0, len(a) - 1)                       # final bar reported separately
            same = np.allclose(a[body], b[body], rtol=0, atol=1e-9, equal_nan=True)
            if not same:
                rep["flow_mismatch"].append(c); rep["ok"] = False
            if c.endswith(SUM_SUFFIXES) and np.isfinite(a[-1]) and np.isfinite(b[-1]):
                rep["final_bar_flow_diff"] = max(rep["final_bar_flow_diff"], float(abs(a[-1] - b[-1])))
        else:
            if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
                rep["book_mismatch"].append(c); rep["ok"] = False
    return rep


def derive_sessions(sessions, target: str, fine_interval: str | None, cache_dir: str,
                    levels: int, clock: str, es_book_source: str, verify_existing: bool):
    """-> (derived list [(date, regime, df)], reports dict). Cache: never overwrites; qc-gated."""
    out, reports = [], {}
    for d, r, fine in sessions:
        label = str(d)[:10]
        coarse = derive_coarse_frame(fine, target, fine_interval)
        rep = {"cached": False, "verified": None}
        qc = ml.session_qc(coarse, date=label)
        rep["qc_ok"] = bool(qc.get("ok"))
        if cache_dir:
            cpath = ml._cache_path(cache_dir, label, target, levels, clock, es_book_source)
            if os.path.exists(cpath):
                rep["cached"] = "existing direct cache kept"
                if verify_existing:
                    direct = pd.read_pickle(cpath)
                    rep["verified"] = _verify(coarse, direct)
            elif rep["qc_ok"]:
                os.makedirs(cache_dir, exist_ok=True)
                tmp = cpath + ".tmp%d" % os.getpid()
                coarse.to_pickle(tmp)
                os.replace(tmp, cpath)
                rep["cached"] = cpath
        out.append((label, r, coarse))
        reports[label] = rep
    return out, reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", required=True, help="fine-grid frames pickle (List[(date[,regime],df)])")
    ap.add_argument("--target", default="1s", help="coarse interval to derive (default 1s)")
    ap.add_argument("--fine-interval", default="", help="fine grid (default: inferred from the index)")
    ap.add_argument("--cache-dir", default="", help="session cache to fill (never overwrites)")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--clock", default="receipt")
    ap.add_argument("--es-book-source", default="aggregated")
    ap.add_argument("--out", default="", help="also write the derived frames pickle here")
    ap.add_argument("--verify-existing", action="store_true",
                    help="compare against any direct coarse cache already present (real-feed "
                         "validation of the derivation, session by session)")
    a = ap.parse_args(argv)

    with open(a.pickle, "rb") as fh:
        raw = pickle.load(fh)
    sessions = [rec if len(rec) == 3 else (rec[0], "auto", rec[1]) for rec in raw]
    print("deriving %s frames from %d fine session(s) (%s)"
          % (a.target, len(sessions), a.pickle))
    try:
        derived, reports = derive_sessions(sessions, a.target, a.fine_interval or None,
                                           a.cache_dir, a.levels, a.clock, a.es_book_source,
                                           a.verify_existing)
    except ValueError as exc:
        print("GRID MISALIGNMENT -- nothing derived: %s" % exc)
        return 2
    bad = 0
    for label, rep in reports.items():
        v = rep["verified"]
        vtxt = ""
        if v is not None:
            if v["ok"]:
                vtxt = (" | VERIFIED against direct cache (vwap max dev %.3g; final-bar flow "
                        "diff %.0f)" % (v["vwap_max_dev"], v["final_bar_flow_diff"]))
            else:
                vtxt = (" | MISMATCH vs direct cache: book=%s flow=%s"
                        % (v["book_mismatch"][:4], v["flow_mismatch"][:4]))
                bad += 1
        print("  %s: qc_ok=%s cache=%s%s" % (label, rep["qc_ok"], rep["cached"], vtxt))
    if a.out:
        with open(a.out, "wb") as fh:
            pickle.dump([(d, r, f) for d, r, f in derived], fh, protocol=4)
        print("derived frames -> %s" % a.out)
    if bad:
        print("\n%d session(s) MISMATCH the direct extraction -- the direct caches were kept; "
              "investigate before trusting derivation for uncached sessions." % bad)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
