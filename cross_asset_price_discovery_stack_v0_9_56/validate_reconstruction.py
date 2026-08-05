"""
validate_reconstruction.py
==========================
Benchmark driver: reconstruct SPY (consolidated book from messages) vs. the vendor snapshot, on the
SAME days, and tabulate (a) how closely the two books agree and (b) whether the choice moves the
SPY-vs-ES price-discovery conclusion. This is the robustness table the methodology report calls for
before committing to one book sample-wide.

For each date it builds SPY+ES under both ``book_source`` settings (reconstruct / snapshot), then:
  1. ``lob_reconstruct.validate_against_snapshot`` -> L1 bid/ask/mid match rates, mid mean-abs-diff,
     per-level price agreement (how faithful the rebuild is to the vendor book);
  2. leadership on EACH book -> Hasbrouck IS midpoint, the order-invariant Lien-Shrestha unique IS
     (``MIS``), Gonzalo-Granger component share (``CS``) for ES, a Hayashi-Yoshida lead-lag (bars,
     ES vs SPY), and the return correlation (the DCC input); optionally the DCC mean rho;
  3. the DELTA (reconstruct - snapshot) on each leadership metric -- the headline: does the snapshot's
     conflation/level-aggregation bias change, or flip, the ES-leadership reading?

Reading the result. If match rates are high and the leadership deltas are ~0, the snapshot bias is
immaterial for the claim and you cite the robustness. If a delta is large -- especially a sign change
in the lead-lag or a material shift in IS_mid_ES / CS_ES -- you have justified the reconstruction and
produced a methods contribution. Run it on a calm day, a March-2020 circuit-breaker day, and an
April-2025 ("Liberation Day") day, which is where the construction sensitivities are largest.

Note on ILS. Putnins' Information Leadership Share is intentionally a NotImplementedError stub in
``price_discovery_shares`` (it is orientation-subtle and academically contested; see that module's
docstring on the Shen-Zhang-Zivot 2025 correction). This driver therefore uses the order-invariant
MIS as the unique leadership share and the Hayashi-Yoshida lead-lag as the asynchronous cross-check,
rather than a hand-rolled ILS; if a validated ILS is added to ``pds`` it will surface here unchanged.

The live path needs the MayStreet binary (mstbook-query + mstwx-lakequery); ``compare_books`` /
``run_validation`` operate on already-built frames and are unit-tested on a synthetic ES-leads-SPY DGP
with a deliberately degraded "snapshot" so the validation and the leadership delta are both non-trivial.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

import price_discovery_shares as pds
import lob_reconstruct as lob

log = logging.getLogger(__name__)
EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════════
# Hayashi-Yoshida lead-lag (asynchronous-robust cross-correlation)
# ════════════════════════════════════════════════════════════════════════════
def hayashi_yoshida_leadlag(price_x, price_y, max_lag_bars: int = 10) -> dict:
    """Lead-lag from the normalized cross-correlation of log-returns at integer-bar shifts.

    For the two books sampled on a common grid the Hayashi-Yoshida estimator reduces to the
    standardized cross-correlation ρ(k) = corr(r^x_t, r^y_{t+k}); the lead-lag is argmax_k |ρ(k)|.
    **Convention:** a positive lead-lag means ``price_x`` LEADS ``price_y`` (x's return today lines up
    with y's return k bars later). Pass ``(ES_mid, SPY_mid)`` so a positive value reads "ES leads SPY".
    (On genuinely asynchronous event-time series the overlap-weighted HY covariance should be used;
    on the aligned grid that collapses to this cross-correlation, which is what the books provide.)"""
    x = np.asarray(price_x, float)
    y = np.asarray(price_y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < max_lag_bars + 5:
        return {"lead_lag_bars": np.nan, "peak_corr": np.nan, "contemp_corr": np.nan,
                "lags": np.array([]), "rho": np.array([])}
    rx = np.diff(np.log(x))
    ry = np.diff(np.log(y))
    lags = np.arange(-max_lag_bars, max_lag_bars + 1)
    rho = np.full(lags.size, np.nan)
    for i, k in enumerate(lags):
        if k >= 0:
            a, b = rx[: rx.size - k], ry[k:]
        else:
            a, b = rx[-k:], ry[: ry.size + k]
        if a.size < 5:
            continue
        sa, sb = a.std(), b.std()
        if sa > EPS and sb > EPS:
            rho[i] = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(rho).any():
        return {"lead_lag_bars": np.nan, "peak_corr": np.nan, "contemp_corr": np.nan,
                "lags": lags, "rho": rho}
    pk = int(np.nanargmax(np.abs(rho)))
    contemp = rho[np.where(lags == 0)[0][0]] if (lags == 0).any() else np.nan
    return {"lead_lag_bars": int(lags[pk]), "peak_corr": float(rho[pk]),
            "contemp_corr": float(contemp), "lags": lags, "rho": rho}


# ════════════════════════════════════════════════════════════════════════════
# Leadership panel for one book
# ════════════════════════════════════════════════════════════════════════════
def _mid_series(df: pd.DataFrame, asset: str) -> np.ndarray:
    """Use {asset}_mid when present and mostly finite, else the L1 midpoint."""
    col = f"{asset}_mid"
    if col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce")).mean() > 0.5:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df.get(f"{asset}_bidprice_1"), errors="coerce").to_numpy(float)
    a = pd.to_numeric(df.get(f"{asset}_askprice_1"), errors="coerce").to_numpy(float)
    return (a + b) / 2.0


def leadership_metrics(df: pd.DataFrame, n_lags: int = 5, max_lag_bars: int = 10,
                       with_dcc: bool = False) -> dict:
    """Price-discovery summary for one (SPY,ES) book: ES information shares + lead-lag + return corr."""
    m_spy, m_es = _mid_series(df, "SPY"), _mid_series(df, "ES")
    out = {}
    try:
        est = pds.estimate_day(m_spy, m_es, n_lags=n_lags)
        out.update({"IS_mid_ES": est["IS_mid_ES"], "MIS_ES": est["MIS_ES"], "CS_ES": est["CS_ES"],
                    "IS_mid_SPY": est["IS_mid_SPY"], "n_obs": est["n_obs"]})
    except Exception as exc:                                   # singular VECM etc. -> NaNs, keep going
        log.warning("VECM leadership failed: %s", exc)
        out.update({"IS_mid_ES": np.nan, "MIS_ES": np.nan, "CS_ES": np.nan,
                    "IS_mid_SPY": np.nan, "n_obs": 0})
    hy = hayashi_yoshida_leadlag(m_es, m_spy, max_lag_bars=max_lag_bars)   # +ve => ES leads SPY
    out["hy_lead_bars"] = hy["lead_lag_bars"]
    out["hy_peak_corr"] = hy["peak_corr"]
    ok = np.isfinite(m_spy) & np.isfinite(m_es)
    if ok.sum() > 5:
        rs, re = np.diff(np.log(m_spy[ok])), np.diff(np.log(m_es[ok]))
        out["ret_corr"] = float(np.corrcoef(rs, re)[0, 1]) if rs.std() > EPS and re.std() > EPS else np.nan
    else:
        out["ret_corr"] = np.nan
    if with_dcc:
        try:
            import dcc_garch as dg
            R = np.column_stack([np.diff(np.log(m_spy[ok])), np.diff(np.log(m_es[ok]))])
            res = dg.dcc_garch_x(R, names=("SPY", "ES"))
            out["dcc_rho_mean"] = float(np.nanmean(res["rho"]))
        except Exception as exc:
            log.warning("DCC failed: %s", exc)
            out["dcc_rho_mean"] = np.nan
    return out


# ════════════════════════════════════════════════════════════════════════════
# Compare the two books for one day
# ════════════════════════════════════════════════════════════════════════════
_LEAD_KEYS = ("IS_mid_ES", "MIS_ES", "CS_ES", "IS_mid_SPY", "hy_lead_bars", "hy_peak_corr", "ret_corr")


def compare_books(recon_df: pd.DataFrame, snap_df: pd.DataFrame, asset: str = "SPY",
                  n_lags: int = 5, max_lag_bars: int = 10, with_dcc: bool = False) -> dict:
    """One row: snapshot-vs-reconstruction agreement on the SPY book + ES leadership under each book +
    the reconstruct-minus-snapshot delta on every leadership metric."""
    val = lob.validate_against_snapshot(recon_df, snap_df, asset=asset, levels=10)
    Lr = leadership_metrics(recon_df, n_lags, max_lag_bars, with_dcc)
    Ls = leadership_metrics(snap_df, n_lags, max_lag_bars, with_dcc)
    row = {f"match_{k}": val.get(k) for k in ("bid1_match", "ask1_match", "mid_match",
                                              "mid_mean_abs_diff", "level_price_match", "n")}
    for k in _LEAD_KEYS:
        row[f"recon_{k}"] = Lr.get(k)
        row[f"snap_{k}"] = Ls.get(k)
        if Lr.get(k) is not None and Ls.get(k) is not None \
                and np.isfinite(Lr.get(k, np.nan)) and np.isfinite(Ls.get(k, np.nan)):
            row[f"d_{k}"] = Lr[k] - Ls[k]
        else:
            row[f"d_{k}"] = np.nan
    if with_dcc:
        row["recon_dcc_rho_mean"] = Lr.get("dcc_rho_mean")
        row["snap_dcc_rho_mean"] = Ls.get("dcc_rho_mean")
    return row


# ════════════════════════════════════════════════════════════════════════════
# Drivers
# ════════════════════════════════════════════════════════════════════════════
def _live_pairs(date_specs: Sequence, levels: int = 10, interval: str = "1s",
                clock: str = "receipt", classify: str = "aggressor", **kw):
    """Yield (date, regime, recon_df, snap_df) by extracting each date under both book sources."""
    import mstbook_loader as ml
    for spec in date_specs:
        date, regime = (spec[0], spec[1] if len(spec) > 1 else "auto") if isinstance(spec, (tuple, list)) \
            else (spec, "auto")
        recon = ml.extract_sessions([(date, regime)], levels=levels, interval=interval, with_flow=False,
                                    book_source="reconstruct", clock=clock, classify=classify, **kw)[0][2]
        snap = ml.extract_sessions([(date, regime)], levels=levels, interval=interval, with_flow=False,
                                   book_source="snapshot", classify=classify, **kw)[0][2]
        yield date, regime, recon, snap


def run_validation(date_specs: Sequence, pairs: Optional[Callable] = None, asset: str = "SPY",
                   n_lags: int = 5, max_lag_bars: int = 10, with_dcc: bool = False,
                   **loader_kw) -> pd.DataFrame:
    """Build the benchmark table over ``date_specs``. ``pairs`` is a generator yielding
    (date, regime, recon_df, snap_df); defaults to the live extract under both book sources. Returns a
    per-date DataFrame: SPY match rates, ES leadership under each book, and the reconstruct-snapshot
    deltas."""
    gen = (pairs or _live_pairs)(date_specs, **loader_kw) if pairs is None \
        else pairs(date_specs, **loader_kw)
    rows, idx = [], []
    for date, regime, recon, snap in gen:
        r = compare_books(recon, snap, asset=asset, n_lags=n_lags, max_lag_bars=max_lag_bars, with_dcc=with_dcc)
        r["regime"] = regime
        rows.append(r); idx.append(date)
    out = pd.DataFrame(rows, index=pd.Index(idx, name="date"))
    if len(out):
        cols = ["regime"] + [c for c in out.columns if c != "regime"]
        out = out[cols]
    return out


def format_report(table: pd.DataFrame) -> str:
    """A compact text view: faithfulness (match rates) then the leadership delta that matters."""
    if table.empty:
        return "(no rows)"
    lines = ["snapshot vs. reconstruction — benchmark", "=" * 44]
    for date, r in table.iterrows():
        lines.append(f"\n{date}  [{r.get('regime','?')}]   (n={int(r['match_n']) if np.isfinite(r.get('match_n', np.nan)) else 0})")
        lines.append("  faithfulness : bid1=%.3f ask1=%.3f mid=%.3f  |mid Δ̄|=%.2e  levels=%.3f"
                     % (r.get("match_bid1_match", np.nan), r.get("match_ask1_match", np.nan),
                        r.get("match_mid_match", np.nan), r.get("match_mid_mean_abs_diff", np.nan),
                        r.get("match_level_price_match", np.nan)))
        lines.append("  IS_mid_ES    : recon=%.3f  snap=%.3f  Δ=%+.3f"
                     % (r.get("recon_IS_mid_ES", np.nan), r.get("snap_IS_mid_ES", np.nan), r.get("d_IS_mid_ES", np.nan)))
        lines.append("  MIS_ES       : recon=%.3f  snap=%.3f  Δ=%+.3f"
                     % (r.get("recon_MIS_ES", np.nan), r.get("snap_MIS_ES", np.nan), r.get("d_MIS_ES", np.nan)))
        lines.append("  CS_ES        : recon=%.3f  snap=%.3f  Δ=%+.3f"
                     % (r.get("recon_CS_ES", np.nan), r.get("snap_CS_ES", np.nan), r.get("d_CS_ES", np.nan)))
        lines.append("  HY lead (ES) : recon=%+d bar  snap=%+d bar  Δ=%+d"
                     % (int(r["recon_hy_lead_bars"]) if np.isfinite(r.get("recon_hy_lead_bars", np.nan)) else 0,
                        int(r["snap_hy_lead_bars"]) if np.isfinite(r.get("snap_hy_lead_bars", np.nan)) else 0,
                        int(r["d_hy_lead_bars"]) if np.isfinite(r.get("d_hy_lead_bars", np.nan)) else 0))
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Self-test (synthetic ES-leads-SPY DGP; degraded snapshot; binary not required)
# ════════════════════════════════════════════════════════════════════════════
def _synth_canonical(mids: dict, index: pd.DatetimeIndex, levels: int = 10, seed: int = 0) -> pd.DataFrame:
    """Build a canonical {SPY,ES}_{bid,ask}{price,quantity}_{i} frame around given mid paths."""
    rng = np.random.default_rng(seed)
    T = len(index)
    cols = {}
    for asset, mid in mids.items():
        mid = np.asarray(mid, float)
        half = 0.01 + 0.002 * rng.random(T)
        cols[f"{asset}_mid"] = mid
        for n in range(1, levels + 1):
            off = half + (n - 1) * 0.01
            cols[f"{asset}_bidprice_{n}"] = mid - off
            cols[f"{asset}_askprice_{n}"] = mid + off
            cols[f"{asset}_bidquantity_{n}"] = rng.integers(100, 900, T).astype(float)
            cols[f"{asset}_askquantity_{n}"] = rng.integers(100, 900, T).astype(float)
    return pd.DataFrame(cols, index=index)


def _selftest() -> bool:
    rng = np.random.default_rng(7)
    T = 4000
    idx = pd.date_range("2025-04-23 09:30:00", periods=T, freq="s", tz="America/New_York")
    r_eff = rng.normal(0, 0.02, T)
    eff = 540.0 + np.cumsum(r_eff)                       # common efficient (index) level
    es_mid = eff + rng.normal(0, 0.004, T)              # ES tracks the efficient price closely
    spy_eff = np.empty(T); spy_eff[0] = eff[0]          # SPY reacts ONE bar late -> ES leads SPY
    spy_eff[1:] = eff[:-1]
    spy_mid = spy_eff + rng.normal(0, 0.006, T)

    # "reconstruct" = the faithful book; "snapshot" = SPY conflated to every 3rd bar (ffill), ES same
    recon = _synth_canonical({"SPY": spy_mid, "ES": es_mid}, idx, seed=1)
    spy_snap = pd.Series(spy_mid, index=idx).iloc[::3].reindex(idx).ffill().bfill().to_numpy()
    snap = _synth_canonical({"SPY": spy_snap, "ES": es_mid}, idx, seed=2)

    # 1) HY recovers ES-leads-SPY by +1 bar on the reconstruct book
    hy = hayashi_yoshida_leadlag(_mid_series(recon, "ES"), _mid_series(recon, "SPY"), max_lag_bars=8)
    hy_ok = (hy["lead_lag_bars"] == 1) and (hy["peak_corr"] > 0)
    print("Hayashi-Yoshida: ES leads SPY by %+d bar (peak ρ=%.3f, contemp ρ=%.3f) : %s"
          % (hy["lead_lag_bars"], hy["peak_corr"], hy["contemp_corr"], hy_ok))

    # 2) leadership panel finite and ES-dominant on the reconstruct book
    Lr = leadership_metrics(recon, n_lags=5, max_lag_bars=8)
    lead_ok = (np.isfinite(Lr["IS_mid_ES"]) and 0.0 <= Lr["IS_mid_ES"] <= 1.0
               and np.isfinite(Lr["CS_ES"]) and Lr["IS_mid_ES"] > 0.5)
    print("leadership(recon): IS_mid_ES=%.3f MIS_ES=%.3f CS_ES=%.3f ret_corr=%.3f : %s"
          % (Lr["IS_mid_ES"], Lr["MIS_ES"], Lr["CS_ES"], Lr["ret_corr"], lead_ok))

    # 3) compare_books: faithfulness in [0,1] and < 1 (SPY degraded), deltas finite
    cmp = compare_books(recon, snap, asset="SPY", n_lags=5, max_lag_bars=8)
    faith_ok = (0.0 <= cmp["match_mid_match"] <= 1.0 and cmp["match_mid_match"] < 1.0
                and 0.0 <= cmp["match_bid1_match"] <= 1.0)
    delta_ok = np.isfinite(cmp["d_IS_mid_ES"]) and np.isfinite(cmp["d_CS_ES"])
    print("compare_books: mid match=%.3f (<1 as SPY conflated), IS_mid_ES recon=%.3f snap=%.3f Δ=%+.3f : %s"
          % (cmp["match_mid_match"], cmp["recon_IS_mid_ES"], cmp["snap_IS_mid_ES"], cmp["d_IS_mid_ES"],
             faith_ok and delta_ok))

    # 4) full table builds + report renders
    pairs = lambda specs, **kw: iter([("2025-04-23", "volatile", recon, snap),
                                      ("2020-03-12", "volatile", recon, snap)])
    tab = run_validation([("2025-04-23", "volatile"), ("2020-03-12", "volatile")], pairs=pairs)
    table_ok = (len(tab) == 2 and "d_IS_mid_ES" in tab.columns and "match_mid_match" in tab.columns
                and "regime" in tab.columns)
    print("\n" + format_report(tab))
    print("\ntable builds (%d rows, deltas + match cols present): %s" % (len(tab), table_ok))

    ok = all([hy_ok, lead_ok, faith_ok, delta_ok, table_ok])
    print("\nchecks:", ok)
    return ok


def main_cli(argv=None):
    """Shell entry point: build the benchmark table over a handful of dates and write it to CSV.

    Universe flags mirror the other drivers -- ``--volatile``/``--benchmark`` are comma-lists tagged
    with that regime, ``--dates`` is a comma-list tagged 'auto' (the regime is a label column only; it
    does not change estimation here). Needs the MayStreet binary; use ``--selftest`` to exercise the
    estimators on synthetic data without it."""
    import argparse
    import datetime as _dt
    import os

    p = argparse.ArgumentParser(
        description="Snapshot-vs-reconstruction benchmark for the consolidated SPY book: per date, "
                    "book faithfulness + ES leadership under each book + the reconstruct-snapshot delta.")
    p.add_argument("--volatile", type=lambda s: s.split(","), default=None,
                   help="comma-list of dates (YYYY-MM-DD) tagged 'volatile' (e.g. MWCB / Liberation Day)")
    p.add_argument("--benchmark", type=lambda s: s.split(","), default=None,
                   help="comma-list of calm control dates tagged 'benchmark'")
    p.add_argument("--dates", type=lambda s: s.split(","), default=None,
                   help="comma-list of dates tagged 'auto'")
    p.add_argument("--interval", default="1s", help="sampling grid for both books (default 1s)")
    p.add_argument("--n-levels", type=int, default=10)
    p.add_argument("--clock", choices=["receipt", "exchange"], default="receipt",
                   help="reconstruct ordering clock (default receipt; see lob_reconstruct)")
    p.add_argument("--classify", choices=["aggressor", "tick", "side"], default="aggressor",
                   help="trade-direction rule for the tape (unused here; passed through for parity)")
    p.add_argument("--round-lot", type=int, default=100, help="round-lot NBBO threshold for SPY (100)")
    p.add_argument("--round-lot-only", action="store_true",
                   help="reconstruct a round-lot-only ladder (default is odd-lot-inclusive)")
    p.add_argument("--n-lags", type=int, default=5, help="VECM lags for the information-share fit")
    p.add_argument("--max-lag-bars", type=int, default=10, help="±bars searched for the HY lead-lag")
    p.add_argument("--with-dcc", action="store_true", help="also report the DCC mean rho per book (slower)")
    p.add_argument("--output-dir", default=os.environ.get("OUT_DIR", "/mnt/user-data/outputs"))
    p.add_argument("--out", default=None, help="CSV path (default: <output-dir>/validation_<ts>.csv)")
    p.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if args.selftest:
        _selftest()
        return None

    specs = ([(d.strip(), "volatile") for d in (args.volatile or [])]
             + [(d.strip(), "benchmark") for d in (args.benchmark or [])]
             + [(d.strip(), "auto") for d in (args.dates or [])])
    if not specs:
        p.error("provide --dates and/or --volatile/--benchmark (or --selftest)")

    table = run_validation(specs, n_lags=args.n_lags, max_lag_bars=args.max_lag_bars,
                           with_dcc=args.with_dcc, levels=args.n_levels, interval=args.interval,
                           clock=args.clock, classify=args.classify, round_lot=args.round_lot,
                           odd_lot_inclusive=not args.round_lot_only)
    os.makedirs(args.output_dir, exist_ok=True)
    out = args.out or os.path.join(args.output_dir,
                                   "validation_%s.csv" % _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    table.to_csv(out)
    print(format_report(table))
    print("\nwrote %d row(s) -> %s" % (len(table), out))
    return table


if __name__ == "__main__":
    main_cli()

