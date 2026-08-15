"""
auction_imbalance.py — opening/closing-cross order-imbalance features, and the cash-open → futures
spillover linkage.

WHAT THIS IS (a feature module, not an estimator). The ``mt_order_imbalance`` message is **auction-only**
in this lake: it carries the opening/closing (and venue periodic) cross state, not a continuous-session
order-flow imbalance (that you build from the reconstructed book's depth changes). So this module turns
the auction trajectory into a small set of per-(session, auction) features and relates the SPY opening
cross to the contemporaneous E-mini move.

WHY IT MATTERS FOR THE PAPER. The overnight tariff shock (announced after the 04-02 close) is absorbed by
the continuously-trading E-mini overnight; the cash market only re-opens at 09:30, and the **opening
auction is the mechanism by which cash catches up**. The signed opening imbalance and the
indicative-vs-reference dislocation quantify the cash-side pressure at the open; relating them to the ES
move around the open is a clean, on-thesis cross-asset linkage. Crucially, ES has **no** auction imbalance
(CME runs no opening cross — confirmed empty), so the linkage is necessarily SPY-auction → ES-*continuous*
move, not auction-to-auction. Both legs are on the SAME GPS-disciplined capture clock (the reconstructed
ES book), which is what makes the comparison meaningful.

PRIMARY LISTING. Several venues run crosses for SPY (NYSE Arca = its listing market, plus Nasdaq's own
cross and Cboe periodic auctions). The economically meaningful, price-forming cross is the **primary
listing** (ARCA for SPY); :func:`select_primary_feed` picks it by ``primarylistingmic`` when present, else
by the feed carrying the most complete auction book (populated ``indicativeprice``).

SIGN CONVENTION. ``side="Bid"`` is a **buy** imbalance (+), ``side="Ask"`` a **sell** imbalance (-),
anything else (``Unknown``/blank) contributes 0. All ratios are signed.

CLOCK. Auction messages are ordered on ``clock="receipt"`` (GPS source capture), the same uniform clock
the books use — see lob_reconstruct / mstbook_loader.

KEY LIMITATION (read before interpreting the linkage). The reconstructed session starts at 09:30, so an
ES *pre-open* window has no book data unless you extract with an earlier ``start_time`` (e.g. "9:25");
:func:`link_auction_to_es` returns NaN for any window it cannot cover, and uses a *post-open* ES window by
default, which is always available. To study the pre-open futures path explicitly, reconstruct from ~09:25.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

import mstbook_loader as ml

log = logging.getLogger("auction_imbalance")

# auction-type buckets (lower-cased). Periodic (Cboe/byx) auctions are NOT the official open/close cross
# and are excluded from the open/close features by default.
_OPEN_TYPES = {"opening", "earlyopening"}
_CLOSE_TYPES = {"closing"}
EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════════
# Parsing / venue selection
# ════════════════════════════════════════════════════════════════════════════
def _sign(side: str) -> float:
    s = str(side).strip().lower()
    return 1.0 if s == "bid" else (-1.0 if s == "ask" else 0.0)


_NUM_COLS = ("totalimbalance", "marketimbalance", "pairedquantity", "auctionquantity",
             "referenceprice", "indicativeprice", "nearprice", "farprice", "clearingprice",
             "bidquantity", "askquantity")


def fetch_auction_imbalance(date_str, product: str = "SPY", product_type: str = "direct",
                            tz: str = "America/New_York", clock: str = "receipt",
                            data_source: str = "apu") -> pd.DataFrame:
    """Fetch ``mt_order_imbalance`` for one date via the message layer (mstwx-lakequery) and coerce the
    auction numeric columns. Returns a tz-aware frame indexed by the chosen ``clock`` (empty if none)."""
    try:
        df = ml._fetch_messages(date_str, product, product_type, "mt_order_imbalance", data_source,
                                tz=tz, clock=clock)
    except Exception as exc:
        log.warning("auction imbalance fetch failed for %s %s: %s", product, date_str, exc)
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("side", "auctiontype", "f", "primarylistingmic", "primarylistingparticipant"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def select_primary_feed(df: pd.DataFrame, prefer_mic: str | None = None) -> str | None:
    """Pick the primary-listing venue's feed id (the price-forming cross). Preference order:
    an explicit ``prefer_mic`` mapped to its feed; else the venue named in ``primarylistingmic``
    (when populated); else the feed with the most non-null ``indicativeprice`` (the venue actually
    running a priced cross). Returns the ``f`` value or None."""
    if df is None or len(df) == 0 or "f" not in df.columns:
        return None
    if prefer_mic and "primarylistingmic" in df.columns:
        m = df[df["primarylistingmic"].str.upper() == prefer_mic.upper()]
        if len(m):
            return m["f"].mode().iloc[0]
    if "primarylistingmic" in df.columns:
        plm = df["primarylistingmic"].replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
        if len(plm):
            mic = plm.mode().iloc[0]
            m = df[df["primarylistingmic"] == mic]
            if len(m) and "f" in m.columns:
                return m["f"].mode().iloc[0]
    if "indicativeprice" in df.columns:                       # feed with the most priced messages
        ok = df[np.isfinite(pd.to_numeric(df["indicativeprice"], errors="coerce"))]
        if len(ok):
            return ok["f"].mode().iloc[0]
    return df["f"].mode().iloc[0]


def _auction_rows(df: pd.DataFrame, auction: str, primary_feed: str | None) -> pd.DataFrame:
    """Rows for one auction (open|close) from the primary feed, time-sorted."""
    if df is None or len(df) == 0 or "auctiontype" not in df.columns:
        return pd.DataFrame()
    types = _OPEN_TYPES if auction == "open" else _CLOSE_TYPES
    sub = df[df["auctiontype"].str.lower().isin(types)]
    if primary_feed is not None and "f" in sub.columns:
        sub = sub[sub["f"] == primary_feed]
    return sub.sort_index()


# ════════════════════════════════════════════════════════════════════════════
# Per-auction features
# ════════════════════════════════════════════════════════════════════════════
def auction_features(df: pd.DataFrame, auction: str = "open", last_n: int = 10,
                     primary_feed: str | None = None) -> dict:
    """Features for ONE auction (``"open"`` or ``"close"``) from the primary listing's imbalance
    trajectory: final signed imbalance + ratio, the indicative-vs-reference dislocation (bps), the
    bid/ask-quantity skew, and the pre-cross trajectory (imbalance slope per second, indicative-price
    drift in bps) over the last ``last_n`` messages. All imbalances are signed (Bid +, Ask -)."""
    if primary_feed is None:
        primary_feed = select_primary_feed(df)
    a = _auction_rows(df, auction, primary_feed)
    out = {"auction": auction, "primary_feed": primary_feed, "n_msgs": int(len(a)),
           "signed_imb_final": np.nan, "imb_ratio_final": np.nan, "imb_ratio_bidask": np.nan,
           "indic_ref_bps_final": np.nan, "imb_slope_per_s": np.nan, "indic_drift_bps": np.nan,
           "paired_final": np.nan, "ref_price_final": np.nan, "indic_price_final": np.nan,
           "cross_time": pd.NaT}
    if len(a) == 0:
        return out
    total = a["totalimbalance"].to_numpy(float) if "totalimbalance" in a else np.full(len(a), np.nan)
    sgn = np.array([_sign(s) for s in a["side"]]) if "side" in a else np.zeros(len(a))
    signed = total * sgn
    paired = a["pairedquantity"].to_numpy(float) if "pairedquantity" in a else np.full(len(a), np.nan)
    ref = a["referenceprice"].to_numpy(float) if "referenceprice" in a else np.full(len(a), np.nan)
    indic = a["indicativeprice"].to_numpy(float) if "indicativeprice" in a else np.full(len(a), np.nan)
    bidq = a["bidquantity"].to_numpy(float) if "bidquantity" in a else np.full(len(a), np.nan)
    askq = a["askquantity"].to_numpy(float) if "askquantity" in a else np.full(len(a), np.nan)

    si, pf, rf, indf = signed[-1], paired[-1], ref[-1], indic[-1]
    out["signed_imb_final"] = float(si) if np.isfinite(si) else np.nan
    out["paired_final"] = float(pf) if np.isfinite(pf) else np.nan
    out["ref_price_final"] = float(rf) if np.isfinite(rf) else np.nan
    out["indic_price_final"] = float(indf) if np.isfinite(indf) else np.nan
    out["cross_time"] = a.index[-1]
    if np.isfinite(si):
        denom = abs(si) + (pf if np.isfinite(pf) else 0.0)
        out["imb_ratio_final"] = float(si / denom) if denom > EPS else np.nan
    if np.isfinite(bidq[-1]) and np.isfinite(askq[-1]) and (bidq[-1] + askq[-1]) > EPS:
        out["imb_ratio_bidask"] = float((bidq[-1] - askq[-1]) / (bidq[-1] + askq[-1]))
    if np.isfinite(indf) and np.isfinite(rf) and rf > EPS:
        out["indic_ref_bps_final"] = float((indf - rf) / rf * 1e4)

    tail = a.iloc[-min(last_n, len(a)):]
    if len(tail) >= 2:
        t = (tail.index - tail.index[0]).total_seconds().to_numpy(float)
        st = (tail["totalimbalance"].to_numpy(float)
              * np.array([_sign(s) for s in tail["side"]])) if "side" in tail else np.full(len(tail), np.nan)
        good = np.isfinite(t) & np.isfinite(st)
        if good.sum() >= 2 and np.ptp(t[good]) > EPS:
            out["imb_slope_per_s"] = float(np.polyfit(t[good], st[good], 1)[0])
        it = tail["indicativeprice"].to_numpy(float) if "indicativeprice" in tail else np.full(len(tail), np.nan)
        if np.isfinite(it[0]) and np.isfinite(it[-1]) and np.isfinite(rf) and rf > EPS:
            out["indic_drift_bps"] = float((it[-1] - it[0]) / rf * 1e4)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Cross-asset linkage: SPY opening cross -> ES continuous move
# ════════════════════════════════════════════════════════════════════════════
def _logret_bps_between(series: pd.Series, t0, t1) -> float:
    """Log return (bps) over [t0, t1], measured FIRST->LAST valid observation strictly inside the
    window. This is robust to a stale or NaN-carried quote landing exactly on an edge (the failure that
    made asof(t0)==asof(t1) and returned a spurious 0 on days the future clearly moved). Falls back to
    ``asof`` at the edges only when the window holds <2 observations (a sparse or late-starting book).
    NaN if the window opens before the data, an endpoint is non-finite/non-positive, OR the sampled
    prices are identical (a genuinely flat window -> NaN, never a spurious 0)."""
    s = series.dropna()
    if len(s) < 2 or t1 <= t0:
        return np.nan
    win = s.loc[(s.index >= t0) & (s.index <= t1)]
    if len(win) >= 2:
        p0, p1 = float(win.iloc[0]), float(win.iloc[-1])
    elif t0 >= s.index[0]:                               # sparse window: anchor with asof at edges
        p0, p1 = float(s.asof(t0)), float(s.asof(t1))
    else:
        return np.nan
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0 or p0 == p1:
        return np.nan
    return float(np.log(p1 / p0) * 1e4)


def _window_n(series: pd.Series, window, tz: str, warmup_s: float = 0.0) -> int:
    """Count of valid (non-NaN) observations inside the clock window — a diagnostic so a NaN/flat
    window is distinguishable from a sparsely-populated one in the linkage output."""
    s = series.dropna()
    if len(s) == 0:
        return 0
    day = s.index[0].tz_convert(tz).date() if s.index.tz is not None else s.index[0].date()
    t0 = pd.Timestamp(f"{day} {window[0]}", tz=tz) + pd.Timedelta(seconds=warmup_s)
    t1 = pd.Timestamp(f"{day} {window[1]}", tz=tz)
    return int(((s.index >= t0) & (s.index <= t1)).sum())


def _window_logret_bps(series: pd.Series, window, tz: str, warmup_s: float = 0.0) -> float:
    """Log return (bps) of ``series`` over a clock window on the series' own date, with the START
    shifted forward by ``warmup_s`` (skip the chaotic first seconds after the open) and both edges
    sampled by ``asof`` (robust to NaN book cells). A flat window returns NaN. Pre-open windows on a
    09:30-start session return NaN (the window opens before any data). The post-open move is measured
    from the cash open (09:30), not the indicative cross, which can print minutes earlier."""
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    day = s.index[0].tz_convert(tz).date() if s.index.tz is not None else s.index[0].date()
    t0 = pd.Timestamp(f"{day} {window[0]}", tz=tz) + pd.Timedelta(seconds=warmup_s)
    t1 = pd.Timestamp(f"{day} {window[1]}", tz=tz)
    return _logret_bps_between(series, t0, t1)


def link_auction_to_es(session_df: pd.DataFrame, open_features: dict,
                       pre_window=("09:28", "09:30"), post_window=("09:30", "09:35"),
                       es_col: str = "ES_mid", tz: str = "America/New_York") -> dict:
    """Relate the SPY OPENING cross to the E-mini move around the open. Pairs the opening signed
    imbalance ratio and indicative-vs-reference dislocation with the ES mid log-return (bps) over a
    pre-open and a post-open window. ES has no auction, so this is auction -> continuous move; ES comes
    from the reconstructed book on the same clock. ``pre_window`` is NaN unless the session was
    reconstructed from before 09:30 (the default 09:30 start has no pre-open bars)."""
    es = session_df.get(es_col, pd.Series(dtype=float)) if session_df is not None else pd.Series(dtype=float)
    return {"spy_imb_ratio": open_features.get("imb_ratio_final", np.nan),
            "spy_imb_ratio_bidask": open_features.get("imb_ratio_bidask", np.nan),
            "spy_indic_ref_bps": open_features.get("indic_ref_bps_final", np.nan),
            "spy_imb_slope_per_s": open_features.get("imb_slope_per_s", np.nan),
            "es_pre_open_ret_bps": _window_logret_bps(es, pre_window, tz),
            "es_post_open_ret_bps": _window_logret_bps(es, post_window, tz, warmup_s=30.0),
            "es_post_open_n": _window_n(es, post_window, tz, warmup_s=30.0),
            "cross_time": open_features.get("cross_time", pd.NaT)}


# ════════════════════════════════════════════════════════════════════════════
# Orchestration: panels + cross-sectional linkage summary
# ════════════════════════════════════════════════════════════════════════════
def _xs_summary(link: pd.DataFrame, xcol: str, ycol: str) -> dict:
    """Cross-sectional Pearson r and OLS slope of ``ycol`` on ``xcol`` (NaN-safe, needs >=3 points)."""
    d = link[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3:
        return {"n": int(len(d)), "corr": np.nan, "slope": np.nan, "intercept": np.nan}
    x, y = d[xcol].to_numpy(float), d[ycol].to_numpy(float)
    if np.ptp(x) <= EPS:
        return {"n": int(len(d)), "corr": np.nan, "slope": np.nan, "intercept": np.nan}
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(len(d)), "corr": r, "slope": float(b), "intercept": float(a)}


def run_auction_analysis(sessions, out_dir: str | None = None, write: bool = False, last_n: int = 10,
                         pre_window=("09:28", "09:30"), post_window=("09:30", "09:35"),
                         prefer_mic: str | None = None, data_source: str = "apu",
                         tz: str = "America/New_York", clock: str = "receipt", fetch_fn=None) -> dict:
    """For each ``(label, regime, df)`` session: fetch the auction imbalance, compute open & close
    features, and link the opening cross to the ES move. Returns ``{auction_panel, linkage_panel,
    summary}`` and (if ``write``) writes ``auction_imbalance.csv`` / ``.md`` and ``auction_linkage.csv``.

    ``fetch_fn(label) -> imbalance_df`` is injectable (defaults to :func:`fetch_auction_imbalance` for
    SPY); the summary cross-sectionally relates the SPY opening imbalance to the post-open ES return,
    overall and split by regime."""
    if fetch_fn is None:
        def fetch_fn(label):
            return fetch_auction_imbalance(label, "SPY", "direct", tz=tz, clock=clock, data_source=data_source)

    feat_rows, link_rows = [], []
    for spec in sessions:
        label, regime, df = (spec[0], spec[1], spec[2]) if len(spec) >= 3 else (spec[0], "auto", spec[1])
        imb = fetch_fn(label)
        primary = select_primary_feed(imb, prefer_mic)
        fo = auction_features(imb, "open", last_n, primary)
        fc = auction_features(imb, "close", last_n, primary)
        for f in (fo, fc):
            feat_rows.append({"date": label, "regime": regime, **f})
        lk = link_auction_to_es(df, fo, pre_window, post_window, tz=tz)
        link_rows.append({"date": label, "regime": regime, **lk})

    auction_panel = pd.DataFrame(feat_rows)
    linkage_panel = pd.DataFrame(link_rows)

    summary = {"overall": _xs_summary(linkage_panel, "spy_imb_ratio", "es_post_open_ret_bps")}
    if "regime" in linkage_panel.columns:
        for reg, g in linkage_panel.groupby("regime"):
            summary[reg] = _xs_summary(g, "spy_imb_ratio", "es_post_open_ret_bps")
    summary["overall_indic_vs_es"] = _xs_summary(linkage_panel, "spy_indic_ref_bps", "es_post_open_ret_bps")

    if write:
        out_dir = out_dir or os.environ.get("OUT_DIR", "/mnt/user-data/outputs")
        os.makedirs(out_dir, exist_ok=True)
        auction_panel.to_csv(os.path.join(out_dir, "auction_imbalance.csv"), index=False)
        linkage_panel.to_csv(os.path.join(out_dir, "auction_linkage.csv"), index=False)
        with open(os.path.join(out_dir, "auction_imbalance.md"), "w") as fh:
            fh.write(_format_md(auction_panel, linkage_panel, summary))
    return {"auction_panel": auction_panel, "linkage_panel": linkage_panel, "summary": summary}


def _format_md(auction_panel, linkage_panel, summary) -> str:
    lines = ["# Auction imbalance and cash-open -> futures linkage\n",
             "## Per-(session, auction) features\n",
             "Signed imbalance: Bid (+) buy pressure, Ask (-) sell pressure. `indic_ref_bps` is the "
             "indicative-vs-reference dislocation; `imb_slope_per_s` is the pre-cross imbalance trend.\n"]
    cols = ["date", "regime", "auction", "n_msgs", "signed_imb_final", "imb_ratio_final",
            "indic_ref_bps_final", "imb_slope_per_s"]
    have = [c for c in cols if c in auction_panel.columns]
    if len(auction_panel):
        lines.append("| " + " | ".join(have) + " |")
        lines.append("|" + "|".join(["---"] * len(have)) + "|")
        for _, r in auction_panel.iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) for c in have) + " |")
    lines.append("\n## Cross-sectional linkage: SPY opening imbalance -> ES post-open return (bps)\n")
    lines.append("Positive `corr`/`slope` => a sell (buy) opening imbalance co-moves with a negative "
                 "(positive) post-open ES return — i.e. the cash open and the future agree in direction.\n")
    lines.append("| sample | n | corr | slope | intercept |")
    lines.append("|---|---|---|---|---|")
    for k in ("overall", "benchmark", "volatile", "calm", "stress", "overall_indic_vs_es"):
        if k in summary:
            s = summary[k]
            lines.append("| %s | %d | %s | %s | %s |"
                         % (k, s["n"], _fmt(s["corr"]), _fmt(s["slope"]), _fmt(s["intercept"])))
    return "\n".join(lines) + "\n"


def _fmt(v):
    if isinstance(v, (pd.Timestamp,)) or v is pd.NaT:
        return str(v)
    try:
        f = float(v)
        return "%.4g" % f if np.isfinite(f) else "nan"
    except (TypeError, ValueError):
        return str(v)


# ════════════════════════════════════════════════════════════════════════════
# Self-test (synthetic mt_order_imbalance; binary not required)
# ════════════════════════════════════════════════════════════════════════════
def _synth_imbalance(date="2025-04-03", tz="America/New_York") -> pd.DataFrame:
    """Build a synthetic auction-imbalance frame: a primary ARCA cross (full fields) plus a sparser
    Nasdaq cross, for both the open (growing SELL imbalance, indicative below reference -> gap down) and
    the close (BUY imbalance, indicative above reference)."""
    rows = []
    # OPEN on ARCA (xdp_arca_integrated): 8 msgs 09:28:00 -> 09:29:45, sell imbalance grows 500->1900,
    # indicative declines 558 -> 555 vs reference 560 (gap down).
    for k in range(8):
        t = pd.Timestamp(f"{date} 09:28:{k*6:02d}", tz=tz)
        rows.append((t, "xdp_arca_integrated", "ARCX", "Ask", 500 + 200 * k, 100000, 560.0,
                     558.0 - 0.4 * k, 561.0, 90000, 30000, "Opening"))
    # a sparser Nasdaq (total_view) open cross -> must NOT be picked as primary
    for k in range(2):
        t = pd.Timestamp(f"{date} 09:29:{30 + k*10:02d}", tz=tz)
        rows.append((t, "total_view", "", "Ask", 300, 40000, 560.0, np.nan, np.nan, 0, 0, "Opening"))
    # CLOSE on ARCA: 6 msgs 15:58:00 -> 15:58:50, buy imbalance 800->1800, indicative 541->543 vs ref 540
    for k in range(6):
        t = pd.Timestamp(f"{date} 15:58:{k*10:02d}", tz=tz)
        rows.append((t, "xdp_arca_integrated", "ARCX", "Bid", 800 + 200 * k, 120000, 540.0,
                     541.0 + 0.4 * k, 539.0, 150000, 110000, "Closing"))
    cols = ["ts", "f", "primarylistingmic", "side", "totalimbalance", "pairedquantity", "referenceprice",
            "indicativeprice", "nearprice", "bidquantity", "askquantity", "auctiontype"]
    df = pd.DataFrame(rows, columns=cols).set_index("ts")
    df.index.name = "receipttimestamp"
    return df


def _synth_session(date="2025-04-03", tz="America/New_York", es_open=5720.0, drift_bps=-30.0):
    """A 09:30->09:40 1s session frame whose ES_mid drifts by ``drift_bps`` over the first 5 min
    (post-open), to plant a known SPY-sell-imbalance -> ES-down association."""
    idx = pd.date_range(f"{date} 09:30:00", f"{date} 09:40:00", freq="1s", tz=tz)
    n = len(idx)
    ramp = np.linspace(0, drift_bps / 1e4, n)
    es = es_open * np.exp(ramp)
    spy = 572.0 * np.exp(ramp * 0.9)
    return pd.DataFrame({"ES_mid": es, "SPY_mid": spy}, index=idx)


def _selftest() -> bool:
    print("auction_imbalance self-test\n")
    tz = "America/New_York"
    imb = _synth_imbalance("2025-04-03", tz)

    # primary feed = ARCA (xdp_arca_integrated), not the sparse Nasdaq cross
    primary = select_primary_feed(imb)
    prim_ok = primary == "xdp_arca_integrated"
    print("primary listing feed selected:", primary, "->", prim_ok)

    # OPEN features: sell imbalance (negative), indicative below reference (negative bps), building
    fo = auction_features(imb, "open", last_n=8, primary_feed=primary)
    open_ok = (fo["n_msgs"] == 8 and fo["signed_imb_final"] < 0 and fo["imb_ratio_final"] < 0
               and fo["indic_ref_bps_final"] < 0 and fo["imb_slope_per_s"] < 0)
    # indicative 555.2 vs ref 560 -> ~ -857 bps
    bps_ok = abs(fo["indic_ref_bps_final"] - ((555.2 - 560.0) / 560.0 * 1e4)) < 5.0
    print("OPEN: signed_imb=%.0f ratio=%.3f indic_ref=%.0f bps slope/s=%.1f : %s"
          % (fo["signed_imb_final"], fo["imb_ratio_final"], fo["indic_ref_bps_final"],
             fo["imb_slope_per_s"], open_ok and bps_ok))

    # CLOSE features: buy imbalance (positive), indicative above reference (positive bps)
    fc = auction_features(imb, "close", last_n=6, primary_feed=primary)
    close_ok = (fc["n_msgs"] == 6 and fc["signed_imb_final"] > 0 and fc["indic_ref_bps_final"] > 0)
    print("CLOSE: signed_imb=%.0f indic_ref=%.0f bps : %s"
          % (fc["signed_imb_final"], fc["indic_ref_bps_final"], close_ok))

    # linkage on one session: sell imbalance + ES down post-open
    sess_df = _synth_session("2025-04-03", tz, drift_bps=-30.0)
    lk = link_auction_to_es(sess_df, fo, tz=tz)
    link_ok = (np.isnan(lk["es_pre_open_ret_bps"])          # 09:30-start session -> no pre-open bars
               and lk["es_post_open_ret_bps"] < 0           # ES fell post-open
               and lk["spy_imb_ratio"] < 0)                 # SPY sell imbalance
    print("LINK: spy_imb_ratio=%.3f es_pre=%s es_post=%.1f bps (pre NaN as expected): %s"
          % (lk["spy_imb_ratio"], lk["es_pre_open_ret_bps"], lk["es_post_open_ret_bps"], link_ok))

    # regression: the windowing bug that returned a spurious 0.0 on days the future moved.
    #  (a) DENSE window that moves -> a real, finite, non-zero return (first->last within window).
    #  (b) equal-at-edges / no in-window obs -> NaN, never 0.0 (so it drops, not biases the regression).
    idx = pd.date_range("2025-04-04 09:30:00", "2025-04-04 09:40:00", freq="s", tz=tz)
    dense = pd.Series(5400.0 * np.exp(np.linspace(0, -0.004, len(idx))), index=idx)  # ES sliding down
    r_dense = _logret_bps_between(dense, pd.Timestamp("2025-04-04 09:30:30", tz=tz),
                                  pd.Timestamp("2025-04-04 09:35:00", tz=tz))
    # sparse: one quote before the window and one after, nothing inside -> asof edges are equal
    sparse = pd.Series([5400.0, 5390.0], index=pd.DatetimeIndex(
        ["2025-04-04 09:30:00", "2025-04-04 09:36:00"]).tz_localize(tz))
    r_sparse = _logret_bps_between(sparse, pd.Timestamp("2025-04-04 09:30:30", tz=tz),
                                   pd.Timestamp("2025-04-04 09:35:00", tz=tz))
    win_ok = (np.isfinite(r_dense) and r_dense < -1.0      # captured the real ~ -40 bps slide
              and np.isnan(r_sparse))                       # equal-edge window -> NaN, not 0.0
    print("WINDOW: dense move=%.1f bps (finite, non-zero); equal-edge=%s (NaN not 0.0) : %s"
          % (r_dense, r_sparse, win_ok))

    # cross-sectional: 4 sessions, sell imbalance -> ES down, buy imbalance -> ES up (positive corr)
    def _imb_for(sign_dir):
        df = _synth_imbalance("2025-04-03", tz).copy()
        if sign_dir > 0:                                    # flip the open to a BUY imbalance
            mask = df["auctiontype"].str.lower().isin(_OPEN_TYPES)
            df.loc[mask, "side"] = "Bid"
            df.loc[mask, "indicativeprice"] = 562.0          # above reference
        return df
    sessions, fetch_map = [], {}
    for i, (lbl, reg, sd) in enumerate([
            ("2025-04-01", "benchmark", -10.0), ("2025-04-02", "benchmark", 12.0),
            ("2025-04-03", "volatile", -40.0), ("2025-04-04", "volatile", 35.0)]):
        sessions.append((lbl, reg, _synth_session(lbl, tz, drift_bps=sd)))
        fetch_map[lbl] = _imb_for(1 if sd > 0 else -1)
    res = run_auction_analysis(sessions, write=False, last_n=8, fetch_fn=lambda l: fetch_map[l])
    s = res["summary"]["overall"]
    panel_ok = (len(res["auction_panel"]) == 8 and len(res["linkage_panel"]) == 4   # open+close x 4
                and s["n"] == 4 and np.isfinite(s["corr"]) and s["corr"] > 0.5 and s["slope"] > 0)
    print("PANEL: %d feature rows, %d linkage rows | xs corr(imb_ratio, ES post)=%.2f slope=%.1f : %s"
          % (len(res["auction_panel"]), len(res["linkage_panel"]), s["corr"], s["slope"], panel_ok))

    # markdown renders
    md = _format_md(res["auction_panel"], res["linkage_panel"], res["summary"])
    md_ok = "cash-open" in md and "corr" in md and len(md) > 200

    ok = all([prim_ok, open_ok, bps_ok, close_ok, link_ok, win_ok, panel_ok, md_ok])
    print("\nchecks:", ok)
    return ok


if __name__ == "__main__":
    _selftest()
