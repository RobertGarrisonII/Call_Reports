"""market_state.py -- regulatory state as session columns: LULD price bands and Rule 201 (SSR).

Both are MECHANICAL constraints on quoting, and both are confounds for this paper specifically.

**LULD (limit up / limit down).** Every NMS security has a price band computed from a trailing
reference price. A quote may not be displayed outside it, and if the NBB/NBO sits at a band edge for
15 seconds the security enters a five-minute Limit State and then a trading pause. So as the mid
approaches a band, quoted spreads, depth and cancellation rates change *because they must*, not
because anyone learned anything. On a volatility day the mid spends real time near a band, and the
paper reads spread and order flow as responses to information.

**Rule 201 (the alternative uptick rule).** Once a covered security falls 10% from the prior close,
short sale orders may not be EXECUTED or DISPLAYED at or below the national best bid, for the rest
of that day and the next. The restriction is one-sided by construction: it constrains the sell side
and leaves the buy side alone. The paper's central objects -- Table 5's cross-market order-flow
corners and Table 7's directional agreement -- are counts of SIGNED order flow. A day with SSR on
has mechanically asymmetric sell-side flow, which will look exactly like directional tandem trading.
March 2020 is full of such days.

Neither belongs in the results as a discovery. Both belong as controls, or at minimum as a sample
note that says which sessions were constrained.

Coverage, measured rather than assumed
--------------------------------------
On the 2020-03-09 SPY status stream the `shortsaleindicator` field IS populated on every venue row
(`ShortSaleRestrictionNotInEffect` throughout), while `luldlowerlimit` / `luldupperlimit` /
`limituplimitdownindicator` are **empty on every row**. The band fields exist in the schema but the
direct venue feeds do not carry them for that date -- the LULD bands are disseminated by the SIP
(CTA/UTP) rather than by the individual exchange feeds. So:

  * SSR is usable today, from the data you already fetch.
  * LULD is wired end to end and will populate itself the moment a source carries it, and
    ``coverage()`` reports the fraction of rows that actually had a band so nobody mistakes an
    all-NaN column for "the bands were never near".

That distinction is the point of ``coverage``: a control that is silently all-NaN is worse than no
control, because the regression still runs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

TZ = "America/New_York"

# after .astype(str) a null prints as one of these, depending on the reader's dtype choice
_NULL = ("", "nan", "<NA>", "None", "NaN", "null", "NULL", "\\N")

# shortsaleindicator vocabulary. The feeds spell the restriction several ways; anything that says
# "InEffect" without "Not" is the restriction being ON.
_SSR_ON = ("shortsalerestrictionineffect", "ineffect", "true", "y", "yes", "1")
_SSR_OFF = ("shortsalerestrictionnotineffect", "notineffect", "false", "n", "no", "0")


def _clean(s: pd.Series) -> pd.Series:
    """-> stripped strings with the null tokens turned into NaN."""
    out = s.astype(str).str.strip()
    return out.mask(out.str.lower().isin([t.lower() for t in _NULL]))


def _ssr_flag(v) -> float:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return np.nan
    t = str(v).strip().lower().replace(" ", "").replace("_", "")
    if t in _SSR_OFF or "notineffect" in t:
        return 0.0
    if t in _SSR_ON or "ineffect" in t:
        return 1.0
    return np.nan


def state_series(status: pd.DataFrame, tz: str = TZ) -> pd.DataFrame:
    """Collapse an ``mt_product_status`` frame to ONE time series of regulatory state.

    The stream is per venue and every venue repeats the same security-level state, so the rows are
    sorted by time and the last value of each field wins at each instant (a step function). SSR and
    the LULD bands are security-level facts, not venue opinions -- taking a venue's view would make
    the control depend on which feed happened to publish last.

    -> DataFrame indexed by event time with columns ssr, luld_lower, luld_upper, luld_indicator."""
    cols = ["ssr", "luld_lower", "luld_upper", "luld_indicator"]
    if status is None or len(status) == 0:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=tz))
    idx = status.index
    if getattr(idx, "tz", None) is None:
        idx = pd.DatetimeIndex(idx).tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    out = pd.DataFrame(index=idx)
    out["ssr"] = ([_ssr_flag(v) for v in _clean(status["shortsaleindicator"])]
                  if "shortsaleindicator" in status.columns else np.nan)
    for src, dst in (("luldlowerlimit", "luld_lower"), ("luldupperlimit", "luld_upper")):
        out[dst] = (pd.to_numeric(_clean(status[src]), errors="coerce").to_numpy(float)
                    if src in status.columns else np.nan)
    out["luld_indicator"] = (_clean(status["limituplimitdownindicator"]).to_numpy()
                             if "limituplimitdownindicator" in status.columns else np.nan)
    return out.sort_index()


def coverage(status: pd.DataFrame) -> dict:
    """What fraction of status rows actually carried each field. A control that is silently all-NaN
    is worse than no control, because the regression still runs and the coefficient is meaningless."""
    st = state_series(status)
    n = len(st)
    if n == 0:
        return {"n_rows": 0, "ssr": 0.0, "luld_lower": 0.0, "luld_upper": 0.0, "luld_indicator": 0.0}
    rep = {"n_rows": n}
    for c in ("ssr", "luld_lower", "luld_upper", "luld_indicator"):
        rep[c] = float(st[c].notna().mean()) if c in st.columns else 0.0
    return rep


def attach_market_state(df: pd.DataFrame, status: pd.DataFrame, asset: str = "SPY",
                        tz: str = TZ) -> pd.DataFrame:
    """Add the regulatory-state columns to a session frame, as-of aligned onto its grid.

    Columns added (all prefixed with ``asset``):

      {A}_ssr                 1 while Rule 201 is in effect, 0 while it is not, NaN if unknown
      {A}_luld_lower/_upper   the price band in dollars
      {A}_luld_indicator      the venue's own limit-state string, where published
      {A}_luld_band_bps       band width relative to the mid, in bps -- the width the quote must fit in
      {A}_dist_upper_bps      (upper - mid) / mid in bps: how much room the ask has before the band
      {A}_dist_lower_bps      (mid - lower) / mid in bps
      {A}_luld_binding        1 where the best ask is at/above the upper band or the best bid at/below
                              the lower band, i.e. the constraint is actually touching the book

    A step function, forward-filled: the state holds until the venue changes it. No look-ahead."""
    if df is None or len(df) == 0:
        return df
    st = state_series(status, tz=tz)
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = pd.DatetimeIndex(idx).tz_localize(tz)

    if len(st):
        # as-of: carry each state FORWARD onto the grid, never backward (that would be look-ahead).
        # luld_indicator is an object column; ffill on mixed dtypes warns under pandas 2.x, so the
        # numeric and string halves are filled separately.
        # searchsorted rather than reindex+ffill: the indicator is an object column, and pandas 2.x
        # emits a downcasting FutureWarning when ffilling one. This is also the as-of rule stated
        # directly -- for each grid point take the LAST state row at or before it, -1 meaning the
        # grid point precedes any state row (unknown, not zero).
        pos = np.searchsorted(st.index.values, idx.values, side="right") - 1
        al = pd.DataFrame(index=idx)
        for c in ("ssr", "luld_lower", "luld_upper"):
            v = pd.to_numeric(st[c], errors="coerce").to_numpy(float)
            al[c] = np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan)
        iv = st["luld_indicator"].to_numpy(dtype=object)
        al["luld_indicator"] = np.where(pos >= 0, iv[np.clip(pos, 0, None)], None)
    else:
        al = pd.DataFrame(index=idx, columns=["ssr", "luld_lower", "luld_upper", "luld_indicator"],
                          dtype=float)

    out = df.copy()
    out[f"{asset}_ssr"] = pd.to_numeric(al["ssr"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(al["luld_lower"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(al["luld_upper"], errors="coerce").to_numpy(float)
    out[f"{asset}_luld_lower"] = lo
    out[f"{asset}_luld_upper"] = hi
    out[f"{asset}_luld_indicator"] = al["luld_indicator"].to_numpy()

    bid = pd.to_numeric(out.get(f"{asset}_bidprice_1", pd.Series(np.nan, index=out.index)),
                        errors="coerce").to_numpy(float)
    ask = pd.to_numeric(out.get(f"{asset}_askprice_1", pd.Series(np.nan, index=out.index)),
                        errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = (bid + ask) / 2.0
        safe = np.where(np.isfinite(mid) & (mid > 0), mid, np.nan)
        out[f"{asset}_luld_band_bps"] = (hi - lo) / safe * 1e4
        out[f"{asset}_dist_upper_bps"] = (hi - safe) / safe * 1e4
        out[f"{asset}_dist_lower_bps"] = (safe - lo) / safe * 1e4
        binding = ((np.isfinite(hi) & np.isfinite(ask) & (ask >= hi - 1e-9))
                   | (np.isfinite(lo) & np.isfinite(bid) & (bid <= lo + 1e-9)))
    out[f"{asset}_luld_binding"] = np.where(np.isfinite(hi) | np.isfinite(lo),
                                            binding.astype(float), np.nan)
    return out


def summarize(df: pd.DataFrame, asset: str = "SPY") -> dict:
    """Session-level summary for the sample appendix: was the session constrained, and how much."""
    rep = {}
    ssr = pd.to_numeric(df.get(f"{asset}_ssr", pd.Series(dtype=float)), errors="coerce")
    rep["ssr_known"] = float(ssr.notna().mean()) if len(ssr) else 0.0
    rep["ssr_frac_on"] = float(ssr.fillna(0).mean()) if len(ssr) else float("nan")
    rep["ssr_any"] = bool(ssr.fillna(0).max() > 0) if len(ssr) else False
    band = pd.to_numeric(df.get(f"{asset}_luld_band_bps", pd.Series(dtype=float)), errors="coerce")
    rep["luld_known"] = float(band.notna().mean()) if len(band) else 0.0
    rep["luld_band_bps_median"] = float(band.median()) if band.notna().any() else float("nan")
    near = pd.to_numeric(df.get(f"{asset}_dist_upper_bps", pd.Series(dtype=float)), errors="coerce")
    far = pd.to_numeric(df.get(f"{asset}_dist_lower_bps", pd.Series(dtype=float)), errors="coerce")
    closest = pd.concat([near.abs(), far.abs()], axis=1).min(axis=1) if len(near) else pd.Series(dtype=float)
    rep["luld_min_dist_bps"] = float(closest.min()) if closest.notna().any() else float("nan")
    bind = pd.to_numeric(df.get(f"{asset}_luld_binding", pd.Series(dtype=float)), errors="coerce")
    rep["luld_binding_frac"] = float(bind.fillna(0).mean()) if len(bind) else float("nan")
    return rep


def describe(rep: dict, asset: str = "SPY") -> str:
    lines = []
    if not rep.get("ssr_known"):
        lines.append(f"{asset} SSR (Rule 201): NOT REPORTED by this source -- the control cannot be "
                     f"built for this session; do not treat its absence as 'unrestricted'")
    else:
        lines.append("%s SSR (Rule 201): in effect on %.1f%% of the session (known on %.0f%% of rows)"
                     % (asset, 100 * rep.get("ssr_frac_on", 0.0), 100 * rep["ssr_known"]))
        if rep.get("ssr_any"):
            lines.append("    Short sales cannot execute or display at or below the NBB while this is "
                         "on. Sell-side order flow is mechanically constrained, which is a confound "
                         "for any signed-order-flow result (Tables 5 and 7).")
    if not rep.get("luld_known"):
        lines.append(f"{asset} LULD bands: NOT REPORTED by this source (the fields exist in "
                     f"mt_product_status but the direct venue feeds leave them empty; the bands come "
                     f"from the SIP). Columns are present and all-NaN -- do NOT use them as a control "
                     f"until a source populates them.")
    else:
        lines.append("%s LULD bands: known on %.0f%% of rows, median width %.0f bps, closest "
                     "approach %.0f bps, binding on %.2f%% of snapshots"
                     % (asset, 100 * rep["luld_known"], rep.get("luld_band_bps_median", float("nan")),
                        rep.get("luld_min_dist_bps", float("nan")),
                        100 * rep.get("luld_binding_frac", 0.0)))
    return "\n".join(lines)


def _selftest() -> bool:
    ok = []
    idx = pd.to_datetime(["2020-03-09 09:00", "2020-03-09 10:00", "2020-03-09 11:00"]).tz_localize(TZ)
    status = pd.DataFrame({"f": ["a", "a", "a"],
                           "shortsaleindicator": ["ShortSaleRestrictionNotInEffect",
                                                  "ShortSaleRestrictionInEffect", ""],
                           "luldlowerlimit": ["", "270.00", "265.00"],
                           "luldupperlimit": ["", "290.00", "285.00"],
                           "limituplimitdownindicator": ["", "", "LimitUp"]}, index=idx)
    st = state_series(status)
    a = (st["ssr"].tolist()[:2] == [0.0, 1.0]) and np.isnan(st["ssr"].iloc[2])
    print("(1) SSR vocabulary: NotInEffect->0, InEffect->1, blank->NaN (not 0) : %s" % a)
    ok.append(a)

    grid = pd.date_range("2020-03-09 09:30", "2020-03-09 12:00", freq="30min", tz=TZ)
    book = pd.DataFrame({"SPY_bidprice_1": 280.0, "SPY_askprice_1": 280.10}, index=grid)
    out = attach_market_state(book, status)
    b = float(out["SPY_ssr"].iloc[0]) == 0.0 and float(out["SPY_ssr"].loc[grid[grid >= idx[1]][0]]) == 1.0
    print("(2) state is carried FORWARD onto the grid, never backward : %s" % b)
    ok.append(b)

    row = out.loc[grid[grid >= idx[1]][0]]
    width = (290.0 - 270.0) / 280.05 * 1e4
    c = abs(float(row["SPY_luld_band_bps"]) - width) < 1.0 and float(row["SPY_dist_upper_bps"]) > 0
    print("(3) band width %.0f bps and distance-to-band derived from the mid : %s"
          % (row["SPY_luld_band_bps"], c))
    ok.append(c)

    # binding: an ask at/above the upper band
    book2 = book.copy(); book2["SPY_askprice_1"] = 291.0; book2["SPY_bidprice_1"] = 290.9
    o2 = attach_market_state(book2, status)
    d = float(o2["SPY_luld_binding"].loc[grid[grid >= idx[1]][0]]) == 1.0
    print("(4) an ask at/through the upper band is flagged binding : %s" % d)
    ok.append(d)

    # the honest case: the fields exist but are empty (2020-03-09 SPY, direct feeds)
    empty = pd.DataFrame({"f": ["a"], "shortsaleindicator": ["ShortSaleRestrictionNotInEffect"],
                          "luldlowerlimit": [""], "luldupperlimit": [""],
                          "limituplimitdownindicator": [""]}, index=idx[:1])
    o3 = attach_market_state(book, empty)
    rep = summarize(o3)
    e = rep["ssr_known"] == 1.0 and rep["luld_known"] == 0.0
    txt = describe(rep)
    f = "NOT REPORTED" in txt and "do NOT use them as a control" in txt
    print("(5) LULD absent but SSR present -> coverage says so (%0.0f%% / %0.0f%%) and the text "
          "refuses to pass an all-NaN column off as a control : %s" % (100 * rep["ssr_known"],
                                                                      100 * rep["luld_known"], e and f))
    ok += [e, f]

    g = coverage(status)["ssr"] > 0.5 and coverage(pd.DataFrame())["n_rows"] == 0
    print("(6) coverage() works on a real frame and on an empty one : %s" % g)
    ok.append(g)

    print("\nmarket-state checks -> %s" % all(ok))
    return all(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
