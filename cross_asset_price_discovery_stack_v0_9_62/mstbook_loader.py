"""
mstbook_loader.py
=================
Fast order-book pull via MayStreet's ``mstbook-query`` CLI, emitted DIRECTLY in the
cross_asset_price_discovery_stack canonical schema so it feeds run_contagion / run_analysis /
mean_variance without a separate adapter step.

Why this module exists
----------------------
The Athena / ``maystreet_data`` path (``market_analysis_fixed``) is slow; ``mstbook-query``
streams the book far faster. This keeps that speed but fixes the things a research stack cares
about, consistent with the rest of this package:

  * **Canonical schema.** Raw ``{prefix}.bid{n}/bsize{n}/ask{n}/asize{n}`` columns are mapped to
    the stack contract ``{ROOT}_{bid|ask}{price|quantity}_{i}`` (ROOT = 'SPY' for direct,
    ``get_futures_root`` for futures, e.g. 'ES'). The output drops straight into the pipeline.
  * **Dtype + NaN hygiene at parse time.** Every numeric column is coerced to float (Athena often
    returns Decimal/None as object), and the "a level counts only if BOTH its price and size are
    finite" rule is applied per level -- a torn level from a shift (one leg NaN) is set to NaN on
    both legs, i.e. treated as absent, exactly like ``lcm._usable_levels`` downstream.
  * **One definition of each metric.** Derived features are computed by CALLING the stack's correct
    implementations (``lcm.weighted_mid`` / ``depth_weighted_mid`` / ``decay_weighted_cost`` /
    ``impact_curve_length``) rather than re-deriving them. In particular the microprice uses the
    correct opposite-side size weighting (a heavy bid leans the fair price UP, not down).
  * **Vectorized, NaN-safe VWAP-to-lot.** Replaces the per-row Python double loop; partial fills
    return NaN (not a downward-biased average), and torn levels are skipped.

The ``mstbook-query`` subprocess cannot run in every environment; ``_to_canonical`` /
``attach_features`` / ``vwap_to_lot`` operate on already-pulled data (or a raw CSV string) and are
unit-tested here on synthetic output. ``query_canonical`` / ``query_pair`` / ``load_sessions`` are
the live entry points.
"""
from __future__ import annotations

import logging
import functools
import os
import re
import subprocess as sp
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

try:                                                   # optional: live progress bar over sessions
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:                                      # graceful fallback to per-session/per-fetch logging
    _HAS_TQDM = False

import liquidity_curve_metrics as lcm

log = logging.getLogger(__name__)
EPS = 1e-12

FUTURES_MONTHS = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Exchange multipliers (point value, USD) for CME equity-index futures.
FUTURES_MULT = {"ES": 50, "MES": 5, "NQ": 20, "MNQ": 2, "RTY": 50, "M2K": 5}
DIRECT_ROUND_LOT = 100  # shares


# ════════════════════════════════════════════════════════════════════════════
# Contract helpers
# ════════════════════════════════════════════════════════════════════════════
def get_third_friday(year: int, month: int) -> date:
    """Third Friday of (year, month) -- CME equity-index expiry."""
    d = date(year, month, 1)
    first_friday = 1 + (4 - d.weekday()) % 7          # weekday: Mon=0..Fri=4
    return date(year, month, first_friday + 14)


def get_front_month_contract(symbol: str, as_of_date: Optional[date] = None,
                             rollover_days: int = 8) -> str:
    """Front-month CME equity-index contract code ``{SYMBOL}{H/M/U/Z}{Y}`` as of a date,
    rolling ``rollover_days`` calendar days before the third-Friday expiry.

    Note: uses a single-digit year ('5' = ...5); fine within a decade. The roll means a window
    spanning an expiry covers two contracts (e.g. ESH0 -> ESM0 across mid-March 2020)."""
    if as_of_date is None:
        as_of_date = date.today()
    for year in (as_of_date.year, as_of_date.year + 1):
        for month, code in FUTURES_MONTHS.items():
            expiry = get_third_friday(year, month)
            if as_of_date <= expiry - timedelta(days=rollover_days):   # timedelta: safe across months
                return f"{symbol.upper()}{code}{str(year)[-1]}"
    raise RuntimeError(f"Could not determine front month for {symbol} as of {as_of_date}")


def roll_window_days(as_of_date: date, rollover_days: int = 8) -> int:
    """SIGNED calendar days to the nearest front-month roll boundary: negative BEFORE, positive after.

    Signed, and both directions, because the two sides are not alike. Measured shares for the
    contract the calendar rule picks:

        2020-03-09   3 days BEFORE   ESH0   93.7%   <- concentrated; an ordinary session
        2020-03-12   AT the boundary ESH0   72.8%   <- split, and the calendar rule still picks right
        2020-03-16   4 days after    ESM0   60.4%   <- the trough
        2020-03-18   6 days after    ESM0   78.0%   <- recovering

    Before the boundary the front month is still the old contract and holds nearly everything. AFTER
    it, the new contract leads but the old one keeps a large share while its open interest unwinds
    (ESH0 open interest is still 2.69 M on 03-16 against ESM0's 1.43 M). So the post-roll days are
    the ones that need the warning -- and a forward-only measure is silent on exactly those, calling
    2020-03-16 "87 days from the June roll".

    The roll is not a switch, it is a week. Measured from the ES trade tapes across the March 2020
    roll, the front month carried:

        2020-03-16   ESM0 3,027,078 RTH lots vs ESH0 1,986,076   ->  60.4%
        2020-03-18   ESM0 2,605,122 RTH lots vs ESH0   732,903   ->  78.0%

    On an ordinary session the front month is essentially all of the volume. Here a single-contract
    ES leg misses 22-40% of futures activity, on two sessions that are IN the volatile panel. The
    calendar rule picks the right contract on both days -- it is not wrong -- but "right contract"
    and "the whole market" are different claims, and only the second is what a price-discovery
    estimate assumes.

    Splicing the two is NOT the fix: they are different instruments carrying a 10-12 index-point
    calendar spread, so a stitched series manufactures a jump at the seam. This is a sample fact to
    report, which is why it is a warning and not a correction."""
    best = 10 ** 6
    for year in (as_of_date.year - 1, as_of_date.year, as_of_date.year + 1):
        for month, _code in FUTURES_MONTHS.items():
            boundary = get_third_friday(year, month) - timedelta(days=rollover_days)
            d = (as_of_date - boundary).days              # <0 before the roll, >0 after
            if abs(d) < abs(best):
                best = d
    return best


def adjacent_contracts(symbol: str, as_of_date: date, rollover_days: int = 8) -> tuple:
    """(previous, front, next) quarterly codes around ``as_of_date``.

    The front month is a calendar rule; the contract it is competing with depends on which side of
    the boundary the date falls. Before the boundary the rival is the DEFERRED month (volume is
    starting to migrate to it); after it, the rival is the one that just stopped being front and is
    still unwinding its open interest. Both are returned so a roll measurement does not have to
    guess which pair to compare."""
    front = get_front_month_contract(symbol, as_of_date=as_of_date, rollover_days=rollover_days)
    codes = list(FUTURES_MONTHS.values())                # H, M, U, Z in cycle order
    i = codes.index(front[-2])
    yr = int(front[-1])
    prev_code, next_code = codes[(i - 1) % 4], codes[(i + 1) % 4]
    prev_yr = (yr - 1) % 10 if i == 0 else yr            # H rolls back to the previous year's Z
    next_yr = (yr + 1) % 10 if i == len(codes) - 1 else yr
    root = symbol.upper()
    return (f"{root}{prev_code}{prev_yr}", front, f"{root}{next_code}{next_yr}")


def measure_roll_at_extraction(label, symbol: str = "ES", rollover_days: int = 8,
                               window: tuple = (-3, 7), _measure=None):
    """Self-verifying contract pick: measure the roll split AT EXTRACTION for in-window sessions.

    The calendar rule (`get_front_month_contract`) stays the pick -- switching contracts on a
    same-day volume read would make the series definition data-dependent, which is worse for
    replication than a fixed rule with a reported coverage number. What changes is that the rule is
    now CHECKED where it can be wrong: any session within ``window`` (signed days around the roll
    boundary; -3..+7 covers both measured regimes -- pre-roll concentration and the post-roll
    unwind) gets a head-read of `mt_product_statistics` for the front month and its rival, and the
    log carries the measured volume and open-interest split instead of an extrapolation from the
    one roll ever measured (March 2020, a crisis week).

    -> (roll_offset_days, report_dict_or_None). None means out-of-window OR the measurement could
    not run (no `mstwx-lakequery` on PATH, vendor error) -- extraction NEVER fails over this, it
    just says so and falls back to the old unmeasured warning. ``_measure`` injects a stand-in for
    `check_roll.measure` in tests.

    Logging contract (all figures are the PRIOR session's closing totals from the statistics
    stream's pre-open rows -- the cheap discriminator; see check_roll._num for why max not last):
      * measured, calendar pick is the MINORITY contract  -> log.error (the extracted ES leg is
        the quieter book; the session needs a decision, not a silent pass)
      * measured, front share < 90%                       -> log.warning (SPLIT: the single-contract
        leg is not the whole futures market; report the share, do not splice)
      * measured, concentrated                            -> log.info
      * volume and OI disagree on which contract leads    -> extra log.info explaining stock vs flow
    """
    d = _parse_yyyymmdd(str(label).replace("-", ""))
    off = roll_window_days(d, rollover_days)
    if not (window[0] <= off <= window[1]):
        return off, None
    date_str = d.strftime("%Y-%m-%d")
    try:
        if _measure is None:
            import check_roll as _cr                      # lazy: check_roll imports this module
            _measure = _cr.measure
        rep = _measure(date_str, symbol, rollover_days)
    except Exception as exc:
        log.warning("%s: %+d day(s) from the roll boundary and the split could NOT be measured "
                    "(%s). Falling back to the unmeasured warning: volume is probably split "
                    "across contracts; run `check_roll.py --dates %s` where the lake is reachable.",
                    label, off, str(exc).splitlines()[0][:120], date_str)
        return off, None
    if not rep.get("measured"):
        log.warning("%s: %+d day(s) from the roll boundary; measurement ran but could not read "
                    "volume for both %s and %s (%s). The share the calendar pick carries is "
                    "UNKNOWN on this session.", label, off, rep.get("front"), rep.get("rival"),
                    rep.get("note", ""))
        return off, rep
    front, rival = rep["front"], rep["rival"]
    v, o, tr = rep["volume"], rep["open_interest"], rep.get("turnover", {})
    fs, oi_s = rep["front_share"], rep.get("oi_share", float("nan"))

    def _fmt(x):
        return "{:,}".format(int(x)) if np.isfinite(x) else "?"

    split = ("volume %s=%s (%.1f%%) vs %s=%s (%.1f%%); open interest %s=%s (%.1f%%) vs %s=%s "
             "(%.1f%%); turnover (vol/OI) %s=%.2f vs %s=%.2f"
             % (front, _fmt(v.get(front, np.nan)), 100 * fs,
                rival, _fmt(v.get(rival, np.nan)), 100 * (1 - fs),
                front, _fmt(o.get(front, np.nan)),
                100 * oi_s if np.isfinite(oi_s) else float("nan"),
                rival, _fmt(o.get(rival, np.nan)),
                100 * (1 - oi_s) if np.isfinite(oi_s) else float("nan"),
                front, tr.get(front, float("nan")), rival, tr.get(rival, float("nan"))))
    if not rep.get("rule_agrees", True):
        log.error("%s: THE CALENDAR RULE PICKED THE MINORITY CONTRACT: %s carries %.1f%% of the "
                  "two-contract volume at %+d day(s) from the boundary (%s). The extracted ES leg "
                  "is the QUIETER book. This is a sample-definition decision, not an auto-switch: "
                  "either re-extract this session pinned to %s and say so in the appendix, or keep "
                  "the rule and report the measured share -- but do not let it pass silently.",
                  label, front, 100 * fs, off, split, rival)
    elif fs < 0.90:
        log.warning("%s: ROLL SPLIT MEASURED at %+d day(s) from the boundary: %s. The calendar "
                    "pick %s leads but the single-contract ES leg misses %.1f%% of two-contract "
                    "futures volume on this session. Report the measured share in the sample "
                    "appendix; do NOT splice the contracts (10-12 point calendar spread jumps at "
                    "the seam).", label, off, split, front, 100 * (1 - fs))
    else:
        log.info("%s: roll split measured at %+d day(s) from the boundary: %s. Concentrated "
                 "(>=90%%) in the calendar pick %s -- the single-contract leg is effectively the "
                 "whole market here.", label, off, split, front)
    if np.isfinite(oi_s) and (fs >= 0.5) != (oi_s >= 0.5):
        log.info("%s: volume and open interest DISAGREE on which contract leads (vol %.1f%% vs OI "
                 "%.1f%% for %s). OI is a day-stale STOCK of positions settled once a day; volume "
                 "is the FLOW price discovery is made of -- the leg follows volume, and the "
                 "disagreement is recorded rather than assumed away.", label,
                 100 * fs, 100 * oi_s, front)
    return off, rep


def get_contract_expiry(contract_code: str, ref_year: Optional[int] = None) -> date:
    """Expiry (third Friday) for a contract code with a single-digit year, e.g. 'ESH5'.

    The single-digit year is decade-ambiguous; we resolve it to the decade of ``ref_year``
    (default: today's year). Pass ``ref_year`` when working with historical contracts whose
    decade differs from today's (e.g. ref_year=2020 for 'ESH0')."""
    code_to_month = {v: k for k, v in FUTURES_MONTHS.items()}
    year_str, month_code = contract_code[-1], contract_code[-2]
    if month_code not in code_to_month:
        raise ValueError(f"Invalid month code {month_code!r} in {contract_code!r}; "
                         f"valid: {list(code_to_month)}")
    base = ref_year if ref_year is not None else date.today().year
    decade = (base // 10) * 10
    year = decade + int(year_str)
    return get_third_friday(year, code_to_month[month_code])


def get_futures_root(product: str) -> str:
    """'ESM5' -> 'ES', 'RTYU5' -> 'RTY' (strip month code + single-digit year)."""
    return product[:-2].upper()


def canonical_root(product: str, product_type: str) -> str:
    """Canonical asset root used in the stack schema: 'SPY' for direct, futures root for futures."""
    return product.upper() if product_type == "direct" else get_futures_root(product)


# ════════════════════════════════════════════════════════════════════════════
# Raw -> canonical schema
# ════════════════════════════════════════════════════════════════════════════
_FIELD_RE = re.compile(r"^(bid|ask|bsize|asize)(\d+)$")
_FIELD_MAP = {"bid": "bidprice", "ask": "askprice", "bsize": "bidquantity", "asize": "askquantity"}


def _canon_field(field: str, root: str) -> Optional[str]:
    """Map a raw book field to the canonical column name, or None if not a book field."""
    m = _FIELD_RE.match(field)
    if m:
        return f"{root}_{_FIELD_MAP[m.group(1)]}_{int(m.group(2))}"
    if field == "mid":
        return f"{root}_mid"
    return None


def _apply_level_integrity(df: pd.DataFrame, root: str, levels: int) -> None:
    """In place: a level counts only if BOTH legs are finite. A torn level (one leg NaN from a
    shift) is set to NaN on price AND size, so every consumer treats it as absent (this is the
    same rule lcm._usable_levels enforces at metric time; applying it here makes the emitted
    frame clean for any consumer, including ones that bypass the stack readers)."""
    for side in ("bid", "ask"):
        for n in range(1, levels + 1):
            pc, qc = f"{root}_{side}price_{n}", f"{root}_{side}quantity_{n}"
            if pc in df.columns and qc in df.columns:
                bad = ~(np.isfinite(df[pc].to_numpy(float)) & np.isfinite(df[qc].to_numpy(float)))
                if bad.any():
                    df.loc[bad, pc] = np.nan
                    df.loc[bad, qc] = np.nan


def _to_canonical(raw_output: str, product: str, product_type: str, levels: int,
                  prefix: Optional[str] = None, tz: str = "America/New_York",
                  level_integrity: bool = True) -> pd.DataFrame:
    """Parse raw mstbook-query stdout (a CSV string) into a canonical, float, tz-aware frame.

    Columns become {ROOT}_{bid|ask}{price|quantity}_{i} (+ optional {ROOT}_mid). Object columns
    are coerced to numeric; the joint price/size integrity rule is applied per level."""
    if not raw_output.strip():
        raise ValueError("mstbook-query returned empty output.")
    raw = pd.read_csv(StringIO(raw_output), low_memory=False)

    tcol = next((c for c in ("time", "datetime") if c in raw.columns), None)
    if tcol is None:
        raise ValueError(f"Expected a 'time' or 'datetime' column; got {raw.columns.tolist()}")
    idx = pd.to_datetime(raw[tcol])

    root = canonical_root(product, product_type)
    prefix = prefix if prefix is not None else product_type          # raw mstbook prefix
    canon: dict[str, np.ndarray] = {}
    for col in raw.columns:
        if col == tcol or "." not in col:
            continue
        pre, field = col.split(".", 1)
        if pre != prefix:
            continue
        name = _canon_field(field, root)
        if name is not None:
            canon[name] = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)
    if not any(k.endswith("price_1") for k in canon):
        raise ValueError(f"No level-1 book columns parsed for prefix {prefix!r}; "
                         f"raw columns: {raw.columns.tolist()[:8]}...")

    out = pd.DataFrame(canon, index=pd.DatetimeIndex(idx))
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)
    out.index.name = "time"
    out = out.sort_index()
    if level_integrity:
        _apply_level_integrity(out, root, levels)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Derived features -- via the stack's CORRECT implementations
# ════════════════════════════════════════════════════════════════════════════
def attach_features(df: pd.DataFrame, asset: str, n_levels: int = 10,
                    decay_scale: Optional[float] = None) -> pd.DataFrame:
    """Append derived liquidity features to a canonical frame, computed by the stack's own
    (correct) functions so there is exactly one definition of each metric:

      {asset}_mid             simple quote midpoint (level 1)
      {asset}_wmid            inside microprice  (lcm.weighted_mid; opposite-side weighting)
      {asset}_depth_wmid      depth-weighted mid (lcm.depth_weighted_mid across n_levels)
      {asset}_quoted_spread_bps   (ask1 - bid1)/mid * 1e4
      {asset}_auc_decay       decay-weighted cost to demand size, bps (lcm.decay_weighted_cost)
      {asset}_curve_length    impact-curve convexity in [sqrt2, 2] (lcm.impact_curve_length)
    """
    out = df.copy()
    b1 = df[f"{asset}_bidprice_1"].to_numpy(float)
    a1 = df[f"{asset}_askprice_1"].to_numpy(float)
    mid = (b1 + a1) / 2.0
    out[f"{asset}_mid"] = mid
    out[f"{asset}_wmid"] = lcm.weighted_mid(df, asset)
    out[f"{asset}_depth_wmid"] = lcm.depth_weighted_mid(df, asset, n_levels)
    out[f"{asset}_quoted_spread_bps"] = np.where(mid > EPS, (a1 - b1) / mid * 1e4, np.nan)
    out[f"{asset}_auc_decay"] = lcm.decay_weighted_cost(df, asset, n_levels, decay_scale=decay_scale)
    out[f"{asset}_curve_length"] = lcm.impact_curve_length(df, asset, n_levels)
    return out


def vwap_to_lot(df: pd.DataFrame, asset: str, side: Literal["bid", "ask"], lot: float,
                n_levels: int = 10):
    """Vectorized VWAP price to fill ``lot`` units on one side by walking the book.

    side='ask' = the average price to BUY ``lot`` (lift offers); side='bid' = to SELL ``lot``.
    Returns (vwap[T], filled[T], levels_used[T]). Partial fills (book too thin) -> vwap NaN, and
    torn levels are excluded via the joint price/size rule. O(T*N) in numpy (no Python row loop)."""
    P = np.column_stack([df[f"{asset}_{side}price_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
    Q = np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
    P, Q = lcm._usable_levels(P, Q)                       # torn level -> (NaN price, 0 size) = absent
    cumQ = np.cumsum(Q, axis=1)
    cum_prev = cumQ - Q
    fill = np.clip(lot - cum_prev, 0.0, Q)                # per-level quantity consumed
    Peff = np.where(np.isfinite(P), P, 0.0)               # NaN price only where fill is already 0
    cost = np.sum(Peff * fill, axis=1)
    filled = np.sum(fill, axis=1)                         # = min(lot, total usable depth)
    vwap = np.where(filled >= lot - EPS, cost / lot, np.nan)   # NaN unless the lot fully fills
    levels_used = (fill > 0).sum(axis=1).astype(int)
    return vwap, filled, levels_used


def default_lot(product_type: str, product: str) -> float:
    """Convenience default fill size: 100 shares (direct) or 1 contract (futures)."""
    return float(DIRECT_ROUND_LOT) if product_type == "direct" else 1.0


# ════════════════════════════════════════════════════════════════════════════
# mstbook-query runner
# ════════════════════════════════════════════════════════════════════════════
def _run_mstbook_query(cmd: list) -> str:
    """Run mstbook-query, trying '-e time' then '-e datetime'. Returns stdout."""
    last = None
    for e_flag in ("time", "datetime"):
        attempt, skip = [], False
        for tok in cmd:
            if skip:
                attempt.append(e_flag); skip = False; continue
            if tok == "-e":
                skip = True
            attempt.append(tok)
        last = sp.run(attempt, capture_output=True, text=True, check=False)
        if last.returncode == 0:
            return last.stdout
    raise RuntimeError(f"mstbook-query failed with both '-e time' and '-e datetime'.\n"
                       f"stderr: {(last.stderr or '').strip()[:300]}")


def _build_cmd(date_str: str, product: str, prefix: str, interval: str, start_time: str,
               end_time: str, levels: int, data_source: str) -> list:
    cmd = ["mstbook-query", str(date_str).replace("-", ""), "-s", prefix, "-p", product,
           "--data-source", data_source, "-e", "time", "--interval", interval,
           "--start-time", start_time, "--end-time", end_time, "-q", f"{prefix}.mid"]
    for i in range(1, levels + 1):
        cmd += ["-q", f"{prefix}.bid{i}", "-q", f"{prefix}.bsize{i}"]
    for i in range(1, levels + 1):
        cmd += ["-q", f"{prefix}.ask{i}", "-q", f"{prefix}.asize{i}"]
    return cmd


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════
def query_canonical(date_str: str, product: str, product_type: Literal["direct", "futures"],
                    levels: int = 10, interval: str = "1s", start_time: str = "9:30",
                    end_time: str = "16:00", data_source: str = "apu",
                    tz: str = "America/New_York", with_features: bool = False,
                    decay_scale: Optional[float] = None, price_scale: float = 1.0) -> pd.DataFrame:
    """[LEGACY / VALIDATION-ONLY] Pull one product's book via the vendor ``mstbook-query`` snapshot.

    NOT on the production path. ``mstbook-query`` is on a different clock from ``mstwx-lakequery`` and is
    being sunset, so production extraction is message reconstruction (``lob_reconstruct``, the
    ``extract_sessions`` path). This is retained solely so ``validate_reconstruction`` can benchmark a
    reconstruction against the vendor book on an archived day (a one-time historical bias check; expect a
    small timestamp-driven L1/mid disagreement consistent with the cross-tool clock offset -- the
    price/size *levels* are what should agree). Returns the canonical stack schema (float, tz-aware);
    ``with_features`` appends derived liquidity columns; ``price_scale`` multiplies price columns
    (0.01 for CME futures index points)."""
    if product_type not in ("direct", "futures"):
        raise ValueError(f"product_type must be 'direct' or 'futures', got {product_type!r}")
    cmd = _build_cmd(date_str, product, product_type, interval, start_time, end_time,
                     levels, data_source)
    raw = _run_mstbook_query(cmd)
    df = _to_canonical(raw, product, product_type, levels, prefix=product_type, tz=tz)
    if price_scale != 1.0:
        for c in df.columns:
            if "price" in c or c.endswith("_mid"):
                df[c] = df[c].to_numpy(float) * price_scale
    if with_features:
        df = attach_features(df, canonical_root(product, product_type), levels, decay_scale)
    return df


def _parse_yyyymmdd(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y%m%d").date()


def query_pair(date_str: str, regime: str = "auto", es_symbol: str = "ES", levels: int = 10,
               rollover_days: int = 8, **kw) -> tuple:
    """Pull SPY (direct) and the front-month equity-index future (default ES) for one date and
    merge to a single canonical session frame on the timestamp index (outer join; the stack's
    _align_books then coerces/ffills/trims). Returns ``(date_str, regime, df)`` ready for
    run_contagion / mean_variance. The ES contract is chosen by ``get_front_month_contract``."""
    spy = query_canonical(date_str, "SPY", "direct", levels=levels, **kw)
    contract = get_front_month_contract(es_symbol, as_of_date=_parse_yyyymmdd(date_str),
                                        rollover_days=rollover_days)
    es = query_canonical(date_str, contract, "futures", levels=levels, **kw)
    df = spy.join(es, how="outer").sort_index()
    df.attrs["es_contract"] = contract
    return (date_str, regime, df)


def load_sessions(specs: Sequence[tuple], es_symbol: str = "ES", levels: int = 10, **kw) -> list:
    """Build a list of ``(date, regime, df)`` sessions from ``specs`` -- an iterable of
    ``date_str`` or ``(date_str, regime)`` -- via ``query_pair``. Feeds run_contagion directly."""
    out = []
    for spec in specs:
        if isinstance(spec, (tuple, list)):
            date_str, regime = spec[0], (spec[1] if len(spec) > 1 else "auto")
        else:
            date_str, regime = spec, "auto"
        out.append(query_pair(date_str, regime=regime, es_symbol=es_symbol, levels=levels, **kw))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Message-type layer (mstwx-lakequery)  -- the trade tape / order-flow counts
# that the Athena extract used to provide. This is what lets --source extract
# drop SQL entirely.
# ════════════════════════════════════════════════════════════════════════════
MESSAGE_TYPES = ("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade")
# which column drives the event/ordering clock. Under LSEG/MayStreet source capture at CyrusOne
# Aurora I (PCAP tap next to the CME engine) each feed is hardware GPS-stamped at its origin colo,
# so receipt is one uniform UTC methodology across venues -> the defensible default. Exchange time
# carries heterogeneous per-venue publication semantics and is offered only as a robustness lens.
# mt_aggregated_price_update names its clocks last{receipt,exchange}timestamp -- it is a SNAPSHOT of
# the whole ladder after a burst, not a single event, so the column says "as of". Without these
# aliases that type cannot be fetched at all (no recognized clock column -> ValueError).
_CLOCK_COLS = {"receipt": ("receipttimestamp", "lastreceipttimestamp"),
               "exchange": ("exchangetimestamp", "exchange_timestamp", "sourcetimestamp",
                            "lastexchangetimestamp")}
_MSG_QTY_COL = {"mt_add_order": "quantity", "mt_cancel_order": "previousquantity",
                "mt_modify_order": "quantity", "mt_trade": "quantity",
                # MBP (price-level) feeds, for the hybrid book reconstruction (e.g. IEX):
                "mt_price_level_update": "quantity", "mt_modify_price_level": "quantity",
                "mt_delete_price_level": "quantity",
                # feed resets (mt_clear_orders / mt_clear_price_levels) carry no quantity at all -- they
                # void everything the venue has published so far. Tiny and read in FULL (not in
                # _BOOK_MSG_TYPES), so no column pruning can drop what they do carry.
                "mt_clear_orders": "quantity", "mt_clear_price_levels": "quantity",
                # data-quality oracles: gaps in the venue's own sequence, and feed decode errors. Not
                # replayed -- read by the diagnostics to tell DATA loss from a CODE fault.
                "mt_missing_product_messages": "quantity", "mt_error": "quantity",
                # INDEPENDENT VALIDATION of the replay, not inputs to it. mt_aggregated_price_update is
                # the venue's own 10-level ladder (quantity AND order count per level) and
                # mt_product_statistics its session high/low/volume/VWAP/settlement -- both from the same
                # lake, on the same capture clock as the messages we replay, which is exactly the
                # objection that retired mstbook-query as a benchmark. See validate_aggregated.py.
                "mt_aggregated_price_update": "bidquantity_1", "mt_product_statistics": "volume",
                # mt_product_status (no trailing S) is the VENUE STATE stream: haltreason,
                # marketsession (PreOpen/EarlySession/CoreSession/LateSession/Closed), LULD bands and
                # the SymbolClear trading event. It is where the circuit-breaker halt actually lives
                # -- on 2020-03-09 five feeds publish MarketWideCircuitBreakerLevel1 at 09:34:13.0787
                # and clear it 900.0 s later -- so the halt window is read rather than hardcoded.
                "mt_product_status": "roundlotquantity",
                # trade-tape hygiene (busted / corrected prints), scrubbed before aggregation:
                "mt_trade_break": "quantity", "mt_trade_correction": "newquantity",
                # auction (opening/closing cross) imbalance -- feature input (auction_imbalance.py):
                "mt_order_imbalance": "pairedquantity"}


def _run_mstwx_lakequery(cmd: list) -> str:
    """Run mstwx-lakequery and return stdout (raises on non-zero exit)."""
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"mstwx-lakequery failed (rc={res.returncode}): "
                           f"{(res.stderr or '').strip()[:300]}")
    return res.stdout


try:
    import pyarrow as _pa   # noqa: F401 -- the Arrow CSV engine is available but OPT-IN (see below)
    _HAVE_PYARROW = True
    try:
        _pa.set_cpu_count(1)            # avoid N-workers x M-parse-threads oversubscription if it IS used
        _pa.set_io_thread_count(1)
    except Exception:
        pass
except Exception:
    _HAVE_PYARROW = False
# The C engine + memory_map is the DEFAULT parse path. In the parallel-extraction context it is the
# safer choice: single-threaded and mmap-backed, it has a predictable, low per-worker peak and cannot
# oversubscribe across the workers, whereas the Arrow engine spawns a parse-thread pool inside EACH
# worker. The Arrow engine's compact-string benefit is also unreliable here -- `dtype="string"` is
# python-backed on pandas 2.x (object-level memory), and any compaction is lost the moment the replay
# calls .to_numpy()/.astype(str). So Arrow is opt-in (MST_CSV_ENGINE=pyarrow), useful mainly single-worker.
_USE_PYARROW = _HAVE_PYARROW and os.environ.get("MST_CSV_ENGINE", "c").lower() in ("pyarrow", "arrow")

# Columns the book reconstruction + flow aggregation + both clocks actually consume. `usecols` keeps
# only those that are present (a tolerant intersection), dropping the rest of lakequery's wide schema
# before it is ever materialized -- the single biggest *safe* memory lever (no value change at all).
_MSG_NEEDED_COLS = frozenset({
    "receipttimestamp", "exchangetimestamp", "exchange_timestamp", "sourcetimestamp",   # clocks
    # The venue's authoritative within-feed event order. lob_reconstruct._event_arrays orders each
    # feed by this and only interleaves feeds by the clock; without it the replay silently falls back
    # to clock-only ordering, which inverts adjacent events (UDP reordering / coarse engine stamps),
    # lets a Modify resurrect an already-Cancelled order, and pins a phantom level that crosses the
    # top on essentially every snapshot (see test_reconstruct_ordering.py: 100% crossed vs 0%).
    # Pruning it here disabled that fix on ALL real data while the synthetic tests -- which build
    # frames directly and never pass through this reader -- kept passing.
    "sequencenumber",                                                                   # intra-feed order
    "f", "side", "price", "previousprice",
    "quantity", "previousquantity", "newquantity", "pairedquantity",                    # qty variants by type
    "orderreferencenumber", "previousorderreferencenumber",                             # book keys
    "maintainpriority", "admindelete", "aggressorside", "printable",                    # modify/trade hygiene
    "matchid", "tradereferencenumber",                                                  # trade-bust matching (mt_trade)
    # mt_trade only, and both were being pruned away exactly like `sequencenumber` was:
    #   leavesquantity     -- the venue's OWN remaining size on the resting order AFTER this execution.
    #                         Authoritative where a decrement is only arithmetic, and leaves=0 removes a
    #                         fully filled order deterministically (an unremoved one rests forever and
    #                         pins the top). Verified on tape: ref ...546706 prints 1198 -> 1194 -> 1192
    #                         on trades of 2, 4, 2.
    #   executionattribute -- 'Hidden' marks an execution against NON-DISPLAYED liquidity. Those prints
    #                         carry no orderreferencenumber by construction, so they are not evidence of
    #                         a broken replay and must not consume displayed size that is still resting.
    "leavesquantity", "executionattribute",
})
# Prune `usecols` ONLY for the high-volume book/trade message types (millions of rows on crash days).
# The low-volume auxiliary types -- trade-break/correction and auction order-imbalance -- carry
# idiosyncratic columns the scrubber and auction module consume (matchid, newprice, referenceprice,
# indicativeprice, ...). Pruning them saves no memory (they are tiny) and risks dropping a needed
# field, which silently corrupts the bust scrub / auction features -- so those are read in FULL.
_BOOK_MSG_TYPES = frozenset({
    "mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade",
    "mt_price_level_update", "mt_modify_price_level", "mt_delete_price_level",
})
# Non-numeric columns -> read as pandas "string" (Arrow-backed and compact WHEN pyarrow is present;
# python-backed otherwise). Numerics and the ns timestamp are left to engine inference and the existing
# pd.to_numeric coercion, so their values stay byte-identical to the legacy object-inference path.
_MSG_STR_COLS = frozenset({
    "f", "side", "orderreferencenumber", "previousorderreferencenumber",
    "maintainpriority", "admindelete", "aggressorside", "printable",
})


# Where the streamed CSV temp files land. MUST be disk-backed: if it points at a tmpfs /tmp (RAM),
# streaming the multi-GB CSV there would consume RAM and defeat the purpose. Defaults to the working
# directory (on MIDAS the package dir on /home, disk-backed); override with MST_TMPDIR for a scratch FS.
_MST_TMPDIR = os.environ.get("MST_TMPDIR") or os.getcwd()


class MessageFetchError(RuntimeError):
    """A message-type fetch failed outright (the vendor query errored), as opposed to returning
    an empty result. The distinction matters: an EMPTY mt_add_order stream is a statement about
    the day, a FAILED one is a statement about the query, and a book replayed without its adds
    (or without its trades/cancels) is not an incomplete book -- it is a fabricated one. Raised
    with the date and product attached so a parallel session log can attribute it."""


# A vendor query that exits non-zero is usually transient (lake throttling, a dropped connection,
# a worker OOM under N-way parallel extraction) rather than a statement that the data is absent --
# the same date/product/type combination typically succeeds on a retry. One un-retried rc=1 used to
# abort an entire multi-hour extraction, so failures are retried with exponential backoff before
# they are believed. Tune with MST_LAKEQUERY_RETRIES / MST_LAKEQUERY_BACKOFF (seconds).
_LAKEQUERY_RETRIES = max(1, int(os.environ.get("MST_LAKEQUERY_RETRIES", "3")))
_LAKEQUERY_BACKOFF = max(0.0, float(os.environ.get("MST_LAKEQUERY_BACKOFF", "5")))


def _run_mstwx_lakequery_to_file(cmd: list, path: str, retries: Optional[int] = None,
                                 backoff: Optional[float] = None) -> None:
    """Run mstwx-lakequery and stream stdout straight to ``path`` (raises on non-zero exit).
    Unlike :func:`_run_mstwx_lakequery`, the full CSV is never materialized as a Python string --
    on a crash day that string (+ its StringIO copy) is several GB, so writing to a file the parser
    can mmap removes the largest transient at the read spike.

    A non-zero exit is retried ``retries`` times with exponential backoff (``backoff``,
    2x``backoff``, ...); only the last failure raises. Each attempt truncates ``path``, so a
    partial CSV from a failed attempt is never parsed."""
    n = _LAKEQUERY_RETRIES if retries is None else max(1, int(retries))
    b = _LAKEQUERY_BACKOFF if backoff is None else max(0.0, float(backoff))
    err = ""
    for attempt in range(1, n + 1):
        with open(path, "w") as fh:                       # "w" truncates: no partial-attempt residue
            res = sp.run(cmd, stdout=fh, stderr=sp.PIPE, text=True, check=False)
        if res.returncode == 0:
            if attempt > 1:
                log.info("mstwx-lakequery succeeded on attempt %d/%d: %s", attempt, n, " ".join(cmd[:8]))
            return
        err = (res.stderr or "").strip()[:300]
        if attempt < n:
            wait = b * (2 ** (attempt - 1))
            log.warning("mstwx-lakequery rc=%d on attempt %d/%d (%s); retrying in %.0fs: %s",
                        res.returncode, attempt, n, " ".join(cmd[:8]), wait, err.splitlines()[0][:160]
                        if err else "(no stderr)")
            if wait:
                time.sleep(wait)
    raise RuntimeError(f"mstwx-lakequery failed (rc={res.returncode}) after {n} attempt(s): {err}")


def _csv_header(path: str) -> list:
    """Read just the header line of a CSV (no body), returning the column names."""
    try:
        with open(path, "r") as fh:
            line = fh.readline()
    except OSError:
        return []
    return [c.strip() for c in line.rstrip("\n").split(",")] if line else []


def _read_messages_csv(path: str, message_type: str = "") -> pd.DataFrame:
    """Parse a lakequery message CSV from ``path``. For the high-volume **book** message types
    (``_BOOK_MSG_TYPES``) keep only the columns reconstruction needs (the big memory lever); for the
    low-volume auxiliary types (trade-break/correction, auction order-imbalance) read ALL columns,
    since they are tiny and carry idiosyncratic fields the scrubber/auction consume. Non-numeric
    columns are read as compact strings. Prefers the Arrow engine (multithreaded, Arrow-backed
    columns) and falls back to the C engine with ``memory_map`` -- which is mutually exclusive with
    the Arrow engine -- if pyarrow is absent or rejects an argument. Header-driven ``usecols``/``dtype``
    (built from columns actually present) keep both engines happy across the differing schemas."""
    header = _csv_header(path)
    if not header:
        return pd.DataFrame()
    if message_type in _BOOK_MSG_TYPES:
        use = [c for c in header if c in _MSG_NEEDED_COLS] or None   # prune the wide book schema
    else:
        use = None                                                   # aux type: read in full
    sdt = {c: "string" for c in (use or header) if c in _MSG_STR_COLS}
    if _USE_PYARROW:
        try:
            return pd.read_csv(path, engine="pyarrow", usecols=use, dtype=(sdt or None))
        except Exception as exc:                                 # any Arrow incompatibility -> C fallback
            log.debug("pyarrow CSV engine fell back to C: %s", exc)
    return pd.read_csv(path, engine="c", memory_map=True, low_memory=False,
                       usecols=use, dtype=(sdt or None))


def _fetch_messages(date_str: str, product: str, product_type: str, message_type: str,
                    data_source: str = "apu", tz: str = "America/New_York",
                    clock: str = "receipt") -> pd.DataFrame:
    """Fetch raw messages for one type; index = the chosen clock (ns since epoch) as a tz-aware
    index in ``tz`` (UTC is unambiguous from ns, then converted to match the book frames). ``clock``
    selects the timestamp column: ``"receipt"`` (default; the GPS source-capture time, one uniform
    methodology across venues) or ``"exchange"`` (per-venue matching-engine time -- a robustness
    lens, heterogeneous across feeds). The relevant quantity column is coerced to numeric (missing
    -> NaN, never silently 0)."""
    if message_type not in _MSG_QTY_COL:
        raise ValueError(f"Unknown message_type {message_type!r}; valid: {list(_MSG_QTY_COL)}")
    cmd = ["mstwx-lakequery", "--date", str(date_str).replace("-", ""), "-s", product_type,
           "-p", product, "-m", message_type, "--print-headers", "--format", "csv"]
    fd, tmp = tempfile.mkstemp(prefix=f"mstwx_{product}_{message_type}_", suffix=".csv", dir=_MST_TMPDIR)
    os.close(fd)
    try:
        _run_mstwx_lakequery_to_file(cmd, tmp)          # stream stdout to disk (no multi-GB string)
        df = _read_messages_csv(tmp, message_type)      # prune book types; aux types read full
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if df.empty:
        return df
    cands = _CLOCK_COLS.get(clock, _CLOCK_COLS["receipt"])
    tcol = next((c for c in cands if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"clock={clock!r}: expected one of {cands} in {message_type} output; "
                         f"got {df.columns.tolist()}")
    idx = pd.to_datetime(df[tcol], unit="ns", utc=True).dt.tz_convert(tz)
    df = df.drop(columns=[tcol]).set_index(idx)
    df.index.name = "time"
    qty = _MSG_QTY_COL[message_type]
    for c in (qty, "price"):                                  # coerce to numeric (Athena/object-safe)
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _aggregate_messages(df: pd.DataFrame, message_type: str, interval: str = "1s") -> pd.DataFrame:
    """Resample raw messages to fixed buckets: msg_count, total_quantity, bid/ask msg_count and
    quantity; for mt_trade also last_trade_price and trade_vwap. Empty buckets -> count 0,
    quantity/price NaN."""
    cols = ["msg_count", "total_quantity", "bid_msg_count", "ask_msg_count",
            "bid_total_quantity", "ask_total_quantity"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    qcol = _MSG_QTY_COL.get(message_type, "quantity")
    qty = pd.to_numeric(df[qcol], errors="coerce") if qcol in df.columns else pd.Series(np.nan, index=df.index)
    side = df["side"].astype(str).str.strip() if "side" in df.columns else pd.Series("", index=df.index)
    is_bid, is_ask = side.eq("Bid"), side.eq("Ask")
    work = pd.DataFrame({"qty": qty, "is_bid": is_bid.astype(float), "is_ask": is_ask.astype(float),
                         "bid_qty": qty.where(is_bid), "ask_qty": qty.where(is_ask)}, index=df.index)
    if message_type == "mt_trade" and "price" in df.columns:
        px = pd.to_numeric(df["price"], errors="coerce")
        work["price"] = px
        work["pxqty"] = px * qty
    rs = work.resample(interval)
    agg = pd.DataFrame({
        "msg_count": rs["qty"].count(),
        "total_quantity": rs["qty"].sum(min_count=1),
        "bid_msg_count": rs["is_bid"].sum(),
        "ask_msg_count": rs["is_ask"].sum(),
        "bid_total_quantity": rs["bid_qty"].sum(min_count=1),
        "ask_total_quantity": rs["ask_qty"].sum(min_count=1),
    })
    if message_type == "mt_trade" and "price" in work.columns:
        agg["last_trade_price"] = rs["price"].last()
        agg["trade_vwap"] = rs["pxqty"].sum(min_count=1) / rs["qty"].sum(min_count=1)
    agg.index.name = "time"
    return agg


def query_messages(date_str: str, product: str, product_type: str, message_type: str,
                   interval: str = "1s", start_time: Optional[str] = None,
                   end_time: Optional[str] = None, data_source: str = "apu",
                   tz: str = "America/New_York", return_raw: bool = False) -> pd.DataFrame:
    """Fetch + bucket one message type. Optional ``start_time``/``end_time`` ('HH:MM', local tz)
    window the session. ``return_raw=True`` returns the unaggregated messages."""
    _validate = canonical_root(product, product_type)         # validates product_type via root logic
    raw = _fetch_messages(date_str, product, product_type, message_type, data_source, tz=tz)
    if raw.empty:
        return raw
    d = _parse_yyyymmdd(date_str)
    if start_time is not None:
        raw = raw[raw.index >= pd.Timestamp(f"{d} {start_time}", tz=tz)]
    if end_time is not None:
        raw = raw[raw.index <= pd.Timestamp(f"{d} {end_time}", tz=tz)]
    return raw if return_raw else _aggregate_messages(raw, message_type, interval)


def _tick_sign(px: np.ndarray) -> np.ndarray:
    """Lee-Ready tick test: +1 on an uptick, -1 on a downtick, zero-ticks carry the last non-zero
    sign forward. First trade -> 0 (unclassified)."""
    px = np.asarray(px, float)
    s = np.sign(np.diff(px, prepend=px[0] if px.size else 0.0))
    s = pd.Series(np.where(s == 0, np.nan, s)).ffill().fillna(0.0).to_numpy()
    return s


def _trade_sign(df: pd.DataFrame, classify: str = "aggressor", side_buy_label: str = "Bid") -> np.ndarray:
    """Per-trade sign (+1 buyer-initiated, -1 seller-initiated, 0/NaN unclassified).

    classify='aggressor' (default): use ``aggressorside`` ('Buy'/'Sell') where present -- authoritative
      for CME futures -- and fall back to the tick rule for rows without it (equities, where the feed
      leaves aggressorside blank). 'tick': tick rule for all (a single, cross-asset-consistent rule).
      'side': use the ``side`` field (side_buy_label, default 'Bid', = buyer-initiated -- the mapping the
      CME data shows: Buy<->Bid, Sell<->Ask)."""
    n = len(df)
    sign = np.full(n, np.nan)
    if classify in ("aggressor",) and "aggressorside" in df.columns:
        a = df["aggressorside"].astype(str).str.strip().str.lower()
        sign = np.where(a.eq("buy"), 1.0, np.where(a.eq("sell"), -1.0, np.nan))
    if classify == "side" and "side" in df.columns:
        s = df["side"].astype(str).str.strip()
        other = "Ask" if side_buy_label == "Bid" else "Bid"
        sign = np.where(s.eq(side_buy_label), 1.0, np.where(s.eq(other), -1.0, np.nan))
    if classify == "tick" or np.isnan(sign).any():        # tick fallback (equities under 'aggressor')
        tick = _tick_sign(pd.to_numeric(df.get("price"), errors="coerce").to_numpy())
        sign = tick if classify == "tick" else np.where(np.isnan(sign), tick, sign)
    return np.nan_to_num(sign)


def _aggregate_trades(df: pd.DataFrame, interval: str, classify: str = "aggressor",
                      side_buy_label: str = "Bid", drop_nonprintable: bool = True) -> pd.DataFrame:
    """Bucket the trade tape into buy/sell counts and volumes (signed via _trade_sign), last price
    and VWAP. NonPrintable prints (hidden / certain conditions) are dropped by default; odd lots are
    kept (they are real executions -- note they do not move the SIP last/NBBO)."""
    cols = ["trade_count", "trade_volume", "trade_buy", "trade_sell",
            "trade_buy_volume", "trade_sell_volume", "last_trade_price", "trade_vwap"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df
    if drop_nonprintable and "printable" in d.columns:
        d = d[d["printable"].astype(str).str.strip().ne("NonPrintable")]
    if d.empty:
        return pd.DataFrame(columns=cols)
    px = pd.to_numeric(d.get("price"), errors="coerce")
    qty = pd.to_numeric(d.get("quantity"), errors="coerce")
    sign = _trade_sign(d, classify=classify, side_buy_label=side_buy_label)
    buy_m, sell_m = sign > 0, sign < 0
    work = pd.DataFrame({"px": px, "qty": qty, "buy": buy_m.astype(float), "sell": sell_m.astype(float),
                         "buy_vol": qty.where(buy_m), "sell_vol": qty.where(sell_m),
                         "pxqty": px * qty}, index=d.index)
    rs = work.resample(interval)
    return pd.DataFrame({
        "trade_count": rs["qty"].count(),
        "trade_volume": rs["qty"].sum(min_count=1),
        "trade_buy": rs["buy"].sum(),
        "trade_sell": rs["sell"].sum(),
        "trade_buy_volume": rs["buy_vol"].sum(min_count=1),
        "trade_sell_volume": rs["sell_vol"].sum(min_count=1),
        "last_trade_price": rs["px"].last(),
        "trade_vwap": rs["pxqty"].sum(min_count=1) / rs["qty"].sum(min_count=1),
    })


def _scrub_trades(trades: pd.DataFrame, date_str: str, product: str, product_type: str,
                  data_source: str = "apu", tz: str = "America/New_York") -> tuple:
    """Drop busted prints and apply price/size corrections to a raw trade frame before aggregation.

    Exchanges bust and correct trades, and they do so most on volatile days (there are real SPY busts
    and a $10 correction on 2025-04-03, and ES busts across the April-2025 window) -- a tape that keeps
    a print which never settled contaminates trade signing and any trade-based flow exactly when it
    matters. Busts are matched by the first id column shared with ``mt_trade_break`` (``matchid``, else
    ``tradereferencenumber``); corrections overwrite price/quantity by ``matchid`` from
    ``mt_trade_correction``. Returns (scrubbed_trades, stats). NOTE: this scrubs the same-day tape;
    a late bust disseminated the next session is not caught here (rare). The reconstruction book is
    not scrubbed -- a handful of busted reductions are negligible on a level snapshot, and MBP feeds
    self-correct via level updates."""
    if trades is None or len(trades) == 0:
        return trades, {"n0": 0, "broken": 0, "corrected": 0}
    n0 = len(trades)
    try:
        brk = _fetch_messages(date_str, product, product_type, "mt_trade_break", data_source, tz=tz)
    except Exception:
        brk = pd.DataFrame()
    try:
        cor = _fetch_messages(date_str, product, product_type, "mt_trade_correction", data_source, tz=tz)
    except Exception:
        cor = pd.DataFrame()
    broken = 0
    key = next((k for k in ("matchid", "tradereferencenumber")
                if k in trades.columns and len(brk) and k in brk.columns), None)
    if key is not None:
        bad = set(pd.to_numeric(brk[key], errors="coerce").dropna().astype("int64").tolist())
        if bad:
            tk = pd.to_numeric(trades[key], errors="coerce")
            mask = tk.isin(bad)
            broken = int(mask.sum())
            trades = trades.loc[~mask]
    corrected = 0
    if len(cor) and "matchid" in trades.columns and "matchid" in cor.columns:
        cmap = {}
        mid = pd.to_numeric(cor["matchid"], errors="coerce")
        npx = pd.to_numeric(cor.get("newprice"), errors="coerce")
        nqty = pd.to_numeric(cor.get("newquantity"), errors="coerce")
        for j in range(len(cor)):
            if np.isfinite(mid.iloc[j]):
                cmap[int(mid.iloc[j])] = (npx.iloc[j], nqty.iloc[j])
        if cmap:
            tk = pd.to_numeric(trades["matchid"], errors="coerce")
            trades = trades.copy()
            for i, mv in tk.items():
                if np.isfinite(mv) and int(mv) in cmap:
                    np_, nq_ = cmap[int(mv)]
                    if np.isfinite(np_) and "price" in trades.columns:
                        trades.at[i, "price"] = np_
                    if np.isfinite(nq_) and "quantity" in trades.columns:
                        trades.at[i, "quantity"] = nq_
                    corrected += 1
    return trades, {"n0": n0, "broken": broken, "corrected": corrected}


def trade_flow(date_str: str, product: str, product_type: Literal["direct", "futures"],
               interval: str = "1s", start_time: str = "9:30", end_time: str = "16:00",
               data_source: str = "apu", tz: str = "America/New_York", classify: str = "aggressor",
               side_buy_label: str = "Bid", price_scale: float = 1.0, scrub: bool = True) -> pd.DataFrame:
    """Per-bucket trade-tape flow for one product, columns named for the canonical root:
    {ROOT}_trade_buy / _trade_sell (counts), _trade_buy_volume / _trade_sell_volume, _trade_px
    (last price) and _trade_vwap.

    Buy/sell are signed by ``classify`` (see _trade_sign): CME futures carry an explicit
    ``aggressorside`` (Buy<->Bid); equities leave it blank, so the default falls back to the tick
    rule for them. ``price_scale`` multiplies the reported prices -- CME equity-index futures print
    integer hundredths of an index point (e.g. ESU5 543775 = 5437.75), so pass 0.01 for index-point
    units; it is irrelevant to the buy/sell counts and to any log-return use (a constant factor
    cancels)."""
    raw = _fetch_messages(date_str, product, product_type, "mt_trade", data_source, tz=tz)
    if raw.empty:
        return pd.DataFrame()
    d = _parse_yyyymmdd(date_str)
    if start_time is not None:
        raw = raw[raw.index >= pd.Timestamp(f"{d} {start_time}", tz=tz)]
    if end_time is not None:
        raw = raw[raw.index <= pd.Timestamp(f"{d} {end_time}", tz=tz)]
    if scrub:
        raw, sc = _scrub_trades(raw, date_str, product, product_type, data_source, tz=tz)
        if sc["broken"] or sc["corrected"]:
            log.info("%s %s trade scrub: dropped %d busted, corrected %d (of %d prints)",
                     date_str, product, sc["broken"], sc["corrected"], sc["n0"])
    agg = _aggregate_trades(raw, interval, classify=classify, side_buy_label=side_buy_label)
    if agg.empty:
        return pd.DataFrame()
    root = canonical_root(product, product_type)
    out = pd.DataFrame(index=agg.index)
    out[f"{root}_trade_buy"] = agg["trade_buy"].to_numpy(float)
    out[f"{root}_trade_sell"] = agg["trade_sell"].to_numpy(float)
    out[f"{root}_trade_buy_volume"] = agg["trade_buy_volume"].to_numpy(float)
    out[f"{root}_trade_sell_volume"] = agg["trade_sell_volume"].to_numpy(float)
    out[f"{root}_trade_px"] = agg["last_trade_price"].to_numpy(float) * price_scale
    out[f"{root}_trade_vwap"] = agg["trade_vwap"].to_numpy(float) * price_scale
    return out


def attach_flow(book: pd.DataFrame, date_str: str, spy_product: str, es_product: str,
                interval: str = "1s", tz: str = "America/New_York", classify: str = "aggressor",
                side_buy_label: str = "Bid", futures_scale: float = 1.0, **kw) -> pd.DataFrame:
    """Reindex SPY + ES trade flow onto the book's bar grid and append the trade columns. Bars with
    no trade get count/volume 0; the price is left NaN (counts_from_frame ffills it for returns).
    ``futures_scale`` is applied to the ES trade prices (see trade_flow.price_scale)."""
    out = book.copy()
    for prod, ptype, scale in ((spy_product, "direct", 1.0), (es_product, "futures", futures_scale)):
        try:
            f = trade_flow(date_str, prod, ptype, interval=interval, tz=tz, classify=classify,
                           side_buy_label=side_buy_label, price_scale=scale, **kw)
        except Exception as exc:                    # attribute it: parallel logs interleave sessions
            raise MessageFetchError("%s %s: trade-tape fetch failed, so the signed order-flow "
                                    "counts (Tables 5/7) would be silently absent for this leg: %s"
                                    % (date_str, prod, str(exc).splitlines()[0][:200])) from exc
        if f.empty:
            continue
        f = f.reindex(book.index)
        root = canonical_root(prod, ptype)
        for c in (f"{root}_trade_buy", f"{root}_trade_sell",
                  f"{root}_trade_buy_volume", f"{root}_trade_sell_volume"):
            out[c] = np.nan_to_num(f[c].to_numpy(float))      # no trade in bar -> 0
        for c in (f"{root}_trade_px", f"{root}_trade_vwap"):
            out[c] = f[c].to_numpy(float)
    return out


def counts_from_frame(df: pd.DataFrame, assets=("SPY", "ES")):
    """Stack ``counts_fn`` built from REAL trade-tape columns attached by ``attach_flow``:
    returns (spy_buy, spy_sell, es_buy, es_sell, spy_ret, es_ret) aligned to df rows, where the
    return is the bps log-change of the (ffilled) trade price. Returns None if the trade columns
    are absent -- the caller then falls back to the book-OFI proxy."""
    if not all(f"{a}_trade_buy" in df.columns for a in assets):
        return None
    res = {}
    for a in assets:
        buy = np.nan_to_num(df[f"{a}_trade_buy"].to_numpy(float))
        sell = np.nan_to_num(df[f"{a}_trade_sell"].to_numpy(float))
        px = pd.Series(df.get(f"{a}_trade_px", df.get(f"{a}_trade_vwap")).to_numpy(float)).ffill().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = np.concatenate([[np.nan], np.diff(np.log(np.where(px > EPS, px, np.nan))) * 1e4])
        res[a] = (buy, sell, ret)
    return (res[assets[0]][0], res[assets[0]][1], res[assets[1]][0], res[assets[1]][1],
            res[assets[0]][2], res[assets[1]][2])


def session_is_ssr(df) -> tuple:
    """(restricted, known) for one session frame.

    Reads the extraction-time summary first (attrs["market_state"]), then the SPY_ssr column.
    ``known=False`` means the source never reported the indicator -- and absence is NOT
    'unrestricted': five sessions of the 2026-08-04 run are in that state. Callers must treat
    unknown as unknown, which is why this returns two booleans instead of one."""
    try:
        ms = df.attrs.get("market_state") if hasattr(df, "attrs") else None
        if ms and ms.get("ssr_known"):
            return bool(ms.get("ssr_frac_on", 0.0) > 0), True
        if "SPY_ssr" in df.columns:
            v = pd.to_numeric(df["SPY_ssr"], errors="coerce")
            if v.notna().any():
                return bool(v.fillna(0).mean() > 0), True
    except Exception:
        pass
    return False, False


def has_trade_flow(sessions, assets=("SPY", "ES")) -> bool:
    """True if the session frames carry attach_flow's trade columns (so counts_from_frame works)."""
    if not sessions:
        return False
    df = sessions[0][-1]
    return all(f"{a}_trade_buy" in df.columns for a in assets)


def session_qc(df: pd.DataFrame, assets: Sequence[str] = ("SPY", "ES"),
               crossed_tol: float = 0.005, date=None) -> dict:
    """Cheap per-session integrity check run at EXTRACTION time, on the frame that is about to be
    written to the dataset. It answers the two questions a full-length frame cannot answer by its
    shape: is each leg actually there, and does its top violate the no-crossing invariant?

    A matching engine cannot cross, so a crossed reconstructed top is always a replay fault -- and
    the failure mode that motivated this is a frame of the right length (23,401 rows), with the
    right column names, whose book is garbage because a message-type fetch failed and left the
    replay without its adds (phantom levels) or without its trades/cancels (levels that never
    leave). ``crossed_tol`` is the fraction of finite snapshots allowed to cross before the session
    is called degraded; consolidated multi-venue books DO legitimately cross for brief moments at
    sub-second scale, so it is a small positive number, not zero.

    **Halt snapshots are excluded from the crossed fraction.** During an MWCB Level 1 halt the
    exchanges stop matching but do not cancel resting orders, so a limit order priced through the
    last trade sits unmatched for the full fifteen minutes: a CORRECTLY reconstructed book is
    crossed for those 900 seconds. Judging a March-2020 session on the raw fraction flags the very
    phenomenon the paper studies as a data fault -- on 2020-03-09 the halt is 901 of 23,401
    snapshots (3.85%) against an observed crossed rate of 3.88%. Both figures are reported;
    ``ok`` is decided on the one outside the halt.

    -> {'ok': bool, 'reasons': [str], 'halt_snapshots': int, '<ASSET>': {'n': int, 'n_finite': int,
        'crossed_frac': float, 'crossed_frac_ex_halt': float, 'median_bid': float}}"""
    rep: dict = {"ok": True, "reasons": [], "n_rows": int(len(df))}
    attrs = df.attrs if hasattr(df, "attrs") else {}

    def _mask(key):
        """Halt mask for one leg. The tape wins: df.attrs[...] is what haltreason actually said."""
        try:
            import market_halts as mh
            # ABSENT falls back to the union, and then to the built-in table -- that is the path a
            # session takes when the status fetch failed, where a hand-entered date is better than
            # nothing. PRESENT-but-empty is a positive statement from the tape, "this leg did not
            # halt", and must stay empty: otherwise a pause on the other leg, or a date in the MWCB
            # table, would excuse this book's crossing.
            if key in attrs:
                wins = attrs[key]
                if not wins:
                    return np.zeros(len(df), bool)
            else:
                wins = attrs.get("halt_windows")
            wins = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in wins] if wins else None
            return mh.halt_mask(df.index, date=date, windows=wins) if len(df) else np.zeros(0, bool)
        except Exception:                                # never let the QC fail on the halt table
            return np.zeros(len(df), bool)

    # Each book is judged against ITS OWN halt, not the union. SPY crossing during a CME Velocity
    # Logic pause is still a replay fault -- NYSE never stopped matching -- and vice versa. The
    # union (df.attrs["halt_windows"]) is for pair estimates, which need both legs live.
    halt = _mask("halt_windows")
    rep["halt_snapshots"] = int(halt.sum())
    # Per-leg halt counts, because the union alone hides which leg contributed. The 2026-08-04 run
    # reported 900/900/900/901 snapshots on the four MWCB days -- the SPY windows exactly -- and
    # there was no way to tell from the output whether the ES windows had been derived at all.
    for _a in assets:
        rep.setdefault("halt_by_leg", {})[_a] = int(_mask(f"halt_windows_{_a}").sum())
    for a in assets:
        halt = _mask(f"halt_windows_{a}")
        n_halt = int(halt.sum())
        bid = pd.to_numeric(df.get(f"{a}_bidprice_1", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        ask = pd.to_numeric(df.get(f"{a}_askprice_1", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        both = np.isfinite(bid) & np.isfinite(ask) if bid.size and ask.size else np.zeros(0, bool)
        n_fin = int(both.sum())
        crossed = (bid > ask + EPS) & both
        frac = float(crossed[both].mean()) if n_fin else float("nan")
        open_mkt = both & ~halt if halt.size == both.size else both
        n_open = int(open_mkt.sum())
        frac_ex = float(crossed[open_mkt].mean()) if n_open else float("nan")
        rep[a] = {"n": int(len(df)), "n_finite": n_fin,
                  "crossed_frac": frac, "crossed_frac_ex_halt": frac_ex, "n_open": n_open,
                  "halt_snapshots": n_halt,
                  "median_bid": float(np.nanmedian(bid)) if bid.size and np.isfinite(bid).any() else float("nan")}
        if n_fin == 0:
            rep["ok"] = False
            rep["reasons"].append(f"{a}: no usable top of book (0 of {len(df)} snapshots have a "
                                  f"finite bid AND ask) -- the leg is missing, not thin")
        elif np.isfinite(frac_ex) and frac_ex > crossed_tol:
            rep["ok"] = False
            extra = (f" ({frac:.1%} including the {n_halt} halt snapshot(s), which "
                     f"are expected to cross)" if n_halt else "")
            rep["reasons"].append(f"{a}: top is CROSSED on {frac_ex:.1%} of {n_open} snapshots "
                                  f"OUTSIDE any halt{extra} (tolerance {crossed_tol:.1%}) -- the "
                                  f"replay is dropping removals or inventing levels")
        # A leg taken from the venue ladder cannot cross, so the invariant above is vacuous for it.
        # Check the properties a ladder must have instead (monotone depth, coverage, staleness).
        _src_a = attrs.get(f"book_source_{a}") or (attrs.get("book_source")
                                                   if f"{a}_bidprice_2" in df.columns else None)
        if _src_a == "aggregated":
            try:
                import aggregated_book as _ab
                li = _ab.ladder_integrity(df, asset=a)
                rep[a]["ladder"] = {k: v for k, v in li.items() if k != "reasons"}
                if not li["ok"]:
                    rep["ok"] = False
                    rep["reasons"].extend(li["reasons"])
            except Exception:
                pass
        if n_halt and np.isfinite(frac) and frac > crossed_tol:
            rep["reasons"].append(f"{a}: {frac:.1%} crossed overall but only {frac_ex:.2%} outside "
                                  f"the {n_halt}-snapshot halt on this leg -- the crossing "
                                  f"IS the halt (matching stops, resting orders do not), so the "
                                  f"book is correct. Exclude halt snapshots from any estimate: a "
                                  f"halted market has no valid midpoint")
    # Roll split, when the session sat near a contract boundary and extraction measured it
    # (measure_roll_at_extraction). Surfaced so the QC table carries the coverage number beside the
    # integrity columns. A minority pick does NOT flip ``ok`` -- the book itself is sound and
    # re-extracting under the same rule would produce the same pick, so a hard fail would loop --
    # but it is recorded as a reason: which contract defines the leg is a sample decision the
    # extraction log.error already forces, and the table must not let it pass silently either.
    rm = attrs.get("roll_measurement")
    if rm:
        rep["roll"] = {"offset_days": attrs.get("roll_offset_days"),
                       "front": rm.get("front"), "rival": rm.get("rival"),
                       "measured": bool(rm.get("measured")),
                       "front_share": rm.get("front_share", float("nan")),
                       "oi_share": rm.get("oi_share", float("nan")),
                       "rule_agrees": rm.get("rule_agrees")}
        if rm.get("measured") and not rm.get("rule_agrees", True):
            rep["reasons"].append(
                "ES: calendar rule picked the MINORITY contract %s (%.1f%% of two-contract "
                "volume vs %s) -- the leg is the quieter book; re-extract pinned to %s and say so "
                "in the appendix, or keep the rule and report the measured share"
                % (rm.get("front"), 100 * rm.get("front_share", float("nan")),
                   rm.get("rival"), rm.get("rival")))
    elif "roll_offset_days" in attrs:
        rep["roll"] = {"offset_days": attrs.get("roll_offset_days"), "measured": False}
    return rep


def _spec_parts(spec) -> tuple:
    """(label, regime) from a date spec that may be 'YYYY-MM-DD' or ('YYYY-MM-DD', regime)."""
    if isinstance(spec, (tuple, list)):
        return spec[0], (spec[1] if len(spec) > 1 else "auto")
    return spec, "auto"


def _cache_path(cache_dir: str, label: str, interval: str, levels: int, clock: str,
                es_book_source: str = "aggregated") -> str:
    """Cache filename for one extracted session. The interval / level count / clock are part of the
    name because they change the frame: a cache hit must not silently hand back a 1s book to a run
    that asked for 10ms. The ES book source is part of it for the same reason -- a ladder frame and
    a replay frame carry the same columns and pass the same shape checks, so nothing downstream
    would notice a run with --es-book-source replay silently resuming from aggregated frames."""
    ymd = str(label).replace("-", "")
    tag = "" if es_book_source in ("", None) else f"_{es_book_source}"
    return os.path.join(cache_dir, f"session_{ymd}_{interval}_L{int(levels)}_{clock}{tag}.pkl")


def heartbeat_writer(hb_dir: str, label: str, interval: str, user_cb=None):
    """File-based per-session liveness for PARALLEL extraction. Console heartbeats cannot stream
    cleanly from process workers, which made a multi-hour fine-grid STAGE 2b look HUNG -- the
    2026-08-06 run went silent for the whole 10ms pull. Each call atomically rewrites one small
    JSON (``<hb_dir>/<label>_<interval>.json``) with the current phase and elapsed time;
    ``extraction_status.py`` tabulates the directory (point ``watch -n 30`` at it). Never raises:
    a full disk or unwritable dir costs the heartbeat, not the session."""
    t0 = time.time()
    path = os.path.join(hb_dir, "%s_%s.json" % (str(label).replace("/", "-"), interval)) if hb_dir else ""

    def _beat(msg):
        if path:
            try:
                import json as _json
                os.makedirs(hb_dir, exist_ok=True)
                tmp = path + ".tmp%d" % os.getpid()
                with open(tmp, "w") as fh:
                    _json.dump({"session": str(label), "interval": interval, "phase": str(msg),
                                "elapsed_s": round(time.time() - t0, 1), "pid": os.getpid(),
                                "updated_epoch": time.time(),
                                "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
                os.replace(tmp, path)
            except Exception:
                pass
        if user_cb is not None:
            user_cb(msg)
    return _beat


def _extract_one_session(spec, cfg: dict, progress_cb=None):
    """Picklable per-session worker (one trading day): reconstruct SPY + front-month ES books on one
    GPS clock, attach the trade tape, and return ``(label, regime, df, diag_msg, warn_msg, qc)``.
    ``cfg`` is a dict of primitives only (so it pickles to loky/process workers); ``progress_cb`` is
    chained behind a file heartbeat (see ``heartbeat_writer``) so parallel workers stay observable.

    When ``cfg['cache_dir']`` is set the finished frame is written there and, if ``cfg['resume']``,
    a previous run's frame is reused instead of re-fetching -- a session costs 10-25 minutes of
    vendor I/O, so a run that dies on day 23 of 24 should not repeat the first 22. Only sessions
    that PASS ``session_qc`` are cached: a degraded day is worth retrying, not memoizing."""
    import lob_reconstruct as lob                          # sole extraction path: both legs from messages
    label, regime = _spec_parts(spec)
    ymd = label.replace("-", "")
    cache_dir = cfg.get("cache_dir") or ""
    # file heartbeat: observable liveness for every worker, serial or parallel (the passed
    # progress_cb -- serial console updates -- is chained behind it)
    progress_cb = heartbeat_writer(os.path.join(cache_dir, "progress") if cache_dir else "",
                                   label, cfg["interval"], user_cb=progress_cb)
    progress_cb("starting")
    cpath = (_cache_path(cache_dir, label, cfg["interval"], cfg["levels"], cfg["clock"],
                         cfg.get("es_book_source", "aggregated"))
             if cache_dir else "")
    # A cache file written before the source tag existed has the legacy name. Rather than force a
    # re-extraction of every session (hours of vendor I/O) over a rename, a legacy file is accepted
    # IF the frame's own recorded source matches the requested one -- the frame knows what built it
    # (attrs["book_source_ES"]; absent means the pre-v0.9.42 replay). A mismatch re-extracts: a
    # ladder frame and a replay frame carry identical columns, so nothing downstream would notice.
    _want_src = cfg.get("es_book_source", "aggregated")
    _legacy = (_cache_path(cache_dir, label, cfg["interval"], cfg["levels"], cfg["clock"], "")
               if cache_dir else "")
    for _cp in ([cpath, _legacy] if cpath else []):
        if not (_cp and cfg.get("resume") and os.path.exists(_cp)):
            continue
        try:
            df = pd.read_pickle(_cp)
            _got_src = df.attrs.get("book_source_ES") if hasattr(df, "attrs") else None
            if _got_src != _want_src:
                if _got_src is None:
                    # v0.9.42-45 frames: the join dropped the ES frame's attrs, so a frame from that
                    # window cannot PROVE which source built it -- and a pre-v0.9.42 replay frame
                    # looks exactly the same. Mixed sources in one dataset is the inconsistency the
                    # aggregated default exists to prevent, so unknown re-extracts. One-time cost:
                    # frames written from v0.9.46 on always carry the tag.
                    log.warning("%s: cached frame at %s predates ES book-source tagging and cannot "
                                "prove which source built it -- re-extracting once; the fresh frame "
                                "is tagged and future resumes will hit", label, _cp)
                else:
                    log.warning("%s: cached frame at %s was built with es_book_source=%s but this "
                                "run asked for %s -- re-extracting rather than silently reusing it "
                                "(the two carry identical columns, so nothing downstream would "
                                "notice)", label, _cp, _got_src, _want_src)
                continue
            qc = session_qc(df)
            progress_cb("DONE (cache hit)")
            return (label, regime, df, "reused cached %s: %d rows (%s)" % (label, len(df), _cp), None, qc)
        except Exception as exc:                           # a corrupt/partial cache must never win
            log.warning("%s: cache read failed (%s); re-extracting", label, exc)
    contract = get_front_month_contract(cfg["es_symbol"], as_of_date=_parse_yyyymmdd(ymd),
                                        rollover_days=cfg["rollover_days"])
    # Self-verifying pick: a session near the roll boundary MEASURES the volume/OI split between
    # the calendar pick and its rival at extraction time (cheap head-read of mt_product_statistics)
    # instead of quoting the one roll ever measured. The pick itself stays the calendar rule --
    # switching on a same-day volume read would make the series definition data-dependent -- but a
    # minority pick is now a loud log.error at extraction, not a post-hoc discovery. The full
    # report lands in df.attrs["roll_measurement"] and the QC table.
    _roll_d, _roll_rep = measure_roll_at_extraction(label, cfg["es_symbol"],
                                                    cfg["rollover_days"])
    # SPY: consolidated multi-venue hybrid MBO+MBP. ES: single-venue CME order-by-order (MBO) — the
    # mt_price_level_* types are empty for futures; integer-hundredths -> index points via price_scale.
    spy = lob.reconstruct_session(ymd, "SPY", "direct", levels=cfg["levels"], interval=cfg["interval"],
                                  round_lot=cfg["round_lot"], odd_lot_inclusive=cfg["odd_lot_inclusive"],
                                  session=(cfg["start_time"], cfg["end_time"]), tz=cfg["tz"],
                                  data_source=cfg["data_source"], clock=cfg["clock"], progress_cb=progress_cb)
    # THE ES LEG COMES FROM CME'S OWN LADDER (mt_aggregated_price_update), not from a replay.
    # CME is one venue and its ladder IS the book, so there is nothing to consolidate; the ladder is
    # populated in every era checked while the message families are not (2024 order-by-order, 2020
    # price-level, and the one MBP type the old fetch list carried empty in both); and it gives ONE
    # mechanism across a 2020-2026 panel instead of two. validate_aggregated.py already compares the
    # replay against this same ladder and the driver runs it as a gate, so this is a switch to the
    # benchmark the replay was measured against. See aggregated_book.py for what is given up.
    if cfg.get("es_book_source", "aggregated") == "aggregated":
        import aggregated_book as ab
        es = ab.session_from_aggregated(ymd, contract, "futures", levels=cfg["levels"],
                                        interval=cfg["interval"],
                                        session=(cfg["start_time"], cfg["end_time"]), tz=cfg["tz"],
                                        data_source=cfg["data_source"], clock=cfg["clock"],
                                        price_scale=cfg["futures_scale"], progress_cb=progress_cb)
    else:
        # the message replay, kept reachable: --es-book-source replay. Needed if a future question
        # asks for futures QUEUE dynamics, which the aggregated ladder cannot answer.
        es = lob.reconstruct_session(ymd, contract, "futures", levels=cfg["levels"],
                                 interval=cfg["interval"], round_lot=cfg["round_lot"], odd_lot_inclusive=cfg["odd_lot_inclusive"],
                                 session=(cfg["start_time"], cfg["end_time"]), tz=cfg["tz"],
                                 data_source=cfg["data_source"], clock=cfg["clock"],
                                 price_scale=cfg["futures_scale"], progress_cb=progress_cb,
                                 # EVERY candidate type, with the family chosen from row counts at
                                 # run time (lob.select_book_family). CME's capture changes shape
                                 # across eras -- 2024 ES is order-by-order with the MBP types empty,
                                 # 2020 ES is price-level via mt_modify_price_level /
                                 # mt_delete_price_level -- and `mt_price_level_update`, the one MBP
                                 # type the old fetch list carried, is empty in BOTH. That is how the
                                 # four 2020 sessions came back with no ES leg at all: the fetch
                                 # asked for order-by-order on a price-level capture and got a
                                 # session's worth of nothing.
                                 #
                                 # Hardcoding "2020 = MBP" would repeat the mistake one era later,
                                 # so nothing here is keyed to a date. The clear types are included
                                 # for the same reason as before: empty on a normal day, but a Globex
                                 # reset on a crash day would otherwise be invisible, and an empty
                                 # query costs nothing next to a 10-25 minute session.
                                 select_family=True,
                                 message_types=("mt_add_order", "mt_cancel_order", "mt_modify_order",
                                                "mt_trade", "mt_price_level_update",
                                                "mt_modify_price_level", "mt_delete_price_level",
                                                "mt_clear_orders", "mt_clear_price_levels"))
    df = spy.join(es, how="outer").sort_index()
    # when each leg actually traded -- the resume signal for a CME halt window (see market_halts)
    _activity = {"SPY": spy.attrs.get("activity_seconds"), "ES": es.attrs.get("activity_seconds")}
    if cfg["with_flow"]:
        if progress_cb is not None:
            progress_cb(f"{label} trade flow")
        df = attach_flow(df, ymd, "SPY", contract, interval=cfg["interval"], tz=cfg["tz"],
                         classify=cfg["classify"], side_buy_label=cfg["side_buy_label"],
                         futures_scale=cfg["futures_scale"], start_time=cfg["start_time"],
                         end_time=cfg["end_time"], data_source=cfg["data_source"])
    # The halt window comes from the TAPE (mt_product_status.haltreason), not from a table of four
    # dates I typed in. A halted market has no valid midpoint and its book is legitimately crossed,
    # so every consumer needs the boundary -- and needs it right. One tiny query per session.
    #
    # BOTH legs, not just SPY. The two markets stop for different reasons and at different times:
    # 2020-03-12 has a 900 s MWCB Level 1 halt on SPY at 09:35:44 and a 6.4 s
    # SuspendedBySurveillance on ESH0 at 09:36:45 -- a CME Velocity Logic pause that fires INSIDE the
    # equity halt. Fetching only the equity stream made the futures pause invisible, and it is
    # exactly the kind of cross-asset event this paper is about. It also cost the ES price limits
    # (below), which CME does publish.
    progress_cb("halt/status streams")
    for _asset, _prod, _ptype, _scale in (("SPY", "SPY", "direct", 1.0),
                                          ("ES", contract, "futures", cfg["futures_scale"])):
        try:
            st = _fetch_messages(ymd, _prod, _ptype, "mt_product_status", cfg["data_source"],
                                 tz=cfg["tz"], clock=cfg["clock"])
            import market_halts as _mh
            hres = _mh.windows_from_status(st, tz=cfg["tz"], activity=_activity.get(_asset))
            for _e in hres["extended"]:
                log.info("%s: %s: halt window extended %.1f s past the status flag's clear, to "
                         "the resumption of trading at %s (pre-halt median inter-trade gap %.3f s, "
                         "so a silence that long is not ordinary: ratio %.0f). The CME flag marks "
                         "the STOP, not the duration.", label, _asset, _e["extra_seconds"],
                         _e["resume"].strftime("%H:%M:%S.%f")[:-3], _e["pre_halt_median_gap_s"],
                         _e["quiet_ratio"])
            # set unconditionally: an EMPTY list is the positive statement "this leg did not halt",
            # which is what stops the other leg's pause from excusing this book (see session_qc).
            df.attrs[f"halt_windows_{_asset}"] = [(a.isoformat(), b.isoformat())
                                                  for a, b in hres["windows"]]
            df.attrs[f"halt_reasons_{_asset}"] = sorted(hres["reasons"])
            # the UNextended spans, so the flag's own claim is never lost behind the correction
            df.attrs[f"halt_flag_windows_{_asset}"] = [(a.isoformat(), b.isoformat())
                                                       for a, b in hres["flag_windows"]]
            if hres["windows"]:
                log.info("%s: %s: %d halt window(s) from the tape: %s (%s)", label, _asset,
                         len(hres["windows"]),
                         ", ".join("%s-%s" % (a.strftime("%H:%M:%S"), b.strftime("%H:%M:%S"))
                                   for a, b in hres["windows"]), ", ".join(sorted(hres["reasons"])))
            if hres["unknown_reasons"]:
                # halts are whitelisted, so an unrecognized reason is deliberately NOT a halt --
                # surfaced here to be classified rather than silently assumed either way.
                log.warning("%s: %s: unclassified haltreason value(s) %s -- treated as NOT a halt; "
                            "add to market_halts._HALT_TOKENS if any is one", label, _asset,
                            ", ".join(sorted(hres["unknown_reasons"])[:6]))
            # The same stream carries the two REGULATORY CONSTRAINTS on quoting: the LULD/price band
            # and Rule 201 (SSR). Both bound what a quote is allowed to be, so a response measured
            # near one is a rule, not price discovery -- and SSR is one-sided, which is a direct
            # confound for the signed-order-flow tables. Session columns; see market_state.py.
            # price_scale: CME publishes ES limits in integer hundredths (563025 = 5630.25), the
            # same convention as its prices, so the futures leg needs the same scale as its book.
            import market_state as _ms
            df = _ms.attach_market_state(df, st, asset=_asset, tz=cfg["tz"], price_scale=_scale)
            df.attrs[f"market_state_coverage_{_asset}"] = _ms.coverage(st)
            _sum = _ms.summarize(df, asset=_asset)
            df.attrs[f"market_state_{_asset}"] = _sum
            if _asset == "SPY":                   # back-compat for readers written before the ES leg
                df.attrs["market_state_coverage"] = df.attrs["market_state_coverage_SPY"]
                df.attrs["market_state"] = _sum
            if _sum.get("ssr_any"):
                log.warning("%s: Rule 201 SHORT SALE RESTRICTION in effect on %.0f%% of the session "
                            "-- sell-side flow is mechanically constrained (short sales cannot "
                            "execute or display at or below the NBB), which is a confound for "
                            "Tables 5 and 7", label, 100 * _sum.get("ssr_frac_on", 0.0))
        except Exception as exc:                  # never fail a session over the status stream
            log.warning("%s: %s: could not read mt_product_status (%s); halt windows fall back to "
                        "the built-in table", label, _asset, str(exc).splitlines()[0][:160])
    # `halt_windows` (unsuffixed) is the UNION across legs: a pair estimate needs BOTH legs matching,
    # so either one halted means the cross-asset relationship is undefined for those snapshots. The
    # per-asset lists stay separate because the crossed-book invariant is per book -- excusing a
    # crossed SPY top with a CME pause would hide a real replay fault.
    _union = sorted([tuple(w) for a in ("SPY", "ES") for w in df.attrs.get(f"halt_windows_{a}", [])])
    if _union:
        df.attrs["halt_windows"] = _union
        df.attrs["halt_reasons"] = sorted(set(df.attrs.get("halt_reasons_SPY", []))
                                          | set(df.attrs.get("halt_reasons_ES", [])))
    df.attrs["es_contract"] = contract
    df.attrs["roll_offset_days"] = int(_roll_d)
    if _roll_rep is not None:                      # in a roll window and the measurement ran
        df.attrs["roll_measurement"] = _roll_rep
    if locals().get("_es_from_ladder"):            # so session_qc knows to run the LADDER checks
        df.attrs["book_source_ES"] = "aggregated"  # (the crossed test is vacuous on that leg)
    progress_cb("qc")
    qc = session_qc(df, crossed_tol=float(cfg.get("crossed_tol", 0.005)))
    df.attrs["qc"] = qc
    med_spy = qc["SPY"]["median_bid"]
    med_es = qc["ES"]["median_bid"]
    ratio = med_es / med_spy if np.isfinite(med_spy) and med_spy else float("nan")
    # The ES book source goes in the completion line because the log is where a reader learns it.
    # The 2026-08-04 run printed "book=reconstruct" on all 24 sessions while every ES leg came from
    # the ladder -- the string was hardcoded from the pre-v0.9.42 path.
    _src = df.attrs.get("book_source_ES", "replay")
    diag = ("extracted %s (ES=%s, ES book=%s): %d rows, flow=%s | median SPY=%.2f ES=%.2f "
            "(ES/SPY=%.1f)" % (label, contract, _src, len(df), cfg["with_flow"], med_spy, med_es, ratio))
    warns = []
    if np.isfinite(ratio) and ratio > 100:
        warns.append("ES/SPY median ratio %.0f looks scaled (CME raw integers); pass "
                     "futures_scale=0.01 for index-point units (log/bps results are unaffected)."
                     % ratio)
    if not qc["ok"]:
        warns.append("%s DEGRADED -- %s" % (label, "; ".join(qc["reasons"])))
    if qc["ok"] and cpath:                                 # only memoize a session worth reusing
        try:
            progress_cb("caching frame")
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cpath + f".tmp{os.getpid()}"
            df.to_pickle(tmp)
            os.replace(tmp, cpath)                         # atomic: a reader never sees a partial file
        except Exception as exc:
            log.warning("%s: could not cache session to %s (%s)", label, cpath, exc)
    progress_cb("DONE")
    return (label, regime, df, diag, ("\n".join(warns) if warns else None), qc)


def _guarded_session(spec, cfg: dict, progress_cb=None):
    """``_extract_one_session`` that CANNOT take the batch down with it.

    joblib's default is fail-fast: the first worker exception is re-raised out of the generator and
    every completed result is discarded. On a 24-day extract that means one transient vendor error
    on the 23rd day throws away ~3 hours of finished work. A single day failing is a fact about
    that day; it is not a reason to lose the other 23. The failure is returned (df=None) with the
    exception text, and ``extract_sessions`` reports and drops it."""
    label, regime = _spec_parts(spec)
    try:
        return _extract_one_session(spec, cfg, progress_cb=progress_cb)
    except Exception as exc:
        msg = "%s: EXTRACTION FAILED (%s: %s)" % (label, type(exc).__name__,
                                                  str(exc).splitlines()[0][:300])
        try:                                        # the status board must show the failure too
            heartbeat_writer(os.path.join(cfg.get("cache_dir") or "", "progress")
                             if cfg.get("cache_dir") else "",
                             label, cfg.get("interval", "?"))("FAILED: %s" % type(exc).__name__)
        except Exception:
            pass
        # logged by extract_sessions' _on_done in the PARENT process (a worker-side log record may
        # never reach the parent's handlers, and logging it in both places double-prints it)
        return (label, regime, None, msg, None,
                {"ok": False, "reasons": [f"extraction raised {type(exc).__name__}"],
                 "error": str(exc)[:2000]})


def _idx_call(i, func, item):
    """Index-tagged call so completion-order results can be reordered back to input order."""
    return i, func(item)


def _parallel_sessions(func, items, n_jobs: int, backend: str, on_done):
    """Map ``func`` over ``items`` PRESERVING input order, calling ``on_done(result)`` in COMPLETION
    order (live progress). Backends: 'sequential'; 'threading' (shared-memory, used for the self-test);
    'process' (joblib loky if importable -> spawned workers, else stdlib ProcessPoolExecutor). loky is
    the default for real runs: it is robust to worker crashes and, being spawn-based, sidesteps the
    fork-after-numpy hazards of a bare process pool. Worker count = ``n_jobs`` (memory-bounded)."""
    n = len(items)
    results: list = [None] * n
    if n_jobs <= 1 or n <= 1 or backend == "sequential":
        for i, it in enumerate(items):
            results[i] = func(it); on_done(results[i])
        return results
    if backend == "threading":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(func, it): i for i, it in enumerate(items)}
            for f in as_completed(futs):
                results[futs[f]] = f.result(); on_done(results[futs[f]])
        return results
    try:                                                   # process backend: joblib loky preferred
        from joblib import Parallel, delayed
        for i, r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(_idx_call)(i, func, it) for i, it in enumerate(items)):
            results[i] = r; on_done(r)
        return results
    except (ImportError, TypeError):                       # stdlib fallback (or old joblib w/o return_as)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(func, it): i for i, it in enumerate(items)}
            for f in as_completed(futs):
                results[futs[f]] = f.result(); on_done(results[futs[f]])
        return results


def extract_sessions(date_specs: Sequence, es_symbol: str = "ES", levels: int = 10,
                     interval: str = "1s", start_time: str = "9:30", end_time: str = "16:00",
                     data_source: str = "apu", tz: str = "America/New_York", with_flow: bool = True,
                     classify: str = "aggressor", side_buy_label: str = "Bid",
                     futures_scale: float = 0.01, rollover_days: int = 8,
                     book_source: str = "reconstruct", es_book_source: str = "aggregated",
                     round_lot: int = 100,
                     odd_lot_inclusive: bool = True, clock: str = "receipt",
                     progress: Optional[bool] = None, max_workers: int = 1,
                     backend: Optional[str] = None, cache_dir: str = "", resume: bool = True,
                     qc_action: str = "warn", crossed_tol: float = 0.005,
                     report: Optional[dict] = None) -> list:
    """CLI replacement for the Athena extract: for each date pull SPY + front-month ES books
    (canonical schema, hygiene applied) and -- when ``with_flow`` -- attach the trade-tape counts
    (signed via ``classify``; see _trade_sign), returning ``[(date_label, regime, df)]`` exactly
    like the old ``market_analysis_fixed.main``.

    ``futures_scale`` multiplies the ES book AND trade prices. It defaults to **0.01** because the CME
    message feed prints integer hundredths of an index point (543775 = 5437.75), which is the sole
    extraction path now; pass 1.0 only if a future's feed is already in index-point units. A per-session
    diagnostic logs the median SPY and ES mids so a scale mismatch (ES/SPY far from ~10x) is visible. ``date_specs`` is an iterable of ``date_label`` or
    ``(date_label, regime)``; the label may be 'YYYY-MM-DD' (preserved for downstream regime/registry
    matching) or 'YYYYMMDD'.

    Both legs are reconstructed from ``mstwx-lakequery`` messages via ``lob_reconstruct`` -- the SOLE
    extraction path. SPY is the consolidated multi-venue NBBO/ladder (hybrid MBO+MBP, odd-lot-inclusive
    10-level plus strict round-lot NBBO); ES is a single-venue CME order-by-order (MBO) replay. They share
    one GPS-disciplined capture clock, which is required for a coherent SPY<->ES lead-lag. ``book_source``
    is retained only for back-compat: ``"snapshot"`` now warns and reconstructs anyway, because the
    vendor ``mstbook-query`` book sits on a different clock and is being sunset (it remains importable
    for one-time historical bias benchmarking via ``validate_reconstruction``, never on the live path).

    ``max_workers`` parallelizes the SESSION loop (each day is one independent, I/O-heavy unit): >1
    fans the days out across processes, so set it to the memory-safe count (peak RAM is ~max_workers x
    one full day of messages; the window has ~12 sessions, the natural cap). ``backend`` selects the
    engine -- ``None``/'process' uses joblib loky (else a stdlib process pool), 'threading' shares
    memory, 'sequential' forces the serial path. The serial path (max_workers<=1) keeps the per-fetch
    heartbeat; the parallel path streams a per-session completion line (and a tqdm bar) as each day lands.

    ``progress`` controls the per-session progress display: ``None`` (default) shows a tqdm bar when
    stderr is a TTY and tqdm is installed, else falls back to per-session/per-fetch INFO logging (so a
    redirected log still shows liveness); ``True``/``False`` force it on/off.

    **Failure handling.** No single day can abort the batch: a session that raises is logged, dropped,
    and named in the closing summary (``_guarded_session``). Every returned session is checked by
    ``session_qc`` -- ``qc_action`` is ``'warn'`` (default: keep it, but say so loudly), ``'drop'``
    (exclude it from the returned list) or ``'raise'`` (stop the run). If EVERY session fails the
    call raises, because an empty universe downstream is indistinguishable from a small one.
    ``cache_dir`` memoizes each good session to disk and ``resume`` reuses it, so a re-run after a
    partial failure costs only the missing days. Pass a dict as ``report`` to receive
    ``{'ok': [...], 'failed': [(label, msg)], 'degraded': [(label, reasons)]}``."""
    specs = [s for s in date_specs]
    show_bar = _HAS_TQDM and (sys.stderr.isatty() if progress is None else bool(progress))
    bar = _tqdm(total=len(specs), desc="extract", unit="session") if show_bar else None
    if book_source == "snapshot":
        log.warning("book_source='snapshot' is retired: mstbook-query sits on a different clock from the "
                    "message lake (which corrupts the SPY<->ES lead-lag) and is being sunset. "
                    "Reconstructing both legs from mstwx-lakequery instead.")
    if qc_action not in ("warn", "drop", "raise"):
        raise ValueError(f"qc_action must be warn|drop|raise, got {qc_action!r}")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        log.info("session cache: %s (resume=%s) -- a completed day is reused instead of re-fetched",
                 cache_dir, resume)
    cfg = {"es_symbol": es_symbol, "levels": levels, "interval": interval, "start_time": start_time,
           "end_time": end_time, "data_source": data_source, "tz": tz, "with_flow": with_flow,
           "classify": classify, "side_buy_label": side_buy_label, "futures_scale": futures_scale,
           "rollover_days": rollover_days, "round_lot": round_lot, "odd_lot_inclusive": odd_lot_inclusive,
           "clock": clock, "cache_dir": cache_dir, "resume": bool(resume), "crossed_tol": crossed_tol,
           "es_book_source": es_book_source}

    def _emit(msg):                                         # per-session summary line
        _tqdm.write(msg) if bar is not None else log.info("%s", msg)

    def _on_done(res):
        if res[2] is None:
            log.error("%s", res[3])
        else:
            _emit(res[3])
        if res[4]:
            log.warning("%s", res[4])
        if bar is not None:
            bar.update(1)

    parallel = (max_workers or 1) > 1 and len(specs) > 1 and backend != "sequential"
    be = backend or ("process" if parallel else "sequential")
    if parallel:
        log.info("extracting %d sessions in parallel (backend=%s, max_workers=%d); per-session "
                 "heartbeats stream to %s -- run `python extraction_status.py --cache-dir %s` "
                 "(or watch -n 30 ...) to see what every worker is doing; a completion line "
                 "prints here as each day lands", len(specs), be, max_workers,
                 os.path.join(cache_dir or "<cache>", "progress"), cache_dir or "<cache>")
        triples = _parallel_sessions(functools.partial(_guarded_session, cfg=cfg),
                                     specs, max_workers, be, _on_done)
    else:
        def _hb(msg):                                       # serial: per-fetch heartbeat (liveness)
            if bar is not None:
                bar.set_postfix_str(msg)
            elif progress is not False:
                log.info("    ... %s", msg)
        triples = []
        for spec in specs:
            res = _guarded_session(spec, cfg, progress_cb=_hb)
            _on_done(res)
            triples.append(res)
    if bar is not None:
        bar.close()

    # ── closing summary: what landed, what died, what landed but is not usable ────────────────
    out, failed, degraded = [], [], []
    for t in triples:
        if t is None:                                       # defensive: a backend that lost a result
            continue
        qc = t[5] if len(t) > 5 else None
        if t[2] is None:
            failed.append((t[0], t[3]))
            continue
        if qc is not None and not qc.get("ok", True):
            degraded.append((t[0], list(qc.get("reasons", []))))
            if qc_action == "drop":
                continue
        out.append((t[0], t[1], t[2]))
    if report is not None:
        report.clear()
        report.update({"ok": [o[0] for o in out], "failed": failed, "degraded": degraded,
                       "requested": [_spec_parts(s)[0] for s in specs]})
    if failed:
        log.error("EXTRACTION SUMMARY: %d of %d session(s) FAILED and were dropped: %s",
                  len(failed), len(specs), "; ".join(f"{d} ({m.split('(', 1)[-1].rstrip(')')[:120]})"
                                                     for d, m in failed))
    if degraded:
        log.warning("EXTRACTION SUMMARY: %d session(s) violate the book invariant (%s): %s",
                    len(degraded), "dropped" if qc_action == "drop" else "KEPT -- estimate on them "
                    "only if you can explain why", "; ".join(f"{d}: {' | '.join(r)}" for d, r in degraded))
        if qc_action == "raise":
            raise RuntimeError("qc_action='raise': %d degraded session(s): %s"
                               % (len(degraded), ", ".join(d for d, _ in degraded)))
    if not out:
        raise RuntimeError("extract_sessions: no usable session out of %d requested (%d failed, "
                           "%d degraded). Re-run with the vendor query fixed; do NOT estimate on an "
                           "empty universe." % (len(specs), len(failed), len(degraded)))
    if failed or degraded:
        log.info("EXTRACTION SUMMARY: returning %d usable session(s) of %d requested", len(out), len(specs))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Self-test (synthetic mstbook output; binary not required)
# ════════════════════════════════════════════════════════════════════════════
def _synth_raw_csv(prefix: str, levels: int, T: int, seed: int, base: float,
                   heavy_bid_row: int = 0, torn_ask_level: Optional[int] = None,
                   torn_rows: tuple = (40, 55)) -> tuple:
    """Build a synthetic mstbook-query CSV string + the underlying efficient-price path.
    Row ``heavy_bid_row`` gets a very heavy bid (to test microprice direction); ``torn_ask_level``
    (if given) has its size blanked on ``torn_rows`` (a shift leaving one leg empty)."""
    rng = np.random.default_rng(seed)
    eff = base + np.cumsum(rng.normal(0, 0.02, T))        # common random-walk efficient price
    half = 0.01 + 0.002 * rng.random(T)                   # half-spread
    ts = pd.date_range("2020-03-12 09:30:00", periods=T, freq="s", tz="America/New_York")
    cols = {"time": ts.tz_localize(None).astype(str)}     # mstbook prints naive local strings
    cols[f"{prefix}.mid"] = eff
    for n in range(1, levels + 1):
        off = half + (n - 1) * 0.01
        bsz = rng.integers(2, 40, T).astype(float)
        asz = rng.integers(2, 40, T).astype(float)
        if n == 1:
            bsz[heavy_bid_row] = 5000.0                   # heavy bid -> microprice should lean UP
            asz[heavy_bid_row] = 50.0
        cols[f"{prefix}.bid{n}"] = eff - off
        cols[f"{prefix}.bsize{n}"] = bsz
        cols[f"{prefix}.ask{n}"] = eff + off
        cols[f"{prefix}.asize{n}"] = asz
    raw = pd.DataFrame(cols)
    if torn_ask_level is not None:
        lo, hi = torn_rows
        col = f"{prefix}.asize{torn_ask_level}"
        raw.iloc[lo:hi + 1, raw.columns.get_loc(col)] = np.nan   # torn: price present, size missing
    return raw.to_csv(index=False), eff


def _synth_trade_csv(T: int, seed: int, base: float, futures: bool = False) -> str:
    """Synthetic mstwx-lakequery mt_trade output matching the real schema: receipttimestamp (ns),
    price, quantity, aggressorside, side, printable. Futures carry an explicit aggressorside
    (Buy<->Bid); equities leave aggressorside blank and include some NonPrintable prints."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-03-12 09:30:00", periods=T, freq="s", tz="America/New_York")
    px = base + np.cumsum(rng.normal(0, 0.02, T))
    qty = rng.integers(1, 20, T)
    cols = {"receipttimestamp": ts.tz_convert("UTC").as_unit("ns").astype("int64"),
            "price": px, "quantity": qty}
    if futures:
        aggr = np.where(rng.random(T) < 0.55, "Buy", "Sell")
        cols["aggressorside"] = aggr
        cols["side"] = np.where(aggr == "Buy", "Bid", "Ask")          # Buy<->Bid, Sell<->Ask
        cols["printable"] = "Printable"
    else:
        cols["aggressorside"] = ""                                    # equities: blank -> tick rule
        cols["side"] = np.where(rng.random(T) < 0.5, "Bid", "Ask")
        printable = np.where(rng.random(T) < 0.15, "NonPrintable", "Printable")
        cols["printable"] = printable
    return pd.DataFrame(cols).to_csv(index=False)


def _selftest() -> bool:
    import cross_asset_pd_liquidity as ca
    import price_discovery_shares as pds
    L, T = 4, 360
    ok = True

    # ---- parse SPY (direct) with a heavy-bid row 0 and a torn ask level 2 ----
    raw_spy, _ = _synth_raw_csv("direct", L, T, seed=1, base=550.0,
                                heavy_bid_row=0, torn_ask_level=2, torn_rows=(40, 55))
    spy = _to_canonical(raw_spy, "SPY", "direct", L)
    schema_ok = all(f"SPY_{f}_{i}" in spy.columns
                    for f in ("bidprice", "askprice", "bidquantity", "askquantity")
                    for i in range(1, L + 1))
    dtype_ok = all(spy[c].dtype == float for c in spy.columns)
    tz_ok = spy.index.tz is not None
    print("parse -> canonical")
    print("    schema present:", schema_ok, "| all float:", dtype_ok, "| tz-aware:", tz_ok)

    # torn level: BOTH legs NaN on the torn rows, intact elsewhere
    torn = spy.loc[spy.index[40:56]]
    intact = spy.loc[spy.index[100:110]]
    torn_ok = (torn["SPY_askprice_2"].isna().all() and torn["SPY_askquantity_2"].isna().all()
               and intact["SPY_askprice_2"].notna().all() and intact["SPY_askquantity_2"].notna().all())
    print("    torn level (one-leg NaN) -> both legs NaN, treated as absent:", torn_ok)

    # ---- features via stack functions: microprice DIRECTION (their bug) ----
    feat = attach_features(spy, "SPY", n_levels=L)
    r0 = feat.iloc[0]
    mp_up = r0["SPY_wmid"] > r0["SPY_mid"]                # heavy bid -> microprice ABOVE mid
    their_wmid = ((r0["SPY_askprice_1"] * r0["SPY_askquantity_1"]
                   + r0["SPY_bidprice_1"] * r0["SPY_bidquantity_1"])
                  / (r0["SPY_askquantity_1"] + r0["SPY_bidquantity_1"]))   # own-side (wrong)
    feat_fin = (np.isfinite(feat["SPY_auc_decay"]).mean() > 0.95
                and np.isfinite(feat["SPY_curve_length"]).mean() > 0.95)
    print("features (stack implementations)")
    print("    heavy-bid microprice: correct wmid=%.4f > mid=%.4f (%s) | their own-side=%.4f (%s)"
          % (r0["SPY_wmid"], r0["SPY_mid"], "UP" if mp_up else "DOWN",
             their_wmid, "UP" if their_wmid > r0["SPY_mid"] else "DOWN"))
    print("    auc_decay & curve_length finite:", feat_fin)

    # ---- vectorized vwap_to_lot == reference row-walk ----
    vw, filled, used = vwap_to_lot(spy, "SPY", "ask", lot=120.0, n_levels=L)
    Pr = np.column_stack([spy[f"SPY_askprice_{i}"].to_numpy(float) for i in range(1, L + 1)])
    Qr = np.column_stack([spy[f"SPY_askquantity_{i}"].to_numpy(float) for i in range(1, L + 1)])
    Pr, Qr = lcm._usable_levels(Pr, Qr)
    ref = np.full(len(spy), np.nan)
    for i in range(len(spy)):
        rem, cost = 120.0, 0.0
        for n in range(L):
            if rem <= 0:
                break
            f = min(Qr[i, n], rem)
            cost += (0.0 if not np.isfinite(Pr[i, n]) else Pr[i, n]) * f
            rem -= f
        ref[i] = cost / 120.0 if rem <= EPS else np.nan
    both = np.isfinite(vw) & np.isfinite(ref)
    vwap_ok = bool(np.allclose(vw[both], ref[both])) and bool((np.isfinite(vw) == np.isfinite(ref)).all())
    print("vwap_to_lot")
    print("    vectorized == reference row-walk (incl. NaN-on-partial):", vwap_ok)

    # ---- end-to-end: SPY + ES front month -> pair -> stack consumes it ----
    raw_es, _ = _synth_raw_csv("futures", L, T, seed=2, base=550.0)   # cointegrated w/ SPY
    es = _to_canonical(raw_es, "ESM0", "futures", L)
    pair = spy.join(es, how="outer").sort_index()
    wm = lcm.weighted_mid(pair, "ES"); ofi = ca.order_flow_imbalance(pair, "SPY", L)
    a, Om, res = pds._fit_vecm_fixed(np.log(ca._mid(pair, "SPY")), np.log(ca._mid(pair, "ES")), n_lags=5)
    is_vec = np.atleast_1d(pds.hasbrouck_is(a, Om)[2])
    pair_ok = (float(np.isfinite(wm).mean()) > 0.95 and float(np.isfinite(ofi).mean()) > 0.95
               and bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(Om)))
               and bool(np.all((is_vec >= -1e-9) & (is_vec <= 1.0 + 1e-9)))
               and "ES_askprice_1" in pair.columns and "SPY_askprice_1" in pair.columns)
    print("pair -> stack")
    print("    canonical ES root from 'ESM0':", get_futures_root("ESM0"),
          "| weighted_mid & OFI finite & VECM IS in [0,1]:", pair_ok)

    # ---- contract helpers ----
    fm_2020 = get_front_month_contract("ES", as_of_date=date(2020, 3, 11))
    fm_late = get_front_month_contract("ES", as_of_date=date(2020, 3, 16))
    exp = get_contract_expiry("ESH0", ref_year=2020)
    helpers_ok = (fm_2020 == "ESH0" and fm_late == "ESM0"
                  and exp == get_third_friday(2020, 3) and get_third_friday(2020, 3) == date(2020, 3, 20))
    print("contract helpers")
    print("    front month 3/11=%s, 3/16=%s (rolls); ESH0 expiry=%s:" % (fm_2020, fm_late, exp),
          helpers_ok)

    # ---- message layer: aggressorside (futures) + tick fallback & NonPrintable filter (equities) ----
    eq_csv = _synth_trade_csv(T, seed=7, base=550.0, futures=False)     # aggressor blank, has NonPrintable
    fut_csv = _synth_trade_csv(T, seed=8, base=550.0, futures=True)     # explicit aggressorside
    n_buy_fut = fut_csv.count(",Buy,")
    orig_book, orig_msg = _run_mstbook_query, _run_mstwx_lakequery
    orig_to_file = _run_mstwx_lakequery_to_file
    def _patch_to_file(csv_fn):                        # adapt a cmd->csv-string stub to the streaming API
        def _w(cmd, path):
            with open(path, "w") as fh:
                fh.write(csv_fn(cmd))
        return _w
    def _patch_msg(csv_fn):                            # _fetch_messages now reads via the streaming path,
        globals()["_run_mstwx_lakequery"] = csv_fn     # so patch BOTH runners to feed the synthetic CSV
        globals()["_run_mstwx_lakequery_to_file"] = _patch_to_file(csv_fn)
    try:
        _patch_msg(lambda cmd: (fut_csv if "futures" in cmd else eq_csv))
        tf_eq = trade_flow("20200312", "SPY", "direct", interval="1s", classify="aggressor")
        tf_fut = trade_flow("20200312", "ESM0", "futures", interval="1s", classify="aggressor")
        n_printable_eq = eq_csv.count(",Printable\n") + eq_csv.count(",Printable\r")
        eq_ok = (tf_eq["SPY_trade_buy"].sum() > 0 and tf_eq["SPY_trade_sell"].sum() > 0
                 and (tf_eq["SPY_trade_buy"].sum() + tf_eq["SPY_trade_sell"].sum()) <= n_printable_eq + 1)
        # futures: aggressorside-signed buy count must equal the number of 'Buy' rows
        fut_ok = abs(tf_fut["ES_trade_buy"].sum() - n_buy_fut) < 1e-6
        print("message layer (mstwx-lakequery)")
        print("    equity tick-fallback buy=%d sell=%d, NonPrintable dropped (<=%d printable): %s"
              % (int(tf_eq["SPY_trade_buy"].sum()), int(tf_eq["SPY_trade_sell"].sum()), n_printable_eq, eq_ok))
        print("    futures aggressorside buy=%d == #Buy rows %d: %s"
              % (int(tf_fut["ES_trade_buy"].sum()), n_buy_fut, fut_ok))
        msg_ok = eq_ok and fut_ok

        # ---- CLI extract end-to-end (BOTH legs reconstructed from lakequery messages; no snapshot) ----
        em_ts = pd.date_range("2020-03-12 09:30:00", periods=40, freq="s",
                              tz="America/New_York").tz_convert("UTC").as_unit("ns").astype("int64")
        spy_rows, es_rows = [], []
        for k, t in enumerate(em_ts):
            spy_rows += [(t, "bats_edgx", "Bid", 550.00 - 0.01 * (k % 3), 100, f"sb{k}"),
                         (t, "bats_edgx", "Ask", 550.05 + 0.01 * (k % 3), 100, f"sa{k}")]
            es_rows += [(t, "cme_globex30_cme", "Bid", 550000 - 100 * (k % 3), 5, f"eb{k}"),  # int hundredths
                        (t, "cme_globex30_cme", "Ask", 550050 + 100 * (k % 3), 8, f"ea{k}")]
        spy_add = pd.DataFrame(spy_rows, columns=["receipttimestamp", "f", "side", "price", "quantity",
                                                  "orderreferencenumber"]).to_csv(index=False)
        es_add = pd.DataFrame(es_rows, columns=["receipttimestamp", "f", "side", "price", "quantity",
                                                "orderreferencenumber"]).to_csv(index=False)

        def _disp_ex(cmd):                              # dispatch on product_type + message type
            pt = "futures" if "futures" in cmd else "direct"
            m = cmd[cmd.index("-m") + 1]
            if pt == "direct":                          # SPY: order-based venue + tape
                return {"mt_add_order": spy_add, "mt_trade": eq_csv}.get(m, "")
            return {"mt_add_order": es_add, "mt_trade": fut_csv}.get(m, "")   # ES: CME MBO + tape
        _patch_msg(_disp_ex)
        # reconstruct_session does `import mstbook_loader` -> when this file is run as __main__ that is a
        # SECOND module object; patch its runners too so the book legs see the synthetic stream. (No-op
        # in production, where the package is imported once.)
        import mstbook_loader as _mlmod
        _mlmod._run_mstwx_lakequery = _disp_ex
        _mlmod._run_mstwx_lakequery_to_file = _patch_to_file(_disp_ex)
        sess_ex = extract_sessions([("2020-03-12", "stress")], levels=L, interval="1s", with_flow=True,
                                   start_time="9:30", end_time="9:41", futures_scale=0.01)
        df_ex = sess_ex[0][2]
        flow_attached = all(c in df_ex.columns for c in
                            ("SPY_trade_buy", "SPY_trade_sell", "ES_trade_buy", "ES_trade_sell"))
        has_flow = has_trade_flow(sess_ex)
        cnt = counts_from_frame(df_ex)
        counts_ok = (cnt is not None and len(cnt) == 6
                     and all(np.asarray(x).shape[0] == len(df_ex) for x in cnt))
        import importlib                                   # dynamic: the data layer must not statically
        pt = importlib.import_module("paper_tables")        # depend on the manuscript table builder
        t5 = pt.table_imbalance_contingency(sess_ex, counts_from_frame)
        # ES median mid should be in index points (~5500) after price_scale, not raw integers
        es_scaled = abs(float(np.nanmedian(df_ex.get("ES_bidprice_1", pd.Series(np.nan)).to_numpy(float)))
                        - 5500.0) < 5.0
        extract_ok = (flow_attached and has_flow and counts_ok and t5 is not None and es_scaled
                      and "SPY_bidprice_1" in df_ex.columns and "ES_askprice_1" in df_ex.columns)
        print("CLI extract (sole path: SPY consolidated + ES CME MBP, one clock, no snapshot)")
        print("    flow attached:", flow_attached, "| has_trade_flow:", has_flow,
              "| counts_from_frame -> 6-tuple aligned:", counts_ok, "| Table 5 builds:", t5 is not None)

        # ---- parallel session extraction: threading backend must match the serial path exactly ----
        # (loky/process workers can't see this monkeypatch; the dispatch logic is shared with threading,
        #  so threading-equivalence validates the refactor. A separate joblib smoke confirms the engine.)
        two = [("2020-03-12", "stress"), ("2020-03-13", "calm")]
        kw = dict(levels=L, interval="1s", with_flow=True, start_time="9:30", end_time="9:41",
                  futures_scale=0.01)
        sess_par = extract_sessions(two, max_workers=2, backend="threading", **kw)
        sess_ser = extract_sessions(two, max_workers=1, **kw)
        order_ok = [s[0] for s in sess_par] == ["2020-03-12", "2020-03-13"]   # input order preserved
        frames_ok = (len(sess_par) == 2 and all(
            sess_par[i][2].shape == sess_ser[i][2].shape and np.allclose(
                sess_par[i][2].get("SPY_bidprice_1", pd.Series(dtype=float)).to_numpy(float),
                sess_ser[i][2].get("SPY_bidprice_1", pd.Series(dtype=float)).to_numpy(float),
                equal_nan=True) for i in range(2)))
        try:                                                  # engine smoke (logic shared w/ threading)
            from joblib import Parallel, delayed
            joblib_ok = Parallel(n_jobs=2)(delayed(abs)(x) for x in (-1, -2, -3)) == [1, 2, 3]
        except Exception:
            joblib_ok = True                                  # absent -> stdlib ProcessPool covers prod
        parallel_ok = order_ok and frames_ok and joblib_ok
        print("    parallel extract (threading==serial): order=%s frames=%s | joblib engine=%s : %s"
              % (order_ok, frames_ok, joblib_ok, parallel_ok))

        # ---- trade break / correction scrub ----
        ts0 = pd.Timestamp("2025-04-03 10:00:00", tz="America/New_York").tz_convert("UTC").value
        tr = pd.DataFrame({"receipttimestamp": [ts0, ts0 + 10**9, ts0 + 2 * 10**9, ts0 + 3 * 10**9],
                           "price": [540.00, 540.10, 553.45, 540.20], "quantity": [100, 100, 100, 100],
                           "aggressorside": ["", "", "", ""], "side": ["Bid", "Ask", "Bid", "Ask"],
                           "printable": ["Printable"] * 4, "matchid": [101, 102, 103, 104]})
        brk = pd.DataFrame({"receipttimestamp": [ts0 + 30 * 10**9], "matchid": [102],
                            "price": [540.10], "quantity": [100]})                 # 102 busted
        cor = pd.DataFrame({"receipttimestamp": [ts0 + 3600 * 10**9], "matchid": [103],
                            "oldprice": [553.45], "newprice": [543.45], "newquantity": [100]})  # $10 fix
        csv = {"mt_trade": tr.to_csv(index=False), "mt_trade_break": brk.to_csv(index=False),
               "mt_trade_correction": cor.to_csv(index=False)}
        _patch_msg(lambda cmd: csv.get(cmd[cmd.index("-m") + 1], ""))
        trades = _fetch_messages("20250403", "SPY", "direct", "mt_trade")
        scrubbed, sc = _scrub_trades(trades, "20250403", "SPY", "direct")
        mids = set(pd.to_numeric(scrubbed["matchid"]).tolist())
        corr_px = float(pd.to_numeric(scrubbed.loc[pd.to_numeric(scrubbed["matchid"]) == 103, "price"]).iloc[0])
        scrub_ok = (sc["broken"] == 1 and sc["corrected"] == 1 and len(scrubbed) == 3
                    and 102 not in mids and abs(corr_px - 543.45) < 1e-9)
        print("trade scrub (busts/corrections)")
        print("    matchid 102 dropped, 103 corrected 553.45->%.2f (broken=%d corrected=%d, %d->%d prints): %s"
              % (corr_px, sc["broken"], sc["corrected"], sc["n0"], len(scrubbed), scrub_ok))

        # ---- read-path equivalence: usecols + string-dtype + (pyarrow | C+mmap) reproduces the legacy
        #      object-inference read at the _fetch_messages level (the frame reconstruction consumes),
        #      including a degenerate empty-field row and extra JUNK columns that usecols must drop ----
        wide_hdr = ("receipttimestamp,f,side,price,quantity,orderreferencenumber,previousprice,"
                    "previousquantity,maintainpriority,admindelete,JUNK1,JUNK2\n")
        t0v = pd.Timestamp("2025-04-03 09:30:00", tz="America/New_York").tz_convert("UTC").value
        erows = []
        for k in range(40):                                    # clean integer refs (no nulls -> keys match exactly)
            tt = t0v + k * 10**9
            erows.append(f"{tt},bats_edgx,Bid,{550.00-0.01*(k%4):.2f},100,{1000+k},,,,,jx,jy")
            erows.append(f"{tt},bats_edgx,Ask,{550.06+0.01*(k%4):.2f},120,{2000+k},,,,,jx,jy")
        erows.append(f"{t0v+41*10**9},bats_edgx,,,,3001,,,,,jx,jy")   # empty side/price/qty (valid ref)
        add_eqv = wide_hdr + "\n".join(erows) + "\n"
        _patch_msg(lambda cmd: add_eqv if cmd[cmd.index("-m") + 1] == "mt_add_order" else "")
        f_new = _fetch_messages("20250403", "SPY", "direct", "mt_add_order")
        def _legacy_read(path, message_type=""):               # the pre-change read: full object inference
            with open(path) as fh:
                txt = fh.read()
            return pd.read_csv(StringIO(txt), low_memory=False)
        orig_read = globals().get("_read_messages_csv")
        globals()["_read_messages_csv"] = _legacy_read
        try:
            f_old = _fetch_messages("20250403", "SPY", "direct", "mt_add_order")
        finally:
            globals()["_read_messages_csv"] = orig_read
        _numc = ("price", "quantity", "previousprice", "previousquantity")
        num_ok = all(np.allclose(pd.to_numeric(f_new[c], errors="coerce").to_numpy(float),
                                 pd.to_numeric(f_old[c], errors="coerce").to_numpy(float), equal_nan=True)
                     for c in _numc if c in f_new.columns and c in f_old.columns)
        key_ok = (list(f_new["orderreferencenumber"].astype(str))           # book keys identical (clean refs)
                  == list(f_old["orderreferencenumber"].astype(str)))
        drop_ok = "JUNK1" not in f_new.columns and "JUNK2" not in f_new.columns   # usecols dropped the noise
        idx_ok = f_new.index.equals(f_old.index)                            # same tz-aware event index
        readpath_ok = bool(num_ok and key_ok and drop_ok and idx_ok)
        print("read-path equivalence (usecols + string-dtype vs legacy object read)")
        print("    numerics==: %s | order-ref keys==: %s | JUNK cols dropped: %s | index==: %s -> %s"
              % (num_ok, key_ok, drop_ok, idx_ok, readpath_ok))
    finally:
        globals()["_run_mstbook_query"], globals()["_run_mstwx_lakequery"] = orig_book, orig_msg
        globals()["_run_mstwx_lakequery_to_file"] = orig_to_file
        try:
            import mstbook_loader as _mlmod
            _mlmod._run_mstwx_lakequery = orig_msg
            _mlmod._run_mstwx_lakequery_to_file = orig_to_file
        except Exception:
            pass

    ok = all([schema_ok, dtype_ok, tz_ok, torn_ok, mp_up, feat_fin, vwap_ok, pair_ok, helpers_ok,
              msg_ok, extract_ok, parallel_ok, scrub_ok, readpath_ok])
    print("\nchecks:", ok)
    return ok


if __name__ == "__main__":
    _selftest()
