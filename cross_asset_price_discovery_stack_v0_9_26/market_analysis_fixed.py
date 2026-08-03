#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import maystreet_data
from typing import Tuple, List, Callable, Optional, Dict
import time
import concurrent.futures as cf
import logging
import sys
import re
from datetime import datetime, timedelta, date as date_type


# ════════════════════════════════════════════════════════════════════════════
# Logging — configured in main / __main__ only, not at import time
# ════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


def _configure_logging(timestamp: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(
                f"/home/garrisonr/paddrik_pk_liquidity_analysis/logs/"
                f"market_analysis_run_on_{timestamp}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ════════════════════════════════════════════════════════════════════════════
# Instrument profiles
# ════════════════════════════════════════════════════════════════════════════

INSTRUMENT_PROFILES: Dict[str, dict] = {
    'SPY': {
        'tick_size':      0.01,
        'typical_spread': 0.01,
        'lambda_dollar':  10.0,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'equity',
    },
    'QQQ': {
        'tick_size':      0.01,
        'typical_spread': 0.01,
        'lambda_dollar':  10.0,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'equity',
    },
    'IWM': {
        'tick_size':      0.01,
        'typical_spread': 0.01,
        'lambda_dollar':  10.0,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'equity',
    },
    'ES': {
        'tick_size':      0.25,
        'typical_spread': 0.25,
        'lambda_dollar':  0.4,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'futures',
    },
    'NQ': {
        'tick_size':      0.25,
        'typical_spread': 0.25,
        'lambda_dollar':  0.4,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'futures',
    },
    'RTY': {
        'tick_size':      0.10,
        'typical_spread': 0.10,
        'lambda_dollar':  1.0,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'futures',
    },
    'BTCUSD': {
        'tick_size':      0.01,
        'typical_spread': 1.00,
        'lambda_dollar':  0.001,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'crypto_spot',
    },
    'BTC': {
        'tick_size':      5.00,
        'typical_spread': 5.00,
        'lambda_dollar':  0.0002,
        'lambda_pct':    500.0,
        'lambda_ticks':    1.0,
        'asset_class':   'crypto_futures',
    },
}

_DEFAULT_PROFILE = {
    'tick_size':      0.01,
    'typical_spread': 0.01,
    'lambda_dollar':  10.0,
    'lambda_pct':    500.0,
    'lambda_ticks':    1.0,
    'asset_class':   'unknown',
}


def get_instrument_profile(product: str) -> dict:
    base = re.sub(r'[A-Z]\d+$', '', product.upper())
    profile = INSTRUMENT_PROFILES.get(base) or INSTRUMENT_PROFILES.get(product.upper())
    if profile is None:
        logger.warning(
            f"No instrument profile for '{product}' (base='{base}'). "
            f"Using default profile."
        )
        return _DEFAULT_PROFILE
    return profile


CONFIG = {
    'spy_product':   'SPY',
    'es_pattern':    '^ES[A-Z][0-9]$',
    'tables': {
        'nbbo':   'p_mst_data_lake.mt_aggregated_price_update',
        'add':    'p_mst_data_lake.mt_add_order',
        'cancel': 'p_mst_data_lake.mt_cancel_order',
        'modify': 'p_mst_data_lake.mt_modify_order',
        'trade':  'p_mst_data_lake.mt_trade',
    },
    'futures_exchange': 'cme_globex30_cme',
    'spy_depth_feeds': [
        'xdp_nyse_integrated',
        'xdp_arca_integrated',
        'xdp_american_integrated',
        'xdp_national_integrated',
        'xdp_chicago_integrated',
        'total_view',
        'total_view_psx',
        'total_view_bx',
        'bats_edgx',
        'bats_edga',
        'bzx',
        'byx',
        'memoir_depth',
        'miax_pearl_equities_dom',
    ],
    'auc_method':        'all',
    'lambda_decay':       0.5,
    'n_levels':           10,
    'chunk_hours':         2,
    'min_chunk_minutes':  15,
    'retry_delay_s':      30,
}


# ════════════════════════════════════════════════════════════════════════════
# Debug helper
# ════════════════════════════════════════════════════════════════════════════

def debug_sql(sql: str, context: str = '') -> None:
    """Print SQL with line numbers to make Athena parse errors easy to locate."""
    lines = sql.split('\n')
    logger.debug(f"=== Generated SQL {context} ===")
    for i, line in enumerate(lines, 1):
        logger.debug(f"{i:4d}: {line}")
    logger.debug(f"=== End SQL {context} ===")


# ════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ════════════════════════════════════════════════════════════════════════════

def validate_date(date: str) -> bool:
    """
    Return True only if *date* is a valid YYYY-MM-DD string that refers to a
    day strictly in the past.  Today is rejected because intraday data is
    likely incomplete.
    """
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        logger.error(f"Invalid date format: {date}. Expected YYYY-MM-DD")
        return False

    if date_obj >= date_type.today():
        logger.warning(
            f"Date {date} is today or in the future — "
            f"data may be incomplete, skipping."
        )
        return False
    return True


def interval_to_seconds(interval: str) -> int:
    """
    Convert an interval string such as '1s', '5min', '1H' to an integer
    number of seconds.

    Supported unit suffixes
    -----------------------
    s          seconds
    min / T    minutes
    h / H      hours
    d / D      days
    """
    match = re.match(r'^(\d+)([a-zA-Z]+)$', interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval!r}")
    amount = int(match.group(1))
    unit   = match.group(2)
    unit_map = {
        's':   1,
        'min': 60,
        'T':   60,
        'h':   3600,
        'H':   3600,
        'd':   86400,
        'D':   86400,
    }
    if unit not in unit_map:
        raise ValueError(f"Unsupported time unit: {unit!r}")
    return amount * unit_map[unit]


# ════════════════════════════════════════════════════════════════════════════
# SQL fragment builders
# ════════════════════════════════════════════════════════════════════════════

def _build_level_columns_sql(n: int) -> str:
    """NBBO wide columns — trailing comma on every line (safe inside SELECT list)."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"            CAST(nbbo.bidprice_{i}    AS DOUBLE) AS bidprice_{i},")
        lines.append(f"            CAST(nbbo.askprice_{i}    AS DOUBLE) AS askprice_{i},")
        lines.append(f"            CAST(nbbo.bidquantity_{i} AS DOUBLE) AS bidquantity_{i},")
        lines.append(f"            CAST(nbbo.askquantity_{i} AS DOUBLE) AS askquantity_{i},")
    return "\n".join(lines)


def _build_null_level_columns(n: int) -> str:
    """NULL placeholders — trailing comma on every line."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"            CAST(NULL AS DOUBLE) AS bidprice_{i},")
        lines.append(f"            CAST(NULL AS DOUBLE) AS askprice_{i},")
        lines.append(f"            CAST(NULL AS DOUBLE) AS bidquantity_{i},")
        lines.append(f"            CAST(NULL AS DOUBLE) AS askquantity_{i},")
    return "\n".join(lines)


def _build_last_value_columns(n: int) -> str:
    """LAST_VALUE window columns — trailing comma on every line."""
    lines = []
    for i in range(1, n + 1):
        for col in (
            f"bidprice_{i}", f"askprice_{i}",
            f"bidquantity_{i}", f"askquantity_{i}",
        ):
            lines.append(
                f"            LAST_VALUE({col}) IGNORE NULLS OVER w AS {col},"
            )
    return "\n".join(lines)


def _build_level_passthrough_sql(n: int, start: int = 2) -> str:
    """
    Pass-through level columns for the filtered_events CTE.

    Level 1 is already projected with short aliases (bidprice, askprice, …)
    immediately above the call site, so we default to *start=2* to avoid
    duplicating those columns.  Pass start=1 only when the short-alias block
    is absent.

    Every generated line carries a trailing comma because more columns follow
    in the enclosing SELECT.
    """
    lines = []
    for i in range(start, n + 1):
        lines.append(
            f"            bidprice_{i}, askprice_{i},"
            f" bidquantity_{i}, askquantity_{i},"
        )
    return "\n".join(lines)


def _book_col_names(n: int) -> List[str]:
    """Per-level book column names emitted into the final frame (bare; the asset
    prefix is applied later in fetch_aggregated_product, giving {ASSET}_bidprice_i)."""
    names: List[str] = []
    for i in range(1, n + 1):
        names += [f"bidprice_{i}", f"askprice_{i}", f"bidquantity_{i}", f"askquantity_{i}"]
    return names


def _book_time_buckets_cols(n: int) -> str:
    """Per-level book passthrough into the time_buckets CTE. The levels are already
    forward-filled per event in with_nbbo, so they pass through unchanged here.
    No leading/trailing comma (slots in after the *_auc fragment)."""
    return ",\n        ".join(_book_col_names(n))


def _book_aggregated_cols(n: int) -> str:
    """Bucket-CLOSE book snapshot — the latest event's book within each bucket,
    mirroring how `spread` is aggregated (MAX_BY by exchange timestamp).
    No leading/trailing comma."""
    return ",\n        ".join(
        f"MAX_BY({c}, exchangetimestamp) AS {c}" for c in _book_col_names(n)
    )


def _book_filled_cols(n: int) -> str:
    """Forward-fill the book across empty buckets, mirroring the `spread` fill.
    Requires the enclosing WINDOW w. No leading/trailing comma."""
    return ",\n        ".join(
        f"COALESCE({c}, LAST_VALUE({c}) IGNORE NULLS OVER w) AS {c}"
        for c in _book_col_names(n)
    )


def _book_final_cols(n: int) -> str:
    """Per-level book passthrough into the final projection. No leading/trailing comma."""
    return ",\n    ".join(_book_col_names(n))


# ── Distance-expression factories ────────────────────────────────────────────

def _dist_dollar(p: str) -> str:
    return f"ABS({p} - mid_price)"


def _dist_pct(p: str) -> str:
    return f"ABS({p} - mid_price) / NULLIF(mid_price, 0.0)"


def _make_dist_ticks(tick_size: float) -> Callable[[str], str]:
    """Return a distance-in-ticks factory bound to *tick_size*."""
    def _dist(p: str) -> str:
        return f"ABS({p} - mid_price) / {tick_size:.10f}"
    return _dist


def _build_auc_terms(
    side: str,
    lambda_val: float,
    dist_fn: Callable[[str], str],
    n: int,
) -> str:
    """
    Build the summed trapezoidal AUC expression for one side + one lambda.

    Parameters
    ----------
    side       : 'bid' or 'ask'
    lambda_val : decay parameter (fixed-precision literal to avoid sci notation)
    dist_fn    : callable(price_sql_expr) -> distance_sql_expr
    n          : number of book levels

    Returns a SQL expression string with no trailing comma.
    """
    lambda_lit = f"{lambda_val:.10f}"

    terms = []
    for i in range(1, n + 1):
        p = f"COALESCE({side}price_{i}, 0.0)"
        q = f"COALESCE({side}quantity_{i}, 0.0)"

        if n == 1:
            delta = "1.0"
        elif i == 1:
            if side == 'ask':
                delta = (
                    f"(COALESCE(askprice_{i+1}, askprice_{i})"
                    f" - COALESCE(askprice_{i}, 0.0))"
                )
            else:
                delta = (
                    f"(COALESCE(bidprice_{i}, 0.0)"
                    f" - COALESCE(bidprice_{i+1}, bidprice_{i}))"
                )
        elif i == n:
            if side == 'ask':
                delta = (
                    f"(COALESCE(askprice_{i}, 0.0)"
                    f" - COALESCE(askprice_{i-1}, askprice_{i}))"
                )
            else:
                delta = (
                    f"(COALESCE(bidprice_{i-1}, bidprice_{i})"
                    f" - COALESCE(bidprice_{i}, 0.0))"
                )
        else:
            if side == 'ask':
                delta = (
                    f"(COALESCE(askprice_{i+1}, askprice_{i})"
                    f" - COALESCE(askprice_{i-1}, askprice_{i})) / 2.0"
                )
            else:
                delta = (
                    f"(COALESCE(bidprice_{i-1}, bidprice_{i})"
                    f" - COALESCE(bidprice_{i+1}, bidprice_{i})) / 2.0"
                )

        dist_expr = dist_fn(p)
        weight    = f"EXP(-{lambda_lit} * {dist_expr})"
        terms.append(
            f"            CASE WHEN {side}price_{i} IS NOT NULL\n"
            f"                 THEN {q} * {weight} * ABS({delta})\n"
            f"                 ELSE 0.0 END"
        )
    return "(\n" + "\n            +\n".join(terms) + "\n            )"


def _build_auc_expressions(
    profile: dict,
    n: int,
    method: str,
) -> Dict[str, str]:
    """
    Return an ordered dict of {column_name: sql_expression} for all requested
    AUC columns.  No trailing commas — caller decides punctuation.

    method: 'dollar' | 'pct' | 'ticks' | 'all'
    """
    tick_size     = profile['tick_size']
    lambda_dollar = profile['lambda_dollar']
    lambda_pct    = profile['lambda_pct']
    lambda_ticks  = profile['lambda_ticks']

    dist_ticks = _make_dist_ticks(tick_size)

    result: Dict[str, str] = {}

    if method in ('dollar', 'all'):
        result['bid_auc_dollar'] = _build_auc_terms('bid', lambda_dollar, _dist_dollar, n)
        result['ask_auc_dollar'] = _build_auc_terms('ask', lambda_dollar, _dist_dollar, n)

    if method in ('pct', 'all'):
        result['bid_auc_pct']    = _build_auc_terms('bid', lambda_pct,    _dist_pct,    n)
        result['ask_auc_pct']    = _build_auc_terms('ask', lambda_pct,    _dist_pct,    n)

    if method in ('ticks', 'all'):
        result['bid_auc_ticks']  = _build_auc_terms('bid', lambda_ticks,  dist_ticks,   n)
        result['ask_auc_ticks']  = _build_auc_terms('ask', lambda_ticks,  dist_ticks,   n)

    return result


def _auc_col_names(method: str) -> List[str]:
    """Return ordered list of AUC output column names for a given method."""
    if method == 'all':
        return [
            'bid_auc_dollar', 'ask_auc_dollar', 'auc_imbalance_dollar',
            'bid_auc_pct',    'ask_auc_pct',    'auc_imbalance_pct',
            'bid_auc_ticks',  'ask_auc_ticks',  'auc_imbalance_ticks',
        ]
    return [
        f'bid_auc_{method}',
        f'ask_auc_{method}',
        f'auc_imbalance_{method}',
    ]


def _build_events_with_auc_select(auc_exprs: Dict[str, str]) -> str:
    """
    AUC expressions for the events_with_auc CTE.
    NO leading or trailing comma — joined with ',\\n' between items only.
    """
    parts = [
        f"            {expr} AS {col_name}"
        for col_name, expr in auc_exprs.items()
    ]
    return ",\n".join(parts)


def _build_time_buckets_auc_cols(auc_exprs: Dict[str, str]) -> str:
    """
    Column pass-through for the time_buckets SELECT.
    Trailing comma on every line except the last.
    """
    keys = list(auc_exprs.keys())
    lines = []
    for i, col in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        lines.append(f"            {col}{comma}")
    return "\n".join(lines)


def _build_aggregated_auc_cols(auc_exprs: Dict[str, str]) -> str:
    """MAX_BY aggregation — NO trailing comma."""
    lines = [
        f"            MAX_BY({col}, exchangetimestamp) AS {col}"
        for col in auc_exprs.keys()
    ]
    return ",\n".join(lines)


def _build_filled_auc_cols(auc_exprs: Dict[str, str]) -> str:
    """
    Forward-fill + imbalance for each bid/ask AUC pair.
    NO leading or trailing comma.

    Implementation note
    -------------------
    Presto/Athena does not allow a SELECT alias to be referenced by a sibling
    expression in the same SELECT clause.  The LAST_VALUE window expression is
    therefore repeated verbatim inside the imbalance formula.
    """
    lines = []

    for col in auc_exprs.keys():
        lines.append(
            f"            COALESCE({col},\n"
            f"                LAST_VALUE({col}) IGNORE NULLS OVER w) AS {col}"
        )

    method_suffixes = [
        col[len('bid_auc_'):]
        for col in auc_exprs.keys()
        if col.startswith('bid_auc_')
    ]
    for suffix in method_suffixes:
        bid_col = f'bid_auc_{suffix}'
        ask_col = f'ask_auc_{suffix}'
        imb_col = f'auc_imbalance_{suffix}'
        lines.append(
            f"            (\n"
            f"                COALESCE({bid_col},\n"
            f"                    LAST_VALUE({bid_col}) IGNORE NULLS OVER w)\n"
            f"                - COALESCE({ask_col},\n"
            f"                    LAST_VALUE({ask_col}) IGNORE NULLS OVER w)\n"
            f"            ) / NULLIF(\n"
            f"                COALESCE({bid_col},\n"
            f"                    LAST_VALUE({bid_col}) IGNORE NULLS OVER w)\n"
            f"                + COALESCE({ask_col},\n"
            f"                    LAST_VALUE({ask_col}) IGNORE NULLS OVER w),\n"
            f"                0.0\n"
            f"            ) AS {imb_col}"
        )

    return ",\n".join(lines)


def _build_final_auc_cols(auc_exprs: Dict[str, str]) -> str:
    """Final SELECT column list — AUC cols + imbalance cols.  NO trailing comma."""
    bid_ask_cols = list(auc_exprs.keys())
    method_suffixes = [
        c[len('bid_auc_'):]
        for c in bid_ask_cols
        if c.startswith('bid_auc_')
    ]
    imb_cols = [f'auc_imbalance_{s}' for s in method_suffixes]
    all_cols = bid_ask_cols + imb_cols
    return ",\n        ".join(all_cols)


def _liquidity_action_case() -> str:
    return """
            CASE
                WHEN type = 'TRADE'  THEN 'LD'
                WHEN type = 'ADD'    THEN 'LS'
                WHEN type = 'CANCEL' THEN 'LW'
                WHEN type = 'MODIFY' THEN
                    CASE
                        WHEN (quantity IS NOT NULL AND previousquantity IS NOT NULL
                              AND quantity - previousquantity > 0)
                          OR (quantity IS NOT NULL AND previousquantity IS NULL
                              AND quantity > 0)                          THEN 'LS'
                        WHEN (quantity IS NOT NULL AND previousquantity IS NOT NULL
                              AND quantity - previousquantity < 0)
                          OR (previousquantity IS NOT NULL AND quantity IS NULL
                              AND previousquantity > 0)                  THEN 'LW'
                        ELSE 'LN'
                    END
                ELSE 'Other'
            END"""


def _time_bucket_expr(date: str, interval_seconds: int) -> str:
    """
    Truncate exchange_dt to the nearest interval_seconds boundary anchored
    at 04:00 ET.  Arithmetic is in Unix epoch seconds for DST safety.
    """
    return f"""at_timezone(
                from_unixtime(
                    CAST(
                        to_unixtime(TIMESTAMP '{date} 04:00:00 America/New_York')
                        + FLOOR(
                            (  to_unixtime(exchange_dt)
                             - to_unixtime(TIMESTAMP '{date} 04:00:00 America/New_York')
                            ) / {interval_seconds}
                          ) * {interval_seconds}
                    AS BIGINT)
                ),
                'America/New_York'
            )"""


# ════════════════════════════════════════════════════════════════════════════
# Unified NBBO order book CTE builder
# ════════════════════════════════════════════════════════════════════════════

def _build_unified_nbbo_cte(
    nbbo_table: str,
    nbbo_filter_sql: str,
    nbbo_timestamp_col: str,
    n_levels: int,
    start_ts: str,
    end_ts: str,
) -> str:
    level_arr     = ", ".join(str(i)             for i in range(1, n_levels + 1))
    bid_price_arr = ", ".join(f"bidprice_{i}"    for i in range(1, n_levels + 1))
    bid_qty_arr   = ", ".join(f"bidquantity_{i}" for i in range(1, n_levels + 1))
    ask_price_arr = ", ".join(f"askprice_{i}"    for i in range(1, n_levels + 1))
    ask_qty_arr   = ", ".join(f"askquantity_{i}" for i in range(1, n_levels + 1))

    bid_level_proj = "\n".join(
        f"               CAST(nbbo.bidprice_{i}    AS DOUBLE) AS bidprice_{i},\n"
        f"               CAST(nbbo.bidquantity_{i} AS DOUBLE) AS bidquantity_{i},"
        for i in range(1, n_levels + 1)
    )
    ask_level_proj = "\n".join(
        f"               CAST(nbbo.askprice_{i}    AS DOUBLE) AS askprice_{i},\n"
        f"               CAST(nbbo.askquantity_{i} AS DOUBLE) AS askquantity_{i},"
        for i in range(1, n_levels + 1)
    )

    def _level_passthrough(n: int, trailing_comma: bool) -> str:
        lines = []
        for i in range(1, n + 1):
            is_last = (i == n)
            comma = "," if (not is_last or trailing_comma) else ""
            lines.append(
                f"               bidprice_{i}, bidquantity_{i},\n"
                f"               askprice_{i}, askquantity_{i}{comma}"
            )
        return "\n".join(lines)

    lpf_level_cols    = _level_passthrough(n_levels, trailing_comma=True)
    latest_level_cols = _level_passthrough(n_levels, trailing_comma=False)

    # Pivot: explicit MAX(CASE...) per level — no trailing comma on last
    bid_pivot_lines = []
    ask_pivot_lines = []
    for i in range(1, n_levels + 1):
        sep = "," if i < n_levels else ""
        bid_pivot_lines.append(
            f"        MAX(CASE WHEN lvl = {i} THEN bp END) AS bidprice_{i},\n"
            f"        MAX(CASE WHEN lvl = {i} THEN bq END) AS bidquantity_{i}{sep}"
        )
        ask_pivot_lines.append(
            f"        MAX(CASE WHEN lvl = {i} THEN ap END) AS askprice_{i},\n"
            f"        MAX(CASE WHEN lvl = {i} THEN aq END) AS askquantity_{i}{sep}"
        )
    bid_pivot = "\n".join(bid_pivot_lines)
    ask_pivot = "\n".join(ask_pivot_lines)

    # ── unified_nbbo_typed: re-casts every pivot column so the names are
    #    unambiguous to Athena's column resolver when consumed via UNION ALL.
    #    Without this extra projection layer, Athena loses the pivot aliases
    #    and raises COLUMN_NOT_FOUND on bidprice_1 inside the window CTE.
    typed_bid_lines = []
    typed_ask_lines = []
    for i in range(1, n_levels + 1):
        typed_bid_lines.append(
            f"        CAST(bidprice_{i}    AS DOUBLE) AS bidprice_{i},\n"
            f"        CAST(bidquantity_{i} AS DOUBLE) AS bidquantity_{i},"
        )
        typed_ask_lines.append(
            f"        CAST(askprice_{i}    AS DOUBLE) AS askprice_{i},\n"
            f"        CAST(askquantity_{i} AS DOUBLE) AS askquantity_{i},"
        )
    # Last ask line must not have a trailing comma
    if typed_ask_lines:
        typed_ask_lines[-1] = typed_ask_lines[-1].rstrip(",")

    typed_bid = "\n".join(typed_bid_lines)
    typed_ask = "\n".join(typed_ask_lines)

    return f"""
    nbbo_raw AS (
        SELECT
               nbbo.f                                    AS feed_id,
{bid_level_proj}
{ask_level_proj}
               CAST({nbbo_timestamp_col} AS BIGINT)      AS ts_nanos,
               date_trunc('second',
                   at_timezone(
                       from_unixtime_nanos(CAST({nbbo_timestamp_col} AS BIGINT)),
                       'America/New_York'
                   )
               )                                         AS snap_second
        FROM {nbbo_table} nbbo
        CROSS JOIN config
        WHERE {nbbo_filter_sql}
          AND at_timezone(
                from_unixtime_nanos(CAST({nbbo_timestamp_col} AS BIGINT)),
                'America/New_York'
              ) BETWEEN CAST('{start_ts}' AS TIMESTAMP) - INTERVAL '1' MINUTE
                    AND CAST('{end_ts}'   AS TIMESTAMP)
    ),
    nbbo_latest_per_feed AS (
        SELECT
               feed_id,
               snap_second,
               ts_nanos,
{lpf_level_cols}
               ROW_NUMBER() OVER (
                   PARTITION BY feed_id, snap_second
                   ORDER BY ts_nanos DESC
               ) AS feed_rn
        FROM nbbo_raw
    ),
    nbbo_latest AS (
        SELECT
               feed_id,
               snap_second,
               ts_nanos,
{latest_level_cols}
        FROM nbbo_latest_per_feed
        WHERE feed_rn = 1
    ),
    nbbo_bid_levels AS (
        SELECT snap_second,
               feed_id,
               lv  AS book_level,
               bp  AS bid_price,
               bq  AS bid_qty
        FROM nbbo_latest
        CROSS JOIN UNNEST(
            ARRAY[{level_arr}],
            ARRAY[{bid_price_arr}],
            ARRAY[{bid_qty_arr}]
        ) AS t(lv, bp, bq)
        WHERE bp IS NOT NULL AND bp > 0
          AND bq IS NOT NULL AND bq > 0
    ),
    nbbo_ask_levels AS (
        SELECT snap_second,
               feed_id,
               lv  AS book_level,
               ap  AS ask_price,
               aq  AS ask_qty
        FROM nbbo_latest
        CROSS JOIN UNNEST(
            ARRAY[{level_arr}],
            ARRAY[{ask_price_arr}],
            ARRAY[{ask_qty_arr}]
        ) AS t(lv, ap, aq)
        WHERE ap IS NOT NULL AND ap > 0
          AND aq IS NOT NULL AND aq > 0
    ),
    nbbo_top_bids AS (
        SELECT snap_second, bid_price, bid_qty, bid_feeds
        FROM (
            SELECT snap_second,
                   bid_price,
                   SUM(bid_qty)            AS bid_qty,
                   COUNT(DISTINCT feed_id) AS bid_feeds,
                   ROW_NUMBER() OVER (
                       PARTITION BY snap_second
                       ORDER BY bid_price DESC
                   ) AS bl
            FROM nbbo_bid_levels
            GROUP BY snap_second, bid_price
        ) ranked_bids
        WHERE bl <= {n_levels}
    ),
    nbbo_top_asks AS (
        SELECT snap_second, ask_price, ask_qty, ask_feeds
        FROM (
            SELECT snap_second,
                   ask_price,
                   SUM(ask_qty)            AS ask_qty,
                   COUNT(DISTINCT feed_id) AS ask_feeds,
                   ROW_NUMBER() OVER (
                       PARTITION BY snap_second
                       ORDER BY ask_price ASC
                   ) AS al
            FROM nbbo_ask_levels
            GROUP BY snap_second, ask_price
        ) ranked_asks
        WHERE al <= {n_levels}
    ),
    nbbo_combined AS (
        SELECT snap_second,
               ROW_NUMBER() OVER (
                   PARTITION BY snap_second ORDER BY bid_price DESC
               )                    AS lvl,
               bid_price            AS bp,
               bid_qty              AS bq,
               CAST(NULL AS DOUBLE) AS ap,
               CAST(NULL AS DOUBLE) AS aq
        FROM nbbo_top_bids
        UNION ALL
        SELECT snap_second,
               ROW_NUMBER() OVER (
                   PARTITION BY snap_second ORDER BY ask_price ASC
               )                    AS lvl,
               CAST(NULL AS DOUBLE) AS bp,
               CAST(NULL AS DOUBLE) AS bq,
               ask_price            AS ap,
               ask_qty              AS aq
        FROM nbbo_top_asks
    ),
    nbbo_pivoted AS (
        -- Pivot lvl/bp/bq/ap/aq into wide level columns.
        -- Result column names are bidprice_1..N, askprice_1..N, etc.
        SELECT
            snap_second,
{bid_pivot},
{ask_pivot}
        FROM nbbo_combined
        GROUP BY snap_second
    ),
    unified_nbbo AS (
        -- Re-project with explicit CAST so Athena's column resolver can see
        -- bidprice_1..N by name when this CTE is consumed via UNION ALL.
        -- Without this layer the pivot aliases are invisible to the
        -- LAST_VALUE window in with_nbbo and raise COLUMN_NOT_FOUND.
        SELECT
            snap_second,
            CAST(to_unixtime(snap_second) * 1000000000 AS BIGINT) AS exchangetimestamp,
            snap_second AS exchange_dt,
{typed_bid}
{typed_ask}
        FROM nbbo_pivoted
    ),"""



# ════════════════════════════════════════════════════════════════════════════
# Core windowed query builder — SPY (equity, unified NBBO)
# ════════════════════════════════════════════════════════════════════════════

def _build_depth_query_windowed(
    date: str,
    start_ts: str,
    end_ts: str,
    interval_seconds: int,
    profile: dict,
    auc_method: str,
    n_levels: int,
    nbbo_table: str,
    nbbo_timestamp_col: str,
    nbbo_filter_sql: str,
    order_flow_filter_sql: str,
    config_extra_cols: str,
) -> str:
    null_cols      = _build_null_level_columns(n_levels)
    last_val_cols  = _build_last_value_columns(n_levels)
    level_passthru = _build_level_passthrough_sql(n_levels, start=1)
    liq_action     = _liquidity_action_case()
    bucket_expr    = _time_bucket_expr(date, interval_seconds)

    auc_exprs = _build_auc_expressions(profile, n_levels, auc_method)

    events_auc_cols  = _build_events_with_auc_select(auc_exprs)
    # Append the per-level book passthrough to each *_auc fragment so the snapshot
    # book (bucket close, forward-filled) is carried all the way to the final SELECT.
    time_buckets_auc = _build_time_buckets_auc_cols(auc_exprs) + ",\n        " + _book_time_buckets_cols(n_levels)
    aggregated_auc   = _build_aggregated_auc_cols(auc_exprs)   + ",\n        " + _book_aggregated_cols(n_levels)
    filled_auc       = _build_filled_auc_cols(auc_exprs)       + ",\n        " + _book_filled_cols(n_levels)
    final_auc        = _build_final_auc_cols(auc_exprs)        + ",\n    "       + _book_final_cols(n_levels)

    unified_nbbo_cte = _build_unified_nbbo_cte(
        nbbo_table=nbbo_table,
        nbbo_filter_sql=nbbo_filter_sql,
        nbbo_timestamp_col=nbbo_timestamp_col,
        n_levels=n_levels,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    unified_level_cols = "\n".join(
        f"            CAST(u.bidprice_{i}    AS DOUBLE) AS bidprice_{i},\n"
        f"            CAST(u.askprice_{i}    AS DOUBLE) AS askprice_{i},\n"
        f"            CAST(u.bidquantity_{i} AS DOUBLE) AS bidquantity_{i},\n"
        f"            CAST(u.askquantity_{i} AS DOUBLE) AS askquantity_{i},"
        for i in range(1, n_levels + 1)
    )

    for name, frag in [
        ('aggregated_auc',  aggregated_auc),
        ('filled_auc',      filled_auc),
        ('final_auc',       final_auc),
        ('events_auc_cols', events_auc_cols),
    ]:
        stripped = frag.rstrip()
        if stripped.endswith(','):
            raise ValueError(
                f"Fragment '{name}' has a trailing comma — will cause "
                f"an Athena parse error.\nFragment tail:\n{stripped[-200:]}"
            )

    return f"""
WITH config AS (
    SELECT
        '{date}'                                AS dt,
        CAST('{start_ts}' AS TIMESTAMP)         AS start_time,
        CAST('{end_ts}'   AS TIMESTAMP)         AS end_time,
        {interval_seconds}                      AS interval_seconds,
        {profile['lambda_pct']}                 AS lambda_pct,
        {profile['lambda_dollar']}              AS lambda_dollar,
        {profile['lambda_ticks']}               AS lambda_ticks,
        {profile['tick_size']}                  AS tick_size
        {config_extra_cols}
),
{unified_nbbo_cte}
all_events AS (
    SELECT
        'NBBO'                      AS type,
        u.exchangetimestamp,
{unified_level_cols}
        CAST(NULL AS DOUBLE)        AS price,
        CAST(NULL AS DOUBLE)        AS quantity,
        CAST(NULL AS DOUBLE)        AS previousquantity,
        CAST(NULL AS VARCHAR)       AS side,
        u.exchange_dt
    FROM unified_nbbo u

    UNION ALL

    SELECT
        'ADD'                       AS type,
        add_order.exchangetimestamp,
{null_cols}
        CAST(add_order.price    AS DOUBLE) AS price,
        CAST(add_order.quantity AS DOUBLE) AS quantity,
        CAST(NULL AS DOUBLE)               AS previousquantity,
        add_order.side,
        at_timezone(
            from_unixtime_nanos(CAST(add_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['add']} add_order
    CROSS JOIN config
    WHERE {order_flow_filter_sql.format(alias='add_order')}
      AND at_timezone(
            from_unixtime_nanos(CAST(add_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time

    UNION ALL

    SELECT
        'CANCEL'                    AS type,
        cancel_order.exchangetimestamp,
{null_cols}
        CAST(cancel_order.price            AS DOUBLE) AS price,
        CAST(NULL AS DOUBLE)                           AS quantity,
        CAST(cancel_order.previousquantity AS DOUBLE) AS previousquantity,
        cancel_order.side,
        at_timezone(
            from_unixtime_nanos(CAST(cancel_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['cancel']} cancel_order
    CROSS JOIN config
    WHERE {order_flow_filter_sql.format(alias='cancel_order')}
      AND at_timezone(
            from_unixtime_nanos(CAST(cancel_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time

    UNION ALL

    SELECT
        'MODIFY'                    AS type,
        modify_order.exchangetimestamp,
{null_cols}
        CAST(modify_order.price            AS DOUBLE) AS price,
        CAST(modify_order.quantity         AS DOUBLE) AS quantity,
        CAST(modify_order.previousquantity AS DOUBLE) AS previousquantity,
        modify_order.side,
        at_timezone(
            from_unixtime_nanos(CAST(modify_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['modify']} modify_order
    CROSS JOIN config
    WHERE {order_flow_filter_sql.format(alias='modify_order')}
      AND at_timezone(
            from_unixtime_nanos(CAST(modify_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time

    UNION ALL

    SELECT
        'TRADE'                     AS type,
        trade.exchangetimestamp,
{null_cols}
        CAST(trade.price    AS DOUBLE) AS price,
        CAST(trade.quantity AS DOUBLE) AS quantity,
        CAST(NULL AS DOUBLE)           AS previousquantity,
        trade.side,
        at_timezone(
            from_unixtime_nanos(CAST(trade.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['trade']} trade
    CROSS JOIN config
    WHERE {order_flow_filter_sql.format(alias='trade')}
      AND at_timezone(
            from_unixtime_nanos(CAST(trade.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time
),
with_nbbo AS (
    SELECT
        type,
        exchangetimestamp,
        exchange_dt,
{last_val_cols}
        price,
        quantity,
        previousquantity,
        side
    FROM all_events
    WINDOW w AS (
        ORDER BY exchangetimestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
filtered_events AS (
    SELECT
        type,
        exchange_dt,
        exchangetimestamp,
{level_passthru}
        price,
        quantity,
        previousquantity,
        side,
        (askprice_1 + bidprice_1) / 2.0                             AS mid_price,
        (askprice_1 - bidprice_1)
            / NULLIF((askprice_1 + bidprice_1) / 2.0, 0)            AS spread,
        (bidquantity_1 - askquantity_1)
            / NULLIF(bidquantity_1 + askquantity_1, 0)              AS order_imbalance,
        CASE
            WHEN quantity IS NOT NULL AND previousquantity IS NOT NULL
                THEN quantity - previousquantity
            WHEN quantity IS NOT NULL         THEN  quantity
            WHEN previousquantity IS NOT NULL THEN -previousquantity
            ELSE 0
        END AS quantity_change,
{liq_action}    AS liquidity_action
    FROM with_nbbo
    WHERE type != 'NBBO'
      AND bidprice_1    IS NOT NULL
      AND askprice_1    IS NOT NULL
      AND bidquantity_1 IS NOT NULL
      AND askquantity_1 IS NOT NULL
),
events_with_auc AS (
    SELECT
        *,
{events_auc_cols}
    FROM filtered_events
),
time_buckets AS (
    SELECT
        {bucket_expr} AS time_bucket_utc,
        type,
        exchangetimestamp,
        spread,
        order_imbalance,
        price,
        quantity,
        previousquantity,
        liquidity_action,
        quantity_change,
{time_buckets_auc}
    FROM events_with_auc
    CROSS JOIN config
    WHERE exchange_dt BETWEEN config.start_time AND config.end_time
),
aggregated AS (
    SELECT
        time_bucket_utc,
        MAX_BY(spread,          exchangetimestamp)                   AS spread,
        MAX_BY(order_imbalance, exchangetimestamp)                   AS order_imbalance,
        MAX_BY(price,
            CASE WHEN type = 'TRADE' THEN exchangetimestamp
                 ELSE NULL END)                                      AS last_trade_price,
        COUNT(*)                                                     AS liquidity_actions,
        SUM(CASE WHEN type = 'TRADE'  THEN 1 ELSE 0 END)            AS trade_count,
        SUM(CASE WHEN type = 'CANCEL' THEN 1 ELSE 0 END)            AS cancel_count,
        SUM(CASE WHEN type = 'MODIFY' THEN 1 ELSE 0 END)            AS modify_count,
        SUM(CASE WHEN type = 'ADD'    THEN 1 ELSE 0 END)            AS add_count,
        SUM(CASE WHEN liquidity_action = 'LD'    THEN 1 ELSE 0 END) AS LD,
        SUM(CASE WHEN liquidity_action = 'LS'    THEN 1 ELSE 0 END) AS LS,
        SUM(CASE WHEN liquidity_action = 'LW'    THEN 1 ELSE 0 END) AS LW,
        SUM(CASE WHEN liquidity_action = 'LN'    THEN 1 ELSE 0 END) AS LN,
        SUM(CASE WHEN liquidity_action = 'Other' THEN 1 ELSE 0 END) AS Other,
        SUM(CASE WHEN type = 'TRADE'
            THEN COALESCE(quantity,         0) ELSE 0 END)           AS trade_volume,
        SUM(CASE WHEN type = 'CANCEL'
            THEN COALESCE(previousquantity, 0) ELSE 0 END)           AS cancel_volume,
        SUM(CASE WHEN type = 'ADD'
            THEN COALESCE(quantity,         0) ELSE 0 END)           AS add_volume,
        SUM(CASE WHEN type = 'MODIFY'
            THEN ABS(COALESCE(quantity_change, 0)) ELSE 0 END)       AS modify_volume_abs,
        SUM(CASE WHEN type = 'MODIFY'
            THEN COALESCE(quantity_change,  0) ELSE 0 END)           AS modify_volume_net,
        {aggregated_auc}
    FROM time_buckets
    GROUP BY time_bucket_utc
),
filled AS (
    SELECT
        time_bucket_utc,
        COALESCE(spread,
            LAST_VALUE(spread) IGNORE NULLS OVER w)                  AS spread,
        COALESCE(order_imbalance,
            LAST_VALUE(order_imbalance) IGNORE NULLS OVER w)         AS order_imbalance,
        COALESCE(last_trade_price,
            LAST_VALUE(last_trade_price) IGNORE NULLS OVER w)        AS last_trade_price,
        liquidity_actions,
        trade_count, cancel_count, modify_count, add_count,
        LD, LS, LW, LN, Other,
        trade_volume, cancel_volume, add_volume,
        modify_volume_abs, modify_volume_net,
        {filled_auc}
    FROM aggregated
    WINDOW w AS (
        ORDER BY time_bucket_utc
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
)
SELECT
    to_iso8601(time_bucket_utc) AS time_bucket,
    spread,
    order_imbalance,
    last_trade_price,
    liquidity_actions,
    trade_count, cancel_count, modify_count, add_count,
    LD, LS, LW, LN, Other,
    trade_volume, cancel_volume, add_volume,
    modify_volume_abs, modify_volume_net,
    {final_auc}
FROM filled
ORDER BY time_bucket_utc
"""


# ════════════════════════════════════════════════════════════════════════════
# Core windowed query builder — ES (futures, single exchange)
# ════════════════════════════════════════════════════════════════════════════

def _build_es_query_windowed(
    date: str,
    start_ts: str,
    end_ts: str,
    interval_seconds: int,
    profile: dict,
    auc_method: str,
    n_levels: int,
) -> str:
    level_cols     = _build_level_columns_sql(n_levels)
    null_cols      = _build_null_level_columns(n_levels)
    last_val_cols  = _build_last_value_columns(n_levels)
    level_passthru = _build_level_passthrough_sql(n_levels, start=1)
    liq_action     = _liquidity_action_case()
    bucket_expr    = _time_bucket_expr(date, interval_seconds)

    auc_exprs = _build_auc_expressions(profile, n_levels, auc_method)

    events_auc_cols  = _build_events_with_auc_select(auc_exprs)
    # Append the per-level book passthrough to each *_auc fragment so the snapshot
    # book (bucket close, forward-filled) is carried all the way to the final SELECT.
    time_buckets_auc = _build_time_buckets_auc_cols(auc_exprs) + ",\n        " + _book_time_buckets_cols(n_levels)
    aggregated_auc   = _build_aggregated_auc_cols(auc_exprs)   + ",\n        " + _book_aggregated_cols(n_levels)
    filled_auc       = _build_filled_auc_cols(auc_exprs)       + ",\n        " + _book_filled_cols(n_levels)
    final_auc        = _build_final_auc_cols(auc_exprs)        + ",\n    "       + _book_final_cols(n_levels)

    return f"""
WITH config AS (
    SELECT
        '{date}'                                AS dt,
        '{CONFIG['futures_exchange']}'          AS f,
        '{CONFIG['es_pattern']}'                AS product,
        CAST('{start_ts}' AS TIMESTAMP)         AS start_time,
        CAST('{end_ts}'   AS TIMESTAMP)         AS end_time,
        {interval_seconds}                      AS interval_seconds,
        {profile['lambda_pct']}                 AS lambda_pct,
        {profile['lambda_dollar']}              AS lambda_dollar,
        {profile['lambda_ticks']}               AS lambda_ticks,
        {profile['tick_size']}                  AS tick_size
),
contract_quantities AS (
    SELECT mt.f, mt.product, SUM(mt.quantity) AS total_quantity
    FROM {CONFIG['tables']['trade']} mt
    JOIN config ON mt.dt = config.dt AND mt.f = config.f
    WHERE REGEXP_LIKE(UPPER(mt.product), UPPER(config.product))
      AND UPPER(mt.printable) = 'PRINTABLE'
    GROUP BY mt.f, mt.product
),
front_month AS (
    SELECT product FROM contract_quantities
    ORDER BY total_quantity DESC LIMIT 1
),
all_events AS (
    SELECT
        'NBBO'                      AS type,
        nbbo.firstexchangetimestamp AS exchangetimestamp,
{level_cols}
        CAST(NULL AS DOUBLE)        AS price,
        CAST(NULL AS DOUBLE)        AS quantity,
        CAST(NULL AS DOUBLE)        AS previousquantity,
        CAST(NULL AS VARCHAR)       AS side,
        at_timezone(
            from_unixtime_nanos(CAST(nbbo.firstexchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['nbbo']} nbbo
    JOIN config      ON nbbo.f = config.f AND nbbo.dt = config.dt
    JOIN front_month ON UPPER(nbbo.product) = UPPER(front_month.product)
    WHERE at_timezone(
            from_unixtime_nanos(CAST(nbbo.firstexchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time - INTERVAL '1' MINUTE
                AND config.end_time

    UNION ALL

    SELECT
        'ADD'                       AS type,
        add_order.exchangetimestamp,
{null_cols}
        CAST(add_order.price    AS DOUBLE) AS price,
        CAST(add_order.quantity AS DOUBLE) AS quantity,
        CAST(NULL AS DOUBLE)               AS previousquantity,
        add_order.side,
        at_timezone(
            from_unixtime_nanos(CAST(add_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['add']} add_order
    JOIN config      ON add_order.f = config.f AND add_order.dt = config.dt
    JOIN front_month ON UPPER(add_order.product) = UPPER(front_month.product)
    WHERE at_timezone(
            from_unixtime_nanos(CAST(add_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time

    UNION ALL

    SELECT
        'CANCEL'                    AS type,
        cancel_order.exchangetimestamp,
{null_cols}
        CAST(cancel_order.price            AS DOUBLE) AS price,
        CAST(NULL AS DOUBLE)                           AS quantity,
        CAST(cancel_order.previousquantity AS DOUBLE) AS previousquantity,
        cancel_order.side,
        at_timezone(
            from_unixtime_nanos(CAST(cancel_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['cancel']} cancel_order
    JOIN config      ON cancel_order.f = config.f AND cancel_order.dt = config.dt
    JOIN front_month ON UPPER(cancel_order.product) = UPPER(front_month.product)
    WHERE at_timezone(
            from_unixtime_nanos(CAST(cancel_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time

    UNION ALL

    SELECT
        'MODIFY'                    AS type,
        modify_order.exchangetimestamp,
{null_cols}
        CAST(modify_order.price            AS DOUBLE) AS price,
        CAST(modify_order.quantity         AS DOUBLE) AS quantity,
        CAST(modify_order.previousquantity AS DOUBLE) AS previousquantity,
        modify_order.side,
        at_timezone(
            from_unixtime_nanos(CAST(modify_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['modify']} modify_order
    JOIN config      ON modify_order.f = config.f AND modify_order.dt = config.dt
    JOIN front_month ON UPPER(modify_order.product) = UPPER(front_month.product)
    WHERE at_timezone(
            from_unixtime_nanos(CAST(modify_order.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time

    UNION ALL

    SELECT
        'TRADE'                     AS type,
        trade.exchangetimestamp,
{null_cols}
        CAST(trade.price    AS DOUBLE) AS price,
        CAST(trade.quantity AS DOUBLE) AS quantity,
        CAST(NULL AS DOUBLE)           AS previousquantity,
        trade.side,
        at_timezone(
            from_unixtime_nanos(CAST(trade.exchangetimestamp AS BIGINT)),
            'America/New_York'
        ) AS exchange_dt
    FROM {CONFIG['tables']['trade']} trade
    JOIN config      ON trade.f = config.f AND trade.dt = config.dt
    JOIN front_month ON UPPER(trade.product) = UPPER(front_month.product)
    WHERE at_timezone(
            from_unixtime_nanos(CAST(trade.exchangetimestamp AS BIGINT)),
            'America/New_York'
          ) BETWEEN config.start_time AND config.end_time
      AND UPPER(trade.printable) = 'PRINTABLE'
),
with_nbbo AS (
    SELECT
        type,
        exchangetimestamp,
        exchange_dt,
{last_val_cols}
        price,
        quantity,
        previousquantity,
        side
    FROM all_events
    WINDOW w AS (
        ORDER BY exchangetimestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
filtered_events AS (
    SELECT
        type,
        exchange_dt,
        exchangetimestamp,
{level_passthru}
        price,
        quantity,
        previousquantity,
        side,
        (askprice_1 + bidprice_1) / 2.0                             AS mid_price,
        (askprice_1 - bidprice_1)
            / NULLIF((askprice_1 + bidprice_1) / 2.0, 0)            AS spread,
        (bidquantity_1 - askquantity_1)
            / NULLIF(bidquantity_1 + askquantity_1, 0)              AS order_imbalance,
        CASE
            WHEN quantity IS NOT NULL AND previousquantity IS NOT NULL
                THEN quantity - previousquantity
            WHEN quantity IS NOT NULL         THEN  quantity
            WHEN previousquantity IS NOT NULL THEN -previousquantity
            ELSE 0
        END AS quantity_change,
{liq_action}    AS liquidity_action
    FROM with_nbbo
    WHERE type != 'NBBO'
      AND bidprice_1    IS NOT NULL
      AND askprice_1    IS NOT NULL
      AND bidquantity_1 IS NOT NULL
      AND askquantity_1 IS NOT NULL
),
events_with_auc AS (
    SELECT
        *,
{events_auc_cols}
    FROM filtered_events
),
time_buckets AS (
    SELECT
        {bucket_expr} AS time_bucket_utc,
        type,
        exchangetimestamp,
        spread,
        order_imbalance,
        price,
        quantity,
        previousquantity,
        liquidity_action,
        quantity_change,
{time_buckets_auc}
    FROM events_with_auc
    CROSS JOIN config
    WHERE exchange_dt BETWEEN config.start_time AND config.end_time
),
aggregated AS (
    SELECT
        time_bucket_utc,
        MAX_BY(spread,          exchangetimestamp)                   AS spread,
        MAX_BY(order_imbalance, exchangetimestamp)                   AS order_imbalance,
        MAX_BY(price,
            CASE WHEN type = 'TRADE' THEN exchangetimestamp
                 ELSE NULL END)                                      AS last_trade_price,
        COUNT(*)                                                     AS liquidity_actions,
        SUM(CASE WHEN type = 'TRADE'  THEN 1 ELSE 0 END)            AS trade_count,
        SUM(CASE WHEN type = 'CANCEL' THEN 1 ELSE 0 END)            AS cancel_count,
        SUM(CASE WHEN type = 'MODIFY' THEN 1 ELSE 0 END)            AS modify_count,
        SUM(CASE WHEN type = 'ADD'    THEN 1 ELSE 0 END)            AS add_count,
        SUM(CASE WHEN liquidity_action = 'LD'    THEN 1 ELSE 0 END) AS LD,
        SUM(CASE WHEN liquidity_action = 'LS'    THEN 1 ELSE 0 END) AS LS,
        SUM(CASE WHEN liquidity_action = 'LW'    THEN 1 ELSE 0 END) AS LW,
        SUM(CASE WHEN liquidity_action = 'LN'    THEN 1 ELSE 0 END) AS LN,
        SUM(CASE WHEN liquidity_action = 'Other' THEN 1 ELSE 0 END) AS Other,
        SUM(CASE WHEN type = 'TRADE'
            THEN COALESCE(quantity,         0) ELSE 0 END)           AS trade_volume,
        SUM(CASE WHEN type = 'CANCEL'
            THEN COALESCE(previousquantity, 0) ELSE 0 END)           AS cancel_volume,
        SUM(CASE WHEN type = 'ADD'
            THEN COALESCE(quantity,         0) ELSE 0 END)           AS add_volume,
        SUM(CASE WHEN type = 'MODIFY'
            THEN ABS(COALESCE(quantity_change, 0)) ELSE 0 END)       AS modify_volume_abs,
        SUM(CASE WHEN type = 'MODIFY'
            THEN COALESCE(quantity_change,  0) ELSE 0 END)           AS modify_volume_net,
        {aggregated_auc}
    FROM time_buckets
    GROUP BY time_bucket_utc
),
filled AS (
    SELECT
        time_bucket_utc,
        COALESCE(spread,
            LAST_VALUE(spread) IGNORE NULLS OVER w)                  AS spread,
        COALESCE(order_imbalance,
            LAST_VALUE(order_imbalance) IGNORE NULLS OVER w)         AS order_imbalance,
        COALESCE(last_trade_price,
            LAST_VALUE(last_trade_price) IGNORE NULLS OVER w)        AS last_trade_price,
        liquidity_actions,
        trade_count, cancel_count, modify_count, add_count,
        LD, LS, LW, LN, Other,
        trade_volume, cancel_volume, add_volume,
        modify_volume_abs, modify_volume_net,
        {filled_auc}
    FROM aggregated
    WINDOW w AS (
        ORDER BY time_bucket_utc
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
)
SELECT
    to_iso8601(time_bucket_utc) AS time_bucket,
    spread,
    order_imbalance,
    last_trade_price,
    liquidity_actions,
    trade_count, cancel_count, modify_count, add_count,
    LD, LS, LW, LN, Other,
    trade_volume, cancel_volume, add_volume,
    modify_volume_abs, modify_volume_net,
    {final_auc}
FROM filled
ORDER BY time_bucket_utc
"""


# ════════════════════════════════════════════════════════════════════════════
# Public query builders
# ════════════════════════════════════════════════════════════════════════════

def create_aggregated_spy_query(
    date: str,
    interval: str = '1s',
    lambda_decay: float = None,
    n_levels: int = CONFIG['n_levels'],
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    auc_method: str = CONFIG['auc_method'],
) -> str:
    if not validate_date(date):
        raise ValueError(f"Invalid date: {date}")
    interval_seconds = interval_to_seconds(interval)

    if start_ts is None:
        start_ts = f"{date} 04:00:00"
    if end_ts is None:
        end_ts = f"{date} 20:00:00"

    profile   = get_instrument_profile('SPY')
    feeds_str = "'" + "', '".join(CONFIG['spy_depth_feeds']) + "'"

    config_extra = f",\n        ARRAY[{feeds_str}] AS f"

    nbbo_filter = (
        f"UPPER(nbbo.product) = '{CONFIG['spy_product']}'\n"
        f"          AND nbbo.dt = config.dt\n"
        f"          AND contains(config.f, nbbo.f)"
    )
    of_filter = (
        f"UPPER({{alias}}.product) = '{CONFIG['spy_product']}'\n"
        f"          AND {{alias}}.dt = config.dt\n"
        f"          AND contains(config.f, {{alias}}.f)"
    )

    logger.info(
        f"SPY query | λ_dollar={profile['lambda_dollar']} "
        f"λ_pct={profile['lambda_pct']} λ_ticks={profile['lambda_ticks']} "
        f"tick={profile['tick_size']} method={auc_method}"
    )

    sql = _build_depth_query_windowed(
        date=date,
        start_ts=start_ts,
        end_ts=end_ts,
        interval_seconds=interval_seconds,
        profile=profile,
        auc_method=auc_method,
        n_levels=n_levels,
        nbbo_table=CONFIG['tables']['nbbo'],
        nbbo_timestamp_col='firstexchangetimestamp',
        nbbo_filter_sql=nbbo_filter,
        order_flow_filter_sql=of_filter,
        config_extra_cols=config_extra,
    )
    debug_sql(sql, context=f'SPY {date} {start_ts}-{end_ts}')
    return sql


def create_aggregated_es_query(
    date: str,
    interval: str = '1s',
    lambda_decay: float = None,
    n_levels: int = CONFIG['n_levels'],
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    auc_method: str = CONFIG['auc_method'],
) -> str:
    if not validate_date(date):
        raise ValueError(f"Invalid date: {date}")
    interval_seconds = interval_to_seconds(interval)

    if start_ts is None:
        start_ts = f"{date} 04:00:00"
    if end_ts is None:
        end_ts = f"{date} 20:00:00"

    profile = get_instrument_profile('ES')

    logger.info(
        f"ES query | λ_dollar={profile['lambda_dollar']} "
        f"λ_pct={profile['lambda_pct']} λ_ticks={profile['lambda_ticks']} "
        f"tick={profile['tick_size']} method={auc_method}"
    )

    sql = _build_es_query_windowed(
        date=date,
        start_ts=start_ts,
        end_ts=end_ts,
        interval_seconds=interval_seconds,
        profile=profile,
        auc_method=auc_method,
        n_levels=n_levels,
    )
    debug_sql(sql, context=f'ES {date} {start_ts}-{end_ts}')
    return sql


# ════════════════════════════════════════════════════════════════════════════
# Time-window helpers
# ════════════════════════════════════════════════════════════════════════════

def _generate_chunks(
    date: str,
    chunk_hours: float,
    day_start: str = "04:00:00",
    day_end: str   = "20:00:00",
) -> List[Tuple[str, str]]:
    fmt   = "%Y-%m-%d %H:%M:%S"
    start = datetime.strptime(f"{date} {day_start}", fmt)
    end   = datetime.strptime(f"{date} {day_end}",   fmt)
    delta = timedelta(hours=chunk_hours)
    chunks, cur = [], start
    while cur < end:
        nxt = min(cur + delta, end)
        chunks.append((cur.strftime(fmt), nxt.strftime(fmt)))
        cur = nxt
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# Core fetch with adaptive chunking
# ════════════════════════════════════════════════════════════════════════════

_BASE_EXPECTED_COLUMNS = [
    'spread', 'order_imbalance', 'last_trade_price',
    'liquidity_actions',
    'trade_count', 'cancel_count', 'modify_count', 'add_count',
    'LD', 'LS', 'LW', 'LN', 'Other',
    'trade_volume', 'cancel_volume', 'add_volume',
    'modify_volume_abs', 'modify_volume_net',
]


def _execute_single_chunk(
    date: str,
    product: str,
    query_func: Callable,
    interval: str,
    n_levels: int,
    start_ts: str,
    end_ts: str,
    auc_method: str,
) -> pd.DataFrame:
    sql = query_func(
        date, interval,
        n_levels=n_levels,
        start_ts=start_ts,
        end_ts=end_ts,
        auc_method=auc_method,
    )
    return maystreet_data.query(
        maystreet_data.DataSource.DATA_LAKE,
        sql,
        return_type='df',
    )


def _parse_bucket_utc(s: "pd.Series") -> "pd.Series":
    """Parse the time_bucket column to a UTC tz-aware Series.

    The query emits ``to_iso8601(time_bucket_utc)`` — ISO-8601 with an explicit
    offset, e.g. ``2024-12-17T23:00:00.000-05:00`` — which parses directly. As a
    backstop this also handles the legacy Trino rendering of a ``timestamp with
    time zone`` as an IANA zone NAME, e.g. ``2024-12-17 23:00:00.000
    America/New_York``, which pandas/dateutil cannot parse on its own.
    """
    s = s.astype("string")
    try:
        out = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):                       # pandas < 2.0: no ISO8601 token
        out = pd.to_datetime(s, utc=True, errors="coerce")
    if out.notna().all():
        return out
    # Legacy fallback: strip a trailing ' Region/City' zone name, localize, convert.
    zone = s.str.extract(r"\s([A-Za-z_]+/[A-Za-z_]+)\s*$", expand=False)
    z = zone.dropna().iloc[0] if zone.notna().any() else "UTC"
    naive = s.str.replace(r"\s+[A-Za-z_]+/[A-Za-z_]+\s*$", "", regex=True)
    local = pd.to_datetime(naive, errors="coerce").dt.tz_localize(
        z, ambiguous=True, nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def fetch_aggregated_product(
    date: str,
    product: str,
    query_func: Callable,
    interval: str = '1s',
    n_levels: int = CONFIG['n_levels'],
    chunk_hours: float = CONFIG['chunk_hours'],
    auc_method: str = CONFIG['auc_method'],
) -> pd.DataFrame:
    profile = get_instrument_profile(product)
    logger.info(
        f"Fetching {product} | {date} | interval={interval} "
        f"| λ_pct={profile['lambda_pct']} (cross-asset) "
        f"| λ_dollar={profile['lambda_dollar']} "
        f"| λ_ticks={profile['lambda_ticks']} "
        f"| tick={profile['tick_size']} "
        f"| levels={n_levels} | auc_method={auc_method} "
        f"| initial_chunk={chunk_hours}h"
    )
    t0_total = time.time()

    min_chunk_minutes = CONFIG['min_chunk_minutes']
    current_chunk     = chunk_hours
    pieces: List[pd.DataFrame] = []
    chunks = _generate_chunks(date, current_chunk)
    idx    = 0

    while idx < len(chunks):
        start_ts, end_ts = chunks[idx]
        t0 = time.time()
        logger.info(
            f"  {product} chunk {idx+1}/{len(chunks)} "
            f"[{start_ts[11:]} - {end_ts[11:]}] "
            f"(chunk={current_chunk}h)"
        )

        try:
            df_chunk = _execute_single_chunk(
                date, product, query_func,
                interval, n_levels,
                start_ts, end_ts,
                auc_method=auc_method,
            )
            elapsed = time.time() - t0
            logger.info(
                f"  {product} chunk {idx+1} done in {elapsed:.1f}s "
                f"- {len(df_chunk):,} rows"
            )
            if not df_chunk.empty:
                pieces.append(df_chunk)
            idx += 1

        except Exception as exc:
            err_str = str(exc)
            elapsed = time.time() - t0

            if "exhausted resources" not in err_str:
                logger.error(
                    f"  {product} chunk {idx+1} non-resource error "
                    f"after {elapsed:.1f}s: {exc}"
                )
                raise

            new_chunk_minutes = current_chunk * 60 / 2
            if new_chunk_minutes < min_chunk_minutes:
                logger.error(
                    f"  {product} chunk {idx+1} exhausted resources even at "
                    f"{current_chunk*60:.0f}-minute chunks — giving up on "
                    f"[{start_ts[11:]} - {end_ts[11:]}]."
                )
                idx += 1
                continue

            current_chunk = new_chunk_minutes / 60
            logger.warning(
                f"  {product} chunk {idx+1} exhausted resources after "
                f"{elapsed:.1f}s — halving to {new_chunk_minutes:.0f}-min "
                f"chunks and rebuilding schedule."
            )
            time.sleep(CONFIG['retry_delay_s'])

            remaining = _generate_chunks(
                date, current_chunk,
                day_start=start_ts[11:],
                day_end="20:00:00",
            )
            chunks = chunks[:idx] + remaining
            logger.info(
                f"  Rebuilt schedule: {len(remaining)} new chunks "
                f"from {start_ts[11:]} at {current_chunk*60:.0f}-min granularity"
            )

    if not pieces:
        logger.warning(f"No data for {product} on {date}")
        expected_cols = (
            _BASE_EXPECTED_COLUMNS
            + _auc_col_names(auc_method)
            + _book_col_names(n_levels)
        )
        empty = pd.DataFrame(
            columns=[f'{product}_{c}' for c in expected_cols]
        )
        empty.index.name = 'time_bucket'
        return empty

    df = pd.concat(pieces, ignore_index=True)

    if 'time_bucket' in df.columns:
        df = df.drop_duplicates(subset='time_bucket', keep='last')

    df['time_bucket'] = (
        _parse_bucket_utc(df['time_bucket'])
          .dt.tz_convert('America/New_York')
    )
    df.set_index('time_bucket', inplace=True)
    df.sort_index(inplace=True)
    df.columns = [f'{product}_{c}' for c in df.columns]

    logger.info(
        f"{product} | {date} complete in "
        f"{time.time() - t0_total:.1f}s — shape {df.shape}"
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════

def process_date(
    date: str,
    interval: str = '1s',
    n_levels: int = CONFIG['n_levels'],
    auc_method: str = CONFIG['auc_method'],
) -> Tuple[str, pd.DataFrame]:
    t0 = time.time()
    logger.info(
        f"Processing {date} | interval={interval} | auc_method={auc_method}"
    )

    spy_df = fetch_aggregated_product(
        date, 'SPY', create_aggregated_spy_query,
        interval, n_levels, auc_method=auc_method,
    )
    es_df = fetch_aggregated_product(
        date, 'ES', create_aggregated_es_query,
        interval, n_levels, auc_method=auc_method,
    )

    combined = spy_df.join(es_df, how='outer')
    logger.info(
        f"Finished {date} in {time.time() - t0:.1f}s "
        f"— combined shape {combined.shape}"
    )
    return date, combined


# ════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════

def validate_auc_output(
    df: pd.DataFrame,
    product_prefix: str,
    auc_method: str = CONFIG['auc_method'],
) -> None:
    methods = ['dollar', 'pct', 'ticks'] if auc_method == 'all' else [auc_method]
    issues: List[str] = []

    for m in methods:
        bid_col = f'{product_prefix}_bid_auc_{m}'
        ask_col = f'{product_prefix}_ask_auc_{m}'
        imb_col = f'{product_prefix}_auc_imbalance_{m}'
        label   = (
            f'[{m.upper()}'
            f'{"  <- cross-asset primary" if m == "pct" else ""}]'
        )

        for col in (bid_col, ask_col):
            if col not in df.columns:
                issues.append(f"{label} Missing column: {col}")
                continue
            neg  = (df[col] < 0).sum()
            nans = df[col].isna().sum()
            if neg:
                issues.append(f"{label} {col}: {neg:,} negative values")
            if nans:
                issues.append(f"{label} {col}: {nans:,} NaNs after forward-fill")

        if imb_col in df.columns:
            oor = ((df[imb_col] < -1) | (df[imb_col] > 1)).sum()
            if oor:
                issues.append(
                    f"{label} {imb_col}: {oor:,} values outside [-1, 1]"
                )

    if issues:
        for msg in issues:
            logger.warning(f"[AUC validation] {product_prefix}: {msg}")
    else:
        logger.info(
            f"[AUC validation] {product_prefix}: all checks passed "
            f"(methods: {methods})"
        )

    auc_cols = [
        c for c in df.columns
        if 'auc' in c.lower() and c.startswith(product_prefix)
    ]
    if auc_cols:
        logger.info(
            f"\nAUC statistics ({product_prefix}):\n"
            f"{df[auc_cols].describe().to_string()}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(
    interval: str = '1s',
    dates: List[str] = None,
    n_levels: int = CONFIG['n_levels'],
    auc_method: str = CONFIG['auc_method'],
    max_workers: int = 4,
) -> List[Tuple[str, pd.DataFrame]]:
    logger.info(
        f"Starting collection | interval={interval} "
        f"| levels={n_levels} | auc_method={auc_method} "
        f"| max_workers={max_workers}\n"
        f"  SPY profile: {get_instrument_profile('SPY')}\n"
        f"  ES  profile: {get_instrument_profile('ES')}\n"
        f"  NOTE: Use *_pct AUC columns for SPY vs ES comparison "
        f"(lambda_pct=500 is cross-asset normalised)"
    )

    if dates is None:
        dates = [
            '2025-04-03',
            '2024-12-18',
            '2024-10-31',
            '2024-09-03',
            '2024-01-31',
            '2024-08-06',
            '2025-04-07',
            '2025-04-14',
            '2025-04-17',
            '2025-04-22',
            '2025-04-23',
            '2025-04-24',
        ]

    valid_dates = [d for d in dates if validate_date(d)]
    skipped = len(dates) - len(valid_dates)
    if skipped:
        logger.warning(f"Skipped {skipped} invalid/future date(s).")

    products = [
        ('SPY', create_aggregated_spy_query),
        ('ES',  create_aggregated_es_query),
    ]

    # Fan out one Athena job per (date, product); max_workers caps the number of
    # concurrent queries. The work is I/O-bound on Athena, so threads (which
    # release the GIL during the wait) are the right tool rather than processes.
    done: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
    with cf.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix='athena'
    ) as ex:
        futs = {
            ex.submit(
                fetch_aggregated_product, d, p, builder,
                interval, n_levels, auc_method=auc_method,
            ): (d, p)
            for d in valid_dates
            for p, builder in products
        }
        for f in cf.as_completed(futs):
            d, p = futs[f]
            try:
                done[(d, p)] = f.result()
            except Exception:
                logger.error(f"Failed {p} {d}", exc_info=True)
                done[(d, p)] = None

    # Reassemble in deterministic date order; join SPY + ES per date
    # (mirrors process_date's spy_df.join(es_df, how='outer')).
    results: List[Tuple[str, pd.DataFrame]] = []
    for d in valid_dates:
        spy_df, es_df = done.get((d, 'SPY')), done.get((d, 'ES'))
        if spy_df is None or es_df is None:
            logger.error(f"Skipping {d}: missing product result")
            continue
        results.append((d, spy_df.join(es_df, how='outer')))

    for date, df in results:
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary | {date} | {interval}")
        logger.info(f"  Buckets : {len(df):,}")
        if not df.empty:
            logger.info(f"  Range   : {df.index.min()} -> {df.index.max()}")
            logger.info(
                f"  Memory  : "
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )
            validate_auc_output(df, 'SPY', auc_method)
            validate_auc_output(df, 'ES',  auc_method)

    return results


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _configure_logging(_timestamp)

    AGGREGATION_INTERVAL = '1s'
    N_LEVELS             = CONFIG['n_levels']
    AUC_METHOD           = CONFIG['auc_method']
    MAX_WORKERS          = 4          # concurrent Athena queries cap

    logger.info(
        f"Run parameters: interval={AGGREGATION_INTERVAL} "
        f"| levels={N_LEVELS} | auc_method={AUC_METHOD}"
    )
    logger.info(
        "Lambda guide:\n"
        "  SPY  lambda_dollar=10.0  lambda_pct=500  lambda_ticks=1.0  tick=$0.01\n"
        "  ES   lambda_dollar=0.4   lambda_pct=500  lambda_ticks=1.0  tick=$0.25\n"
        "  BTC  lambda_dollar=0.001 lambda_pct=500  lambda_ticks=1.0  tick=$0.01\n"
        "  Use *_pct columns for ALL cross-asset comparisons."
    )

    try:
        results = main(
            interval=AGGREGATION_INTERVAL,
            n_levels=N_LEVELS,
            auc_method=AUC_METHOD,
            max_workers=MAX_WORKERS,
        )

        if not results:
            logger.error("No data returned — exiting.")
            sys.exit(1)

        logger.info(f"Successfully processed {len(results)} date(s); frames returned in memory.")
        logger.info(
            "This script writes no files by design. Run "
            "`python run_analysis.py --source extract` to produce the consolidated "
            "final dataset and the result tables in ./output/, or import main() "
            "and persist the returned List[(date, DataFrame)] however you prefer."
        )

    except Exception:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)


