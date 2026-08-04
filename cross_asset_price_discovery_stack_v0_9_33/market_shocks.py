"""
market_shocks.py
===============
Halts are one face of a single theme -- contagion, liquidity withdrawal, and price-discovery
breakdown during *material-consequence* events. So the registry is not limited to mechanical
trading halts: it spans market-structure events (MWCB / mass-LULD / SIP & exchange outages /
flash crashes), monetary shocks (surprise FOMC moves, emergency cuts), sovereign-credit
shocks (US rating downgrades), trade-policy shocks (tariff announcements / FX manipulation),
geopolitical shocks (wars, strikes, attacks), and pure volatility-regime breaks. Each is an
*exogenous* event whose first-reaction session is a candidate treatment day.

Two things the hand-picked "volatile day vs ln(sigma) fell" selection got wrong, and what
this module provides instead:

  1. A curated, ANNOTATED registry of real shocks (the exogenous treatment universe), tagged
     by `event_class` (coarse) and `category` (fine), with a `timing` field and an `event_et`
     identifying intraday timestamp.
  2. A principled treatment/control assignment. Treatment = first-reaction sessions of
     registry events. Control = NON-event days matched to each treatment day on an EX-ANTE
     stress level (prior-close VIX / a vol forecast / the long-run vol level), excluded if
     within a buffer of any event. This replaces selecting controls on a vol OUTCOME --
     e.g. ln(sigma_t / sigma_{t-1}) < 0 -- which conditions on (a relative of) the dependent
     variable, mechanically guarantees a volatility gap, and confounds the event with the
     vol level / mean reversion.

IDENTIFICATION DIFFERS BY `timing`, which is why it is a first-class field:
  * halt_reopen   -> regression discontinuity in time at the reopen boundary (`event_et`).
  * scheduled     -> intraday event study around the known release (FOMC 14:00 ET).
  * intraday_news -> intraday event study around the (unscheduled) release time.
  * overnight / weekend -> open-to-open gap on the first reaction session (no clean intraday
                    boundary; `event_et` is blank and `release_times` skips the row).
  * multi_day     -> an episode; `date` is the principal session, span noted in `annotation`.

DATING CONVENTION: `date` is the first TRADING session that prices the news (the event day
itself if it hit intraday; the next session if it broke after the close or on a weekend), so
every row is matchable to a daily series. The literal event time/date is in the annotation.

DATES/TIMES ARE APPROXIMATE AND MUST BE VERIFIED against primary sources before publication
(SEC MWCB and Reg SCI notices, exchange/FINRA halt records, FOMC statements, rating-agency
releases). The current MWCB framework (adopted 2013, replacing DJIA-point thresholds):
Level 1 = S&P 500 -7%, Level 2 = -13% (each a 15-min halt before 15:25 ET, else trade on),
Level 3 = -20% (halt for the day). Pure numpy / pandas.
"""
import numpy as np
import pandas as pd
import datetime as _dt

EPS = 1e-12

# ── the registry ──────────────────────────────────────────────────────────────
# date    : first reaction TRADING session (see dating convention above)
# category: fine-grained type        event_class: coarse bucket
# timing  : identification regime     event_et: identifying intraday time ('' if overnight)
# spx_pct : approx S&P 500 move (%) on `date`  spy_es: plausible SPY/ES halt / ETF-arb channel
_SHOCKS = [
    dict(date="1997-10-27", category="MWCB", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="closed early", halt_level="DJIA-point (old rules)", spx_pct=-6.9, spy_es=True,
         annotation="Only pre-2013 circuit-breaker trigger; 350-pt halt 14:35, 550-pt halt 15:30 closed the day.",
         source="NYSE/SEC 1997 circuit-breaker notices"),
    dict(date="2010-05-06", category="FLASH_CRASH", event_class="MARKET_STRUCTURE", timing="intraday_news",
         event_et="14:45", halt_level="none (pre-LULD)", spx_pct=-3.2, spy_es=True,
         annotation="Flash Crash: S&P 500 ~-9% intraday ~14:32 then recovered; no MWCB. Prompted LULD / single-stock CBs.",
         source="CFTC-SEC 2010 Flash Crash report"),
    dict(date="2011-08-08", category="SOVEREIGN_DOWNGRADE", event_class="SOVEREIGN_CREDIT", timing="overnight",
         event_et="", halt_level="none", spx_pct=-6.7, spy_es=True,
         annotation="'Black Monday': S&P cut US AAA->AA+ after close Fri 08-05; first reaction session 08-08 (-6.66%).",
         source="S&P Global Ratings 2011-08-05 release"),
    dict(date="2013-08-22", category="SIP_OUTAGE", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="15:25", halt_level="Tape C halt", spx_pct=np.nan, spy_es=False,
         annotation="Nasdaq 'flash freeze': UTP SIP quote-dissemination failure halted Tape C 12:14-15:25 ET; broad market ~flat.",
         source="Nasdaq OMX 2013-08-29 statement; SEC"),
    dict(date="2015-07-08", category="EXCHANGE_OUTAGE", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="15:10", halt_level="NYSE venue halt", spx_pct=-1.7, spy_es=False,
         annotation="NYSE technical outage 11:32-15:10 ET; NYSE-listed names kept trading on other venues (limited SPY/ES halt).",
         source="NYSE notices; SEC Reg SCI"),
    dict(date="2015-08-24", category="LULD_MASS", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="09:30", halt_level="~1,278 LULD pauses", spx_pct=-3.9, spy_es=True,
         annotation="August 2015 open: mass LULD pauses and severe ETF NAV dislocations (ETF-arb breakdown). Highly relevant to SPY/ES.",
         source="SEC 2015 ETF/Equity Market Volatility research note"),
    dict(date="2018-02-05", category="VOL_SPIKE", event_class="VOL_REGIME", timing="intraday_news",
         event_et="15:08", halt_level="none", spx_pct=-4.1, spy_es=True,
         annotation="'Volmageddon': late-day VIX +115% (~37); XIV/inverse-VIX collapse and termination. No MWCB.",
         source="press; product termination notices"),
    dict(date="2019-08-05", category="FX_TRADE", event_class="TRADE_POLICY", timing="overnight",
         event_et="", halt_level="none", spx_pct=-3.0, spy_es=True,
         annotation="China let yuan break 7/USD; US named China a currency manipulator that evening. S&P -2.98%.",
         source="press; US Treasury 2019-08-05 statement"),
    dict(date="2020-03-03", category="FED_EMERGENCY", event_class="MONETARY", timing="intraday_news",
         event_et="10:00", halt_level="none", spx_pct=-2.8, spy_es=True,
         annotation="Surprise intermeeting -50bp cut ~10:00 ET; equities fell anyway (cut read as alarm). COVID onset.",
         source="FOMC 2020-03-03 statement"),
    dict(date="2020-03-09", category="MWCB", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="09:49", halt_level="Level 1 (-7%)", spx_pct=-7.6, spy_es=True,
         annotation="MWCB Level 1 ~09:34 ET (oil price war + COVID). One of four March-2020 halts (paper Section 6).",
         source="NYSE/Cboe MWCB notices"),
    dict(date="2020-03-12", category="MWCB", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="09:50", halt_level="Level 1 (-7%)", spx_pct=-9.5, spy_es=True,
         annotation="MWCB Level 1 near the open. March-2020 halt 2 of 4.",
         source="NYSE/Cboe MWCB notices"),
    dict(date="2020-03-15", category="FED_EMERGENCY", event_class="MONETARY", timing="weekend",
         event_et="", halt_level="none", spx_pct=np.nan, spy_es=True,
         annotation="Sunday emergency cut to zero + $700bn QE; compounded the 03-16 MWCB open (which is also a registry row).",
         source="FOMC 2020-03-15 statement"),
    dict(date="2020-03-16", category="MWCB", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="09:45", halt_level="Level 1 (-7%)", spx_pct=-12.0, spy_es=True,
         annotation="MWCB Level 1 at the open; largest single-day drop of the episode. March-2020 halt 3 of 4.",
         source="NYSE/Cboe MWCB notices"),
    dict(date="2020-03-18", category="MWCB", event_class="MARKET_STRUCTURE", timing="halt_reopen",
         event_et="13:11", halt_level="Level 1 (-7%)", spx_pct=-5.2, spy_es=True,
         annotation="MWCB Level 1 ~12:56 ET (a midday trigger, unlike the others). March-2020 halt 4 of 4.",
         source="NYSE/Cboe MWCB notices"),
    dict(date="2022-02-24", category="GEOPOLITICAL", event_class="GEOPOLITICAL", timing="overnight",
         event_et="", halt_level="none", spx_pct=+1.5, spy_es=True,
         annotation="Russia's full-scale invasion of Ukraine (pre-dawn); S&P gapped ~-2.6% then reversed to close +1.50%.",
         source="press"),
    dict(date="2022-06-15", category="FOMC", event_class="MONETARY", timing="scheduled",
         event_et="14:00", halt_level="none", spx_pct=+1.5, spy_es=True,
         annotation="First +75bp hike since 1994 (scheduled 14:00 ET); S&P +1.46% on the day, reversed next session.",
         source="FOMC 2022-06-15 statement"),
    dict(date="2023-08-02", category="SOVEREIGN_DOWNGRADE", event_class="SOVEREIGN_CREDIT", timing="overnight",
         event_et="", halt_level="none", spx_pct=-1.4, spy_es=True,
         annotation="Fitch cut US AAA->AA+ after close 08-01; first reaction session 08-02 (-1.38%).",
         source="Fitch Ratings 2023-08-01 release"),
    dict(date="2023-10-09", category="GEOPOLITICAL", event_class="GEOPOLITICAL", timing="weekend",
         event_et="", halt_level="none", spx_pct=+0.6, spy_es=True,
         annotation="Hamas attack on Israel 10-07 (Sat); first trading session 10-09 (+0.63%, oil up). Bond market closed (holiday).",
         source="press"),
    dict(date="2024-04-15", category="GEOPOLITICAL", event_class="GEOPOLITICAL", timing="weekend",
         event_et="", halt_level="none", spx_pct=-1.2, spy_es=True,
         annotation="Iran's first direct drone/missile attack on Israel 04-13 (Sat night); reaction session 04-15 (-1.20%).",
         source="press"),
    dict(date="2024-08-05", category="VOL_SPIKE", event_class="VOL_REGIME", timing="overnight",
         event_et="", halt_level="none", spx_pct=-3.0, spy_es=True,
         annotation="Yen carry-trade unwind; VIX spiked intraday to ~65 (one of the highest ever); Nikkei -12%. Gapped down at the open. No MWCB.",
         source="press; Cboe VIX data"),
    dict(date="2024-09-18", category="FOMC", event_class="MONETARY", timing="scheduled",
         event_et="14:00", halt_level="none", spx_pct=-0.3, spy_es=True,
         annotation="-50bp cut starting the easing cycle (scheduled 14:00 ET); choppy, S&P ~-0.3% on the day.",
         source="FOMC 2024-09-18 statement"),
    dict(date="2025-04-03", category="TARIFF", event_class="TRADE_POLICY", timing="overnight",
         event_et="", halt_level="none", spx_pct=-4.8, spy_es=True,
         annotation="'Liberation Day' reciprocal tariffs (EO 14257) announced ~16:00 ET 04-02; 04-03 -4.84%, 04-04 -5.97% (~-11% over the week).",
         source="EO 14257; press"),
    dict(date="2025-04-09", category="TARIFF", event_class="TRADE_POLICY", timing="intraday_news",
         event_et="13:18", halt_level="none", spx_pct=+9.5, spy_es=True,
         annotation="90-day reciprocal-tariff pause announced intraday ~13:18 ET; S&P +9.52% -- one of the largest one-day gains on record.",
         source="White House; press"),
    dict(date="2025-05-19", category="SOVEREIGN_DOWNGRADE", event_class="SOVEREIGN_CREDIT", timing="overnight",
         event_et="", halt_level="none", spx_pct=np.nan, spy_es=True,
         annotation="Moody's cut US Aaa->Aa1 after close Fri 05-16 (last AAA agency to do so); reaction 05-19, equities ~flat, Treasury yields rose.",
         source="Moody's Ratings 2025-05-16 release"),
    dict(date="2025-06-13", category="GEOPOLITICAL", event_class="GEOPOLITICAL", timing="overnight",
         event_et="", halt_level="none", spx_pct=-1.1, spy_es=True,
         annotation="Israel's 'Operation Rising Lion' strikes on Iran (~03:00 local 06-13); priced at the US open 06-13, oil spiked, equities ~-1.1%.",
         source="press"),
    dict(date="2025-06-23", category="GEOPOLITICAL", event_class="GEOPOLITICAL", timing="weekend",
         event_et="", halt_level="none", spx_pct=+0.9, spy_es=True,
         annotation="US strikes on Fordow/Natanz/Isfahan ('Operation Midnight Hammer') 06-21/22 (weekend); reaction 06-23 -- oil spike faded (Hormuz open), equities up.",
         source="press"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULED-EVENT CALENDAR -- the impact-blind backbone of the multi-event design.
# ═══════════════════════════════════════════════════════════════════════════════
# Scheduled macro/policy releases have EXOGENOUS, known-ex-ante timestamps, so they
# identify the stress-response function f(state) impact-blind: take EVERY occurrence
# over a wide, regime-dispersed span (NOT a contiguous window, NOT a volatility
# filter). The ones that happen to land during turbulent stretches are how the
# backbone reaches HIGH pre-state without ever selecting on the outcome.
#
# Release dates are DATA, not a rule: holidays shift them, and the 2025 federal
# shutdown cancelled/rescheduled several BLS releases (the Oct-2025 Employment
# Situation was never published; Sep-2025 CPI moved to 2025-10-24). So FOMC and the
# BLS series are stored as VERIFIED tables; only the exchange-calendar OPEX dates are
# generated (third-Friday rule). VERIFY/EXTEND against the primary sources before use:
#   FOMC : federalreserve.gov FOMC schedule press releases. Two-day meeting; policy
#          statement at 14:00 ET on day 2 (the rate-decision day stored here).
#   CPI / Employment Situation : bls.gov/schedule/news_release/{cpi,empsit}.htm, 08:30 ET.
#          2025 shutdown revisions: bls.gov/bls/2025-lapse-revised-release-dates.htm
#   OPEX : quad-witching = third Friday of Mar/Jun/Sep/Dec (CME/Cboe/OCC), settlement
#          flows at the open (ES SOQ, 09:30 ET) and the close.

# FOMC rate-decision days (day 2 of each meeting), 14:00 ET. Source: Fed press releases
# (monetary20220624a 2023; monetary20230623a 2024; monetary20240809a 2025-26). VERIFIED.
FOMC_DECISION_DAYS = {
    2023: ["02-01", "03-22", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
    2024: ["01-31", "03-20", "05-01", "06-12", "07-31", "09-18", "11-07", "12-18"],
    2025: ["01-29", "03-19", "05-07", "06-18", "07-30", "09-17", "10-29", "12-10"],
    2026: ["01-28", "03-18", "04-29", "06-17", "07-29", "09-16", "10-28", "12-09"],
}

# BLS releases at 08:30 ET. Dates are DATA (holiday/shutdown shifts) -> store verified,
# do NOT generate. SEEDED with verified 2025 shutdown-revised points only; COMPLETE the
# full 2023-2026 schedule from bls.gov (the dense body of the backbone depends on it).
# {category: {YYYY: ["MM-DD", ...]}}
BLS_RELEASE_DAYS = {
    "CPI": {
        2025: ["10-24",   # Sep-2025 CPI, shutdown-rescheduled. Source: BLS 092025 reschedule notice.
               "12-18"],  # Nov-2025 CPI, revised. Source: BLS 2025-lapse revised-release-dates.
    },
    "NFP": {
        2025: [],          # NOTE: Oct-2025 Employment Situation was NOT published (shutdown). Extend from BLS.
    },
}

# Loader-registered releases (any agency: BLS/BEA/Census/ISM/...), populated by
# load_release_schedule / load_ics. Empty by default -> the calendar is FOMC+OPEX+seeds only.
_LOADED_RELEASES = []


def _third_friday(year, month):
    """Third Friday of (year, month) -- the standard quad-witching expiration date."""
    d = _dt.date(year, month, 1)
    first_friday = d + _dt.timedelta(days=(4 - d.weekday()) % 7)   # Mon=0..Fri=4
    return first_friday + _dt.timedelta(days=14)


def _mk_scheduled(date_str, category, event_class, event_et, annotation, source,
                  selection="scheduled", timing="scheduled"):
    return dict(date=date_str, category=category, event_class=event_class, timing=timing,
                event_et=event_et, halt_level=None, spx_pct=np.nan, spy_es=True,
                annotation=annotation, source=source, selection=selection)


# Selection provenance -> whether an event is IMPACT-BLIND (its presence in the sample does not depend
# on the SPY/ES reaction, so it carries exogeneity for the benchmark) or SALIENCE-CURATED (it is in
# the sample BECAUSE it was notable -> usable only as excess over the impact-blind benchmark).
#   scheduled            : known-date release (FOMC/CPI/...) -- impact-blind.
#   scheduled_uncertainty: known date, unknown (possibly tail) outcome (election/referendum/OPEC/
#                          deadline/ruling) -- impact-blind; take EVERY one, like FOMC.
#   news_classified      : from a categorized news feed, selected on EX-ANTE news prominence (first-
#                          headline priority, NOT accumulated coverage, which is endogenous) -- impact-blind.
#   salience_curated     : hand-picked because notable/large (the _SHOCKS crisis registry) -- NOT
#                          impact-blind; identify within-event and read only as excess-over-benchmark.
IMPACT_BLIND_SELECTION = {"scheduled", "scheduled_uncertainty", "news_classified"}

# Scheduled-UNCERTAINTY seed (known-date, unknown-outcome). US general elections are GENERATED
# (1st Tue after 1st Mon in Nov, even years); other known-date events are SEEDED -> extend, or load a
# categorized feed with selection="scheduled_uncertainty". Dates are the reaction session. VERIFY.
_SCHED_UNCERTAINTY_SEED = [
    ("2016-06-24", "REFERENDUM", "GEOPOLITICAL",      "Brexit referendum result (vote 06-23 overnight -> 06-24 reaction).", "press"),
    ("2023-06-02", "DEADLINE",   "SOVEREIGN_CREDIT",  "US debt-ceiling resolution window, 2023 (Fiscal Responsibility Act).", "press; US Treasury"),
]


def _election_days(start_year, end_year):
    """US general election days: the Tuesday after the first Monday of November, even years."""
    out = []
    for y in range(start_year, end_year + 1):
        if y % 2 != 0:
            continue
        nov1 = _dt.date(y, 11, 1)
        first_monday = nov1 + _dt.timedelta(days=(0 - nov1.weekday()) % 7)   # Mon=0
        out.append(first_monday + _dt.timedelta(days=1))                     # the Tuesday after
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# NYSE TRADING CALENDAR + the rule-generated 10:00 release cluster
#
# Why a TRADING calendar rather than a federal-holiday one: the 10:00 releases are scheduled by
# BUSINESS-DAY rules, and for this design "business day" must mean "a day the tape exists". A federal
# calendar would place releases on Good Friday (market closed -> no session to extract, the event
# silently returns NaN) and would skip Columbus/Veterans Day (market OPEN -> a perfectly good session
# dropped). Anchoring the rule to the NYSE calendar guarantees every generated date has a session.
#
# Why the 10:00 cluster at all: extraction runs 09:30-16:00, so an onset needs enough RTH bars on BOTH
# sides. A 10:00 release leaves 1,800 pre-bars and the rest of the session after -- comfortably inside
# the default 600/600 windows. The 08:30 block (CPI, NFP, PPI, RETAIL, GDP, PCE, CLAIMS, DURABLE,
# HOUSING, TRADE) and OPEX at 09:30 fall on or before the first bar: pre-window empty, transmission
# NaN, event silently dropped. They are deliberately NOT generated here.
#
# PROVENANCE: these dates are RULE-GENERATED, not a verified table (contrast FOMC_DECISION_DAYS, which
# is transcribed from the Fed's schedule). The rules are the publishers' own stated conventions and are
# stable, but agencies do reschedule. Loading an authoritative schedule via load_release_schedule()
# OVERRIDES a generated date for the same (date, category), so treat these as a dense backbone to be
# corrected by ingest, and spot-check before the dates become load-bearing in a submission.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# Full-day NYSE closures with no calendar rule (weather / national days of mourning). Half-days are
# NOT listed: a 1pm early close still leaves a valid 10:00 onset with a full post-window.
_NYSE_ADHOC_CLOSURES = {
    "2012-10-29", "2012-10-30",        # Hurricane Sandy
    "2018-12-05",                      # G.H.W. Bush national day of mourning
    "2025-01-09",                      # Jimmy Carter national day of mourning
}


def _easter(year):
    """Gregorian Easter Sunday (anonymous algorithm). Good Friday = Easter - 2 is the only NYSE
    holiday with no fixed date and no federal counterpart, so it cannot be skipped."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return _dt.date(year, month, day + 1)


def _nth_weekday(year, month, weekday, n):
    """n-th `weekday` (Mon=0) of the month."""
    d = _dt.date(year, month, 1)
    first = d + _dt.timedelta(days=(weekday - d.weekday()) % 7)
    return first + _dt.timedelta(days=7 * (n - 1))


def _last_weekday_of_month(year, month, weekday):
    """Last `weekday` (Mon=0) of the month -- the Conference Board's release rule (last Tuesday)."""
    nxt = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    last = nxt - _dt.timedelta(days=1)
    return last - _dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d):
    """NYSE observation: a Saturday holiday closes the preceding Friday, a Sunday holiday the
    following Monday."""
    if d.weekday() == 5:
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + _dt.timedelta(days=1)
    return d


_HOLIDAY_CACHE = {}


def nyse_holidays(year):
    """Full-day NYSE closures for `year`."""
    if year in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[year]
    H = set()
    ny = _dt.date(year, 1, 1)                       # New Year's Day. NYSE does NOT pull a Saturday
    if ny.weekday() == 6:                           # New Year back to Dec 31 (it keeps the year's
        H.add(_dt.date(year, 1, 2))                 # final session), so the Saturday case is a no-op.
    elif ny.weekday() != 5:
        H.add(ny)
    H.add(_nth_weekday(year, 1, 0, 3))              # MLK Day: 3rd Monday of January
    H.add(_nth_weekday(year, 2, 0, 3))              # Washington's Birthday: 3rd Monday of February
    H.add(_easter(year) - _dt.timedelta(days=2))    # Good Friday
    H.add(_last_weekday_of_month(year, 5, 0))       # Memorial Day: last Monday of May
    if year >= 2022:                                # NYSE first observed Juneteenth in 2022
        H.add(_observed(_dt.date(year, 6, 19)))
    H.add(_observed(_dt.date(year, 7, 4)))          # Independence Day
    H.add(_nth_weekday(year, 9, 0, 1))              # Labor Day: 1st Monday of September
    H.add(_nth_weekday(year, 11, 3, 4))             # Thanksgiving: 4th Thursday of November
    H.add(_observed(_dt.date(year, 12, 25)))        # Christmas
    _HOLIDAY_CACHE[year] = H
    return H


def is_trading_day(d):
    """True when the NYSE holds a full or half session on date `d`."""
    d = d if isinstance(d, _dt.date) else pd.Timestamp(d).date()
    if d.weekday() >= 5 or d.isoformat() in _NYSE_ADHOC_CLOSURES:
        return False
    return d not in nyse_holidays(d.year)


def nth_trading_day(year, month, n):
    """The n-th NYSE trading day of (year, month) -- ISM's 'first/third business day' rule, anchored to
    the tape so the generated date always has a session."""
    d = _dt.date(year, month, 1)
    seen = 0
    while d.month == month:
        if is_trading_day(d):
            seen += 1
            if seen == n:
                return d
        d += _dt.timedelta(days=1)
    return None


# category -> (rule, annotation). Only releases with a PUBLISHED deterministic rule and a 10:00 ET
# time are generated; JOLTS / NEW_HOME / UMICH are 10:00 but have no clean rule -> ingest them via
# load_release_schedule(), which overrides on (date, category).
_RULE_RELEASES = {
    "ISM_MFG":    (("nth_trading_day", 1), "ISM Manufacturing PMI (1st business day, 10:00 ET)"),
    "ISM_SVC":    (("nth_trading_day", 3), "ISM Services PMI (3rd business day, 10:00 ET)"),
    "CONF_BOARD": (("last_weekday", 1),    "Conference Board Consumer Confidence (last Tuesday, 10:00 ET)"),
}
RULE_RELEASE_CATEGORIES = tuple(_RULE_RELEASES)


def _rule_release_date(rule, year, month):
    kind, arg = rule
    return nth_trading_day(year, month, arg) if kind == "nth_trading_day" \
        else _last_weekday_of_month(year, month, arg)


def rule_release_days(category, lo_year, hi_year):
    """Every generated date for `category` across [lo_year, hi_year], skipping any that lands on a
    non-trading day (belt-and-braces: nth_trading_day already guarantees it, last_weekday does not)."""
    rule, _ = _RULE_RELEASES[category]
    out = []
    for y in range(lo_year, hi_year + 1):
        for m in range(1, 13):
            d = _rule_release_date(rule, y, m)
            if d is not None and is_trading_day(d):
                out.append(d)
    return out


def release_collisions(frame):
    """Dates carrying MORE THAN ONE distinct release time. cross_event_onset builds one session per
    DATE and takes the first matching event row, so a multi-release date silently resolves to one
    arbitrary onset -- e.g. an ISM 10:00 and an FOMC 14:00 on the same session. Returns
    {date: [(category, event_et), ...]} so the choice is made deliberately rather than by sort order."""
    if not len(frame):
        return {}
    f = frame.copy()
    f["_d"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
    out = {}
    for d, g in f.groupby("_d"):
        ets = {str(x) for x in g["event_et"].fillna("") if str(x).strip()}
        if len(ets) > 1:
            out[d] = sorted((str(r["category"]), str(r["event_et"]))
                            for _, r in g.iterrows() if str(r["event_et"]).strip())
    return out


def scheduled_events(start=None, end=None, categories=("FOMC", "CPI", "NFP", "OPEX")):
    """Scheduled-release events (same dict schema as the crisis registry) over [start, end],
    drawn from the verified FOMC/BLS tables and the generated OPEX calendar. Impact-blind by
    construction -- every occurrence in range, regardless of realized move. Dates outside the
    populated tables are simply absent: extend FOMC_DECISION_DAYS / BLS_RELEASE_DAYS from the
    primary sources in the section header (CPI/NFP are only seeded with shutdown-revised points)."""
    if categories is None:
        categories = (("FOMC", "CPI", "NFP", "OPEX", "ELECTION")
                      + RULE_RELEASE_CATEGORIES
                      + tuple(sorted({e[1] for e in _SCHED_UNCERTAINTY_SEED}))
                      + tuple(sorted({e["category"] for e in _LOADED_RELEASES})))
    lo = pd.Timestamp(start) if start is not None else pd.Timestamp("2023-01-01")
    hi = pd.Timestamp(end) if end is not None else pd.Timestamp("2026-12-31")
    out = []
    if "FOMC" in categories:
        for y, days in FOMC_DECISION_DAYS.items():
            for md in days:
                out.append(_mk_scheduled(f"{y}-{md}", "FOMC", "MONETARY", "14:00",
                                         "FOMC rate decision (statement 14:00 ET, day 2 of meeting)",
                                         "federalreserve.gov FOMC schedule"))
    for cat in ("CPI", "NFP"):
        if cat in categories:
            for y, days in BLS_RELEASE_DAYS.get(cat, {}).items():
                for md in days:
                    out.append(_mk_scheduled(f"{y}-{md}", cat, "MACRO", "08:30",
                                             f"{cat} release (08:30 ET)", "bls.gov release schedule"))
    if "OPEX" in categories:
        for y in range(lo.year, hi.year + 1):
            for m in (3, 6, 9, 12):
                out.append(_mk_scheduled(_third_friday(y, m).isoformat(), "OPEX", "MARKET_STRUCTURE",
                                         "09:30", "Quad-witching expiration (3rd Friday; flows at open & close)",
                                         "CME/Cboe/OCC expiration calendar (3rd-Friday rule)"))
    # the rule-generated 10:00 cluster -- the dense, RTH-valid backbone (see the calendar section above).
    # An INGESTED date wins over a generated one for the same category-MONTH: the dedup below is on
    # (date, category), so a rescheduled release would otherwise survive alongside the generated date
    # and double-count that month. Suppressing at month granularity means a partial ingest (say
    # authoritative 2024 only) corrects 2024 while generation still fills the untouched years.
    for cat, (_rule, annot) in _RULE_RELEASES.items():
        if cat in categories:
            loaded_months = {(pd.Timestamp(e["date"]).year, pd.Timestamp(e["date"]).month)
                             for e in _LOADED_RELEASES if e.get("category") == cat}
            for d in rule_release_days(cat, lo.year, hi.year):
                if (d.year, d.month) in loaded_months:
                    continue
                out.append(_mk_scheduled(d.isoformat(), cat, _MACRO_CLASS.get(cat, "MACRO"), "10:00",
                                         annot, "rule-generated (publisher convention) -- VERIFY/override "
                                                "via load_release_schedule"))
    # scheduled-uncertainty stratum (impact-blind unscheduled): elections generated, others seeded
    if "ELECTION" in categories:
        for d in _election_days(lo.year, hi.year):
            out.append(_mk_scheduled(d.isoformat(), "ELECTION", "POLITICAL", "",
                                     "US general election (results overnight; reaction the next session)",
                                     "US general-election calendar (1st Tue after 1st Mon, Nov)",
                                     selection="scheduled_uncertainty", timing="overnight"))
    for date_str, cat, cls, ann, src in _SCHED_UNCERTAINTY_SEED:
        if cat in categories:
            out.append(_mk_scheduled(date_str, cat, cls, "", ann, src,
                                     selection="scheduled_uncertainty", timing="overnight"))
    for e in _LOADED_RELEASES:                      # loader-registered releases (any agency)
        if e.get("category") in categories:
            out.append(dict(e))
    out = [e for e in out if lo <= pd.Timestamp(e["date"]) <= hi]
    seen, ded = set(), []                           # dedup on (date, category); first wins
    for e in sorted(out, key=lambda e: e["date"]):
        k = (e["date"], e["category"])
        if k not in seen:
            seen.add(k); ded.append(e)
    return ded


# ── release-schedule loader (the dates are DATA -> ingest, don't invent) ───────
# Map agency release NAMES (substring, lowercased) to category codes. First match wins.
RELEASE_NAME_TO_CATEGORY = {
    "employment situation": "NFP", "nonfarm": "NFP", "jobs report": "NFP", "the employment situation": "NFP",
    "consumer price index": "CPI",
    "producer price index": "PPI",
    "personal income and outlays": "PCE", "personal consumption expenditures": "PCE", "pce price": "PCE",
    "gross domestic product": "GDP",
    "advance monthly retail": "RETAIL", "retail sales": "RETAIL", "retail trade": "RETAIL",
    "durable goods": "DURABLE", "manufacturers' new orders": "DURABLE",
    "job openings": "JOLTS", "jolts": "JOLTS",
    "unemployment insurance weekly": "CLAIMS", "initial claims": "CLAIMS", "jobless claims": "CLAIMS",
    "ism manufacturing": "ISM_MFG", "manufacturing pmi": "ISM_MFG", "ism report on business manufacturing": "ISM_MFG",
    "ism services": "ISM_SVC", "services pmi": "ISM_SVC", "non-manufacturing": "ISM_SVC",
    "consumer confidence": "CONF_BOARD",
    "consumer sentiment": "UMICH", "michigan": "UMICH",
    "adp national employment": "ADP", "adp employment": "ADP",
    "international trade": "TRADE", "trade balance": "TRADE", "u.s. international trade": "TRADE",
    "housing starts": "HOUSING", "building permits": "HOUSING", "new residential construction": "HOUSING",
    "new home sales": "NEW_HOME", "new residential sales": "NEW_HOME",
    "fomc": "FOMC", "federal funds": "FOMC",
    # scheduled-uncertainty / unscheduled event types (for a categorized news/event feed)
    "election": "ELECTION", "referendum": "REFERENDUM", "opec": "OPEC",
    "debt ceiling": "DEADLINE", "x-date": "DEADLINE", "supreme court": "RULING", "ruling": "RULING",
    "geopolit": "GEOPOLITICAL", "sanction": "GEOPOLITICAL", "military strike": "GEOPOLITICAL",
}

# Typical release times (ET). Used only when the source file has no time column.
DEFAULT_RELEASE_TIME_ET = {
    "FOMC": "14:00", "CPI": "08:30", "NFP": "08:30", "PPI": "08:30", "PCE": "08:30", "GDP": "08:30",
    "RETAIL": "08:30", "DURABLE": "08:30", "TRADE": "08:30", "HOUSING": "08:30", "CLAIMS": "08:30",
    "ADP": "08:15", "JOLTS": "10:00", "ISM_MFG": "10:00", "ISM_SVC": "10:00", "CONF_BOARD": "10:00",
    "UMICH": "10:00", "NEW_HOME": "10:00", "OPEX": "09:30",
    # unscheduled / overnight-reaction types default to "" -> open-to-open gap (set a time in the feed
    # for genuinely intraday events such as a mid-session military strike or sanction headline)
    "ELECTION": "", "REFERENDUM": "", "OPEC": "", "DEADLINE": "", "RULING": "", "GEOPOLITICAL": "",
}
_MACRO_CLASS = {"FOMC": "MONETARY", "OPEX": "MARKET_STRUCTURE", "ELECTION": "POLITICAL",
                "REFERENDUM": "GEOPOLITICAL", "OPEC": "GEOPOLITICAL", "DEADLINE": "SOVEREIGN_CREDIT",
                "RULING": "POLITICAL", "GEOPOLITICAL": "GEOPOLITICAL"}   # everything else -> "MACRO"


def _name_to_category(name, name_map=None):
    s = str(name).strip().lower()
    for key, cat in (name_map or RELEASE_NAME_TO_CATEGORY).items():
        if key in s:
            return cat
    return None


def _norm_time(t):
    """Normalize '8:30', '08:30', '8:30 AM', '2:00 PM', a datetime.time, etc. -> 'HH:MM'."""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    if hasattr(t, "hour") and hasattr(t, "minute"):
        return "%02d:%02d" % (t.hour, t.minute)
    s = str(t).strip().upper().replace(".", "")
    ampm = None
    if s.endswith("AM") or s.endswith("PM"):
        ampm, s = s[-2:], s[:-2].strip()
    try:
        p = s.split(":"); hh = int(p[0]); mm = int(p[1]) if len(p) > 1 else 0
    except Exception:
        return None
    if ampm == "PM" and hh < 12: hh += 12
    if ampm == "AM" and hh == 12: hh = 0
    return "%02d:%02d" % (hh, mm)


def _clean(v):
    """None for missing/NaN/empty values (DataFrame cells come through as NaN, which is truthy)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return None if (s == "" or s.lower() == "nan") else v


def _coerce_rows(source):
    """CSV path / DataFrame / iterable-of-dicts -> list of dicts with lowercased keys."""
    if isinstance(source, str):
        recs = pd.read_csv(source).to_dict("records")
    elif isinstance(source, pd.DataFrame):
        recs = source.to_dict("records")
    else:
        recs = [dict(r) for r in source]
    return [{str(k).strip().lower(): v for k, v in r.items()} for r in recs]


def register_releases(events):
    """Append already-built scheduled-event dicts to the module-level loaded set."""
    _LOADED_RELEASES.extend(events)
    return len(_LOADED_RELEASES)


def clear_loaded():
    """Empty the loaded-releases set (calendar reverts to FOMC+OPEX+seeds)."""
    _LOADED_RELEASES.clear()


def load_release_schedule(source, name_map=None, default_time_et=None, register=True,
                          selection="scheduled"):
    """Ingest an authoritative scheduled-release schedule into the calendar. `source` is one of:
      * a path to a CSV with columns {date, category|release|name, [time_et|time]},
      * a pandas DataFrame with those columns,
      * an iterable of dict-like rows with those keys.
    A `category` column is used as-is; otherwise the `release`/`name` is mapped via
    RELEASE_NAME_TO_CATEGORY (override/extend with `name_map`). Release times come from a time column
    if present, else DEFAULT_RELEASE_TIME_ET[category] (override with `default_time_et`). `selection`
    is the provenance stamped on every loaded row: keep "scheduled" for an agency release schedule;
    pass "news_classified" for a categorized news/event feed (selected on EX-ANTE prominence) or
    "scheduled_uncertainty" for a known-date/unknown-outcome calendar -- both are impact-blind.
    Returns the registry-schema event dicts; with register=True they flow through scheduled_events /
    shocks_frame / event_dates / release_times.

    The dates are DATA -- download the authoritative schedule (bls.gov 'Schedule of Releases', the
    BEA and Census release calendars, or an iCal/CSV export) and feed it here. This loader maps and
    validates; it does not invent dates. Unmapped release names are skipped (extend `name_map`)."""
    times = {**DEFAULT_RELEASE_TIME_ET, **(default_time_et or {})}
    out = []
    for r in _coerce_rows(source):
        d = _clean(r.get("date"))
        if d is None:
            continue
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        cat = _clean(r.get("category")) or _name_to_category(
            _clean(r.get("release")) or _clean(r.get("name")) or "", name_map)
        if not cat:
            continue
        et = _norm_time(_clean(r.get("time_et")) or _clean(r.get("time"))) or times.get(cat, "08:30")
        timing = "overnight" if not et else "scheduled"
        out.append(_mk_scheduled(date_str, cat, _MACRO_CLASS.get(cat, "MACRO"), et,
                                 "%s release (%s ET)" % (cat, et or "overnight"), "loaded release schedule",
                                 selection=selection, timing=timing))
    seen, ded = set(), []
    for e in sorted(out, key=lambda e: e["date"]):
        k = (e["date"], e["category"])
        if k not in seen:
            seen.add(k); ded.append(e)
    if register:
        register_releases(ded)
    return ded


def load_ics(source, name_map=None, default_time_et=None, register=True, selection="scheduled"):
    """Parse an iCal (.ics) export (e.g. the BLS release calendar or an economic-calendar feed) into
    the schedule. `source` is a path or the iCal text itself. Each VEVENT's SUMMARY is mapped to a
    category and DTSTART gives the date (its time is used if present, else the category default).
    `selection` stamps provenance (see load_release_schedule). Minimal parser, no external deps."""
    text = source if ("BEGIN:VCALENDAR" in source or "BEGIN:VEVENT" in source) else \
        open(source, "r", encoding="utf-8", errors="ignore").read()
    rows, summary, dtstart = [], None, None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("BEGIN:VEVENT"):
            summary, dtstart = None, None
        elif line.startswith("SUMMARY") and ":" in line:
            summary = line.split(":", 1)[1]
        elif line.startswith("DTSTART") and ":" in line:
            dtstart = line.split(":", 1)[1].strip()
        elif line.startswith("END:VEVENT") and summary and dtstart:
            d = dtstart[:8]
            date_str = "%s-%s-%s" % (d[:4], d[4:6], d[6:8])
            tm = None
            if "T" in dtstart and len(dtstart) >= 13:
                tt = dtstart.split("T")[1]
                tm = "%s:%s" % (tt[:2], tt[2:4])
            rows.append({"date": date_str, "name": summary, "time": tm})
    return load_release_schedule(rows, name_map=name_map, default_time_et=default_time_et,
                                 register=register, selection=selection)


def shocks_frame(tz="America/New_York", include_scheduled=False,
                 scheduled_categories=("FOMC", "CPI", "NFP", "OPEX"), start=None, end=None):
    """The shock registry as a tz-aware DataFrame (one row per event, sorted by date). With
    include_scheduled=True it also appends the scheduled-event calendar (FOMC/CPI/NFP/OPEX) over
    [start, end], de-duplicated against the hand-annotated crisis rows on (date, category) so the
    richer registry entry always wins. Default is off -> identical to the crisis-only registry."""
    rows = list(_SHOCKS)
    if include_scheduled:
        have = {(e["date"], e.get("category")) for e in rows}
        for e in scheduled_events(start=start, end=end, categories=scheduled_categories):
            if (e["date"], e["category"]) not in have:
                rows.append(e)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(tz)
    if "selection" not in df.columns:                       # the hand-curated crisis registry is
        df["selection"] = "salience_curated"                # salience-curated by construction
    df["selection"] = df["selection"].fillna("salience_curated")
    df["impact_blind"] = df["selection"].isin(IMPACT_BLIND_SELECTION)
    return df.sort_values("date").reset_index(drop=True)


def event_dates(categories=None, classes=None, spy_es_only=False, tz="America/New_York",
                include_scheduled=False, scheduled_categories=("FOMC", "CPI", "NFP", "OPEX"),
                start=None, end=None):
    """DatetimeIndex of event (reaction-session) dates, filtered by fine `categories` (e.g.
    ('MWCB',)), coarse `classes` (e.g. ('GEOPOLITICAL','MONETARY')), and/or the SPY/ES flag.
    Set include_scheduled=True to fold in the scheduled FOMC/CPI/NFP/OPEX backbone over [start,end]."""
    df = shocks_frame(tz, include_scheduled=include_scheduled,
                      scheduled_categories=scheduled_categories, start=start, end=end)
    if categories is not None:
        df = df[df["category"].isin(categories)]
    if classes is not None:
        df = df[df["event_class"].isin(classes)]
    if spy_es_only:
        df = df[df["spy_es"]]
    return pd.DatetimeIndex(df["date"])


def release_times(categories=("MWCB",), classes=None, spy_es_only=False, tz="America/New_York",
                  include_scheduled=False, scheduled_categories=("FOMC", "CPI", "NFP", "OPEX"),
                  start=None, end=None):
    """{date -> intraday Timestamp} from `event_et`, for the event study's release boundary
    (halt reopen, or the scheduled/unscheduled release time). Rows with a blank or non-numeric
    `event_et` (overnight / weekend / 'closed early') are skipped -- those use the open-to-open
    gap on the reaction session instead. include_scheduled=True folds in the FOMC/CPI/NFP/OPEX
    backbone over [start,end] (FOMC 14:00, CPI/NFP 08:30, OPEX 09:30 ET)."""
    df = shocks_frame(tz, include_scheduled=include_scheduled,
                      scheduled_categories=scheduled_categories, start=start, end=end)
    if categories is not None:
        df = df[df["category"].isin(categories)]
    if classes is not None:
        df = df[df["event_class"].isin(classes)]
    if spy_es_only:
        df = df[df["spy_es"]]
    out = {}
    for _, r in df.iterrows():
        t = str(r["event_et"]).lstrip("~")
        try:
            hh, mm = int(t.split(":")[0]), int(t.split(":")[1])
        except Exception:
            continue
        out[r["date"]] = r["date"] + pd.Timedelta(hours=hh, minutes=mm)
    return out


# ── treatment / control assignment ────────────────────────────────────────────
def is_event_day(dates, categories=None, classes=None, spy_es_only=False, buffer_days=0):
    """Boolean mask over `dates`: True if the (calendar) day is a registry event, or within
    `buffer_days` *trading* days (positions in the given index) of one."""
    idx = pd.DatetimeIndex(dates)
    ev = set(event_dates(categories=categories, classes=classes, spy_es_only=spy_es_only).normalize())
    is_ev = np.array([d.normalize() in ev for d in idx])
    if buffer_days <= 0:
        return is_ev
    mask = is_ev.copy()
    for p in np.where(is_ev)[0]:
        mask[max(0, p - buffer_days):min(len(idx), p + buffer_days + 1)] = True
    return mask


def treatment_control_assignment(daily_state, categories=("MWCB",), classes=None, n_controls=2,
                                  buffer_days=5, window_days=60, exact_weekday=False,
                                  caliper=None, spy_es_only=False):
    """Match each treatment (registry-event reaction) day to non-event control days on the
    EX-ANTE stress level. `daily_state` is a Series indexed by trading date of an ex-ante level
    (prior-close VIX, a vol forecast, or the long-run vol level -- NOT a contemporaneous or
    one-day-change vol). Filter the treatment universe with `categories` (fine) and/or
    `classes` (coarse, e.g. ('GEOPOLITICAL',)). Controls are excluded within `buffer_days`
    trading days of ANY event in that universe, drawn within +/- `window_days` of the
    treatment day, optionally same weekday, optionally within a `caliper` on the standardized
    state distance, nearest-first and without replacement. Returns a tidy treatment->control
    table with the state values and distances (the design you would estimate the DiD on)."""
    state = pd.Series(np.asarray(daily_state, float), index=pd.DatetimeIndex(daily_state.index)).dropna()
    idx = state.index
    sd = float(state.std()) or 1.0
    evset = set(event_dates(categories=categories, classes=classes, spy_es_only=spy_es_only).normalize())
    is_ev = np.array([d.normalize() in evset for d in idx])
    contaminated = np.zeros(len(idx), bool)
    for p in np.where(is_ev)[0]:
        contaminated[max(0, p - buffer_days):min(len(idx), p + buffer_days + 1)] = True
    treat_pos = list(np.where(is_ev)[0])
    used, rows = set(), []
    for p in treat_pos:
        ts = state.iloc[p]
        cand = [q for q in range(max(0, p - window_days), min(len(idx), p + window_days + 1))
                if not contaminated[q] and q not in used
                and (not exact_weekday or idx[q].weekday() == idx[p].weekday())]
        cand.sort(key=lambda q: abs(state.iloc[q] - ts))
        n_picked = 0
        for q in cand:
            dist = abs(state.iloc[q] - ts) / sd
            if caliper is not None and dist > caliper:
                break
            used.add(q)
            rows.append({"treatment_date": idx[p], "control_date": idx[q],
                         "treat_state": ts, "control_state": state.iloc[q],
                         "state_dist_sd": dist, "calendar_gap_days": int((idx[q] - idx[p]).days)})
            n_picked += 1
            if n_picked >= n_controls:
                break
    return pd.DataFrame(rows)


def compare_control_rules(vol, upper_q=0.60, treat_frac=0.25, n_controls=2, window_days=60,
                          ewma_halflife=10, treatment_dates=None, seed=0):
    """Show why level-matching beats the ln(sigma_t/sigma_{t-1})<0 rule, on a daily vol
    series. Treatment = `treatment_dates` if given, else a random sample (`treat_frac`) of
    the high-level days (ex-ante level >= `upper_q` quantile) -- i.e. 'hand-picked volatile
    days'. Matched controls = nearest ex-ante level among non-treatment days; naive controls
    = days where the one-day log vol change is negative (the rule used). Reports each group's
    mean ex-ante level and the standardized mean difference (SMD) from treatment: matched
    should be ~0 (balanced); naive is materially non-zero (imbalanced, hence biased). The
    ex-ante level is a lagged EWMA of vol -- never the contemporaneous or one-day-change vol."""
    v = pd.Series(np.asarray(vol, float), index=pd.DatetimeIndex(vol.index)).dropna()
    level = v.shift(1).ewm(halflife=ewma_halflife, min_periods=1).mean()        # ex-ante level
    ln_chg = np.log(v).diff()
    valid = np.isfinite(level.values)
    sd = float(np.nanstd(level.values)) or 1.0
    if treatment_dates is not None:
        tset_dates = set(pd.DatetimeIndex(treatment_dates).normalize())
        treat = np.array([i for i, d in enumerate(v.index) if d.normalize() in tset_dates and valid[i]])
    else:
        thr = np.nanquantile(level.values[valid], upper_q)
        upper = np.where((level.values >= thr) & valid)[0]
        rng = np.random.default_rng(seed)
        treat = np.sort(rng.choice(upper, size=max(1, int(len(upper) * treat_frac)), replace=False))
    treset = set(treat.tolist()); used = set(); matched = []
    for p in treat:
        cand = [q for q in range(max(0, p - window_days), min(len(v), p + window_days + 1))
                if q not in treset and q not in used and valid[q]]
        cand.sort(key=lambda q: abs(level.values[q] - level.values[p]))
        for q in cand[:n_controls]:
            used.add(q); matched.append(q)
    naive = [q for q in np.where((ln_chg.values < 0) & valid)[0] if q not in treset]
    mt = float(np.nanmean(level.values[treat])) if len(treat) else np.nan
    def smd(group):
        if len(group) == 0:
            return np.nan, np.nan
        m = float(np.nanmean(level.values[group])); return m, (mt - m) / sd
    mm, smd_m = smd(matched); mn, smd_n = smd(naive)
    tab = pd.DataFrame({
        "n_days": [len(treat), len(matched), len(naive)],
        "mean_exante_level": [mt, mm, mn],
        "SMD_vs_treatment": [0.0, smd_m, smd_n],
    }, index=["treatment (volatile)", "matched controls (level)", "naive controls (ln-change<0)"])
    return {"table": tab, "smd_matched": smd_m, "smd_naive": smd_n}


# ── self-test ─────────────────────────────────────────────────────────────────
def _selftest():
    print("market_shocks self-test")

    # (1) registry well-formed, broad classes present, canonical events present
    df = shocks_frame()
    ev_norm = set(df["date"].dt.normalize())
    classes = set(df["event_class"].unique())
    has_mar2020 = all(pd.Timestamp(d, tz="America/New_York") in ev_norm
                      for d in ["2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"])
    expect_classes = {"MARKET_STRUCTURE", "MONETARY", "SOVEREIGN_CREDIT", "TRADE_POLICY", "GEOPOLITICAL", "VOL_REGIME"}
    print("\n(1) registry: %d events across %d classes=%s" % (len(df), len(classes), sorted(classes)))
    print("    categories:", sorted(df["category"].unique()))
    print("    March-2020 MWCB present:", has_mar2020,
          "| Liberation Day 2025-04-03:", pd.Timestamp("2025-04-03", tz="America/New_York") in ev_norm,
          "| Iran strikes 2025-06-13/23:", {pd.Timestamp(d, tz="America/New_York") for d in ["2025-06-13", "2025-06-23"]} <= ev_norm)
    print("    checks:", len(df) >= 20
          and {"date", "category", "event_class", "timing", "event_et", "halt_level", "annotation",
               "source", "spx_pct", "spy_es"}.issubset(df.columns)
          and df["date"].is_monotonic_increasing and df["date"].is_unique
          and has_mar2020 and expect_classes.issubset(classes)
          and df["timing"].isin(["halt_reopen", "scheduled", "intraday_news", "overnight", "weekend", "multi_day"]).all())

    # (2) category / class filters and release times
    mwcb = event_dates(categories=("MWCB",))
    geo = event_dates(classes=("GEOPOLITICAL",))
    rel = release_times(categories=("MWCB",))
    rel_fomc = release_times(categories=("FOMC",))
    print("\n(2) filters: MWCB dates=%d, GEOPOLITICAL dates=%d; MWCB reopen times=%d, FOMC release times=%d"
          % (len(mwcb), len(geo), len(rel), len(rel_fomc)))
    print("    e.g. MWCB", {str(k.date()): str(v.time()) for k, v in list(rel.items())[:2]},
          "| FOMC", {str(k.date()): str(v.time()) for k, v in list(rel_fomc.items())[:1]})
    print("    checks:", len(mwcb) == 5 and len(geo) >= 4
          and len(rel) == 4 and all(isinstance(v, pd.Timestamp) for v in rel.values())
          and len(rel_fomc) >= 1 and all(v.hour == 14 for v in rel_fomc.values()))

    # (3) treatment/control matching: controls are clean and balanced on the ex-ante state
    days = pd.bdate_range("2019-06-03", "2020-06-30", tz="America/New_York")
    rng = np.random.default_rng(3)
    s = pd.Series(np.cumsum(rng.normal(0, 0.1, len(days))), index=days); s = (s - s.min() + 1.0)
    design = treatment_control_assignment(s, categories=("MWCB",), n_controls=2, buffer_days=5, window_days=40)
    evset = set(event_dates(categories=("MWCB",)).normalize())
    controls_clean = not design["control_date"].dt.normalize().isin(evset).any()
    print("\n(3) treatment/control matching (MWCB treatment, level-matched controls)")
    print("    %d treatment days in sample, %d matched control rows; mean |state dist|=%.3f sd"
          % (design["treatment_date"].nunique(), len(design), design["state_dist_sd"].mean()))
    print("    controls non-event:", controls_clean, "| no control reused:", design["control_date"].is_unique)
    print("    checks:", design["treatment_date"].nunique() == 4 and len(design) > 0
          and controls_clean and design["control_date"].is_unique and design["state_dist_sd"].mean() < 0.5)

    # (4) level-matching beats the ln(sigma)<0 control rule (balance)
    rng = np.random.default_rng(7); n = 500
    lvl = np.zeros(n)
    for t in range(1, n):
        lvl[t] = 0.95 * lvl[t - 1] + rng.normal(0, 0.3)
    vol = pd.Series(np.exp(0.3 * lvl) * 0.01 * np.exp(rng.normal(0, 0.1, n)),
                    index=pd.bdate_range("2018-01-01", periods=n, tz="America/New_York"))
    cmp = compare_control_rules(vol, upper_q=0.60, treat_frac=0.25, n_controls=2, seed=1)
    print("\n(4) control-rule balance: level-matched vs ln(sigma_t/sigma_{t-1})<0")
    print(cmp["table"].round(4).to_string())
    print("    |SMD matched|=%.3f  vs  |SMD naive|=%.3f" % (abs(cmp["smd_matched"]), abs(cmp["smd_naive"])))
    print("    checks:", abs(cmp["smd_matched"]) < abs(cmp["smd_naive"]) and abs(cmp["smd_matched"]) < 0.3)


if __name__ == "__main__":
    _selftest()
