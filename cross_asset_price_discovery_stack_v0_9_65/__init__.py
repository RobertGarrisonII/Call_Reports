"""
cross_asset_price_discovery_stack
=================================
Cross-asset liquidity & price-discovery research stack (augmenting OFR WP 19-04:
Garrison, Jain & Paddrik). See README.md for the full guide.

The modules import one another with flat names (e.g. ``import cross_asset_pd_liquidity``),
which assumes this directory is on ``sys.path``. Importing this package guarantees that
— it prepends its own directory to ``sys.path`` — so both usage styles work:

    python irf.py                                              # run a module directly (from this dir)
    from cross_asset_price_discovery_stack_v0_3_0 import irf          # or import it as a package

Requires numpy, pandas, scipy (no statsmodels / arch). ``market_analysis_fixed``
additionally needs ``maystreet_data`` (available inside the MIDAS environment); importing
this package does NOT pull it in, so the package itself is importable anywhere. (Avoid
``from cross_asset_price_discovery_stack_v0_3_0 import *`` outside MIDAS, since that would try to
import ``market_analysis_fixed`` and hence ``maystreet_data``.)

Generated output (pickles, CSVs, plots) is written to ``OUTPUT_DIR`` (the bundled
``./output/`` directory).
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

#: Default sink for generated artifacts (the bundled ./output/ directory).
OUTPUT_DIR = _os.path.join(_HERE, "output")
_os.makedirs(OUTPUT_DIR, exist_ok=True)

#: THE version. Single source of truth: the package directory name, the release archive name and
#: the CHANGELOG's top entry are all derived from or checked against it (see check_version.py,
#: which is in the STAGE 1 gate). It drifted 22 releases once -- pinned at 0.9.20 while the
#: CHANGELOG read 0.9.42 and every archive shipped as "v0934" -- so the check is a test, not a
#: convention.
__version__ = "0.9.65"

__all__ = [
    'autoscale',
    'onset_sensitivity',
    'onset_response',
    "run_analysis",
    "legacy_tables",
    "market_analysis_fixed",
    "run_mean_variance",
    "mean_variance",
    "run_contagion",
    "liquidity_curve_metrics",
    "price_discovery_shares",
    "cross_asset_pd_liquidity",
    "cross_impact",
    "dcc_garch",
    "robustness",
    "irf",
    "noise_robust_cov",
    "microstructure_diagnostics",
    "functional_liquidity",
    "liquidity_stress",
    "jump_robust",
    "robust_prices",
    "ecm_sde",
    "tandem_order_flow",
    "correlation_svar",
    "cross_flow",
    "stress_index",
    "mwcb_event_study",
    "rigobon_id",
    "hawkes_cross",
    "markov_switching_vecm",
    "pricing_error",
    "tick_correction",
    "inference",
    "paper_tables",
    "market_shocks",
    "event_study_driver",
    "mstbook_loader",
    "lob_reconstruct",
    "auction_imbalance",
    "copula_garch",
    "efficient_price",
    "state_space_price",
]
