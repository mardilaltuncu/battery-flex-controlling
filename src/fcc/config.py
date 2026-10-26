"""Central assumptions registry.

Every parameter used anywhere in the case is declared through ``param`` so that it carries its provenance.
The resulting table is shown in the notebook and written to the Excel workbook (sheet ``Inputs``).

Provenance labels
-----------------
OBSERVED : taken from public market data (Energy-Charts / SMARD, Bundesnetzagentur)
PRIMER   : mechanics / definitions taken from the energy-trading finance study notes (market rules change, so
           numbers quoted there are treated as illustrative unless marked OBSERVED)
ILLUSTR  : an illustrative assumption made for this case; replace with the company's real figure
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

OBSERVED = "observed data"
PRIMER = "domain primer"
ILLUSTR = "illustrative assumption"

_REG: list[dict] = []


def param(key: str, value, unit: str, source: str, note: str = ""):
    """Register a parameter and return its value."""
    _REG.append({"parameter": key, "value": value, "unit": unit, "source": source, "note": note})
    return value


def assumptions_table() -> pd.DataFrame:
    return pd.DataFrame(_REG).drop_duplicates("parameter", keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Time and market
# --------------------------------------------------------------------------------------------------
TZ = param("timezone", "Europe/Berlin", "tz", PRIMER, "all maths in UTC, local time only for reporting")
DT_H = param("interval_hours", 0.25, "h", PRIMER, "15-minute market time unit (MTU)")
DA_15MIN_START = param("da_15min_start_utc", "2025-09-30 22:00", "UTC", OBSERVED,
                       "day-ahead moved from hourly to 15-minute products on 1 Oct 2025 (local)")

# --------------------------------------------------------------------------------------------------
# Battery technics and costs (per-MW, scaled by asset size)
# --------------------------------------------------------------------------------------------------
RTE = param("round_trip_efficiency", 0.88, "share", PRIMER, "Li-ion typically 85-90 %")
CELL_REPLACEMENT_EUR_PER_MWH_CAP = param("cell_replacement_value", 90_000.0, "EUR/MWh capacity", ILLUSTR)
LIFETIME_CYCLES = param("lifetime_full_cycles", 8_000, "cycles", ILLUSTR)
DEGRADATION_EUR_MWH = param("degradation_cost", CELL_REPLACEMENT_EUR_PER_MWH_CAP / LIFETIME_CYCLES, "EUR/MWh discharged",
                            PRIMER, "= replacement value / (lifetime cycles x capacity MWh)")
MAX_CYCLES_PER_DAY = param("max_cycles_per_day", 1.5, "cycles/day", PRIMER, "typical warranty limit, e.g. 1.5")
SOC_START_FRAC = param("soc_start_fraction", 0.5, "share", ILLUSTR, "daily start and minimum end state of charge")
AVAILABILITY = param("availability", 0.97, "share", ILLUSTR, "share of time the asset can operate")

# Exchange fees (trading fee to the exchange, clearing fee to the CCP)
FEE_DA_EUR_MWH = param("fee_day_ahead", 0.10, "EUR/MWh traded", ILLUSTR, "trading + clearing fee, day-ahead")
FEE_ID_EUR_MWH = param("fee_intraday", 0.25, "EUR/MWh traded", ILLUSTR, "continuous intraday carries higher fees")

# Intraday: continuous re-trading recovers part of the gap between forecast-based and perfect-foresight dispatch
ID_RECOVERY_RHO = param("intraday_gap_recovery", 0.30, "share of gap", ILLUSTR,
                        "share of (perfect foresight - forecast dispatch) revenue recovered by intraday re-trading")

ID_EXTRA_VALUE = param("intraday_extra_value_share", 0.20, "share of DA perfect-foresight value", ILLUSTR,
                       "additional value from intraday volatility beyond the day-ahead (no intraday data downloaded)")
ID_VOLUME_SHARE = param("intraday_volume_share", 0.25, "share of DA traded MWh", ILLUSTR,
                        "intraday re-trading volume relative to the day-ahead volume")

# Imbalance (reBAP is not downloaded here -> modelled)
IMBALANCE_VOL_RATE = param("imbalance_noise_sd", 0.012, "share of scheduled MWh", ILLUSTR,
                           "random metered deviation around the schedule")

# --------------------------------------------------------------------------------------------------
# Reserve capacity (FCR / aFRR capacity): modelled, not observed
# --------------------------------------------------------------------------------------------------
RESERVE_PRICE_START = param("reserve_price_2022", 14.0, "EUR/MW/h", ILLUSTR, "modelled path, not market data")
RESERVE_PRICE_END = param("reserve_price_2026", 4.5, "EUR/MW/h", ILLUSTR,
                          "market saturation: capacity prices fall as storage is added")
RESERVE_PRICE_NOISE = param("reserve_price_monthly_sd", 0.12, "log-sd", ILLUSTR)


@dataclass(frozen=True)
class Asset:
    asset_id: str
    kind: str                 # "stand-alone" | "co-located"
    mw: float
    duration_h: float
    cod_plan: str             # planned commercial operation date
    cod_actual: str
    customer: str
    reserve_share: float      # share of MW committed to reserve capacity
    grid_cap_frac: float = 1.0      # co-located: discharge cap (share of MW) in PV hours
    contract: str = "net_fixed"


ASSETS = [
    Asset("BESS-A", "stand-alone", 60.0, 2.0, "2024-01-01", "2024-01-01", "C1", 0.30, 1.0, "net_fixed"),
    Asset("BESS-B", "co-located", 30.0, 2.0, "2024-07-01", "2024-07-01", "C2", 0.20, 0.40, "tiered_floor"),
    Asset("BESS-C", "stand-alone", 10.0, 1.0, "2025-03-01", "2025-06-01", "C3", 0.00, 1.0, "fixed_min_fee"),
    # contracted, not yet built (pre-COD): only enters the forward-looking forecast; cod_actual unknown
    Asset("BESS-D", "stand-alone", 20.0, 2.0, "2027-03-01", "", "C4", 0.25, 1.0, "net_fixed"),
]
param("portfolio_size_operational", sum(a.mw for a in ASSETS if a.cod_actual), "MW", ILLUSTR,
      "three illustrative operating customer assets")
param("portfolio_size_pre_cod", sum(a.mw for a in ASSETS if not a.cod_actual), "MW", ILLUSTR, "contracted, COD 2027-03 planned")
COD_DELAY_PROBS = param("cod_delay_distribution", {0: 0.35, 3: 0.35, 6: 0.20, 9: 0.10}, "months: probability", PRIMER,
                        "delays are the rule in energy projects -> probability-weight the COD")
FEE_TARIFF_AFTER = param("fee_day_ahead_from_2026_04", 0.12, "EUR/MWh traded", ILLUSTR,
                         "exchange tariff change on 2026-04-01; the rolling forecast still carries the old tariff")
TARIFF_CHANGE_DATE = "2026-04-01"

# Contract terms (Layer 2 module)
SHARE_C1 = param("c1_share_rate", 0.15, "share of net revenue", ILLUSTR, "net base, fixed rate")
SHARE_C2_LOW = param("c2_share_low", 0.12, "share", ILLUSTR, "tiered: up to the threshold")
SHARE_C2_HIGH = param("c2_share_high", 0.20, "share", ILLUSTR, "tiered: above the threshold")
TIER_THRESHOLD_EUR_MW_YR = param("c2_tier_threshold", 80_000.0, "EUR/MW/yr", ILLUSTR)
FLOOR_EUR_MW_YR = param("c2_customer_floor", 55_000.0, "EUR/MW/yr", ILLUSTR,
                        "customer receives at least this per year, optimizer tops up (a put option)")
SHARE_C3 = param("c3_share_rate", 0.20, "share", ILLUSTR)
MIN_FEE_C3_EUR_MONTH = param("c3_min_monthly_fee", 1_500.0, "EUR/month", ILLUSTR,
                             "small, complex customers need a minimum fee (cost to serve)")
SHARE_C4 = param("c4_share_rate", 0.15, "share", ILLUSTR, "pre-COD asset, same terms as C1")

# Company cost base attributed to this portfolio (monthly). Joint = shared by all products.
JOINT_COSTS_EUR_MONTH = {
    "Trading team": param("cost_trading_team", 30_000.0, "EUR/month", ILLUSTR, "share of traders + quants"),
    "Data & IT": param("cost_data_it", 12_000.0, "EUR/month", ILLUSTR, "cloud, market data, tooling"),
    "Balance group (BRP)": param("cost_brp", 4_000.0, "EUR/month", ILLUSTR),
}
DIRECT_COSTS_EUR_MONTH = {
    "Day-ahead": param("direct_da_data", 1_000.0, "EUR/month", ILLUSTR, "product-specific data/licences"),
    "Intraday": param("direct_id_data", 5_000.0, "EUR/month", ILLUSTR, "order-book data, API/message fees"),
    "Reserve capacity": param("direct_res_prequal", 2_500.0, "EUR/month", ILLUSTR, "prequalification, testing"),
}
COST_TO_SERVE_EUR_MONTH = {
    "C1": param("cost_to_serve_c1", 2_500.0, "EUR/month", ILLUSTR, "large stand-alone"),
    "C2": param("cost_to_serve_c2", 5_000.0, "EUR/month", ILLUSTR, "co-located: constraint management"),
    "C3": param("cost_to_serve_c3", 3_000.0, "EUR/month", ILLUSTR, "small asset, same reporting effort"),
    "C4": param("cost_to_serve_c4", 2_500.0, "EUR/month", ILLUSTR, "from COD"),
}
# Activity drivers used by the activity-based allocation (shares sum to 1 per pool)
ALLOC_ACTIVITY = {
    "Trading team": {"Day-ahead": 0.18, "Intraday": 0.67, "Reserve capacity": 0.15},       # trader hours
    "Data & IT": {"Day-ahead": 0.20, "Intraday": 0.55, "Reserve capacity": 0.25},          # messages / compute
    "Balance group (BRP)": {"Day-ahead": 0.35, "Intraday": 0.55, "Reserve capacity": 0.10},  # MWh through BRP
}
for _pool, _keys in ALLOC_ACTIVITY.items():
    for _prod, _w in _keys.items():
        param(f"alloc_activity|{_pool}|{_prod}", _w, "share", ILLUSTR, "activity key (trader hours / messages / MWh)")

# --------------------------------------------------------------------------------------------------
# Reconciliation policy
# --------------------------------------------------------------------------------------------------
TOL_VOL_MWH = param("tolerance_volume_per_interval", 0.005, "MWh", ILLUSTR, "below this a volume difference self-clears")
TOL_AMOUNT_EUR = param("tolerance_amount_per_interval", 0.50, "EUR", ILLUSTR, "below this an amount difference self-clears")
TOL_PRICE_EUR_MWH = param("tolerance_price", 0.005, "EUR/MWh", ILLUSTR)
SYSTEMATIC_ONE_SIDED_SHARE = param("systematic_bias_one_sided_share", 0.90, "share", PRIMER,
                                   "small differences that always point the same way are investigated")

# --------------------------------------------------------------------------------------------------
# Scenario and forecast settings
# --------------------------------------------------------------------------------------------------
MATERIALITY_PCT = param("variance_materiality_pct", 0.05, "share", PRIMER, "written explanation above +-5 %")
MATERIALITY_EUR = param("variance_materiality_eur", 25_000.0, "EUR", PRIMER, "and above +-25 k EUR")
MC_DRAWS = param("monte_carlo_draws", 4_000, "draws", ILLUSTR)
RANDOM_SEED = param("random_seed", 42, "seed", ILLUSTR)

# --------------------------------------------------------------------------------------------------
# Study windows
# --------------------------------------------------------------------------------------------------
HISTORY_START = param("history_start", "2019-01-01", "date", OBSERVED, "market history used for statistics")
BACKTEST_START = param("backtest_start", "2024-01-01", "date", ILLUSTR, "forecast / dispatch evaluation window")
RECON_START = param("recon_start", "2025-10-01", "date", ILLUSTR, "reconciliation window = 12 months of native 15-min DA")
RECON_END = param("recon_end", "2026-09-30", "date", ILLUSTR)
