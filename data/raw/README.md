---
license: cc-by-4.0
tags:
  - electricity
  - energy
  - power-markets
  - germany
  - time-series
pretty_name: Germany Power Market Data
---

# Germany Power Market Data

Part of a country-split collection of historical electricity market data gathered for the [elpriser.org](https://elpriser.org) price-forecasting project. See the [index dataset](https://huggingface.co/datasets/Elpriser/nordic-power-market-data) for the other countries: Denmark, Germany, Norway, Sweden, Finland, Netherlands.

## License & attribution

CC BY 4.0. Source: ENTSO-E Transparency Platform (transparency.entsoe.eu).

## Contents

DE-LU bidding zone (Germany/Luxembourg), ~2018-09/10 → present:

| File | Description |
|---|---|
| `entsoe_delu_dayahead_prices.parquet` | Day-ahead price, EUR/MWh, 15-min resolution |
| `entsoe_delu_generation_per_type.parquet` | Actual generation per production type (ENTSO-E `psrType` codes) |
| `entsoe_delu_load_actual.parquet` | Actual (realised) total system load, MW |
| `entsoe_delu_load_forecast.parquet` | Day-ahead forecast total system load, MW |
| `entsoe_delu_windsolar_forecast.parquet` | Day-ahead wind+solar generation forecast (only ~1 day ahead, archived history) |
| `entsoe_delu_installed_capacity.parquet` | Installed generation capacity per type, yearly snapshots |

Denmark's physical interconnector data with Germany (`entsoe_flow_dk1_delu`, `entsoe_flow_dk2_delu`, `entsoe_ntc_dk1_delu`, `entsoe_ntc_dk2_delu`) lives in the [Denmark dataset](https://huggingface.co/datasets/Elpriser/denmark-power-market).

## Format

Parquet, one row per timestamp (and per production type, where applicable). Column names are preserved as returned by the source API.

## Provenance

Collected 2026-07-03/04 via each provider's public API using free, personally-registered access. Collection scripts: [elpriser.org repo](https://github.com/x2q/elpriser.org), `scripts/data_backup/`.

## Updates

Last refreshed 2026-10-06. Each refresh re-fetches the most recent three days of every series and replaces those rows, so very recent *actual* values (generation, load, flows) can be revised between refreshes; everything older is left exactly as published. Refreshed with [`scripts/data_backup/refresh_hf_datasets.py`](https://github.com/x2q/elpriser.org/blob/main/scripts/data_backup/refresh_hf_datasets.py) in the elpriser.org repository.
