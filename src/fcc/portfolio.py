"""Portfolio engine: from dispatch to monthly drivers and P&L lines per asset.

Real       : day-ahead prices -> perfect-foresight and forecast-based dispatch (see ``dispatch`` / ``forecast``)
Modelled   : reserve capacity prices, intraday recovery, imbalance (all parameters in ``config`` with provenance)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import dispatch, synth
from .config import (ASSETS, AVAILABILITY, DEGRADATION_EUR_MWH, DT_H, FEE_DA_EUR_MWH, FEE_ID_EUR_MWH, FEE_TARIFF_AFTER,
                     ID_EXTRA_VALUE, ID_RECOVERY_RHO, ID_VOLUME_SHARE, IMBALANCE_VOL_RATE, RANDOM_SEED, RESERVE_PRICE_END,
                     RESERVE_PRICE_NOISE, RESERVE_PRICE_START, TARIFF_CHANGE_DATE, TZ, Asset)
from .variance import Drivers, drivers_from_actuals


def variants(assets: list[Asset] | None = None) -> list[tuple[float, float]]:
    return sorted({(a.duration_h, a.grid_cap_frac) for a in (assets or ASSETS)})


def cached(path: Path, fn):
    """Parquet cache for expensive steps. Delete the file to recompute."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def dispatch_sets(price: pd.Series, price_forecast: pd.Series, cache_dir: Path, start_fc: str,
                  assets: list[Asset] | None = None) -> dict:
    """Per-MW schedules for every asset variant: perfect foresight (full history) and forecast-based (from start_fc)."""
    cache_dir = Path(cache_dir)
    out = {}
    fc_idx = price_forecast.index
    for dur, cap in variants(assets):
        tag = f"d{dur:g}_cap{cap:g}"
        pf = cached(cache_dir / f"sched_pf_{tag}.parquet", lambda: dispatch.run_dispatch(price, dur, cap))
        fc = cached(cache_dir / f"sched_fc_{tag}.parquet",
                    lambda: dispatch.run_dispatch(price_forecast, dur, cap))
        out[(dur, cap)] = {"pf": pf, "fc": fc.reindex(fc_idx)}
    return out


def daily_per_mw(sched: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Settled value per MW and local day."""
    st = dispatch.settle(sched, price.reindex(sched.index))
    d = st.groupby(st.index.tz_convert(TZ).tz_localize(None).normalize()).sum()
    return d


def reserve_price_path(months: pd.DatetimeIndex, seed: int = RANDOM_SEED) -> pd.Series:
    """Modelled FCR / aFRR capacity price in EUR per MW and hour (saturation: falling path with AR(1) noise)."""
    rng = np.random.default_rng(seed + 7)
    t0, t1 = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-12-01")
    frac = np.clip(np.asarray((months - t0) / (t1 - t0), dtype=float), 0, 1)
    trend = np.exp(np.log(RESERVE_PRICE_START) + (np.log(RESERVE_PRICE_END) - np.log(RESERVE_PRICE_START)) * frac)
    e = np.zeros(len(months))
    for i in range(1, len(months)):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, RESERVE_PRICE_NOISE)
    return pd.Series(trend * np.exp(e), index=months, name="reserve_eur_mw_h")


def simulate_imbalance(net_mwh: pd.Series, rebap_final: pd.Series, seed: int) -> pd.DataFrame:
    """Monthly-level imbalance: random metering noise plus unplanned outage windows (schedule - meter)."""
    rng = np.random.default_rng(seed)
    n = len(net_mwh)
    f = net_mwh.to_numpy()
    dev = f * rng.normal(0, IMBALANCE_VOL_RATE, n)
    out = np.zeros(n, dtype=bool)
    n_events = int(n / 96 * 30 / 365 * 1.0) + 1                 # ~30 events per year, 2-14 h each
    starts = rng.choice(max(n - 60, 1), n_events, replace=False)
    for s in starts:
        out[s:s + int(rng.integers(8, 56))] = True
    short = np.where(out, f, dev)                                # F - M: outage -> whole schedule undelivered
    cost = short * rebap_final.reindex(net_mwh.index).to_numpy()
    return pd.DataFrame({"imb_mwh": np.abs(short), "imb_cost": cost}, index=net_mwh.index)


def asset_monthly(asset: Asset, sets: dict, price: pd.Series, rebap: pd.DataFrame, reserve_px: pd.Series,
                  start: str, end: str) -> pd.DataFrame:
    """Monthly realised lines and drivers for one asset between ``start`` and ``end`` (local dates)."""
    key = (asset.duration_h, asset.grid_cap_frac)
    pf, fc = sets[key]["pf"], sets[key]["fc"].dropna()
    fc = fc[(fc.index >= pd.Timestamp(start, tz=TZ)) & (fc.index < pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1))]
    px = price.reindex(fc.index)
    day_pf = daily_per_mw(pf.reindex(fc.index), px)
    day_fc = daily_per_mw(fc, px)
    cod = pd.Timestamp(asset.cod_actual)
    online = pd.Series((day_fc.index >= cod).astype(float), index=day_fc.index)

    arb_mw_day = asset.mw * (1 - asset.reserve_share) * online
    res_mw_day = asset.mw * asset.reserve_share * online
    mw_day = asset.mw * online

    gross_da = day_fc["gross_eur"] * arb_mw_day
    dis = day_fc["discharge_mwh"] * arb_mw_day
    chg = day_fc["charge_mwh"] * arb_mw_day
    pf_gross = day_pf["gross_eur"] * arb_mw_day
    pf_dis = day_pf["discharge_mwh"] * arb_mw_day
    gap = (day_pf["net_eur"] - day_fc["net_eur"]).clip(lower=0) * arb_mw_day
    id_rev = ID_RECOVERY_RHO * gap + ID_EXTRA_VALUE * day_pf["net_eur"].clip(lower=0) * arb_mw_day
    traded_da = dis + chg
    traded_id = ID_VOLUME_SHARE * traded_da
    fee_da = pd.Series(np.where(day_fc.index >= pd.Timestamp(TARIFF_CHANGE_DATE), FEE_TARIFF_AFTER, FEE_DA_EUR_MWH),
                       index=day_fc.index)
    fees_da = traded_da * fee_da
    fees_id = traded_id * FEE_ID_EUR_MWH
    fees = fees_da + fees_id
    deg = dis * DEGRADATION_EUR_MWH
    # reserve capacity
    month_of_day = day_fc.index.to_period("M").to_timestamp()
    rp = reserve_px.reindex(month_of_day).to_numpy()
    res_rev = res_mw_day.to_numpy() * 24.0 * AVAILABILITY * rp

    # imbalance on the scheduled net energy of this asset
    net_sched = ((fc["discharge_mw"] - fc["charge_mw"]) * DT_H) * asset.mw * (1 - asset.reserve_share)
    net_sched = net_sched * pd.Series((fc.index.tz_convert(TZ).tz_localize(None).normalize() >= cod).astype(float), index=fc.index)
    imb = simulate_imbalance(net_sched, rebap["rebap_final"], seed=RANDOM_SEED + sum(map(ord, asset.asset_id)))
    imb_d = imb.groupby(imb.index.tz_convert(TZ).tz_localize(None).normalize()).sum()

    day = pd.DataFrame({
        "mw": mw_day, "mw_res": res_mw_day, "dis_mwh": dis, "chg_mwh": chg, "gross_da": gross_da, "id_rev": id_rev,
        "res_rev": res_rev, "pf_gross": pf_gross, "pf_dis": pf_dis, "traded_mwh": traded_da + traded_id,
        "traded_da": traded_da, "traded_id": traded_id, "fees_da": fees_da, "fees_id": fees_id,
        "fees": fees, "deg": deg, "imb_mwh": imb_d["imb_mwh"], "imb_cost": imb_d["imb_cost"],
        "days_online": online,
    })
    m = day.groupby(day.index.to_period("M")).agg(
        {**{c: "sum" for c in day.columns if c not in ("mw", "mw_res")}, "mw": "mean", "mw_res": "mean"})
    m.index = m.index.to_timestamp()
    m["arb_rev"] = m["gross_da"] + m["id_rev"]
    m["net_rev"] = m["arb_rev"] + m["res_rev"] - m["fees"] - m["deg"] - m["imb_cost"]
    m["asset_id"] = asset.asset_id
    m["res_share"] = asset.reserve_share
    return m.reset_index(names="month")


def drivers_row(r: pd.Series) -> Drivers:
    return drivers_from_actuals(mw=r["mw"], res_share=r["res_share"], dis_mwh=r["dis_mwh"], arb_rev=r["arb_rev"],
                                pf_gross=r["pf_gross"], pf_dis=r["pf_dis"], res_rev=r["res_rev"],
                                traded_mwh=r["traded_mwh"], fees=r["fees"], deg=r["deg"], imb_mwh=r["imb_mwh"],
                                imb_cost=r["imb_cost"])


def full_rebap(price: pd.Series, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Modelled imbalance price for the whole history (indexed by UTC timestamp)."""
    return synth.modelled_rebap(price, np.random.default_rng(seed + 11))


def portfolio_monthly(sets: dict, price: pd.Series, rebap: pd.DataFrame, start: str, end: str,
                      assets: list[Asset] | None = None) -> pd.DataFrame:
    months = pd.date_range("2019-01-01", "2027-12-01", freq="MS")
    reserve_px = reserve_price_path(months)
    parts = [asset_monthly(a, sets, price, rebap, reserve_px, start, end)
             for a in (assets or [x for x in ASSETS if x.cod_actual])]
    out = pd.concat(parts, ignore_index=True)
    return out[out["days_online"] > 0].reset_index(drop=True)
