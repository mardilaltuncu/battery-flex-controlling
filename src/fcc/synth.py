"""Synthetic operational records for the reconciliation exercise.

Real input  : day-ahead clearing prices and the dispatch an optimiser would have scheduled from a *forecast*.
Synthetic   : the four record sets a controller reconciles - trading ledger (the "CSV from Data Science"), the exchange /
              CCP statement, the schedule notified to the TSO (Fahrplan) and the meter - plus a modelled imbalance price.

Known breaks are injected with a ground-truth log, so the reconciliation engine can be scored on what it catches.
Nothing here describes a real company; all counterparties and volumes are illustrative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (ASSETS, DT_H, FEE_DA_EUR_MWH, FEE_TARIFF_AFTER, IMBALANCE_VOL_RATE, TARIFF_CHANGE_DATE, TZ, Asset)
from .market import local_day_table


@dataclass
class OpsData:
    ledger: pd.DataFrame           # asset_id, ts_utc, net_mwh (sell +), price_eur_mwh, amount_eur
    ecc: pd.DataFrame              # statement rows as delivered (local time, buy / sell columns)
    fahrplan: pd.DataFrame         # asset_id, ts_utc, schedule_mwh
    meter: pd.DataFrame            # asset_id, ts_utc, meter_mwh
    rebap: pd.DataFrame            # ts_utc, rebap_prelim, rebap_final   (modelled)
    truth: pd.DataFrame            # injected break log (event level)
    truth_intervals: pd.DataFrame  # injected break log (interval level)
    window: tuple[pd.Timestamp, pd.Timestamp]


# --------------------------------------------------------------------------------------------------
def _window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cal = local_day_table(start, (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return cal["start_utc"].iloc[0], cal["start_utc"].iloc[-1]


def modelled_rebap(price: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Imbalance price = day-ahead price + heavy-tailed, sign-symmetric premium. Final differs slightly from prelim."""
    n = len(price)
    scale = 0.12 * np.maximum(price.abs().to_numpy(), 40.0)
    prem = rng.standard_t(3, n) * scale * rng.choice([-1.0, 1.0], n)
    spike = rng.random(n) < 0.002
    prem = np.where(spike, np.sign(prem) * np.abs(prem) * rng.uniform(3, 8, n), prem)
    prelim = price.to_numpy() + prem
    final = prelim * (1 + rng.normal(0, 0.01, n)) + 0.8          # small systematic upward revision
    return pd.DataFrame({"rebap_prelim": prelim.round(2), "rebap_final": final.round(2)}, index=price.index)


def _clean_ledger(asset: Asset, sched_mw: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    arb_mw = asset.mw * (1 - asset.reserve_share)
    net = ((sched_mw["discharge_mw"] - sched_mw["charge_mw"]) * arb_mw * DT_H).round(6)
    df = pd.DataFrame({"net_mwh": net, "price_eur_mwh": price.reindex(net.index)})
    df = df[df["net_mwh"].abs() > 1e-9].copy()
    df["amount_eur"] = df["net_mwh"] * df["price_eur_mwh"]
    return df


def _ecc_rows(asset_id: str, clean: pd.DataFrame, truncate: bool, rng: np.random.Generator) -> pd.DataFrame:
    """Render ledger rows the way an exchange statement shows them: local clock time, separate buy / sell columns."""
    ts = clean.index
    loc = ts.tz_convert(TZ)
    ldate = pd.Series(loc.strftime("%Y-%m-%d"), index=ts)
    ltime = pd.Series(loc.strftime("%H:%M"), index=ts)
    # repeated hour on the 'clock back' day: the wall-clock time exists twice -> A (summer time) / B (winter time)
    naive = loc.tz_localize(None)
    amb = naive.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").isna()
    offset_h = (naive - ts.tz_convert("UTC").tz_localize(None)) / pd.Timedelta(hours=1)
    flag = np.where(amb, np.where(offset_h == 2, "A", "B"), "")
    vol = clean["net_mwh"].abs()
    vol = np.trunc(vol * 1000) / 1000 if truncate else vol.round(3)       # statements show 3 decimals
    sell = np.where(clean["net_mwh"] > 0, vol, 0.0)
    buy = np.where(clean["net_mwh"] < 0, vol, 0.0)
    price = clean["price_eur_mwh"].round(2)
    tariff = np.where(ts >= pd.Timestamp(TARIFF_CHANGE_DATE, tz=TZ).tz_convert("UTC"), FEE_TARIFF_AFTER, FEE_DA_EUR_MWH)
    out = pd.DataFrame({
        "asset_id": asset_id, "delivery_date": ldate.to_numpy(), "time_local": ltime.to_numpy(), "dst_flag": flag,
        "sell_mwh": sell, "buy_mwh": buy, "clearing_price": price.to_numpy(),
        "amount_eur": ((sell - buy) * price.to_numpy()).round(2),
        "fee_eur": ((sell + buy) * tariff).round(4),
    }, index=ts)
    out.index.name = "_ts"
    return out


# --------------------------------------------------------------------------------------------------
def build_ops(price: pd.Series, sched_by_variant: dict, start: str, end: str, seed: int = 42,
              assets: list[Asset] | None = None) -> OpsData:
    """Create the four record sets plus the ground-truth log of injected breaks."""
    assets = assets or [a for a in ASSETS if a.cod_actual]
    rng = np.random.default_rng(seed)
    t0, t1 = _window(start, end)
    px = price[(price.index >= t0) & (price.index < t1)]
    idx = px.index
    ldays = pd.Series(idx.tz_convert(TZ).strftime("%Y-%m-%d"), index=idx)
    days_all = sorted(ldays.unique())
    regular_days = [d for d in days_all if d not in ("2025-10-26", "2026-03-29")]

    ledger, ecc, fahr, meter = {}, {}, {}, {}
    truth, tint = [], []

    def log(typ, asset_id, day, ts_list, desc, owner):
        ts_list = pd.DatetimeIndex(ts_list)
        truth.append({"break_id": f"INJ-{len(truth) + 1:03d}", "type": typ, "asset_id": asset_id, "local_date": day,
                      "n_intervals": len(ts_list), "first_utc": ts_list.min() if len(ts_list) else pd.NaT,
                      "description": desc, "expected_owner": owner})
        for t in ts_list:
            tint.append((asset_id, t, typ))

    def pick(clean_idx: pd.DatetimeIndex, day: str, n: int, block: bool) -> pd.DatetimeIndex:
        sel = clean_idx[ldays.reindex(clean_idx).to_numpy() == day]
        if len(sel) <= n:
            return sel
        if block:
            s = int(rng.integers(0, len(sel) - n))
            return sel[s:s + n]
        return pd.DatetimeIndex(np.sort(rng.choice(sel, n, replace=False)))

    def draw_days(k):
        return list(rng.choice(regular_days, k, replace=False))

    for a in assets:
        key = (a.duration_h, a.grid_cap_frac)
        sched = sched_by_variant[key].reindex(idx).fillna(0.0)
        clean = _clean_ledger(a, sched, px)
        led = clean.copy()
        led.insert(0, "asset_id", a.asset_id)
        ledger[a.asset_id] = led
        trunc = a.asset_id == "BESS-A"
        ecc[a.asset_id] = _ecc_rows(a.asset_id, clean, trunc, rng)
        arb_mw = a.mw * (1 - a.reserve_share)
        fp = clean["net_mwh"].reindex(idx).fillna(0.0)
        fahr[a.asset_id] = fp.copy()
        noise = rng.normal(0, IMBALANCE_VOL_RATE, len(idx))
        meter[a.asset_id] = fp * (1 + noise)

        # genuine unplanned outages: meter shows nothing while the schedule says otherwise (asset-caused imbalance)
        n_out = int(rng.poisson(30))
        starts = rng.choice(len(idx) - 60, n_out, replace=False)
        outage_mask = np.zeros(len(idx), dtype=bool)
        for s in starts:
            outage_mask[s:s + int(rng.integers(8, 56))] = True
        m = meter[a.asset_id].to_numpy().copy()
        m[outage_mask] = 0.0
        meter[a.asset_id] = pd.Series(m, index=idx)
        hit = idx[outage_mask & (fp.abs().to_numpy() > 0)]
        if len(hit):
            log("ASSET_OUTAGE", a.asset_id, "multiple", hit, "unplanned outage windows (meter 0 vs schedule)", "asset / customer")

    # ----------------------------------------------------------------------------------- injected breaks
    # BESS-A: MISSING_IN_ECC (3 days, 2h block), DUPLICATE_ECC (1 day), ROUNDING_BIAS (all year), NOTIFICATION_ERROR
    a_id = "BESS-A"
    for day in draw_days(3):
        blk = pick(ecc[a_id].index, day, 8, True)
        ecc[a_id] = ecc[a_id].drop(blk)
        log("MISSING_IN_ECC", a_id, day, blk, "statement truncated / late file: 2h block absent", "exchange ops / data feed")
    day = draw_days(1)[0]
    blk = pick(ecc[a_id].index, day, 12, False)
    ecc[a_id] = pd.concat([ecc[a_id], ecc[a_id].loc[blk]]).sort_index(kind="stable")
    log("DUPLICATE_ECC", a_id, day, blk, "statement re-sent: rows appear twice", "exchange ops / data feed")
    log("ROUNDING_BIAS", a_id, "all", ledger[a_id].index, "statement truncates (not rounds) volume to 3 decimals", "exchange ops / data feed")
    for day in draw_days(2):
        blk = pick(ledger[a_id].index, day, 8, True)
        fahr[a_id] = fahr[a_id].copy()
        mtr = meter[a_id].copy()
        mtr.loc[blk] = ledger[a_id].loc[blk, "net_mwh"]              # physical flow followed the real trade
        meter[a_id] = mtr
        fahr[a_id].loc[blk] = -ledger[a_id].loc[blk, "net_mwh"]       # notified with the wrong sign
        log("NOTIFICATION_ERROR", a_id, day, blk, "schedule notified to TSO with reversed sign", "optimizer (process error)")

    # BESS-B: UNIT_X4 (2 days, ledger in MW), MISSING_IN_LEDGER (1 day), PRICE_DIFF (1 day), NOT_NOTIFIED (4 days), DST flag lost
    b_id = "BESS-B"
    for day in draw_days(2):
        blk = ledger[b_id].index[ldays.reindex(ledger[b_id].index).to_numpy() == day]
        ledger[b_id].loc[blk, ["net_mwh", "amount_eur"]] *= 4
        log("UNIT_X4", b_id, day, blk, "ledger exported in MW instead of MWh (15-min interval -> x4)", "data science export")
    day = draw_days(1)[0]
    blk = pick(ledger[b_id].index, day, 6, True)
    ledger[b_id] = ledger[b_id].drop(blk)
    log("MISSING_IN_LEDGER", b_id, day, blk, "trades not booked in the ledger export", "data science export")
    day = draw_days(1)[0]
    blk = pick(ecc[b_id].index, day, 18, False)
    delta = rng.choice([-1, 1], len(blk)) * rng.uniform(8, 120, len(blk))
    ecc[b_id].loc[blk, "clearing_price"] = (ecc[b_id].loc[blk, "clearing_price"] + delta).round(2)
    ecc[b_id].loc[blk, "amount_eur"] = ((ecc[b_id].loc[blk, "sell_mwh"] - ecc[b_id].loc[blk, "buy_mwh"])
                                        * ecc[b_id].loc[blk, "clearing_price"]).round(2)
    log("PRICE_DIFF", b_id, day, blk, "corrected clearing price on the statement after first publication", "exchange / pricing")
    for day in draw_days(4):
        blk = pick(ledger[b_id].index, day, 6, True)
        fahr[b_id] = fahr[b_id].copy()
        mtr = meter[b_id].copy()
        fahr[b_id].loc[blk] = 0.0
        mtr.loc[blk] = 0.0
        meter[b_id] = mtr
        log("NOT_NOTIFIED", b_id, day, blk, "trade never notified to TSO and asset idle: schedule = meter = 0", "optimizer (process error)")
    flagless = ecc[b_id].index[(ldays.reindex(ecc[b_id].index).to_numpy() == "2025-10-26")
                               & (ecc[b_id]["dst_flag"] != "").to_numpy()]
    ecc[b_id].loc[flagless, "dst_flag"] = ""
    log("DST_FLAG_MISSING", b_id, "2025-10-26", flagless, "export dropped the A/B marker of the repeated hour", "exchange ops / data feed")

    # BESS-C: SIGN_FLIP (1 day), VOLUME_DIFF partial fills (3 days), METER_BIAS (all year)
    c_id = "BESS-C"
    day = draw_days(1)[0]
    blk = ecc[c_id].index[ldays.reindex(ecc[c_id].index).to_numpy() == day]
    tmp = ecc[c_id].loc[blk, "sell_mwh"].copy()
    ecc[c_id].loc[blk, "sell_mwh"] = ecc[c_id].loc[blk, "buy_mwh"]
    ecc[c_id].loc[blk, "buy_mwh"] = tmp
    ecc[c_id].loc[blk, "amount_eur"] = ((ecc[c_id].loc[blk, "sell_mwh"] - ecc[c_id].loc[blk, "buy_mwh"])
                                        * ecc[c_id].loc[blk, "clearing_price"]).round(2)
    log("SIGN_FLIP", c_id, day, blk, "buy / sell columns swapped for the whole day", "exchange ops / data feed")
    for day in draw_days(3):
        blk = pick(ecc[c_id].index, day, 8, False)
        f = rng.uniform(0.4, 0.6, len(blk))
        for col in ["sell_mwh", "buy_mwh"]:
            ecc[c_id].loc[blk, col] = (ecc[c_id].loc[blk, col] * f).round(3)
        ecc[c_id].loc[blk, "amount_eur"] = ((ecc[c_id].loc[blk, "sell_mwh"] - ecc[c_id].loc[blk, "buy_mwh"])
                                            * ecc[c_id].loc[blk, "clearing_price"]).round(2)
        log("VOLUME_DIFF", c_id, day, blk, "order only partly matched; ledger booked the full order", "trading desk")
    meter[c_id] = meter[c_id] * 0.997
    log("METER_BIAS", c_id, "all", fahr[c_id].index[fahr[c_id].abs() > 0], "meter reads 0.3 % low (calibration)", "asset / metering")

    log("TARIFF_CHANGE", "all", f"from {TARIFF_CHANGE_DATE}", pd.DatetimeIndex([]),
        f"clearing fee {FEE_DA_EUR_MWH:.2f} -> {FEE_TARIFF_AFTER:.2f} EUR/MWh; budget kept the old tariff", "exchange tariff")

    # ----------------------------------------------------------------------------------- shape outputs
    led_all = pd.concat([ledger[a.asset_id] for a in assets]).rename_axis("ts_utc").reset_index()
    # statements are delivered in chronological order per account; the timestamp itself is NOT part of the file
    ecc_all = pd.concat([ecc[a.asset_id].sort_index(kind="stable") for a in assets]).reset_index(drop=True)
    fah_all = pd.concat([pd.DataFrame({"asset_id": a.asset_id, "ts_utc": fahr[a.asset_id].index,
                                       "schedule_mwh": fahr[a.asset_id].to_numpy()}) for a in assets], ignore_index=True)
    met_all = pd.concat([pd.DataFrame({"asset_id": a.asset_id, "ts_utc": meter[a.asset_id].index,
                                       "meter_mwh": meter[a.asset_id].to_numpy()}) for a in assets], ignore_index=True)
    fah_all = fah_all[fah_all["schedule_mwh"].abs() > 1e-9].reset_index(drop=True)
    rebap = modelled_rebap(px, rng).rename_axis("ts_utc").reset_index()
    truth_df = pd.DataFrame(truth)
    tint_df = pd.DataFrame(tint, columns=["asset_id", "ts_utc", "type"])
    return OpsData(led_all, ecc_all, fah_all, met_all, rebap, truth_df, tint_df, (t0, t1))
