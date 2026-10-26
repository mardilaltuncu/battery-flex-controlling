"""Interval-by-interval reconciliation.

Core (what the role asks for): internal MWh vs the exchange settlement statement, every 15-minute interval, every
number traceable to a source row. Extension: the four-way match (trading ledger, exchange / CCP statement, schedule
notified to the TSO, meter) that separates data errors, process errors and asset-caused imbalance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (DT_H, SYSTEMATIC_ONE_SIDED_SHARE, TOL_AMOUNT_EUR, TOL_PRICE_EUR_MWH, TOL_VOL_MWH, TZ)

OWNER = {
    "MISSING_IN_ECC": ("exchange ops / data feed", "ECC statement incomplete or late", "re-request statement"),
    "MISSING_IN_LEDGER": ("data science export", "trades absent from ledger export", "re-run export, check booking cut-off"),
    "DUPLICATE_ECC": ("exchange ops / data feed", "statement rows delivered twice", "de-duplicate on (account, interval)"),
    "UNIT_X4": ("data science export", "MW vs MWh confusion (15 min = factor 4)", "fix export unit, re-load day"),
    "SIGN_FLIP": ("exchange ops / data feed", "buy / sell convention reversed", "map buy/sell columns, re-load day"),
    "PRICE_DIFF": ("exchange / pricing", "clearing price corrected after first publication", "take final price, true-up"),
    "VOLUME_DIFF": ("trading desk", "partial fill / cancellation not reflected in ledger", "book executed volume"),
    "ROUNDING_BIAS": ("exchange ops / data feed", "systematic truncation instead of rounding", "agree rounding rule, adjust"),
    "DST_FLAG_MISSING": ("exchange ops / data feed", "A/B marker of the repeated hour missing", "infer by sequence, ask for fixed export"),
    "NOT_NOTIFIED": ("optimizer (process error)", "trade missing from the schedule notified to the TSO", "notify schedule, avoidable imbalance"),
    "NOTIFICATION_ERROR": ("optimizer (process error)", "schedule differs from trade (e.g. sign)", "correct schedule, avoidable imbalance"),
    "ASSET_OUTAGE": ("asset / customer", "unplanned outage: meter below schedule", "availability KPI; imbalance to asset side"),
    "METER_BIAS": ("asset / metering", "systematic meter deviation vs schedule", "meter calibration check"),
    "UNEXPLAINED": ("unassigned", "difference without a known signature", "manual deep-dive"),
}


# --------------------------------------------------------------------------------------------------
# Parsing: exchange statements speak local time, we compute in UTC
# --------------------------------------------------------------------------------------------------
def parse_ecc(ecc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert statement rows to UTC. Repeated hour on the 'clock back' day uses the A/B marker; a missing marker is
    inferred from row order and reported."""
    e = ecc.copy()
    e["row_id"] = np.arange(len(e))
    naive = pd.to_datetime(e["delivery_date"] + " " + e["time_local"])
    amb = naive.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    is_amb = amb.isna()
    key = e["asset_id"] + "|" + e["delivery_date"] + "|" + e["time_local"]
    seq = e.groupby(key).cumcount()
    flag = e["dst_flag"].fillna("")
    use_dst = np.where(flag == "A", True, np.where(flag == "B", False, seq.to_numpy() == 0))
    notes = {"ambiguous_rows": int(is_amb.sum()), "ambiguous_without_marker": int((is_amb & (flag == "")).sum()),
             "nonexistent_local_times": 0}
    ts = pd.Series(pd.NaT, index=e.index, dtype="datetime64[ns, UTC]")
    ok = ~is_amb
    ts[ok] = amb[ok].dt.tz_convert("UTC")
    if is_amb.any():
        sub = naive[is_amb]
        res = [t.tz_localize(TZ, ambiguous=bool(d)) for t, d in zip(sub, use_dst[is_amb.to_numpy()])]
        ts[is_amb] = pd.DatetimeIndex(res).tz_convert("UTC")
    e["ts_utc"] = ts
    e["inferred_dst_marker"] = is_amb & (flag == "")
    return e, notes


def naive_local_join_stats(ledger: pd.DataFrame, ecc: pd.DataFrame) -> dict:
    """What happens if both sides are joined on naive local clock time (the classic mistake)."""
    led = ledger.copy()
    led["clock"] = led["ts_utc"].dt.tz_convert(TZ).dt.tz_localize(None)
    e = ecc.copy()
    e["clock"] = pd.to_datetime(e["delivery_date"] + " " + e["time_local"])
    j = led.merge(e, on=["asset_id", "clock"], how="outer", indicator=True)
    fan_out = j.groupby(["asset_id", "clock"]).size()
    return {
        "rows_joined": len(j),
        "keys_with_fan_out (>1 match)": int((fan_out > 1).sum()),
        "one_sided_rows": int((j["_merge"] != "both").sum()),
    }


# --------------------------------------------------------------------------------------------------
# Two-way core: ledger vs exchange statement
# --------------------------------------------------------------------------------------------------
def reconcile_two_way(ledger: pd.DataFrame, ecc_parsed: pd.DataFrame, tol_vol: float = TOL_VOL_MWH,
                      tol_amt: float = TOL_AMOUNT_EUR, tol_price: float = TOL_PRICE_EUR_MWH) -> pd.DataFrame:
    """One row per (asset, interval) with differences and a rule-based classification."""
    e = ecc_parsed.copy()
    e["net_ecc"] = e["sell_mwh"] - e["buy_mwh"]
    cnt = e.groupby(["asset_id", "ts_utc"]).size().rename("ecc_rows")
    e1 = e.drop_duplicates(["asset_id", "ts_utc"], keep="first")[
        ["asset_id", "ts_utc", "net_ecc", "clearing_price", "amount_eur", "fee_eur", "row_id", "inferred_dst_marker"]]
    e1 = e1.merge(cnt.reset_index(), on=["asset_id", "ts_utc"])
    l = ledger.rename(columns={"net_mwh": "net_led", "price_eur_mwh": "price_led", "amount_eur": "amount_led"})
    j = l.merge(e1, on=["asset_id", "ts_utc"], how="outer", indicator=True)
    j["net_led"] = j["net_led"].fillna(0.0)
    j["net_ecc"] = j["net_ecc"].fillna(0.0)
    j["amount_led"] = j["amount_led"].fillna(0.0)
    j["amount_ecc"] = j["amount_eur"].fillna(0.0)
    j["ecc_rows"] = j["ecc_rows"].fillna(0).astype(int)
    j["diff_mwh"] = j["net_ecc"] - j["net_led"]
    j["diff_eur"] = j["amount_ecc"] - j["amount_led"]
    j["diff_price"] = j["clearing_price"] - j["price_led"]
    ratio = np.divide(j["net_ecc"], j["net_led"], out=np.full(len(j), np.nan), where=j["net_led"].abs() > 1e-9)
    mag_diff = j["net_ecc"].abs() - j["net_led"].abs()
    j["mag_diff_mwh"] = mag_diff

    typ = np.full(len(j), "", dtype=object)
    both = (j["_merge"] == "both").to_numpy()
    typ[(j["_merge"] == "left_only").to_numpy()] = "MISSING_IN_ECC"
    typ[(j["_merge"] == "right_only").to_numpy()] = "MISSING_IN_LEDGER"
    dup = both & (j["ecc_rows"].to_numpy() > 1)
    typ[dup] = "DUPLICATE_ECC"
    # amount tolerance scales with price: a volume difference inside tolerance may cost more than 0.50 EUR at 700 EUR/MWh
    amt_tol = np.maximum(tol_amt, tol_vol * j["price_led"].abs().fillna(0).to_numpy() + 0.01)
    within = both & ~dup & (j["diff_mwh"].abs().to_numpy() <= tol_vol) & \
        (j["diff_price"].abs().fillna(0).to_numpy() <= tol_price) & (j["diff_eur"].abs().to_numpy() <= amt_tol)
    typ[within] = "CLEAR"
    rest = both & ~dup & ~within
    r = np.nan_to_num(ratio, nan=0.0)
    unit = rest & ((np.abs(r - 0.25) < 0.01) | (np.abs(r - 4.0) < 0.02))
    sign = rest & ~unit & (np.abs(r + 1.0) < 0.01)
    price = rest & ~unit & ~sign & (j["diff_mwh"].abs().to_numpy() <= tol_vol) & (j["diff_price"].abs().to_numpy() > tol_price)
    vol = rest & ~unit & ~sign & ~price & (j["diff_mwh"].abs().to_numpy() > tol_vol)
    typ[unit], typ[sign], typ[price], typ[vol] = "UNIT_X4", "SIGN_FLIP", "PRICE_DIFF", "VOLUME_DIFF"
    typ[rest & (typ == "")] = "UNEXPLAINED"
    j["status"] = typ
    # duplicated rows: exposure is the extra row(s), not a price / volume difference
    j.loc[dup, "diff_eur"] = j.loc[dup, "amount_ecc"] * (j.loc[dup, "ecc_rows"] - 1)
    j.loc[dup, "diff_mwh"] = j.loc[dup, "net_ecc"] * (j.loc[dup, "ecc_rows"] - 1)

    # systematic small differences that always point the same way (checked per asset and month)
    j["local_month"] = j["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m")
    tiny = j[(j["status"] == "CLEAR") & (j["mag_diff_mwh"].abs() > 1e-9)]
    sys_flags = []
    for (asset, month), g in tiny.groupby(["asset_id", "local_month"]):
        share = float((g["mag_diff_mwh"] < 0).mean())
        one_sided = max(share, 1 - share)
        if len(g) >= 50 and one_sided >= SYSTEMATIC_ONE_SIDED_SHARE:
            sys_flags.append((asset, month, len(g), one_sided, g["mag_diff_mwh"].sum()))
    j.attrs["systematic"] = pd.DataFrame(sys_flags, columns=["asset_id", "local_month", "n_small_diffs", "one_sided_share",
                                                             "net_mag_diff_mwh"])
    if len(sys_flags):
        keys = {(a, m) for a, m, *_ in sys_flags}
        mask = (j["status"] == "CLEAR") & (j["mag_diff_mwh"].abs() > 1e-9) & \
               pd.Series([(a, m) in keys for a, m in zip(j["asset_id"], j["local_month"])], index=j.index)
        j.loc[mask, "status"] = "ROUNDING_BIAS"
    j["local_date"] = j["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    return j.sort_values(["asset_id", "ts_utc"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Break register
# --------------------------------------------------------------------------------------------------
MONTHLY_FINDINGS = ("ROUNDING_BIAS", "METER_BIAS")


def break_register(rec: pd.DataFrame, as_of: str | None = None, eur_col: str = "diff_eur", mwh_col: str = "diff_mwh",
                   skip: tuple = ("CLEAR", "OK")) -> pd.DataFrame:
    """Group flagged intervals into break events (asset x day x type) with owner, root cause and EUR impact.
    Systematic findings are one break per asset and month, not hundreds of tiny daily items."""
    r = rec[~rec["status"].isin(skip)].copy()
    if r.empty:
        return pd.DataFrame()
    month = r["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m")
    r["local_date"] = np.where(r["status"].isin(MONTHLY_FINDINGS), month, r["local_date"])
    g = r.groupby(["asset_id", "local_date", "status"])
    reg = g.agg(n_intervals=("ts_utc", "size"), first_utc=("ts_utc", "min"), last_utc=("ts_utc", "max"),
                mwh_diff=(mwh_col, "sum"), eur_impact=(eur_col, "sum")).reset_index().rename(columns={"status": "type"})
    reg["abs_eur"] = reg["eur_impact"].abs()
    reg["owner"] = reg["type"].map(lambda t: OWNER.get(t, OWNER["UNEXPLAINED"])[0])
    reg["root_cause"] = reg["type"].map(lambda t: OWNER.get(t, OWNER["UNEXPLAINED"])[1])
    reg["next_step"] = reg["type"].map(lambda t: OWNER.get(t, OWNER["UNEXPLAINED"])[2])
    ref = pd.Timestamp(as_of or rec["ts_utc"].max().tz_convert(TZ).date(), tz=TZ)
    reg["age_days"] = (ref - reg["first_utc"].dt.tz_convert(TZ).dt.normalize()).dt.days
    reg["severity"] = pd.cut(reg["abs_eur"], [-1, 100, 2_500, 25_000, np.inf], labels=["low", "medium", "high", "critical"])
    reg = reg.sort_values("abs_eur", ascending=False).reset_index(drop=True)
    reg.insert(0, "break_id", [f"BRK-{i + 1:04d}" for i in range(len(reg))])
    reg["status"] = "open"
    return reg


def summary(rec: pd.DataFrame) -> dict:
    """Interval counts. Rounding-bias intervals are individually inside tolerance; they are one systematic finding."""
    n = len(rec)
    within = int(rec["status"].isin(["CLEAR", "ROUNDING_BIAS"]).sum())
    systematic = int((rec["status"] == "ROUNDING_BIAS").sum())
    return {
        "intervals_compared": n,
        "gross_volume_mwh": float(rec["net_led"].abs().sum()),
        "gross_amount_eur": float(rec["amount_led"].abs().sum()),
        "within_tolerance_%": 100 * within / n,
        "of which part of a systematic finding": systematic,
        "intervals outside tolerance (breaks)": n - within,
        "net_diff_eur": float(rec["diff_eur"].sum()),
        "abs_diff_eur": float(rec["diff_eur"].abs().sum()),
    }


# --------------------------------------------------------------------------------------------------
# Four-way match
# --------------------------------------------------------------------------------------------------
def reconcile_four_way(rec2: pd.DataFrame, fahrplan: pd.DataFrame, meter: pd.DataFrame, rebap: pd.DataFrame,
                       tol_abs: float = 0.01, tol_rel: float = 0.05) -> pd.DataFrame:
    """Add the schedule (Fahrplan) and the meter to the two-way result and classify what the first two cannot see."""
    base = rec2[["asset_id", "ts_utc", "net_led", "net_ecc", "amount_led", "price_led", "clearing_price"]].copy()
    f = fahrplan.rename(columns={"schedule_mwh": "fpl"})
    m = meter.rename(columns={"meter_mwh": "met"})
    keys = pd.concat([base[["asset_id", "ts_utc"]], f[["asset_id", "ts_utc"]], m[["asset_id", "ts_utc"]]]).drop_duplicates()
    x = keys.merge(base, on=["asset_id", "ts_utc"], how="left").merge(f, on=["asset_id", "ts_utc"], how="left") \
        .merge(m, on=["asset_id", "ts_utc"], how="left").merge(rebap, on="ts_utc", how="left")
    for c in ["net_led", "net_ecc", "amount_led", "fpl", "met"]:
        x[c] = x[c].fillna(0.0)
    x["price"] = x["price_led"].fillna(x["clearing_price"])
    x["local_date"] = x["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    x["local_month"] = x["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m")

    tol_f = np.maximum(tol_abs, tol_rel * x["fpl"].abs())
    led_vs_fpl = (x["net_led"] - x["fpl"]).abs() > np.maximum(tol_abs, tol_rel * x["net_led"].abs())
    fpl_vs_met = (x["fpl"] - x["met"]).abs() > tol_f
    met_zero = x["met"].abs() < 1e-9
    fpl_zero = x["fpl"].abs() < 1e-9
    met_follows_trade = (x["met"] - x["net_led"]).abs() <= np.maximum(tol_abs, tol_rel * x["net_led"].abs())

    cls = np.full(len(x), "OK", dtype=object)
    cls[(led_vs_fpl & fpl_zero & met_zero & (x["net_led"].abs() > 1e-9)).to_numpy()] = "NOT_NOTIFIED"
    cls[(led_vs_fpl & ~(fpl_zero & met_zero) & met_follows_trade).to_numpy()] = "NOTIFICATION_ERROR"
    cls[(~led_vs_fpl & fpl_vs_met & met_zero).to_numpy()] = "ASSET_OUTAGE"
    cls[(~led_vs_fpl & fpl_vs_met & ~met_zero).to_numpy()] = "METER_DEVIATION"
    x["status"] = cls
    x["imbalance_mwh"] = x["fpl"] - x["met"]                            # >0: short (delivered less than scheduled)
    x["imbalance_cost_eur"] = x["imbalance_mwh"] * x["rebap_prelim"]
    # exposure of a process error: the mismatched energy settles at the imbalance price instead of the traded price
    mism = np.where(x["status"] == "NOT_NOTIFIED", x["net_led"].abs(), (x["net_led"] - x["fpl"]).abs())
    x["exposure_eur"] = np.where(x["status"].isin(["NOT_NOTIFIED", "NOTIFICATION_ERROR"]),
                                 mism * (x["rebap_prelim"] - x["price"]).abs(), 0.0)

    # systematic meter deviation per asset-month (statistical, because every single interval is inside tolerance)
    ok = x[(x["status"] == "OK") & (x["fpl"].abs() > 1e-9)].copy()
    ok["rel"] = (ok["met"] - ok["fpl"]) / ok["fpl"].abs() * np.sign(ok["fpl"])
    sysrows = []
    for (a, mth), g in ok.groupby(["asset_id", "local_month"]):
        if len(g) < 100:
            continue
        mu, sd = g["rel"].mean(), g["rel"].std()
        z = mu / (sd / np.sqrt(len(g))) if sd > 0 else 0.0
        if abs(z) > 4 and abs(mu) > 0.001:
            sysrows.append({"asset_id": a, "local_month": mth, "mean_rel_dev_%": 100 * mu, "z": z, "n": len(g),
                            "energy_mwh": float((g["met"] - g["fpl"]).abs().sum())})
    x.attrs["meter_bias"] = pd.DataFrame(sysrows)
    if sysrows:
        flagged = {(r["asset_id"], r["local_month"]) for r in sysrows}
        mask = (x["status"] == "OK") & (x["fpl"].abs() > 1e-9) & \
               pd.Series([(a, m_) in flagged for a, m_ in zip(x["asset_id"], x["local_month"])], index=x.index)
        x.loc[mask, "status"] = "METER_BIAS"
    return x


def breaks_four_way(x: pd.DataFrame) -> pd.DataFrame:
    """Four-way break register. EUR = imbalance cost for asset-side items, exposure for process errors."""
    r = x[~x["status"].isin(["OK", "METER_DEVIATION"])].copy()
    proc = r["status"].isin(["NOT_NOTIFIED", "NOTIFICATION_ERROR"])
    r["eur"] = np.where(proc, r["exposure_eur"], r["imbalance_cost_eur"])
    r["mwh"] = np.where(r["status"] == "NOT_NOTIFIED", r["net_led"], r["imbalance_mwh"])
    return break_register(r.assign(diff_eur=r["eur"], diff_mwh=r["mwh"]), skip=())


# --------------------------------------------------------------------------------------------------
# Scoring against the injected truth
# --------------------------------------------------------------------------------------------------
def scorecard(truth_intervals: pd.DataFrame, rec2: pd.DataFrame, x4: pd.DataFrame) -> pd.DataFrame:
    """Interval-level recall / precision per injected break type. Bias types are scored as asset-level detections."""
    pred2 = rec2[["asset_id", "ts_utc", "status"]].rename(columns={"status": "pred"})
    pred4 = x4[["asset_id", "ts_utc", "status"]].rename(columns={"status": "pred"})
    rows = []
    two_way = {"MISSING_IN_ECC", "MISSING_IN_LEDGER", "DUPLICATE_ECC", "UNIT_X4", "SIGN_FLIP", "PRICE_DIFF", "VOLUME_DIFF"}
    four_way = {"NOT_NOTIFIED", "NOTIFICATION_ERROR", "ASSET_OUTAGE"}
    for typ, g in truth_intervals.groupby("type"):
        if typ in {"ROUNDING_BIAS", "METER_BIAS"}:
            pred = rec2 if typ == "ROUNDING_BIAS" else x4
            hit_assets = set(pred.loc[pred["status"] == typ, "asset_id"])
            truth_assets = set(g["asset_id"])
            rows.append({"type": typ, "injected": len(truth_assets), "unit": "asset", "caught": len(truth_assets & hit_assets),
                         "recall_%": 100 * len(truth_assets & hit_assets) / len(truth_assets),
                         "flagged_wrongly": len(hit_assets - truth_assets)})
            continue
        if typ == "DST_FLAG_MISSING":
            # success = the marker-less rows of the repeated hour still landed on the right UTC interval
            j = g.merge(rec2[["asset_id", "ts_utc", "status"]], on=["asset_id", "ts_utc"], how="left")
            ok = j["status"].isin(["CLEAR", "ROUNDING_BIAS"]).sum()
            rows.append({"type": typ, "injected": len(g), "unit": "interval", "caught": int(ok),
                         "recall_%": 100 * ok / len(g), "flagged_wrongly": 0})
            continue
        pred = pred2 if typ in two_way else pred4 if typ in four_way else None
        if pred is None:
            continue
        j = g.merge(pred, on=["asset_id", "ts_utc"], how="left")
        caught = int((j["pred"] == typ).sum())
        flagged_all = pred[pred["pred"] == typ]
        truth_keys = set(zip(g["asset_id"], g["ts_utc"]))
        wrong = sum((a, t) not in truth_keys for a, t in zip(flagged_all["asset_id"], flagged_all["ts_utc"]))
        rows.append({"type": typ, "injected": len(g), "unit": "interval", "caught": caught,
                     "recall_%": 100 * caught / len(g), "flagged_wrongly": int(wrong)})
    return pd.DataFrame(rows).sort_values("type").reset_index(drop=True)
