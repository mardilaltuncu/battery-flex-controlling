"""Market data: loading, a 15-minute UTC grid, DST-aware day structure and data-quality audits."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter

from .config import DT_H, TZ


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------
def load_price(path: str | Path) -> pd.DataFrame:
    """Day-ahead price on a gap-free 15-minute UTC grid.

    Until 30 Sep 2025 the day-ahead auction cleared hourly products, so every hourly price is applied to its four
    quarter-hours (``native_res_min == 60``). From 1 Oct 2025 prices are cleared per quarter-hour.
    """
    raw = pd.read_parquet(path).sort_values("ts_utc").reset_index(drop=True)
    raw["ts_utc"] = pd.to_datetime(raw["ts_utc"], utc=True)
    hourly = raw["resolution_min"] == 60
    # unknown resolution (last row of a file or a gap) -> infer from the neighbours
    res = raw["resolution_min"].copy()
    res = res.fillna(res.ffill()).fillna(15)
    raw["native_res_min"] = res.astype(int)
    reps = (raw["native_res_min"] // 15).to_numpy()
    idx = np.repeat(np.arange(len(raw)), reps)
    offs = np.concatenate([np.arange(r) for r in reps]) * 15
    grid = raw.iloc[idx].copy()
    grid["ts_utc"] = grid["ts_utc"] + pd.to_timedelta(offs, unit="m").to_numpy()
    grid = grid.drop(columns=["resolution_min"]).set_index("ts_utc")
    grid = grid[~grid.index.duplicated(keep="first")]
    _ = hourly  # (kept for readability)
    return grid


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    f = pd.read_parquet(path)
    f["ts_utc"] = pd.to_datetime(f["ts_utc"], utc=True)
    f = f.set_index("ts_utc").sort_index()
    f = f[~f.index.duplicated(keep="first")]
    return f.resample("15min").mean()


# --------------------------------------------------------------------------------------------------
# Local-day structure (DST)
# --------------------------------------------------------------------------------------------------
def local_day_table(start: str, end: str) -> pd.DataFrame:
    """One row per local delivery day with the number of 15-minute intervals (92 / 96 / 100)."""
    days = pd.date_range(start, end, freq="D")
    s = days.tz_localize(TZ)                      # local midnight
    e = (days + pd.Timedelta(days=1)).tz_localize(TZ)
    n = ((e.tz_convert("UTC") - s.tz_convert("UTC")) / pd.Timedelta(minutes=15)).astype(int)
    return pd.DataFrame({"local_date": days.date, "start_utc": s.tz_convert("UTC"), "n_expected": np.asarray(n)})


def add_local_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add local date / time-of-day fields to a frame indexed by tz-aware UTC timestamps."""
    out = df.copy()
    loc = out.index.tz_convert(TZ)
    out["local_date"] = loc.date
    out["local_hour"] = loc.hour
    out["local_minute"] = loc.minute
    return out


def interval_count_check(index: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    """Compare delivered vs expected intervals per local day. Only DST days may differ from 96."""
    cal = local_day_table(start, end).set_index("local_date")
    loc = index.tz_convert(TZ)
    got = pd.Series(1, index=pd.Index(loc.date, name="local_date")).groupby(level=0).sum()
    cal["n_found"] = got.reindex(cal.index).fillna(0).astype(int)
    cal["gap"] = cal["n_found"] - cal["n_expected"]
    return cal


# --------------------------------------------------------------------------------------------------
# German public holidays (for the forecast calendar features)
# --------------------------------------------------------------------------------------------------
def de_holidays(years: range | list[int]) -> set[date]:
    out: set[date] = set()
    for y in years:
        e = easter(y)
        out |= {date(y, 1, 1), e - timedelta(days=2), e + timedelta(days=1), date(y, 5, 1), e + timedelta(days=39),
                e + timedelta(days=50), date(y, 10, 3), date(y, 12, 25), date(y, 12, 26)}
    return out


# --------------------------------------------------------------------------------------------------
# Data-quality audits
# --------------------------------------------------------------------------------------------------
def audit_price(p: pd.DataFrame, start: str, end: str) -> dict:
    """Headline integrity numbers for the day-ahead series."""
    cal = interval_count_check(p.index, start, end)
    dst = cal[cal["n_expected"] != 96]
    off = cal[(cal["n_expected"] == 96) & (cal["gap"] != 0)]
    return {
        "intervals": len(p),
        "first_utc": p.index.min(),
        "last_utc": p.index.max(),
        "duplicated_timestamps": int(p.index.duplicated().sum()),
        "missing_prices": int(p["price_eur_mwh"].isna().sum()),
        "local_days_checked": len(cal),
        "dst_days": len(dst),
        "dst_days_complete": int((dst["gap"] == 0).sum()),
        "regular_days_with_gap": len(off),
        "negative_price_intervals": int((p["price_eur_mwh"] < 0).sum()),
        "min_price": float(p["price_eur_mwh"].min()),
        "max_price": float(p["price_eur_mwh"].max()),
    }


def annual_stats(p: pd.DataFrame) -> pd.DataFrame:
    year = p.index.tz_convert(TZ).year                      # local calendar year
    g = p["price_eur_mwh"].groupby(year)
    out = pd.DataFrame({
        "mean": g.mean(), "std": g.std(), "min": g.min(), "max": g.max(),
        "negative_hours": (p["price_eur_mwh"] < 0).groupby(year).sum() * DT_H,
    }).round(2)
    out.index.name = "year"
    return out


def audit_secondary_source(hf_path: str | Path, ref: pd.DataFrame) -> pd.DataFrame:
    """Why a second, popular price file was NOT used: compare it with the reference per year.

    The file mixes two series (every timestamp appears twice around the hour / quarter-hour), contains no negative
    price before 2026 although Germany had hundreds of negative-price hours per year, and reports values that a
    day-ahead clearing price cannot take. We test this against the reference series instead of trusting a README.
    """
    hf = pd.read_parquet(hf_path)
    hf["ts"] = pd.to_datetime(hf["datetime_utc"], utc=True)
    hf["occ"] = hf.groupby("ts").cumcount()
    hf["year"] = hf["ts"].dt.year
    r = ref["price_eur_mwh"]
    rows = []
    for y, g in hf.groupby("year"):
        if y < 2019 or y > 2025:
            continue
        ry = r[r.index.year == y]
        rows.append({
            "year": y,
            "reference mean": ry.mean(),
            "file mean (all rows)": g["price_eur_mwh"].mean(),
            "rows per distinct timestamp": len(g) / g["ts"].nunique(),
            "reference negative hours": (ry < 0).sum() * DT_H,
            "file negative rows": int((g["price_eur_mwh"] < 0).sum()),
            "reference min": ry.min(),
            "file min": g["price_eur_mwh"].min(),
            "file max": g["price_eur_mwh"].max(),
            "reference max": ry.max(),
        })
    return pd.DataFrame(rows).round(2).set_index("year")
