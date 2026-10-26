"""Pull DE-LU day-ahead prices and system fundamentals from the Energy-Charts API.

Source : Fraunhofer ISE Energy-Charts (https://api.energy-charts.info), data from
         Bundesnetzagentur | SMARD.de, licence CC BY 4.0.
Output : <out_dir>/price_de_lu.parquet         ts_utc, price_eur_mwh, resolution_min
         <out_dir>/fundamentals_de.parquet     ts_utc, load_mw, residual_load_mw, solar_mw, wind_onshore_mw, wind_offshore_mw

Each request chunk is cached under <out_dir>/_cache, so a re-run resumes where it stopped
(the API rate-limits with HTTP 429).

Usage  : py -P fetch_data.py <out_dir> [start=2019-01-01] [end=today-1]
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.energy-charts.info"
HEADERS = {"User-Agent": "flex-controlling-case/1.0 (analytics portfolio project)"}
PAUSE_S = 3.0
FUND_MAP = {
    "Load": "load_mw",
    "Residual load": "residual_load_mw",
    "Solar": "solar_mw",
    "Wind onshore": "wind_onshore_mw",
    "Wind offshore": "wind_offshore_mw",
}


def chunks(start: date, end: date, months: int):
    """Calendar-aligned windows of `months` months (12 = year, 3 = quarter)."""
    cur = date(start.year, ((start.month - 1) // months) * months + 1, 1)
    while cur <= end:
        m = cur.month - 1 + months
        nxt = date(cur.year + m // 12, m % 12 + 1, 1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


def get_json(path: str, params: dict, retries: int = 8) -> dict:
    last = None
    for k in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", params=params, headers=HEADERS, timeout=120)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 20 * (k + 1)))
                print(f"  429 rate limit, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = exc
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"{path} {params} failed: {last}")


def fetch_price(a: date, b: date) -> pd.DataFrame:
    j = get_json("/price", {"bzn": "DE-LU", "start": a.isoformat(), "end": b.isoformat()})
    return pd.DataFrame({
        "ts_utc": pd.to_datetime(j["unix_seconds"], unit="s", utc=True),
        "price_eur_mwh": pd.to_numeric(pd.Series(j["price"]), errors="coerce"),
    })


def fetch_fund(a: date, b: date) -> pd.DataFrame:
    j = get_json("/public_power", {"country": "de", "start": a.isoformat(), "end": b.isoformat()})
    df = pd.DataFrame({"ts_utc": pd.to_datetime(j["unix_seconds"], unit="s", utc=True)})
    for pt in j["production_types"]:
        col = FUND_MAP.get(pt["name"])
        if col:
            df[col] = pd.to_numeric(pd.Series(pt["data"]), errors="coerce")
    return df


def cached(cache: Path, name: str, fn, a: date, b: date) -> pd.DataFrame:
    f = cache / f"{name}_{a}_{b}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn(a, b)
    df.to_parquet(f, index=False)
    time.sleep(PAUSE_S)
    return df


def main() -> None:
    out = Path(sys.argv[1])
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2019, 1, 1)
    end = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date.today() - timedelta(days=1)

    prices = []
    for a, b in chunks(start, end, 12):
        prices.append(cached(cache, "price", fetch_price, a, b))
        print(f"price {a}..{b}: {len(prices[-1])} rows", flush=True)
    funds = []
    for a, b in chunks(start, end, 3):
        funds.append(cached(cache, "fund", fetch_fund, a, b))
        print(f"fundamentals {a}..{b}: {len(funds[-1])} rows", flush=True)

    p = pd.concat(prices, ignore_index=True).drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)
    f = pd.concat(funds, ignore_index=True).drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)
    # native resolution per row: gap to the next timestamp (minutes)
    gap = (p["ts_utc"].shift(-1) - p["ts_utc"]).dt.total_seconds().div(60)
    p["resolution_min"] = gap.where(gap.isin([15, 60]))
    p.to_parquet(out / "price_de_lu.parquet", index=False)
    f.to_parquet(out / "fundamentals_de.parquet", index=False)
    print("price:", p.shape, p.ts_utc.min(), "->", p.ts_utc.max())
    print("fundamentals:", f.shape, f.ts_utc.min(), "->", f.ts_utc.max())


if __name__ == "__main__":
    main()
