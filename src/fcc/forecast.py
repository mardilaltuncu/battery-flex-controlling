"""Short-term (day-ahead) price forecasting with strict information discipline.

A forecast for local delivery day D is produced on D-1 around 10:00, before the auction closes at 12:00. It may
therefore use day-ahead prices up to and including D-1 (published on D-2) but only *actual* system data up to D-2 -
D-1 actuals are not complete yet. Training is walk-forward: each quarter is predicted by a model fitted on earlier
data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import RANDOM_SEED, TZ
from .market import de_holidays

PRICE_LAGS_H = (24, 48, 168)
FUND_LAG_H = 48
FEATURES = [
    "lag_24h", "lag_48h", "lag_168h", "d1_mean", "d1_max", "d1_min", "d1_std", "mean_7d", "std_7d",
    "tod", "dow", "month", "is_holiday", "post_holiday",
    "res_load_lag48", "solar_lag48", "wind_lag48", "res_share_lag48",
]


def build_features(price: pd.Series, fund: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature frame on the 15-minute UTC grid. ``price`` must be gap-free."""
    idx = price.index
    df = pd.DataFrame({"price": price}, index=idx)
    for h in PRICE_LAGS_H:
        df[f"lag_{h}h"] = price.shift(h * 4)

    loc = idx.tz_convert(TZ)
    ldate = pd.Series(loc.date, index=idx)
    daily = price.groupby(ldate.to_numpy()).agg(["mean", "max", "min", "std"])
    prev = daily.shift(1)                                     # previous local delivery day
    for c in ["mean", "max", "min", "std"]:
        df[f"d1_{c}"] = ldate.map(prev[c])
    df["mean_7d"] = ldate.map(daily["mean"].shift(1).rolling(7).mean())
    df["std_7d"] = ldate.map(daily["mean"].shift(1).rolling(7).std())

    hol = de_holidays(range(idx.year.min(), idx.year.max() + 1))
    df["tod"] = loc.hour + loc.minute / 60.0
    df["dow"] = loc.dayofweek
    df["month"] = loc.month
    df["is_holiday"] = [d in hol for d in loc.date]
    df["post_holiday"] = [(pd.Timestamp(d) - pd.Timedelta(days=1)).date() in hol for d in loc.date]

    if fund is not None:
        f = fund.reindex(idx)
        wind = f[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1)
        df["res_load_lag48"] = f["residual_load_mw"].shift(FUND_LAG_H * 4)
        df["solar_lag48"] = f["solar_mw"].shift(FUND_LAG_H * 4)
        df["wind_lag48"] = wind.shift(FUND_LAG_H * 4)
        df["res_share_lag48"] = ((f["solar_mw"] + wind) / f["load_mw"]).shift(FUND_LAG_H * 4)
    else:
        for c in ["res_load_lag48", "solar_lag48", "wind_lag48", "res_share_lag48"]:
            df[c] = np.nan

    # standard weekday-aware naive benchmark: D-1 on Tue-Fri, D-7 on Mon, Sat, Sun and after/on a holiday
    use_week = (df["dow"].isin([0, 5, 6])) | df["is_holiday"] | df["post_holiday"]
    df["naive"] = np.where(use_week, df["lag_168h"], df["lag_24h"])
    return df


def walk_forward(df: pd.DataFrame, start: str, end: str, step_months: int = 3, min_rows: int = 20_000,
                 params: dict | None = None) -> pd.Series:
    """Quarterly re-fit, strictly out-of-sample predictions for [start, end]."""
    prm = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
               random_state=RANDOM_SEED)
    prm.update(params or {})
    start_ts = pd.Timestamp(start, tz=TZ).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz=TZ).tz_convert("UTC") + pd.Timedelta(days=1)
    preds = []
    cur = start_ts
    feats = [c for c in FEATURES if df[c].notna().any()]
    while cur < end_ts:
        nxt = min((cur.tz_convert(TZ).tz_localize(None) + pd.DateOffset(months=step_months)).tz_localize(TZ)
                  .tz_convert("UTC"), end_ts)
        train = df[(df.index < cur - pd.Timedelta(days=2))].dropna(subset=feats + ["price"])
        test = df[(df.index >= cur) & (df.index < nxt)]
        if len(train) < min_rows:
            raise ValueError(f"not enough training rows before {cur}")
        model = HistGradientBoostingRegressor(**prm).fit(train[feats], train["price"])
        preds.append(pd.Series(model.predict(test[feats]), index=test.index))
        cur = nxt
    return pd.concat(preds).rename("ml")


def metrics(y: pd.Series, yhat: pd.Series, near_zero: float = 5.0) -> dict:
    """Error metrics. MAPE is shown with and without near-zero prices to make its blind spot visible."""
    d = pd.concat([y.rename("y"), yhat.rename("f")], axis=1).dropna()
    e = d["f"] - d["y"]
    nz = d["y"].abs() >= near_zero
    return {
        "MAE": e.abs().mean(),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "WAPE %": 100 * e.abs().sum() / d["y"].abs().sum(),
        "bias (mean error)": e.mean(),
        "MAPE % (|y|>=5)": 100 * (e[nz].abs() / d.loc[nz, "y"].abs()).mean(),
        "MAPE % (all)": 100 * (e.abs() / d["y"].abs().replace(0, np.nan)).mean(),
        "n": len(d),
    }
