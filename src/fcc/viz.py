"""Chart helpers: one consistent, colour-blind-checked style for every figure in the case.

Palette = reference categorical slots in fixed order (blue, orange, aqua; at most three series per chart), diverging
blue (favourable) <-> red (adverse) with neutral grey totals. Solid hairline grid, one y-axis per chart, thin marks.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
POS, NEG, NEUTRAL = "#2a78d6", "#e34948", "#898781"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.dpi": 110,
    })


def eur_fmt(scale: str = "k"):
    div = {"k": 1e3, "M": 1e6, "": 1.0}[scale]
    return mticker.FuncFormatter(lambda v, _: f"{v / div:,.0f}{scale}" if scale != "M" else f"{v / div:,.1f}M")


def lines(df: pd.DataFrame, cols: list[str], title: str, ylabel: str, scale: str = "k", ax=None):
    """Up to three series, fixed slot order, legend + endpoint label on the first series."""
    assert len(cols) <= 3
    ax = ax or plt.subplots(figsize=(10, 3.8))[1]
    for i, c in enumerate(cols):
        ax.plot(df.index, df[c], color=SLOTS[i], label=c, lw=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(eur_fmt(scale))
    if len(cols) > 1:
        ax.legend(loc="upper left", ncol=len(cols))
    return ax


def bars(s: pd.Series, title: str, xlabel: str = "", fmt: str = "{:,.0f}", ax=None, color: str = SLOTS[0]):
    """Single-series horizontal bars with a value label at the bar end."""
    ax = ax or plt.subplots(figsize=(8, 0.42 * len(s) + 1.0))[1]
    s = s.iloc[::-1]
    ax.barh(s.index.astype(str), s.to_numpy(), color=color, height=0.6)
    span = max(abs(s.max()), abs(s.min()), 1e-9)
    for y, v in enumerate(s.to_numpy()):
        ax.text(v + (0.01 * span if v >= 0 else -0.01 * span), y, fmt.format(v), va="center",
                ha="left" if v >= 0 else "right", color=INK2, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    return ax


def grouped_bars(df: pd.DataFrame, cols: list[str], title: str, ylabel: str, scale: str = "k", ax=None):
    assert len(cols) <= 3
    ax = ax or plt.subplots(figsize=(9, 3.8))[1]
    n, k = len(df), len(cols)
    w = 0.8 / k
    x = np.arange(n)
    for i, c in enumerate(cols):
        ax.bar(x + (i - (k - 1) / 2) * w, df[c].to_numpy(), width=w * 0.92, color=SLOTS[i], label=c)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, df.index.astype(str))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(eur_fmt(scale))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=k)
    return ax


def waterfall(br: pd.DataFrame, title: str, label_min_share: float = 0.03, ax=None):
    """Bridge chart: grey totals, blue favourable / red adverse steps, labels only on material steps.
    Steps that are exactly zero are left out of the chart (they stay in the table)."""
    ax = ax or plt.subplots(figsize=(11, 4.6))[1]
    steps = br.reset_index(drop=True)
    keep = [0] + [i for i in range(1, len(steps) - 1) if abs(steps["eur"].iloc[i]) >= 0.5] + [len(steps) - 1]
    steps = steps.iloc[keep].reset_index(drop=True)
    run = steps["eur"].iloc[0]
    span = max(abs(steps["eur"].iloc[0]), abs(steps["eur"].iloc[-1]))
    top = max(steps["eur"].iloc[0], steps["eur"].iloc[-1])
    for i, r in steps.iterrows():
        if i == 0 or i == len(steps) - 1:
            ax.bar(i, r["eur"], color=NEUTRAL, width=0.62)
            ax.text(i, r["eur"] + 0.015 * span, f"{r['eur'] / 1e3:,.0f}k", ha="center", va="bottom", color=INK, fontsize=9)
            continue
        v = r["eur"]
        ax.bar(i, v, bottom=run, color=POS if v >= 0 else NEG, width=0.62)
        top = max(top, run + max(v, 0))
        if abs(v) >= label_min_share * span:
            y = run + v + 0.015 * span if v >= 0 else run + v - 0.015 * span
            ax.text(i, y, f"{v / 1e3:+,.0f}k", ha="center", va="bottom" if v >= 0 else "top", color=INK2, fontsize=9)
        run += v
    ax.set_ylim(0, top * 1.10)
    ax.set_xticks(range(len(steps)), steps["step"], rotation=30, ha="right")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(eur_fmt("k"))
    ax.grid(axis="x", visible=False)
    return ax


def tornado(t: pd.DataFrame, title: str, scale: str = "k", ax=None):
    """Effect of each driver's low / high value versus the base; sorted by range."""
    ax = ax or plt.subplots(figsize=(10, 0.5 * len(t) + 1.2))[1]
    t = t.iloc[::-1].reset_index(drop=True)
    for i, r in t.iterrows():
        for v in (r["low"], r["high"]):
            ax.barh(i, v, color=POS if v >= 0 else NEG, height=0.55)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(t)), t["driver"])
    ax.set_title(title)
    ax.xaxis.set_major_formatter(eur_fmt(scale))
    ax.set_xlabel(f"change vs base ({'EUR' if scale == '' else 'EUR ' + scale})")
    ax.grid(axis="y", visible=False)
    return ax


def histogram(x: pd.Series, marks: dict, title: str, scale: str = "M", ax=None):
    """Distribution with labelled reference lines (e.g. P50 / P90 empirical and normal approximation)."""
    ax = ax or plt.subplots(figsize=(10, 3.8))[1]
    ax.hist(x, bins=60, color=SLOTS[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.6)
    ymax = ax.get_ylim()[1]
    for i, (name, v) in enumerate(marks.items()):
        ax.axvline(v, color=INK2 if "normal" not in name else MUTED, lw=1.4,
                   linestyle="-" if "normal" not in name else (0, (4, 3)))
        ax.text(v, ymax * (0.97 - 0.11 * i), f" {name}", color=INK2, fontsize=9, va="top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_title(title)
    ax.xaxis.set_major_formatter(eur_fmt(scale))
    ax.set_ylabel("draws")
    ax.grid(axis="x", visible=False)
    return ax
