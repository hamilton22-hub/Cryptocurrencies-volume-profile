#!/usr/bin/env python3
"""Load and normalize trade exports for R-based research."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def _is_open(exit_time) -> bool:
    return isinstance(exit_time, str) and "OPEN" in exit_time.upper()


def load_export(path: Path | str) -> pd.DataFrame:
    import json

    with open(path) as fh:
        payload = json.load(fh)

    rows = []
    for month in payload["months"]:
        for t in month["trades"]:
            row = dict(t)
            row["month"] = month["month"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    # open trades keep NaT exit
    exit_parsed = []
    for v in df["exit_time"]:
        if _is_open(v):
            exit_parsed.append(pd.NaT)
        else:
            exit_parsed.append(pd.to_datetime(v, utc=True, errors="coerce"))
    df["exit_time_parsed"] = exit_parsed
    df["is_open"] = df["exit_time"].map(_is_open)
    df["skip_metrics"] = df["skip_metrics"].fillna(False).astype(bool) if "skip_metrics" in df.columns else False
    if "gf_blocked" in df.columns:
        df["gf_blocked"] = df["gf_blocked"].fillna(False).astype(bool)
    else:
        df["gf_blocked"] = False
    for c in ["entry_price", "exit_price", "sl_usd", "r", "r_base", "pyr_r", "pnl", "macro_mult"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["direction"] = df["direction"].str.upper()
    df["sign"] = np.where(df["direction"] == "LONG", 1.0, -1.0)
    df["stop_price"] = np.where(
        df["direction"] == "LONG",
        df["entry_price"] - df["sl_usd"],
        df["entry_price"] + df["sl_usd"],
    )
    df["in_metrics"] = (~df["skip_metrics"].astype(bool)) & (~df["is_open"])
    df["trade_id"] = (
        df["entry_time"].astype(str)
        + "|"
        + df["direction"]
        + "|"
        + df["setup"].astype(str)
        + "|"
        + df["entry_price"].round(4).astype(str)
        + "|"
        + df["sl_usd"].round(4).astype(str)
    )
    return df.sort_values("entry_time").reset_index(drop=True)


def load_gf_on(path: Path | str | None = None) -> pd.DataFrame:
    return load_export(path or DATA / "gf_on.json")


def load_gf_off(path: Path | str | None = None) -> pd.DataFrame:
    return load_export(path or DATA / "gf_off.json")


def metrics_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["in_metrics"]].copy().reset_index(drop=True)


def summarize_r(df: pd.DataFrame) -> dict:
    r = df["r"].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]
    zeros = (r == 0).sum()
    equity = r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    streak = 0
    max_streak = 0
    for x in r:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    gp = wins.sum() if len(wins) else 0.0
    gl = -losses.sum() if len(losses) else 0.0
    return {
        "n": int(len(r)),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "zeros": int(zeros),
        "wr": float((r > 0).mean() * 100) if len(r) else np.nan,
        "sum_r": float(r.sum()),
        "expectancy": float(r.mean()) if len(r) else np.nan,
        "median": float(r.median()) if len(r) else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        "pf": float(gp / gl) if gl > 0 else np.nan,
        "max_dd": float(dd.min()) if len(r) else np.nan,
        "max_loss_streak": int(max_streak),
        "best": float(r.max()) if len(r) else np.nan,
        "worst": float(r.min()) if len(r) else np.nan,
        "tail_gt5": int((r > 5).sum()),
        "tail_r_gt5": float(r[r > 5].sum()) if (r > 5).any() else 0.0,
    }


if __name__ == "__main__":
    on = metrics_subset(load_gf_on())
    off = metrics_subset(load_gf_off())
    print("GF ON", summarize_r(on))
    print("GF OFF", summarize_r(off))
    blocked = off.loc[off["gf_blocked"]]
    print("Blocked", summarize_r(blocked))
