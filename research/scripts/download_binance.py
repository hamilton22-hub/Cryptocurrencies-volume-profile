#!/usr/bin/env python3
"""Download Binance Vision public data for ETHUSDT research."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1] / "data"
BASE = "https://data.binance.vision/data/futures/um"
SYMBOL = "ETHUSDT"


def month_range(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)


def day_range(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def download_zip(url: str, timeout: int = 60) -> bytes | None:
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as fh:
            return pd.read_csv(fh)


def fetch_monthly_klines(start: dt.date, end: dt.date, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    months = list(month_range(start, end))
    for y, m in tqdm(months, desc="klines 15m"):
        cache = out_dir / f"ETHUSDT-15m-{y}-{m:02d}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        url = f"{BASE}/monthly/klines/{SYMBOL}/15m/{SYMBOL}-15m-{y}-{m:02d}.zip"
        content = download_zip(url)
        if content is None:
            continue
        df = read_zip_csv(content)
        # Binance Vision kline columns
        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        if df.shape[1] >= 12:
            df.columns = cols[: df.shape[1]]
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_parquet(cache, index=False)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    out = out_dir / "ETHUSDT_15m.parquet"
    all_df.to_parquet(out, index=False)
    return out


def fetch_monthly_funding(start: dt.date, end: dt.date, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for y, m in tqdm(list(month_range(start, end)), desc="funding"):
        cache = out_dir / f"ETHUSDT-funding-{y}-{m:02d}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        url = f"{BASE}/monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{y}-{m:02d}.zip"
        content = download_zip(url)
        if content is None:
            continue
        df = read_zip_csv(content)
        # calc_time, funding_interval_hours, last_funding_rate
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if "calc" in cl or cl == "time":
                rename[c] = "calc_time"
            elif "funding" in cl and "rate" in cl:
                rename[c] = "funding_rate"
            elif "interval" in cl:
                rename[c] = "funding_interval_hours"
        df = df.rename(columns=rename)
        if "calc_time" in df.columns:
            df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True, errors="coerce")
            if df["calc_time"].isna().all():
                df["calc_time"] = pd.to_datetime(df["calc_time"], utc=True, errors="coerce")
        if "funding_rate" in df.columns:
            df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df.to_parquet(cache, index=False)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    if "calc_time" in all_df.columns:
        all_df = all_df.drop_duplicates("calc_time").sort_values("calc_time")
    out = out_dir / "ETHUSDT_funding.parquet"
    all_df.to_parquet(out, index=False)
    return out


def list_metric_days(start: dt.date, end: dt.date) -> list[dt.date]:
    return list(day_range(start, end))


def fetch_daily_metrics(start: dt.date, end: dt.date, out_dir: Path, workers: int = 16) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    days = list_metric_days(start, end)
    frames = []
    missing = []

    def one(day: dt.date):
        cache = out_dir / f"ETHUSDT-metrics-{day.isoformat()}.parquet"
        if cache.exists():
            return day, pd.read_parquet(cache), None
        url = f"{BASE}/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{day.isoformat()}.zip"
        try:
            content = download_zip(url)
        except Exception as e:
            return day, None, str(e)
        if content is None:
            return day, None, "404"
        df = read_zip_csv(content)
        df.to_parquet(cache, index=False)
        return day, df, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, d) for d in days]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="metrics 5m"):
            day, df, err = fut.result()
            if df is not None:
                frames.append(df)
            else:
                missing.append((day, err))

    if not frames:
        raise RuntimeError("No metrics downloaded")

    all_df = pd.concat(frames, ignore_index=True)
    # normalize columns
    colmap = {c: c.strip() for c in all_df.columns}
    all_df = all_df.rename(columns=colmap)
    # create_time is typical
    time_col = None
    for c in all_df.columns:
        if c.lower() in {"create_time", "timestamp", "time"}:
            time_col = c
            break
    if time_col:
        all_df[time_col] = pd.to_datetime(all_df[time_col], utc=True, errors="coerce")
        all_df = all_df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(time_col)
    out = out_dir / "ETHUSDT_metrics.parquet"
    all_df.to_parquet(out, index=False)
    miss_path = out_dir / "metrics_missing.csv"
    pd.DataFrame(missing, columns=["day", "error"]).to_csv(miss_path, index=False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--metrics-start", default="2021-12-01")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    mstart = dt.date.fromisoformat(args.metrics_start)

    k = fetch_monthly_klines(start, end, ROOT / "klines")
    print("klines ->", k)
    f = fetch_monthly_funding(start, end, ROOT / "funding")
    print("funding ->", f)
    if not args.skip_metrics:
        m = fetch_daily_metrics(mstart, end, ROOT / "metrics", workers=args.workers)
        print("metrics ->", m)


if __name__ == "__main__":
    main()
