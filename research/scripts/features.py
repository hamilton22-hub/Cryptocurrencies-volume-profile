#!/usr/bin/env python3
"""Market feature engineering aligned to trade entry times (no look-ahead)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def load_klines(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "klines" / "ETHUSDT_15m.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
    df["taker_imbalance"] = 2 * df["taker_buy_ratio"] - 1
    df["ret"] = df["close"].pct_change()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["atr14"] = (df["high"] - df["low"]).rolling(14).mean()
    return df


def load_funding(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "funding" / "ETHUSDT_funding.parquet"
    df = pd.read_parquet(path)
    if "calc_time" not in df.columns:
        # try first datetime-like column
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                df = df.rename(columns={c: "calc_time"})
                break
    df = df.sort_values("calc_time").drop_duplicates("calc_time")
    return df


def load_metrics(path: Path | None = None) -> pd.DataFrame | None:
    path = path or DATA / "metrics" / "ETHUSDT_metrics.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    rename = {
        "sum_open_interest": "oi",
        "sum_open_interest_value": "oi_value",
        "count_toptrader_long_short_ratio": "top_trader_ls",
        "sum_toptrader_long_short_ratio": "top_trader_pos_ls",
        "count_long_short_ratio": "global_ls",
        "sum_taker_long_short_vol_ratio": "taker_ls",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "create_time" not in df.columns:
        return df
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["create_time"]).sort_values("create_time").drop_duplicates("create_time")
    for c in ["oi", "oi_value", "top_trader_ls", "top_trader_pos_ls", "global_ls", "taker_ls"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "oi" in df.columns:
        df["oi_ret"] = np.log(df["oi"].replace(0, np.nan)).diff()
        df["oi_vol_1h"] = df["oi_ret"].rolling(12).std()
        df["oi_vol_24h"] = df["oi_ret"].rolling(288).std()
        df["dead_oi_ratio"] = df["oi_vol_1h"] / df["oi_vol_24h"]
    if "global_ls" in df.columns:
        df["global_ls_log"] = np.log(df["global_ls"].replace(0, np.nan))
        df["global_ls_log_diff_1h"] = df["global_ls_log"].diff(12)  # 5m * 12 = 1h
    return df


def _asof_idx(times: pd.DatetimeIndex, query: pd.Timestamp) -> int | None:
    """Index of last bar with open_time <= query."""
    pos = times.searchsorted(query, side="right") - 1
    if pos < 0:
        return None
    return int(pos)


def attach_path_features(trades: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    """Compute MFE/MAE and stop-path features using 15m OHLC after entry."""
    times = pd.DatetimeIndex(klines["open_time"])
    opens = klines["open"].to_numpy()
    highs = klines["high"].to_numpy()
    lows = klines["low"].to_numpy()
    closes = klines["close"].to_numpy()
    tbuy = klines["taker_buy_volume"].to_numpy()
    vol = klines["volume"].to_numpy()

    rows = []
    for _, t in trades.iterrows():
        i0 = _asof_idx(times, t["entry_time"])
        if i0 is None:
            rows.append({})
            continue
        # features from already closed bars only: use bars strictly before entry bar close
        # entry_time is typically candle open; use bars with open_time < entry_time for pre-entry
        i_pre = times.searchsorted(t["entry_time"], side="left") - 1
        feat = {}
        if i_pre >= 0:
            # 1h = 4 bars, 4h=16, 24h=96, 72h=288, 7d=672
            def window(n):
                a = max(0, i_pre - n + 1)
                return slice(a, i_pre + 1)

            w1, w4, w24, w72, w7d = window(4), window(16), window(96), window(288), window(672)
            sign = t["sign"]
            # directed move
            feat["mom_1h"] = sign * (closes[i_pre] / closes[max(0, i_pre - 3)] - 1) if i_pre >= 3 else np.nan
            feat["mom_4h"] = sign * (closes[i_pre] / closes[max(0, i_pre - 15)] - 1) if i_pre >= 15 else np.nan
            # ranges
            for name, w in [("range_4h", w4), ("range_24h", w24), ("range_72h", w72), ("range_7d", w7d)]:
                hi = np.nanmax(highs[w])
                lo = np.nanmin(lows[w])
                feat[name] = (hi - lo) / closes[i_pre] if closes[i_pre] else np.nan
            # directed position in 24h range
            hi24 = np.nanmax(highs[w24])
            lo24 = np.nanmin(lows[w24])
            mid = (closes[i_pre] - lo24) / (hi24 - lo24) if hi24 > lo24 else np.nan
            feat["pos_24h_raw"] = mid
            feat["pos_24h_dir"] = mid if sign > 0 else (1 - mid if mid == mid else np.nan)
            # distance inside extreme
            if sign > 0:
                feat["dist_inside_24h"] = (hi24 - closes[i_pre]) / closes[i_pre]
            else:
                feat["dist_inside_24h"] = (closes[i_pre] - lo24) / closes[i_pre]
            # taker imbalance 1h
            tb = np.nansum(tbuy[w1])
            vv = np.nansum(vol[w1])
            feat["taker_imb_1h"] = sign * (2 * tb / vv - 1) if vv > 0 else np.nan
            # vol compression
            atr_now = np.nanmean(highs[w4] - lows[w4])
            atr_ref = np.nanmean(highs[w72] - lows[w72])
            feat["vol_comp_4h_72h"] = atr_now / atr_ref if atr_ref else np.nan
            feat["atr15"] = float(klines["atr14"].iloc[i_pre]) if i_pre < len(klines) else np.nan
            feat["stop_pct"] = float(t["sl_usd"] / t["entry_price"]) if t["entry_price"] else np.nan
            feat["stop_atr"] = float(t["sl_usd"] / feat["atr15"]) if feat.get("atr15") else np.nan
            feat["stop_24h"] = float(t["sl_usd"] / ((hi24 - lo24))) if hi24 > lo24 else np.nan
            # previous day range (prior 96 bars ending 96 bars ago)
            if i_pre >= 191:
                prev = slice(i_pre - 191, i_pre - 95)
                feat["range_prev_day"] = (np.nanmax(highs[prev]) - np.nanmin(lows[prev])) / closes[i_pre]
            else:
                feat["range_prev_day"] = np.nan
            # trend efficiency 24h
            net = abs(closes[i_pre] - closes[max(0, i_pre - 95)])
            path = np.nansum(np.abs(np.diff(closes[w24])))
            feat["trend_eff_24h"] = net / path if path else np.nan
            # volume vs 7d median
            med7 = np.nanmedian(vol[w7d])
            feat["vol_vs_7d"] = float(vol[i_pre] / med7) if med7 else np.nan

        # path after entry until exit
        if pd.isna(t["exit_time_parsed"]):
            i1 = min(i0 + 1, len(closes) - 1)
        else:
            i1 = times.searchsorted(t["exit_time_parsed"], side="right") - 1
            i1 = max(i0, min(i1, len(closes) - 1))
        path_slice = slice(i0, i1 + 1)
        sign = t["sign"]
        entry = float(t["entry_price"])
        stop = float(t["stop_price"])
        risk = float(t["sl_usd"]) if t["sl_usd"] else np.nan
        if risk and risk > 0:
            if sign > 0:
                mfe_px = np.nanmax(highs[path_slice])
                mae_px = np.nanmin(lows[path_slice])
                feat["mfe_r"] = (mfe_px - entry) / risk
                feat["mae_r"] = (mae_px - entry) / risk
            else:
                mfe_px = np.nanmin(lows[path_slice])
                mae_px = np.nanmax(highs[path_slice])
                feat["mfe_r"] = (entry - mfe_px) / risk
                feat["mae_r"] = (entry - mae_px) / risk
            # stop touch classification
            if sign > 0:
                touched = np.where(lows[path_slice] <= stop)[0]
                close_through = np.where(closes[path_slice] <= stop)[0]
            else:
                touched = np.where(highs[path_slice] >= stop)[0]
                close_through = np.where(closes[path_slice] >= stop)[0]
            feat["stop_touched"] = bool(len(touched))
            feat["stop_close_through"] = bool(len(close_through))
            feat["stop_wick_only"] = bool(len(touched) and not len(close_through))
            if len(touched):
                feat["bars_to_stop"] = int(touched[0])
            # first 4h after entry
            h4 = slice(i0, min(i0 + 16, len(closes)))
            if sign > 0:
                feat["move_4h"] = (closes[min(i0 + 15, len(closes) - 1)] - entry) / entry
                feat["max_bar_4h"] = np.nanmax((highs[h4] - lows[h4]) / closes[h4])
            else:
                feat["move_4h"] = (entry - closes[min(i0 + 15, len(closes) - 1)]) / entry
                feat["max_bar_4h"] = np.nanmax((highs[h4] - lows[h4]) / closes[h4])
            tb = np.nansum(tbuy[h4])
            vv = np.nansum(vol[h4])
            feat["taker_imb_4h_after"] = sign * (2 * tb / vv - 1) if vv > 0 else np.nan
            # entry touch confirmation
            if sign > 0:
                touched_entry = np.any((lows[path_slice] <= entry) & (highs[path_slice] >= entry))
            else:
                touched_entry = np.any((lows[path_slice] <= entry) & (highs[path_slice] >= entry))
            # also check if entry inside first candle
            feat["entry_in_first_bar"] = bool(lows[i0] <= entry <= highs[i0])
            feat["entry_touched_before_exit"] = bool(touched_entry)
        rows.append(feat)
    feat_df = pd.DataFrame(rows)
    return pd.concat([trades.reset_index(drop=True), feat_df], axis=1)


def attach_funding(trades: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    f = funding.sort_values("calc_time").copy()
    left = trades.sort_values("entry_time").copy()
    left["entry_time"] = pd.to_datetime(left["entry_time"], utc=True).astype("datetime64[ns, UTC]")
    f["calc_time"] = pd.to_datetime(f["calc_time"], utc=True).astype("datetime64[ns, UTC]")
    right = f[["calc_time", "funding_rate"]].rename(columns={"calc_time": "entry_time"})
    merged = pd.merge_asof(left, right, on="entry_time", direction="backward")
    merged["funding_dir"] = merged["sign"] * merged["funding_rate"]
    merged["against_funding"] = merged["funding_dir"] < 0
    return merged.sort_index()


def attach_oi_features(trades: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or "create_time" not in metrics.columns:
        out = trades.copy()
        out["oi_available"] = False
        return out
    m = metrics.sort_values("create_time").copy()
    if "oi_ret" in m.columns and "dead_oi_ratio" not in m.columns:
        m["oi_vol_1h"] = m["oi_ret"].rolling(12).std()
        m["oi_vol_24h"] = m["oi_ret"].rolling(288).std()
        m["dead_oi_ratio"] = m["oi_vol_1h"] / m["oi_vol_24h"]
    keep = ["create_time"]
    for c in ["oi", "oi_value", "global_ls", "global_ls_log_diff_1h", "dead_oi_ratio", "taker_ls", "top_trader_ls"]:
        if c in m.columns:
            keep.append(c)
    left = trades.sort_values("entry_time").copy()
    left["entry_time"] = pd.to_datetime(left["entry_time"], utc=True).astype("datetime64[ns, UTC]")
    right = m[keep].copy()
    right["create_time"] = pd.to_datetime(right["create_time"], utc=True).astype("datetime64[ns, UTC]")
    right = right.rename(columns={"create_time": "entry_time"})
    merged = pd.merge_asof(left, right, on="entry_time", direction="backward")
    merged["oi_available"] = merged["oi"].notna() if "oi" in merged.columns else False
    if "global_ls_log_diff_1h" in merged.columns:
        merged["retail_chase"] = merged["sign"] * merged["global_ls_log_diff_1h"]
    # post-entry OI change over 4h
    if "oi" in m.columns:
        oi_times = pd.DatetimeIndex(pd.to_datetime(m["create_time"], utc=True))
        oi_vals = m["oi"].to_numpy()
        d_oi = []
        for _, t in merged.iterrows():
            i0 = oi_times.searchsorted(t["entry_time"], side="right") - 1
            if i0 < 0 or not t.get("oi_available", False):
                d_oi.append(np.nan)
                continue
            i1 = min(i0 + 48, len(oi_vals) - 1)  # 5m*48=4h
            if oi_vals[i0] and oi_vals[i0] == oi_vals[i0]:
                d_oi.append(oi_vals[i1] / oi_vals[i0] - 1)
            else:
                d_oi.append(np.nan)
        merged["oi_chg_4h"] = d_oi
    return merged


def node_exhaustion_mask(trades: pd.DataFrame, lookback_hours: float = 72.0, n_fails: int = 3) -> pd.Series:
    """True if trade should be blocked by NODE_EXHAUSTION shadow rule.

    Returns a boolean Series aligned to the input index.
    """
    order = trades.sort_values("entry_time").index.to_list()
    block = pd.Series(False, index=trades.index)
    hist = []
    for idx in order:
        row = trades.loc[idx]
        cutoff = row["entry_time"] - pd.Timedelta(hours=lookback_hours)
        hist = [h for h in hist if h["entry_time"] >= cutoff]
        same = [
            h
            for h in hist
            if h["direction"] == row["direction"]
            and abs(h["entry_price"] - row["entry_price"]) <= float(row["sl_usd"])
            and h["r"] < 0
        ]
        if len(same) >= n_fails:
            block.loc[idx] = True
        gf_blocked = bool(row["gf_blocked"]) if "gf_blocked" in trades.columns and pd.notna(row.get("gf_blocked")) else False
        in_metrics = bool(row["in_metrics"]) if "in_metrics" in trades.columns else True
        if in_metrics and not gf_blocked:
            hist.append(
                {
                    "entry_time": row["entry_time"],
                    "direction": row["direction"],
                    "entry_price": float(row["entry_price"]),
                    "r": float(row["r"]),
                }
            )
    return block


if __name__ == "__main__":
    print("feature module ok")
