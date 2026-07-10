#!/usr/bin/env python3
"""DEPRECATED diagnostic script.

Do not use its ``stop_hunt`` labels or counterfactual results for decisions.
Use ``analyze_causal_policies.py`` instead.

Historical intent:
- Decompose stopped trades into logical structures:
- entry style: market-like vs limit/retest
- timeframe: 1H vs 15M
- stop fate: stop-hunt (counterfactual 12h MFE without retest) vs hopeless deep invalidation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_klines  # noqa: E402
from load_trades import load_gf_on, metrics_subset, summarize_r  # noqa: E402

ART = ROOT / "artifacts"
REP = ROOT / "reports"
ART.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)


def analyze_trade_paths(trades: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    times = pd.DatetimeIndex(klines["open_time"])
    opens = klines["open"].to_numpy(float)
    highs = klines["high"].to_numpy(float)
    lows = klines["low"].to_numpy(float)
    closes = klines["close"].to_numpy(float)
    atr = (klines["high"] - klines["low"]).rolling(14).mean().to_numpy(float)

    rows = []
    for _, t in trades.iterrows():
        entry_time = t["entry_time"]
        exit_time = t["exit_time_parsed"]
        entry = float(t["entry_price"])
        stop = float(t["stop_price"])
        risk = float(t["sl_usd"])
        sign = float(t["sign"])
        i0 = times.searchsorted(entry_time, side="left")
        if i0 >= len(times):
            rows.append({})
            continue
        # align: if entry_time falls inside a bar, use that bar
        if i0 > 0 and times[i0] > entry_time:
            i0 -= 1
        i0 = max(0, min(i0, len(times) - 1))

        if pd.isna(exit_time):
            i_exit = min(i0 + 1, len(times) - 1)
        else:
            i_exit = times.searchsorted(exit_time, side="right") - 1
            i_exit = max(i0, min(i_exit, len(times) - 1))

        # --- entry style ---
        # bars until price first touches entry (fill confirmation)
        fill_bar = None
        for j in range(i0, min(i0 + 20, i_exit + 1)):
            if lows[j] <= entry <= highs[j]:
                fill_bar = j
                break
        bars_to_fill = (fill_bar - i0) if fill_bar is not None else np.nan
        entry_in_signal_bar = bool(lows[i0] <= entry <= highs[i0])
        # distance of entry from signal-bar open in R
        dist_from_open_r = sign * (entry - opens[i0]) / risk if risk else np.nan
        # market-like: filled on signal bar and entry near aggressive continuation
        # limit/retest: either delayed fill OR entry on opposite side of signal bar (pullback)
        if fill_bar is None:
            entry_style = "unknown"
        elif bars_to_fill == 0 and abs(dist_from_open_r) <= 0.35:
            # filled immediately near open -> market-ish / aggressive
            entry_style = "market_like"
        elif bars_to_fill == 0 and dist_from_open_r * sign < 0:
            # same bar but entry on pullback side of open
            entry_style = "limit_retest"
        elif bars_to_fill is not None and bars_to_fill >= 1:
            entry_style = "limit_retest"
        else:
            # same bar, entry stretched with move
            entry_style = "market_like" if abs(dist_from_open_r) > 0.35 else "limit_retest"

        # refine: if entry is between open and stop (pullback toward invalidation) => limit
        if entry_in_signal_bar and risk:
            toward_stop = sign * (opens[i0] - entry) / risk  # >0 if entry closer to stop than open
            if toward_stop > 0.15 and bars_to_fill == 0:
                entry_style = "limit_retest"

        # --- path until recorded exit ---
        path = slice(i0, i_exit + 1)
        if sign > 0:
            mfe_px = np.nanmax(highs[path])
            mae_px = np.nanmin(lows[path])
            mfe_r = (mfe_px - entry) / risk
            mae_r = (mae_px - entry) / risk
            stop_touch_idx = np.where(lows[path] <= stop)[0]
            close_thru_idx = np.where(closes[path] <= stop)[0]
        else:
            mfe_px = np.nanmin(lows[path])
            mae_px = np.nanmax(highs[path])
            mfe_r = (entry - mfe_px) / risk
            mae_r = (entry - mae_px) / risk
            stop_touch_idx = np.where(highs[path] >= stop)[0]
            close_thru_idx = np.where(closes[path] >= stop)[0]

        stop_touched = len(stop_touch_idx) > 0
        stop_close_through = len(close_thru_idx) > 0
        stop_wick_only = stop_touched and not stop_close_through
        bars_to_stop = int(stop_touch_idx[0]) if stop_touched else np.nan

        # adverse excursion beyond stop in R (how deep past -1R)
        beyond_stop_r = max(0.0, -1.0 - mae_r) if mae_r == mae_r else np.nan
        # situational depth: beyond stop in ATR units at entry
        atr0 = atr[i0] if i0 < len(atr) and atr[i0] == atr[i0] else np.nan
        beyond_stop_atr = (beyond_stop_r * risk / atr0) if atr0 and atr0 == atr0 else np.nan

        # --- counterfactual after first stop touch: next 12h (48 bars) ---
        cf_mfe_r = np.nan
        cf_mfe_no_retest_r = np.nan
        cf_reclaimed = False
        cf_reached_1r = False
        cf_reached_2r = False
        cf_reached_5r = False
        hours_to_cf_mfe = np.nan
        first_stop_abs = None

        if stop_touched:
            first_stop_abs = i0 + int(stop_touch_idx[0])
            # start AFTER the stop-touch bar
            j0 = first_stop_abs + 1
            j1 = min(first_stop_abs + 48, len(times) - 1)  # 12h
            if j0 <= j1:
                # reclaim: price returns to entry side of stop within 12h
                if sign > 0:
                    reclaim_bars = np.where(closes[j0 : j1 + 1] > stop)[0]
                    # MFE from entry over 12h window after stop
                    cf_mfe_r = (np.nanmax(highs[j0 : j1 + 1]) - entry) / risk
                    # path without retesting stop: walk forward until stop retested
                    end = j1
                    for j in range(j0, j1 + 1):
                        if lows[j] <= stop:
                            end = j - 1
                            break
                    if end >= j0:
                        cf_mfe_no_retest_r = (np.nanmax(highs[j0 : end + 1]) - entry) / risk
                    else:
                        cf_mfe_no_retest_r = np.nan
                else:
                    reclaim_bars = np.where(closes[j0 : j1 + 1] < stop)[0]
                    cf_mfe_r = (entry - np.nanmin(lows[j0 : j1 + 1])) / risk
                    end = j1
                    for j in range(j0, j1 + 1):
                        if highs[j] >= stop:
                            end = j - 1
                            break
                    if end >= j0:
                        cf_mfe_no_retest_r = (entry - np.nanmin(lows[j0 : end + 1])) / risk
                    else:
                        cf_mfe_no_retest_r = np.nan

                cf_reclaimed = len(reclaim_bars) > 0
                cf_reached_1r = cf_mfe_r >= 1.0 if cf_mfe_r == cf_mfe_r else False
                cf_reached_2r = cf_mfe_r >= 2.0 if cf_mfe_r == cf_mfe_r else False
                cf_reached_5r = cf_mfe_r >= 5.0 if cf_mfe_r == cf_mfe_r else False
                # time to CF MFE
                if sign > 0:
                    peak_j = j0 + int(np.nanargmax(highs[j0 : j1 + 1]))
                else:
                    peak_j = j0 + int(np.nanargmin(lows[j0 : j1 + 1]))
                hours_to_cf_mfe = (peak_j - first_stop_abs) * 0.25

        # MFE in first 12h from entry regardless of stop (for context)
        h12 = slice(i0, min(i0 + 48, len(times)))
        if sign > 0:
            mfe_12h = (np.nanmax(highs[h12]) - entry) / risk
            mae_12h = (np.nanmin(lows[h12]) - entry) / risk
        else:
            mfe_12h = (entry - np.nanmin(lows[h12])) / risk
            mae_12h = (entry - np.nanmax(highs[h12])) / risk

        rows.append(
            {
                "bars_to_fill": bars_to_fill,
                "entry_in_signal_bar": entry_in_signal_bar,
                "dist_from_open_r": dist_from_open_r,
                "entry_style": entry_style,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "mfe_12h": mfe_12h,
                "mae_12h": mae_12h,
                "stop_touched": stop_touched,
                "stop_close_through": stop_close_through,
                "stop_wick_only": stop_wick_only,
                "bars_to_stop": bars_to_stop,
                "beyond_stop_r": beyond_stop_r,
                "beyond_stop_atr": beyond_stop_atr,
                "atr_at_entry": atr0,
                "cf_mfe_r": cf_mfe_r,
                "cf_mfe_no_retest_r": cf_mfe_no_retest_r,
                "cf_reclaimed": cf_reclaimed,
                "cf_reached_1r": cf_reached_1r,
                "cf_reached_2r": cf_reached_2r,
                "cf_reached_5r": cf_reached_5r,
                "hours_to_cf_mfe": hours_to_cf_mfe,
            }
        )
    return pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def classify_stop_fate(df: pd.DataFrame) -> pd.DataFrame:
    """Classify negative / stopped trades into hunt vs hopeless vs other."""
    out = df.copy()
    out["stop_fate"] = "not_applicable"

    # universe: trades that touched stop OR finished near full stop
    stopped = out["stop_touched"].fillna(False) | (out["r"] <= -0.95)
    neg_or_stopped = stopped & (out["r"] <= 0.1)  # include tiny leftovers

    # situational deep threshold: beyond-stop depth > median of stopped trades
    # in same timeframe, using beyond_stop_r and beyond_stop_atr
    for tf, g_idx in out.groupby("timeframe").groups.items():
        sub = out.loc[g_idx]
        stopped_sub = sub.loc[sub["stop_touched"].fillna(False) | (sub["r"] <= -0.95)]
        med_r = stopped_sub["beyond_stop_r"].median()
        med_atr = stopped_sub["beyond_stop_atr"].median()
        # also use 75th percentile as "clearly deep"
        p75_r = stopped_sub["beyond_stop_r"].quantile(0.75)
        p75_atr = stopped_sub["beyond_stop_atr"].quantile(0.75)
        out.loc[g_idx, "_med_beyond_r"] = med_r
        out.loc[g_idx, "_p75_beyond_r"] = p75_r
        out.loc[g_idx, "_med_beyond_atr"] = med_atr
        out.loc[g_idx, "_p75_beyond_atr"] = p75_atr

    # STOP HUNT candidate:
    # - early-ish stop (within first 4h = 16 bars) OR wick-only
    # - after stop, within 12h, MFE without retest of stop >= 1R (or CF MFE >= 1R with reclaim)
    # - not deeply beyond stop
    early = out["bars_to_stop"].fillna(999) <= 16
    shallow = out["beyond_stop_r"].fillna(0) <= out["_med_beyond_r"].fillna(0.2)
    good_cf = (
        (out["cf_mfe_no_retest_r"].fillna(-9) >= 1.0)
        | ((out["cf_reclaimed"].fillna(False)) & (out["cf_mfe_r"].fillna(-9) >= 1.0))
    )
    hunt = neg_or_stopped & good_cf & (out["stop_wick_only"].fillna(False) | early) & shallow

    # HOPELESS:
    # - deep breach: beyond_stop_r >= p75 OR beyond_stop_atr >= p75
    # - OR close-through with poor CF (cf_mfe < 0.5 and no reclaim to +1R)
    deep = (out["beyond_stop_r"] >= out["_p75_beyond_r"]) | (
        out["beyond_stop_atr"] >= out["_p75_beyond_atr"]
    )
    poor_cf = out["cf_mfe_r"].fillna(-9) < 0.5
    hopeless = neg_or_stopped & (
        (deep & (out["stop_close_through"].fillna(False) | (out["beyond_stop_r"] > 0.25)))
        | (out["stop_close_through"].fillna(False) & poor_cf & ~good_cf)
    )

    # ambiguous stopped
    other_stop = neg_or_stopped & ~hunt & ~hopeless

    out.loc[hunt, "stop_fate"] = "stop_hunt"
    out.loc[hopeless, "stop_fate"] = "hopeless"
    out.loc[other_stop, "stop_fate"] = "ambiguous_stop"

    # winners that never stopped
    out.loc[(out["r"] > 0) & ~out["stop_touched"].fillna(False), "stop_fate"] = "clean_winner"
    out.loc[(out["r"] > 0) & out["stop_touched"].fillna(False), "stop_fate"] = "winner_survived_stop_touch"

    return out


def summarize_group(df: pd.DataFrame, name: str) -> dict:
    s = summarize_r(df)
    s["label"] = name
    if "cf_mfe_r" in df.columns and len(df):
        s["med_cf_mfe"] = float(df["cf_mfe_r"].median()) if df["cf_mfe_r"].notna().any() else np.nan
        s["med_cf_mfe_noret"] = (
            float(df["cf_mfe_no_retest_r"].median()) if df["cf_mfe_no_retest_r"].notna().any() else np.nan
        )
        s["pct_cf_ge1"] = float((df["cf_mfe_r"] >= 1).mean() * 100) if df["cf_mfe_r"].notna().any() else np.nan
        s["pct_cf_ge2"] = float((df["cf_mfe_r"] >= 2).mean() * 100) if df["cf_mfe_r"].notna().any() else np.nan
        s["pct_wick"] = float(df["stop_wick_only"].mean() * 100) if "stop_wick_only" in df else np.nan
        s["med_beyond_r"] = float(df["beyond_stop_r"].median()) if "beyond_stop_r" in df else np.nan
        s["med_bars_to_stop"] = float(df["bars_to_stop"].median()) if "bars_to_stop" in df else np.nan
    return s


def counterfactual_hold_value(df: pd.DataFrame) -> dict:
    """If stop-hunt trades were held to min(cf_mfe_no_retest, cap) instead of stop."""
    hunts = df.loc[df["stop_fate"] == "stop_hunt"].copy()
    if not len(hunts):
        return {}
    # conservative: take cf_mfe_no_retest capped at +2R / +5R as hypothetical exit proxy
    out = {"n": int(len(hunts)), "canonical_sum": float(hunts["r"].sum())}
    for cap in [1.0, 2.0, 5.0]:
        hyp = hunts["cf_mfe_no_retest_r"].clip(upper=cap).fillna(hunts["r"])
        # if no-retest path missing, fall back to reclaim cf_mfe capped
        miss = hunts["cf_mfe_no_retest_r"].isna()
        hyp.loc[miss] = hunts.loc[miss, "cf_mfe_r"].clip(upper=cap).fillna(hunts.loc[miss, "r"])
        out[f"hyp_sum_cap{cap:g}R"] = float(hyp.sum())
        out[f"delta_cap{cap:g}R"] = float(hyp.sum() - hunts["r"].sum())
    return out


def main():
    print("Loading...")
    trades = metrics_subset(load_gf_on())
    klines = load_klines()
    print("Path analysis...")
    feat = analyze_trade_paths(trades, klines)
    feat = classify_stop_fate(feat)
    feat.to_parquet(ART / "stop_structure_features.parquet", index=False)
    feat.to_csv(ART / "stop_structure_features.csv", index=False)

    # Overall entry style / TF
    tables = []
    for col in ["entry_style", "timeframe"]:
        for k, g in feat.groupby(col):
            tables.append(summarize_group(g, f"{col}={k}"))
    # cross
    for (es, tf), g in feat.groupby(["entry_style", "timeframe"]):
        tables.append(summarize_group(g, f"{es}|{tf}"))

    # stop fate overall and crosses
    for k, g in feat.groupby("stop_fate"):
        tables.append(summarize_group(g, f"fate={k}"))
    for (fate, tf), g in feat.groupby(["stop_fate", "timeframe"]):
        tables.append(summarize_group(g, f"fate={fate}|{tf}"))
    for (fate, es), g in feat.groupby(["stop_fate", "entry_style"]):
        tables.append(summarize_group(g, f"fate={fate}|{es}"))
    for (fate, es, tf), g in feat.groupby(["stop_fate", "entry_style", "timeframe"]):
        tables.append(summarize_group(g, f"fate={fate}|{es}|{tf}"))

    tab = pd.DataFrame(tables)
    tab.to_csv(ART / "stop_structure_summary.csv", index=False)

    # Focus: negative stopped universe
    stopped = feat.loc[feat["stop_fate"].isin(["stop_hunt", "hopeless", "ambiguous_stop"])]
    fate_counts = stopped.groupby("stop_fate").agg(
        n=("r", "size"),
        sum_r=("r", "sum"),
        mean_r=("r", "mean"),
        med_beyond=("beyond_stop_r", "median"),
        med_cf=("cf_mfe_r", "median"),
        pct_cf1=("cf_reached_1r", "mean"),
        pct_cf2=("cf_reached_2r", "mean"),
        pct_wick=("stop_wick_only", "mean"),
        med_bars=("bars_to_stop", "median"),
    )
    fate_counts.to_csv(ART / "stop_fate_core.csv")

    # Contingency entry_style x timeframe x fate
    ct = (
        feat.groupby(["timeframe", "entry_style", "stop_fate"])
        .agg(n=("r", "size"), sum_r=("r", "sum"), mean_r=("r", "mean"))
        .reset_index()
    )
    ct.to_csv(ART / "structure_contingency.csv", index=False)

    hyp = counterfactual_hold_value(feat)
    # also by TF / entry style
    hyp_parts = {}
    for key, g in feat.groupby(["timeframe", "entry_style"]):
        hyp_parts[f"{key[0]}|{key[1]}"] = counterfactual_hold_value(g)

    # Pre-entry predictability of hunt vs hopeless (only among stopped)
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pred = {}
    stopped2 = feat.loc[feat["stop_fate"].isin(["stop_hunt", "hopeless"])].copy()
    stopped2["y_hunt"] = (stopped2["stop_fate"] == "stop_hunt").astype(int)
    feats_cols = [
        c
        for c in [
            "stop_pct",
            "mom_1h",
            "dist_from_open_r",
            "bars_to_fill",
            "atr_at_entry",
        ]
        if c in stopped2.columns
    ]
    # add stop_pct
    stopped2["stop_pct"] = stopped2["sl_usd"] / stopped2["entry_price"]
    # simple directional mom from path already not precomputed here — skip if missing
    # Use only known-at-entry proxies available in this frame
    pre_cols = ["stop_pct", "dist_from_open_r", "atr_at_entry"]
    # bars_to_fill is known at fill, borderline for pre-entry; include as structure feature
    pre_cols2 = pre_cols + ["bars_to_fill"]
    oos = stopped2["entry_time"] >= pd.Timestamp("2023-01-01", tz="UTC")
    for cols, label in [(pre_cols, "strict_pre"), (pre_cols2, "with_fill_delay")]:
        d = stopped2.dropna(subset=cols + ["y_hunt"])
        tr = d["entry_time"] < pd.Timestamp("2023-01-01", tz="UTC")
        te = ~tr
        if tr.sum() < 40 or te.sum() < 40:
            continue
        Xtr, Xte = d.loc[tr, cols], d.loc[te, cols]
        ytr, yte = d.loc[tr, "y_hunt"], d.loc[te, "y_hunt"]
        sc = StandardScaler()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(sc.fit_transform(Xtr), ytr)
        proba = model.predict_proba(sc.transform(Xte))[:, 1]
        pred[label] = {
            "auc": float(roc_auc_score(yte, proba)) if yte.nunique() > 1 else None,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "hunt_rate_test": float(yte.mean()),
        }

    # Share of negative R explained
    neg = feat.loc[feat["r"] < 0]
    expl = {}
    for fate in ["stop_hunt", "hopeless", "ambiguous_stop"]:
        g = neg.loc[neg["stop_fate"] == fate]
        expl[fate] = {
            "n": int(len(g)),
            "share_n": float(len(g) / len(neg)) if len(neg) else 0,
            "sum_r": float(g["r"].sum()),
            "share_r": float(g["r"].sum() / neg["r"].sum()) if neg["r"].sum() != 0 else 0,
        }

    summary = {
        "entry_style_counts": feat["entry_style"].value_counts().to_dict(),
        "timeframe_counts": feat["timeframe"].value_counts().to_dict(),
        "stop_fate_counts": feat["stop_fate"].value_counts().to_dict(),
        "fate_core": fate_counts.reset_index().to_dict(orient="records"),
        "counterfactual_hunt": hyp,
        "counterfactual_by_structure": hyp_parts,
        "predict_hunt_vs_hopeless": pred,
        "neg_r_explanation": expl,
        "thresholds": {
            "note": "deep = beyond_stop >= TF p75; hunt = early/wick + CF MFE>=1R without deep breach",
        },
    }
    with open(ART / "stop_structure_research.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Markdown report
    lines = []
    lines.append("# Структуры стопов: hunt vs hopeless × market/limit × 1H/15M\n")
    lines.append("В экспорте нет явного флага market/limit — тип входа восстановлен по path.\n")
    lines.append("## Определения\n")
    lines.append("- **limit_retest**: отложенный fill (≥1 бар) или вход на стороне отката к стопу внутри сигнальной свечи.")
    lines.append("- **market_like**: fill на сигнальной свече, вход около open / по ходу импульса.")
    lines.append("- **stop_hunt**: ранний/wick вынос, неглубокий beyond-stop, и в следующие 12ч MFE ≥1R без ретеста стопа (или после reclaim).")
    lines.append("- **hopeless**: глубокий вынос (≥ p75 beyond-stop в R или ATR внутри ТФ) и/или close-through без нормального CF.\n")
    lines.append("## Счётчики\n")
    lines.append(f"- entry_style: `{summary['entry_style_counts']}`")
    lines.append(f"- timeframe: `{summary['timeframe_counts']}`")
    lines.append(f"- stop_fate: `{summary['stop_fate_counts']}`\n")
    lines.append("## Ядро стоп-судеб\n")
    lines.append(fate_counts.to_markdown())
    lines.append("\n## Отрицательный R: чем объясняется\n")
    lines.append("```json\n" + json.dumps(expl, indent=2) + "\n```\n")
    lines.append("## Контрфакт для stop_hunt (если удержать до CF MFE без ретеста)\n")
    lines.append("```json\n" + json.dumps(hyp, indent=2) + "\n```\n")
    lines.append("## Разрез contingency\n")
    lines.append(ct.to_markdown(index=False))
    lines.append("\n## Предсказуемость hunt vs hopeless до/в момент входа\n")
    lines.append("```json\n" + json.dumps(pred, indent=2) + "\n```\n")
    (REP / "stop_structure_decomposition.md").write_text("\n".join(lines))

    print(json.dumps(summary, indent=2, default=str)[:4000])
    print("Wrote", REP / "stop_structure_decomposition.md")


if __name__ == "__main__":
    raise SystemExit(
        "DEPRECATED: this script contains circular/non-causal diagnostics. "
        "Run research/scripts/analyze_causal_policies.py instead."
    )
