#!/usr/bin/env python3
"""
Negative-trade filter research for TBX breakout system.

Trusts the exported `r` sequence (not Extended UI cards).
Evaluates pre-entry filters, path/stop classes, shadow rules, and OI candidates
with walk-forward and multiple-testing awareness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    attach_funding,
    attach_oi_features,
    attach_path_features,
    load_funding,
    load_klines,
    load_metrics,
    node_exhaustion_mask,
)
from load_trades import load_gf_off, load_gf_on, metrics_subset, summarize_r  # noqa: E402

ART = ROOT / "artifacts"
REP = ROOT / "reports"
ART.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)


def classify_negatives(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    r = out["r"]
    out["neg_class"] = "non_negative"
    neg = r < 0
    # full stop-ish
    full_stop = neg & (r <= -0.95)
    # path-based
    close_stop = neg & out.get("stop_close_through", False)
    wick_stop = neg & out.get("stop_wick_only", False)
    reached_1r = out.get("mfe_r", 0) >= 1.0
    out.loc[neg & close_stop, "neg_class"] = "close_through_stop"
    out.loc[neg & wick_stop & ~close_stop, "neg_class"] = "wick_breach_stop"
    out.loc[neg & reached_1r & full_stop, "neg_class"] = "reached_1r_then_full_stop"
    out.loc[neg & reached_1r & ~full_stop, "neg_class"] = "reached_1r_then_neg_exit"
    # remaining negatives
    mask = neg & (out["neg_class"] == "non_negative")
    out.loc[mask, "neg_class"] = "partial_or_structure_exit"
    # override priority for full stop without path info
    out.loc[full_stop & ~out.get("stop_touched", False).fillna(False), "neg_class"] = "full_stop_no_path_touch"
    return out


def filter_impact(base: pd.DataFrame, mask_remove: pd.Series, label: str) -> dict:
    keep = base.loc[~mask_remove]
    rem = base.loc[mask_remove]
    sb, sk, sr = summarize_r(base), summarize_r(keep), summarize_r(rem)
    return {
        "label": label,
        "removed_n": int(mask_remove.sum()),
        "removed_sum_r": sr["sum_r"],
        "removed_wr": sr["wr"],
        "removed_pf": sr["pf"],
        "removed_tail_gt5": sr["tail_gt5"],
        "removed_tail_r": sr["tail_r_gt5"],
        "base_sum_r": sb["sum_r"],
        "keep_sum_r": sk["sum_r"],
        "delta_sum_r": sk["sum_r"] - sb["sum_r"],
        "base_wr": sb["wr"],
        "keep_wr": sk["wr"],
        "delta_wr": sk["wr"] - sb["wr"],
        "base_pf": sb["pf"],
        "keep_pf": sk["pf"],
        "delta_pf": (sk["pf"] - sb["pf"]) if sk["pf"] == sk["pf"] and sb["pf"] == sb["pf"] else np.nan,
        "base_mdd": sb["max_dd"],
        "keep_mdd": sk["max_dd"],
        "delta_mdd": sk["max_dd"] - sb["max_dd"],
        "improves_total_r": sk["sum_r"] >= sb["sum_r"] - 1e-9,
        "improves_pf": (sk["pf"] >= sb["pf"] - 1e-9) if sk["pf"] == sk["pf"] else False,
        "no_tail_loss": sr["tail_gt5"] == 0,
    }


def yearly_removed(base: pd.DataFrame, mask_remove: pd.Series) -> pd.DataFrame:
    tmp = base.copy()
    tmp["year"] = tmp["entry_time"].dt.year
    tmp["removed"] = mask_remove.values
    rows = []
    for y, g in tmp.groupby("year"):
        rem = g.loc[g["removed"], "r"]
        rows.append(
            {
                "year": int(y),
                "n": int(g["removed"].sum()),
                "sum_r": float(rem.sum()) if len(rem) else 0.0,
                "wr": float((rem > 0).mean() * 100) if len(rem) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def walk_forward_quantile_filter(
    df: pd.DataFrame,
    feature: str,
    q: float,
    side: str,
    oos_start: str = "2023-01-01",
) -> dict:
    """Expanding-history quantile filter evaluated on OOS."""
    d = df.dropna(subset=[feature]).sort_values("entry_time").copy()
    oos = d["entry_time"] >= pd.Timestamp(oos_start, tz="UTC")
    remove = pd.Series(False, index=d.index)
    hist_vals = []
    for idx, row in d.iterrows():
        if len(hist_vals) >= 50:
            thr = np.nanquantile(hist_vals, q)
            if side == "low" and row[feature] <= thr:
                if oos.loc[idx]:
                    remove.loc[idx] = True
            if side == "high" and row[feature] >= thr:
                if oos.loc[idx]:
                    remove.loc[idx] = True
        hist_vals.append(row[feature])
    base_oos = d.loc[oos]
    rem_oos = remove.loc[oos]
    # align
    return filter_impact(base_oos, rem_oos.reindex(base_oos.index).fillna(False), f"WF {feature} {side} q={q}")


def simple_intersection_filter(df: pd.DataFrame, oos_start: str = "2023-01-01") -> dict:
    """Least-bad compromise from prior research: tight stop/ATR + weak 1h momentum."""
    d = df.dropna(subset=["stop_atr", "mom_1h"]).sort_values("entry_time").copy()
    oos = d["entry_time"] >= pd.Timestamp(oos_start, tz="UTC")
    remove = pd.Series(False, index=d.index)
    hist_stop, hist_mom = [], []
    for idx, row in d.iterrows():
        if len(hist_stop) >= 50:
            thr_stop = np.nanquantile(hist_stop, 0.10)
            thr_mom = np.nanquantile(hist_mom, 0.25)
            if row["stop_atr"] <= thr_stop and row["mom_1h"] <= thr_mom and oos.loc[idx]:
                remove.loc[idx] = True
        hist_stop.append(row["stop_atr"])
        hist_mom.append(row["mom_1h"])
    base_oos = d.loc[oos]
    return filter_impact(base_oos, remove.loc[oos].reindex(base_oos.index).fillna(False), "WF stop_atr low10 ∩ mom_1h low25")


def ml_stop_filter(df: pd.DataFrame, oos_start: str = "2023-01-01") -> list[dict]:
    feats = [
        "stop_pct",
        "stop_atr",
        "stop_24h",
        "mom_1h",
        "vol_comp_4h_72h",
        "range_24h",
        "range_72h",
        "taker_imb_1h",
        "pos_24h_dir",
        "dist_inside_24h",
        "trend_eff_24h",
        "vol_vs_7d",
        "range_prev_day",
    ]
    d = df.dropna(subset=[c for c in feats if c in df.columns]).sort_values("entry_time").copy()
    feats = [c for c in feats if c in d.columns]
    d["y_toxic"] = (d["r"] <= -0.5).astype(int)
    d["y_full"] = ((d["r"] <= -0.95) | d.get("stop_touched", False).fillna(False)).astype(int)
    d["y_neg"] = (d["r"] < 0).astype(int)

    train = d["entry_time"] < pd.Timestamp(oos_start, tz="UTC")
    test = ~train
    if train.sum() < 100 or test.sum() < 100:
        return []

    results = []
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"),
        "et": ExtraTreesClassifier(n_estimators=300, max_depth=4, random_state=42, class_weight="balanced"),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    for target in ["y_toxic", "y_full", "y_neg"]:
        Xtr, Xte = d.loc[train, feats], d.loc[test, feats]
        ytr, yte = d.loc[train, target], d.loc[test, target]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        for name, model in models.items():
            try:
                if name == "logreg":
                    model.fit(Xtr_s, ytr)
                    proba = model.predict_proba(Xte_s)[:, 1]
                else:
                    model.fit(Xtr, ytr)
                    proba = model.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, proba) if yte.nunique() > 1 else np.nan
                # remove top 5% risk
                thr = np.quantile(proba, 0.95)
                rem = pd.Series(False, index=d.index)
                rem.loc[test] = proba >= thr
                impact = filter_impact(d.loc[test], rem.loc[test], f"ML {name}/{target} top5%")
                impact["auc"] = float(auc)
                results.append(impact)
            except Exception as e:
                results.append({"label": f"ML {name}/{target}", "error": str(e)})
    return results


def evaluate_shadow_rules(df: pd.DataFrame, oos_start: str = "2023-01-01") -> list[dict]:
    d = df.sort_values("entry_time").copy()
    # NODE_EXHAUSTION
    exh = node_exhaustion_mask(d)
    oos = d["entry_time"] >= pd.Timestamp(oos_start, tz="UTC")
    out = []
    out.append(filter_impact(d.loc[oos], exh.loc[oos], "NODE_EXHAUSTION OOS"))
    out.append(filter_impact(d, exh, "NODE_EXHAUSTION ALL"))

    # DEAD_OI / RETAIL_CHASE where available
    if "dead_oi_ratio" in d.columns:
        dead = d["oi_available"].fillna(False) & (d["dead_oi_ratio"] <= 0.25)
        out.append(filter_impact(d.loc[d["oi_available"].fillna(False)], dead.loc[d["oi_available"].fillna(False)], "DEAD_OI covered"))
    if "retail_chase" in d.columns:
        chase = d["oi_available"].fillna(False) & (d["retail_chase"] >= 0.025)
        out.append(filter_impact(d.loc[d["oi_available"].fillna(False)], chase.loc[d["oi_available"].fillna(False)], "RETAIL_CHASE covered"))
        if "dead_oi_ratio" in d.columns:
            both = dead | chase
            covered = d["oi_available"].fillna(False)
            out.append(filter_impact(d.loc[covered], both.loc[covered], "DEAD_OI ∪ RETAIL_CHASE covered"))
    return out


def search_new_candidates(df: pd.DataFrame, oos_start: str = "2023-01-01") -> pd.DataFrame:
    """Grid of physically motivated pre-entry rules; keep only those with negative removed Total R and zero tail loss on OOS."""
    d = df.sort_values("entry_time").copy()
    oos = d["entry_time"] >= pd.Timestamp(oos_start, tz="UTC")
    base = d.loc[oos]
    candidates = []

    # expanding quantile singles
    for feat, side, qs in [
        ("stop_pct", "low", [0.05, 0.10]),
        ("stop_atr", "low", [0.05, 0.10]),
        ("stop_24h", "low", [0.05, 0.10]),
        ("mom_1h", "low", [0.10, 0.25]),
        ("taker_imb_1h", "low", [0.10, 0.25]),
        ("vol_comp_4h_72h", "low", [0.10, 0.20]),
        ("range_prev_day", "low", [0.10, 0.20]),
        ("range_72h", "low", [0.10, 0.20]),
        ("dist_inside_24h", "high", [0.80, 0.90]),
        ("pos_24h_dir", "high", [0.90]),
        ("trend_eff_24h", "high", [0.90]),
        ("vol_vs_7d", "low", [0.10]),
    ]:
        if feat not in d.columns:
            continue
        for q in qs:
            # build expanding mask on full d, evaluate on oos
            remove = pd.Series(False, index=d.index)
            hist = []
            for idx, row in d.iterrows():
                val = row[feat]
                if len(hist) >= 80 and val == val:
                    thr = np.nanquantile(hist, q if side == "low" else q)
                    if side == "low" and val <= thr and oos.loc[idx]:
                        remove.loc[idx] = True
                    if side == "high" and val >= thr and oos.loc[idx]:
                        remove.loc[idx] = True
                if val == val:
                    hist.append(val)
            impact = filter_impact(base, remove.loc[oos].reindex(base.index).fillna(False), f"{feat}|{side}|q{q}")
            impact["family"] = "quantile"
            candidates.append(impact)

    # hour-of-week / weekend
    d["hour"] = d["entry_time"].dt.hour
    d["dow"] = d["entry_time"].dt.dayofweek
    weekend = oos & (d["dow"] >= 5)
    impact = filter_impact(base, weekend.loc[oos].reindex(base.index).fillna(False), "weekend")
    impact["family"] = "calendar"
    candidates.append(impact)

    # very weak directed taker + tight stop intersection
    if {"stop_atr", "taker_imb_1h"}.issubset(d.columns):
        remove = pd.Series(False, index=d.index)
        hs, ht = [], []
        for idx, row in d.iterrows():
            if len(hs) >= 80 and row["stop_atr"] == row["stop_atr"] and row["taker_imb_1h"] == row["taker_imb_1h"]:
                if row["stop_atr"] <= np.nanquantile(hs, 0.15) and row["taker_imb_1h"] <= np.nanquantile(ht, 0.25) and oos.loc[idx]:
                    remove.loc[idx] = True
            if row["stop_atr"] == row["stop_atr"]:
                hs.append(row["stop_atr"])
            if row["taker_imb_1h"] == row["taker_imb_1h"]:
                ht.append(row["taker_imb_1h"])
        impact = filter_impact(base, remove.loc[oos].reindex(base.index).fillna(False), "stop_atr low15 ∩ taker_imb low25")
        impact["family"] = "intersection"
        candidates.append(impact)

    # series: 2 fails same direction within 24h near price
    remove = pd.Series(False, index=d.index)
    hist = []
    for idx, row in d.iterrows():
        cutoff = row["entry_time"] - pd.Timedelta(hours=24)
        hist = [h for h in hist if h["entry_time"] >= cutoff]
        fails = [
            h
            for h in hist
            if h["direction"] == row["direction"]
            and abs(h["entry_price"] - row["entry_price"]) <= 1.5 * row["sl_usd"]
            and h["r"] < 0
        ]
        if len(fails) >= 2 and oos.loc[idx]:
            remove.loc[idx] = True
        hist.append({"entry_time": row["entry_time"], "direction": row["direction"], "entry_price": row["entry_price"], "r": row["r"]})
    impact = filter_impact(base, remove.loc[oos].reindex(base.index).fillna(False), "2fails_24h_near_1.5R")
    impact["family"] = "series"
    candidates.append(impact)

    # OI extensions if present
    if "dead_oi_ratio" in d.columns:
        for thr in [0.15, 0.20, 0.25, 0.30, 0.35]:
            covered = d["oi_available"].fillna(False)
            rem = covered & (d["dead_oi_ratio"] <= thr) & oos
            base_c = d.loc[covered & oos]
            if len(base_c) < 30:
                continue
            impact = filter_impact(base_c, rem.loc[base_c.index], f"DEAD_OI<={thr}")
            impact["family"] = "oi"
            candidates.append(impact)
    if "retail_chase" in d.columns:
        for thr in [0.015, 0.020, 0.025, 0.030, 0.040]:
            covered = d["oi_available"].fillna(False)
            rem = covered & (d["retail_chase"] >= thr) & oos
            base_c = d.loc[covered & oos]
            if len(base_c) < 30:
                continue
            impact = filter_impact(base_c, rem.loc[base_c.index], f"RETAIL_CHASE>={thr}")
            impact["family"] = "oi"
            candidates.append(impact)

    return pd.DataFrame(candidates).sort_values(["improves_total_r", "no_tail_loss", "delta_sum_r"], ascending=[False, False, False])


def permutation_p(base_r: np.ndarray, removed_idx: np.ndarray, n_perm: int = 2000, seed: int = 42) -> float:
    """Probability that a random set of same size has removed sum_r <= observed."""
    rng = np.random.default_rng(seed)
    obs = base_r[removed_idx].sum()
    k = len(removed_idx)
    if k == 0:
        return 1.0
    count = 0
    n = len(base_r)
    for _ in range(n_perm):
        idx = rng.choice(n, size=k, replace=False)
        if base_r[idx].sum() <= obs:
            count += 1
    return count / n_perm


def time_stop_experiment(df: pd.DataFrame) -> dict:
    """If after 1h: r_path < -0.25, MFE < 1, taker against -> force close at 1h mark."""
    # approximate using move_4h/4 as proxy is weak; use path features if bars available via mfe/mae only
    # We approximate 1h state with directed move over first 4 bars stored? Not stored.
    # Use available: if mae_r <= -0.25 and mfe_r < 1 and eventual r
    d = df.dropna(subset=["mae_r", "mfe_r"]).copy()
    bad = (d["mae_r"] <= -0.25) & (d["mfe_r"] < 1.0)
    # canonical result of group
    group = d.loc[bad]
    # hypothetical: close at -0.25R instead of final r when final would be whatever
    hyp = d["r"].copy()
    hyp.loc[bad] = -0.25
    return {
        "group_n": int(bad.sum()),
        "group_wr": float((group["r"] > 0).mean() * 100) if len(group) else np.nan,
        "group_sum_r": float(group["r"].sum()) if len(group) else 0.0,
        "canonical_sum": float(d["r"].sum()),
        "hyp_sum": float(hyp.sum()),
        "delta": float(hyp.sum() - d["r"].sum()),
    }


def main():
    print("Loading trades...")
    on_all = load_gf_on()
    off_all = load_gf_off()
    on = metrics_subset(on_all)
    off = metrics_subset(off_all)
    blocked = off.loc[off["gf_blocked"] & off["in_metrics"]]

    summary = {
        "gf_on": summarize_r(on),
        "gf_off": summarize_r(off),
        "gf_blocked": summarize_r(blocked),
    }
    print(json.dumps(summary, indent=2))

    print("Loading market data...")
    klines = load_klines()
    funding = load_funding()
    metrics = load_metrics()
    print("klines", len(klines), "funding", len(funding), "metrics", None if metrics is None else len(metrics))

    print("Attaching features...")
    feat = attach_path_features(on, klines)
    feat = attach_funding(feat, funding)
    feat = attach_oi_features(feat, metrics)
    feat = classify_negatives(feat)
    feat.to_parquet(ART / "gf_on_features.parquet", index=False)

    # also features for off blocked comparison sample
    off_feat = attach_path_features(off.loc[off["in_metrics"]], klines)
    off_feat = attach_funding(off_feat, funding)
    off_feat = attach_oi_features(off_feat, metrics)
    off_feat.to_parquet(ART / "gf_off_features.parquet", index=False)

    # Negative class table
    neg = feat.loc[feat["r"] < 0]
    neg_table = (
        neg.groupby("neg_class")
        .agg(n=("r", "size"), total_r=("r", "sum"), mean_r=("r", "mean"))
        .sort_values("total_r")
        .reset_index()
    )
    neg_table.to_csv(ART / "negative_classes.csv", index=False)
    print(neg_table)

    # Stop flush stats
    touched = feat.loc[(feat["r"] < 0) & feat["stop_touched"].fillna(False)]
    stop_stats = {
        "neg_stop_touched_n": int(len(touched)),
        "wick_share": float(touched["stop_wick_only"].mean()) if len(touched) else np.nan,
        "winners_touch_m05": float(((feat["r"] > 0) & (feat["mae_r"] <= -0.5)).mean()) if "mae_r" in feat else np.nan,
        "winners_touch_m075": float(((feat["r"] > 0) & (feat["mae_r"] <= -0.75)).mean()) if "mae_r" in feat else np.nan,
        "winners_touch_m09": float(((feat["r"] > 0) & (feat["mae_r"] <= -0.9)).mean()) if "mae_r" in feat else np.nan,
        "winners_touch_m1": float(((feat["r"] > 0) & (feat["mae_r"] <= -1.0)).mean()) if "mae_r" in feat else np.nan,
        "winner_r_after_m09": float(feat.loc[(feat["r"] > 0) & (feat["mae_r"] <= -0.9), "r"].sum()),
        "winner_r_after_m1": float(feat.loc[(feat["r"] > 0) & (feat["mae_r"] <= -1.0), "r"].sum()),
    }
    # reclaim after stop among negatives that touched stop: did price later reach +1R?
    # approximate with mfe before exit already in trade; for post-stop need full path — use mfe_r on negatives with wick
    stop_stats["wick_neg_later_mfe_ge1"] = float(
        ((feat["r"] < 0) & feat["stop_wick_only"].fillna(False) & (feat["mfe_r"] >= 1)).mean()
    )

    print("Walk-forward simple filters...")
    wf_rows = []
    for feat_name, side, q in [
        ("stop_pct", "low", 0.05),
        ("stop_atr", "low", 0.05),
        ("stop_atr", "low", 0.10),
        ("stop_24h", "low", 0.05),
        ("mom_1h", "low", 0.25),
        ("vol_comp_4h_72h", "low", 0.20),
    ]:
        if feat_name in feat.columns:
            wf_rows.append(walk_forward_quantile_filter(feat, feat_name, q, side))
    wf_rows.append(simple_intersection_filter(feat))
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(ART / "walkforward_simple_filters.csv", index=False)

    print("ML filters...")
    ml_rows = ml_stop_filter(feat)
    ml_df = pd.DataFrame(ml_rows)
    ml_df.to_csv(ART / "ml_filters.csv", index=False)

    print("Shadow rules...")
    shadow_rows = evaluate_shadow_rules(feat)
    shadow_df = pd.DataFrame(shadow_rows)
    shadow_df.to_csv(ART / "shadow_rules.csv", index=False)

    print("Candidate search...")
    cand_df = search_new_candidates(feat)
    cand_df.to_csv(ART / "candidate_filters.csv", index=False)

    # permutation for NODE_EXHAUSTION and best candidate with improves_total_r & no_tail_loss
    oos = feat["entry_time"] >= pd.Timestamp("2023-01-01", tz="UTC")
    base_oos = feat.loc[oos].reset_index(drop=True)
    exh = node_exhaustion_mask(feat).loc[oos].reset_index(drop=True)
    p_exh = permutation_p(base_oos["r"].to_numpy(), np.where(exh.to_numpy())[0])
    good = cand_df.loc[cand_df["improves_total_r"] & cand_df["no_tail_loss"]].copy()
    good.to_csv(ART / "promising_candidates.csv", index=False)

    ts = time_stop_experiment(feat)

    # GF rule marginals from off file
    rules = []
    if "gf_rules" in off.columns:
        blocked2 = off.loc[off["gf_blocked"] & off["in_metrics"]].copy()
        # unique single-rule blocks
        def rules_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    return json.loads(x.replace("'", '"'))
                except Exception:
                    return [x]
            return []

        blocked2["rules"] = blocked2["gf_rules"].map(rules_list)
        blocked2["n_rules"] = blocked2["rules"].map(len)
        singles = blocked2.loc[blocked2["n_rules"] == 1].copy()
        singles["rule"] = singles["rules"].map(lambda x: x[0] if x else None)
        for rule, g in singles.groupby("rule"):
            s = summarize_r(g)
            rules.append({"rule": rule, **s})
    rules_df = pd.DataFrame(rules).sort_values("sum_r") if rules else pd.DataFrame()
    rules_df.to_csv(ART / "gf_single_rule_marginals.csv", index=False)

    # OI tail mechanism summary
    oi_summary = {}
    if "oi_chg_4h" in feat.columns:
        covered = feat.loc[feat["oi_available"].fillna(False)]
        tails = covered.loc[covered["r"] > 5]
        negs = covered.loc[covered["r"] < 0]
        oi_summary = {
            "covered_n": int(len(covered)),
            "tail_n": int(len(tails)),
            "tail_long_oi_up": float(((tails["direction"] == "LONG") & (tails["oi_chg_4h"] > 0.005)).mean()) if len(tails) else np.nan,
            "tail_short_oi_down": float(((tails["direction"] == "SHORT") & (tails["oi_chg_4h"] < -0.005)).mean()) if len(tails) else np.nan,
            "tail_oi_chg_long_med": float(tails.loc[tails["direction"] == "LONG", "oi_chg_4h"].median()) if (tails["direction"] == "LONG").any() else np.nan,
            "tail_oi_chg_short_med": float(tails.loc[tails["direction"] == "SHORT", "oi_chg_4h"].median()) if (tails["direction"] == "SHORT").any() else np.nan,
            "neg_median_oi_chg": float(negs["oi_chg_4h"].median()) if len(negs) else np.nan,
        }

    report = {
        "summary": summary,
        "stop_stats": stop_stats,
        "time_stop": ts,
        "node_exhaustion_perm_p": p_exh,
        "oi_summary": oi_summary,
        "n_candidates": int(len(cand_df)),
        "n_promising": int(len(good)),
        "best_compromise": wf_df.sort_values("delta_wr", ascending=False).head(3).to_dict(orient="records"),
        "shadow": shadow_df.to_dict(orient="records"),
        "promising_top": good.head(15).to_dict(orient="records"),
    }
    with open(ART / "research_summary.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # Markdown report
    lines = []
    lines.append("# Исследование фильтров отрицательных сделок (TBX / GF ON)\n")
    lines.append("Доверяем последовательности `r` из экспорта. Extended-карточки исключены как артефакт.\n")
    lines.append("## База\n")
    lines.append(f"- GF ON: N={summary['gf_on']['n']}, SumR={summary['gf_on']['sum_r']:.2f}, E[R]={summary['gf_on']['expectancy']:.3f}, WR={summary['gf_on']['wr']:.2f}%, PF={summary['gf_on']['pf']:.2f}, MDD={summary['gf_on']['max_dd']:.2f}R")
    lines.append(f"- GF OFF: N={summary['gf_off']['n']}, SumR={summary['gf_off']['sum_r']:.2f}, E[R]={summary['gf_off']['expectancy']:.3f}, PF={summary['gf_off']['pf']:.2f}, MDD={summary['gf_off']['max_dd']:.2f}R")
    lines.append(f"- GF blocked: N={summary['gf_blocked']['n']}, SumR={summary['gf_blocked']['sum_r']:.2f}, E[R]={summary['gf_blocked']['expectancy']:.3f}, PF={summary['gf_blocked']['pf']:.2f}\n")
    lines.append("## Классы отрицательных сделок\n")
    lines.append(neg_table.to_markdown(index=False))
    lines.append("\n## Стоп-геометрия\n")
    for k, v in stop_stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Walk-forward простые фильтры (OOS 2023–2026)\n")
    lines.append(wf_df.to_markdown(index=False))
    lines.append("\n## Shadow-правила\n")
    lines.append(shadow_df.to_markdown(index=False))
    lines.append(f"\nPermutation p (NODE_EXHAUSTION removed sum ≤ obs): {p_exh:.4f}\n")
    lines.append("\n## Time-stop эксперимент\n")
    lines.append(json.dumps(ts, indent=2))
    lines.append("\n## OI механизм хвоста\n")
    lines.append(json.dumps(oi_summary, indent=2))
    lines.append("\n## Кандидаты с улучшением Total R и без потери хвоста >5R (OOS)\n")
    if len(good):
        lines.append(good.head(20)[["label", "family", "removed_n", "removed_sum_r", "delta_sum_r", "delta_wr", "delta_pf", "removed_tail_gt5"]].to_markdown(index=False))
    else:
        lines.append("_Ни один кандидат не улучшил Total R без потери хвоста._")
    lines.append("\n## ML-фильтры (фрагмент)\n")
    if len(ml_df):
        cols = [c for c in ["label", "auc", "removed_n", "removed_sum_r", "delta_sum_r", "delta_wr", "removed_tail_gt5"] if c in ml_df.columns]
        lines.append(ml_df[cols].head(20).to_markdown(index=False))
    lines.append("\n## Вердикт\n")
    lines.append("См. итоговый вывод в конце пайплайна и `research_summary.json`.")
    (REP / "negative_trade_filter_research.md").write_text("\n".join(lines))
    print("Wrote report", REP / "negative_trade_filter_research.md")


if __name__ == "__main__":
    main()
