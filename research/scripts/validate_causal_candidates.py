#!/usr/bin/env python3
"""Validate frozen causal candidates and correct exploratory rule families."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_causal_policies import (  # noqa: E402
    EARLY_EXIT_RULES,
    ExtensionPolicy,
    NUMERIC_FEATURES,
    _json_safe,
)
from load_trades import load_gf_on, metrics_subset, summarize_r  # noqa: E402

ART = ROOT / "artifacts"


def _monthly_sign_flip(
    values: np.ndarray,
    months: np.ndarray,
    masks: np.ndarray,
    observed_threshold: float,
    candidate_mask: np.ndarray | None = None,
    iterations: int = 10_000,
) -> dict:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    values = np.asarray(values, dtype=float)
    masks = np.asarray(masks, dtype=float)
    months = np.asarray(months)
    if masks.ndim != 2 or masks.shape[1] != len(values) or len(months) != len(values):
        raise ValueError("values, months, and mask columns must align")
    if not np.isfinite(values).all() or not np.isfinite(masks).all():
        raise ValueError("sign-flip inputs must be finite")
    if not np.isfinite(observed_threshold):
        raise ValueError("observed threshold must be finite")
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=float)
        if candidate_mask.shape != values.shape or not np.isfinite(candidate_mask).all():
            raise ValueError("candidate mask must be finite and aligned")
    rng = np.random.default_rng(42)
    unique_months, month_codes = np.unique(months, return_inverse=True)
    monthly_family_effects = np.zeros((masks.shape[0], len(unique_months)))
    weighted_masks = masks * values
    for month_code in range(len(unique_months)):
        monthly_family_effects[:, month_code] = weighted_masks[
            :, month_codes == month_code
        ].sum(axis=1)
    monthly_candidate_effects = None
    if candidate_mask is not None:
        monthly_candidate_effects = np.zeros(len(unique_months))
        candidate_effects = candidate_mask * values
        for month_code in range(len(unique_months)):
            monthly_candidate_effects[month_code] = candidate_effects[
                month_codes == month_code
            ].sum()
    family_hits = 0
    candidate_hits = 0
    for _ in range(iterations):
        signs = rng.choice([-1.0, 1.0], size=len(unique_months))
        family_hits += bool(
            np.max(monthly_family_effects @ signs) >= observed_threshold
        )
        if monthly_candidate_effects is not None:
            candidate_hits += bool(
                np.sum(monthly_candidate_effects * signs) >= observed_threshold
            )
    result = {
        "family_p": (family_hits + 1) / (iterations + 1),
        "iterations": iterations,
    }
    if candidate_mask is not None:
        result["unadjusted_p"] = (candidate_hits + 1) / (iterations + 1)
    return result


def rescue_candidate(adverse: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    policy = ExtensionPolicy(0.65, 1.25, 0.50)
    delta_col = f"{policy.name}__delta_full"
    risknorm_delta_col = f"{policy.name}__delta_risknorm"
    result_col = f"{policy.name}__full_r"
    hit_col = f"{policy.name}__original_stop_hit"
    outcome_col = f"{policy.name}__outcome"
    label_end_col = f"{policy.name}__label_end_time"

    data = adverse[
        (adverse["timeframe"] == "15M") & (adverse["checkpoint"] == 0.65)
    ].copy()
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    discovery = (data["decision_time"] < cutoff) & (
        data[label_end_col] < cutoff
    )
    volume_threshold = float(data.loc[discovery, "volume_vs_pre24h"].quantile(0.5))
    flow_threshold = float(
        data.loc[discovery, "taker_imb_since_fill"].quantile(0.5)
    )
    selected = data[
        (data["volume_vs_pre24h"] <= volume_threshold)
        & (data["taker_imb_since_fill"] >= flow_threshold)
    ].copy()
    selected["delta_full"] = selected[delta_col]
    selected["delta_risknorm"] = selected[risknorm_delta_col]
    selected["year"] = selected["entry_time"].dt.year
    economic = selected["delta_full"].notna()

    ledger = metrics_subset(load_gf_on()).sort_values("entry_time").copy()
    ledger["r_canonical"] = ledger["r"]
    replacement = dict(
        zip(selected.loc[economic, "trade_id"], selected.loc[economic, result_col])
    )
    replacement_values = ledger["trade_id"].map(replacement)
    ledger["r"] = ledger["r"].where(
        ~ledger["trade_id"].isin(replacement), replacement_values
    )

    changed = selected[selected[hit_col] & economic].copy()
    changed["month"] = changed["entry_time"].dt.tz_localize(None).dt.to_period("M").astype(str)
    # Unadjusted block p for the frozen rule.
    candidate_mask = np.ones(len(changed))
    unadjusted = _monthly_sign_flip(
        changed["delta_full"].to_numpy(),
        changed["month"].to_numpy(),
        np.ones((1, len(changed))),
        float(changed["delta_full"].sum()),
        candidate_mask,
    )

    summary = {
        "name": "LOW_ENERGY_ADVERSE_PROBE",
        "policy": policy.name,
        "volume_threshold": volume_threshold,
        "flow_threshold": flow_threshold,
        "decision_events": int(len(selected)),
        "original_stop_hits": int(selected[hit_col].sum()),
        "reconciled_stop_hits": int((selected[hit_col] & economic).sum()),
        "unreconciled_stop_hits": int((selected[hit_col] & ~economic).sum()),
        "outcomes": selected[outcome_col].value_counts().to_dict(),
        "gross_delta_full_reconciled_subset": float(selected["delta_full"].sum()),
        "gross_delta_risknorm": float(selected["delta_risknorm"].sum()),
        "delta_by_year": (
            selected.groupby("year")["delta_full"].sum().round(6).to_dict()
        ),
        "net_delta_by_cost_per_changed_trade": {
            str(cost): float(selected["delta_full"].sum() - hit_col_count * cost)
            for cost in (0.02, 0.05, 0.10, 0.20)
            for hit_col_count in [int((selected[hit_col] & economic).sum())]
        },
        "modified_ledger_entry_order_approx": summarize_r(ledger),
        "unadjusted_monthly_block_p": unadjusted["unadjusted_p"],
        "r_reconciliation_tolerance": 0.10,
    }
    return summary, selected


def rescue_family_correction(
    adverse: pd.DataFrame,
    selected: pd.DataFrame,
    iterations: int = 10_000,
) -> dict:
    policy = ExtensionPolicy(0.65, 1.25, 0.50)
    delta_col = f"{policy.name}__delta_full"
    label_end_col = f"{policy.name}__label_end_time"
    data = adverse[
        (adverse["timeframe"] == "15M") & (adverse["checkpoint"] == 0.65)
    ].dropna(subset=[delta_col]).copy().reset_index(drop=True)
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    discovery = (data["decision_time"] < cutoff) & (
        data[label_end_col] < cutoff
    )
    oos = data["decision_time"] >= cutoff
    features = [
        c
        for c in NUMERIC_FEATURES
        if c in data.columns and data.loc[discovery, c].notna().sum() >= 20
    ]

    conditions: list[tuple[str, pd.Series]] = []
    for feature in features:
        for quantile in (0.2, 0.3, 0.5, 0.7, 0.8):
            threshold = data.loc[discovery, feature].quantile(quantile)
            for side in ("low", "high"):
                mask = (
                    data[feature] <= threshold
                    if side == "low"
                    else data[feature] >= threshold
                )
                if mask[discovery].sum() >= 5 and mask[oos].sum() >= 5:
                    conditions.append(
                        (f"{feature}|{side}|q{quantile}", mask.fillna(False))
                    )

    rules: list[tuple[str, pd.Series]] = []
    for name, mask in conditions:
        if data.loc[discovery & mask, delta_col].sum() > 0:
            rules.append((name, mask))
    for i, (name_a, mask_a) in enumerate(conditions):
        feature_a = name_a.split("|")[0]
        for name_b, mask_b in conditions[i + 1 :]:
            if feature_a == name_b.split("|")[0]:
                continue
            mask = mask_a & mask_b
            if (
                mask[discovery].sum() >= 5
                and mask[oos].sum() >= 5
                and data.loc[discovery & mask, delta_col].sum() > 0
            ):
                rules.append((f"{name_a}&{name_b}", mask))

    deduplicated: dict[tuple[bool, ...], tuple[str, pd.Series]] = {}
    for name, mask in rules:
        deduplicated.setdefault(tuple(mask.to_numpy()), (name, mask))
    rules = list(deduplicated.values())
    values = data.loc[oos, delta_col].to_numpy()
    selected_ids = set(selected["trade_id"])
    candidate_mask = data.loc[oos, "trade_id"].isin(selected_ids).to_numpy(float)
    candidate_oos_delta = float(np.sum(candidate_mask * values))
    if not rules:
        return {
            "scope": "feature rules within fixed q0.65/E1.25/T0.50 policy",
            "conditions": len(conditions),
            "unique_discovery_positive_rules": 0,
            "candidate_oos_delta": candidate_oos_delta,
            "candidate_oos_decisions": int(candidate_mask.sum()),
            "family_p": None,
            "unadjusted_p": None,
            "reason": "No rule had positive paired delta in the discovery sample.",
        }

    matrix = np.asarray([mask[oos].to_numpy(float) for _, mask in rules])
    observed = matrix @ values
    months = (
        data.loc[oos, "entry_time"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .to_numpy()
    )
    correction = _monthly_sign_flip(
        values,
        months,
        matrix,
        candidate_oos_delta,
        candidate_mask,
        iterations,
    )
    correction.update(
        {
            "scope": "feature rules within fixed q0.65/E1.25/T0.50 policy",
            "conditions": len(conditions),
            "unique_rules": len(rules),
            "candidate_oos_delta": candidate_oos_delta,
            "candidate_oos_decisions": int(candidate_mask.sum()),
            "best_oos_delta": float(observed.max()),
        }
    )
    return correction


def early_exit_family_correction(
    timed: pd.DataFrame, iterations: int = 10_000
) -> dict:
    data = timed[
        (timed["timeframe"] == "15M")
        & (timed["decision_time"] >= pd.Timestamp("2023-01-01", tz="UTC"))
    ].dropna(subset=["delta_early_exit"]).copy()
    trades = (
        data[["trade_id", "entry_time"]]
        .drop_duplicates()
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    position = {trade_id: i for i, trade_id in enumerate(trades["trade_id"])}
    names = []
    vectors = []
    for bars, current_max, mfe_max, taker_against in EARLY_EXIT_RULES:
        checkpoint = data[data["checkpoint_bars"] == bars]
        mask = (checkpoint["current_r"] <= current_max) & (
            checkpoint["mfe_sofar"] <= mfe_max
        )
        if taker_against:
            mask &= checkpoint["taker_imb_since_fill"] < 0
        acted = checkpoint[mask]
        if len(acted) < 5:
            continue
        vector = np.zeros(len(trades))
        action_positions = acted["trade_id"].map(position).to_numpy(int)
        vector[action_positions] = acted["delta_early_exit"].to_numpy()
        names.append(f"{bars}|{current_max}|{mfe_max}|{taker_against}")
        vectors.append(vector)
    matrix = np.asarray(vectors)
    observed = matrix.sum(axis=1)
    best_index = int(np.argmax(observed))
    months = (
        trades["entry_time"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .to_numpy()
    )
    correction = _monthly_sign_flip(
        np.ones(len(trades)),
        months,
        matrix,
        float(observed[best_index]),
        candidate_mask=matrix[best_index],
        iterations=iterations,
    )
    yearly = pd.DataFrame(
        {
            "year": trades["entry_time"].dt.year,
            "delta": matrix[best_index],
        }
    ).groupby("year")["delta"].sum()
    correction.update(
        {
            "rules": len(names),
            "best_rule": names[best_index],
            "best_delta": float(observed[best_index]),
            "best_delta_by_year": yearly.round(6).to_dict(),
        }
    )
    return correction


def giveback_validation(details: pd.DataFrame, iterations: int = 10_000) -> dict:
    data = details[details["timeframe"] == "15M"].copy()
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    discovery = data["entry_time"] < cutoff
    discovery_delta = (
        data.loc[discovery]
        .groupby("close_threshold_r")["delta_r"]
        .sum()
        .sort_values(ascending=False)
    )
    eligible_thresholds = discovery_delta[discovery_delta > 0].index.tolist()
    oos = data[data["entry_time"] >= cutoff].copy()
    trades = (
        oos[["trade_id", "entry_time"]]
        .drop_duplicates()
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if not eligible_thresholds or not len(trades):
        return {
            "eligible_discovery_positive_rules": 0,
            "reason": "No discovery-positive 15M giveback rule.",
        }
    position = {trade_id: i for i, trade_id in enumerate(trades["trade_id"])}
    vectors = []
    for threshold in eligible_thresholds:
        vector = np.zeros(len(trades))
        acted = oos[oos["close_threshold_r"] == threshold]
        vector[acted["trade_id"].map(position).to_numpy(int)] = acted[
            "delta_r"
        ].to_numpy()
        vectors.append(vector)
    matrix = np.asarray(vectors)
    oos_delta = matrix.sum(axis=1)
    chosen_threshold = float(discovery_delta.index[0])
    chosen_index = eligible_thresholds.index(chosen_threshold)
    months = (
        trades["entry_time"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .to_numpy()
    )
    correction = _monthly_sign_flip(
        np.ones(len(trades)),
        months,
        matrix,
        float(oos_delta[chosen_index]),
        candidate_mask=matrix[chosen_index],
        iterations=iterations,
    )
    chosen = data[data["close_threshold_r"] == chosen_threshold].copy()
    correction.update(
        {
            "eligible_discovery_positive_rules": len(eligible_thresholds),
            "chosen_threshold_r": chosen_threshold,
            "discovery_delta": float(discovery_delta.loc[chosen_threshold]),
            "oos_delta": float(oos_delta[chosen_index]),
            "full_sample_actions": int(len(chosen)),
            "full_sample_delta": float(chosen["delta_r"].sum()),
            "full_sample_tail_actions": int(chosen["tail_gt5"].sum()),
            "delta_by_year": (
                chosen.assign(year=chosen["entry_time"].dt.year)
                .groupby("year")["delta_r"]
                .sum()
                .round(6)
                .to_dict()
            ),
        }
    )
    return correction


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    adverse = pd.read_parquet(ART / "causal_adverse_checkpoints.parquet")
    timed = pd.read_parquet(ART / "causal_timed_checkpoints.parquet")
    giveback_details = pd.read_csv(
        ART / "causal_giveback_details.csv", parse_dates=["entry_time"]
    )
    rescue, selected = rescue_candidate(adverse)
    family = rescue_family_correction(adverse, selected)
    early = early_exit_family_correction(timed)
    giveback = giveback_validation(giveback_details)
    selected.to_csv(ART / "causal_rescue_candidate_trades.csv", index=False)
    result = {
        "rescue_candidate": rescue,
        "rescue_within_policy_rule_search": family,
        "early_exit_multiple_testing": early,
        "giveback_validation": giveback,
    }
    result = _json_safe(result)
    with open(ART / "causal_candidate_validation.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str, allow_nan=False)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
