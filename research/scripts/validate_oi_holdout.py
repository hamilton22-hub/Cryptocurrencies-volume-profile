#!/usr/bin/env python3
"""Validate OI shadow rules on true 2024-2026 holdout (full Binance metrics)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import attach_oi_features, load_metrics, node_exhaustion_mask  # noqa: E402
from load_trades import load_gf_on, metrics_subset, summarize_r  # noqa: E402
from analyze_filters import filter_impact, permutation_p, yearly_removed  # noqa: E402

ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def main():
    on = metrics_subset(load_gf_on())
    metrics = load_metrics()
    feat = attach_oi_features(on, metrics)
    print("OI coverage", feat["oi_available"].mean(), "n", feat["oi_available"].sum())

    discovery = (feat["entry_time"] >= "2022-01-01") & (feat["entry_time"] < "2024-01-01")
    holdout = feat["entry_time"] >= "2024-01-01"
    covered = feat["oi_available"].fillna(False)

    results = []

    # DEAD_OI thresholds
    for thr in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for name, mask_period in [("discovery_2022_2023", discovery), ("holdout_2024_2026", holdout), ("all_covered", covered)]:
            base = feat.loc[covered & mask_period] if name != "all_covered" else feat.loc[covered]
            rem = (base["dead_oi_ratio"] <= thr)
            impact = filter_impact(base, rem, f"DEAD_OI<={thr} [{name}]")
            impact["year_breakdown"] = yearly_removed(base, rem).to_dict(orient="records")
            results.append(impact)

    # RETAIL_CHASE
    for thr in [0.015, 0.02, 0.025, 0.03, 0.04]:
        for name, mask_period in [("discovery_2022_2023", discovery), ("holdout_2024_2026", holdout), ("all_covered", covered)]:
            base = feat.loc[covered & mask_period] if name != "all_covered" else feat.loc[covered]
            rem = base["retail_chase"] >= thr
            impact = filter_impact(base, rem, f"RETAIL_CHASE>={thr} [{name}]")
            impact["year_breakdown"] = yearly_removed(base, rem).to_dict(orient="records")
            results.append(impact)

    # Combined fixed thresholds from prior research
    for name, mask_period in [("discovery_2022_2023", discovery), ("holdout_2024_2026", holdout), ("all_covered", covered)]:
        base = feat.loc[covered & mask_period] if name != "all_covered" else feat.loc[covered]
        rem = (base["dead_oi_ratio"] <= 0.25) | (base["retail_chase"] >= 0.025)
        impact = filter_impact(base, rem, f"DEAD_OI∪RETAIL_CHASE [{name}]")
        impact["year_breakdown"] = yearly_removed(base, rem).to_dict(orient="records")
        # permutation on holdout
        if name == "holdout_2024_2026":
            idx = np.where(rem.to_numpy())[0]
            impact["perm_p"] = permutation_p(base["r"].to_numpy(), idx, n_perm=5000)
        results.append(impact)

    # NODE_EXHAUSTION on full GF ON with OOS splits
    exh = node_exhaustion_mask(feat)
    for name, mask_period in [
        ("oos_2023_2026", feat["entry_time"] >= "2023-01-01"),
        ("holdout_2024_2026", holdout),
        ("all", pd.Series(True, index=feat.index)),
    ]:
        base = feat.loc[mask_period]
        rem = exh.loc[mask_period]
        impact = filter_impact(base, rem, f"NODE_EXHAUSTION [{name}]")
        impact["year_breakdown"] = yearly_removed(base, rem).to_dict(orient="records")
        if name != "all":
            impact["perm_p"] = permutation_p(base["r"].to_numpy(), np.where(rem.to_numpy())[0], n_perm=5000)
        results.append(impact)

    # Triple combo on holdout where OI available
    base = feat.loc[holdout]
    rem = exh.loc[holdout] | (
        covered.loc[holdout]
        & ((feat.loc[holdout, "dead_oi_ratio"] <= 0.25) | (feat.loc[holdout, "retail_chase"] >= 0.025))
    )
    impact = filter_impact(base, rem, "TRIPLE_SHADOW holdout_2024_2026")
    impact["year_breakdown"] = yearly_removed(base, rem).to_dict(orient="records")
    impact["perm_p"] = permutation_p(base["r"].to_numpy(), np.where(rem.to_numpy())[0], n_perm=5000)
    results.append(impact)

    df = pd.DataFrame(results)
    # flatten for csv
    flat = df.drop(columns=[c for c in ["year_breakdown"] if c in df.columns])
    flat.to_csv(ART / "oi_holdout_validation.csv", index=False)
    with open(ART / "oi_holdout_validation.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    # Print key lines
    keys = [
        "DEAD_OI<=0.25 [discovery_2022_2023]",
        "DEAD_OI<=0.25 [holdout_2024_2026]",
        "RETAIL_CHASE>=0.025 [discovery_2022_2023]",
        "RETAIL_CHASE>=0.025 [holdout_2024_2026]",
        "DEAD_OI∪RETAIL_CHASE [holdout_2024_2026]",
        "NODE_EXHAUSTION [oos_2023_2026]",
        "NODE_EXHAUSTION [holdout_2024_2026]",
        "TRIPLE_SHADOW holdout_2024_2026",
    ]
    for r in results:
        if r["label"] in keys:
            print(
                r["label"],
                "n=",
                r["removed_n"],
                "remR=",
                round(r["removed_sum_r"], 2),
                "dR=",
                round(r["delta_sum_r"], 2),
                "tail=",
                r["removed_tail_gt5"],
                "p=",
                r.get("perm_p"),
            )


if __name__ == "__main__":
    main()
