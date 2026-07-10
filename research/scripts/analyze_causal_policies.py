#!/usr/bin/env python3
"""Point-in-time policy research for adverse excursions and early exits.

This supersedes the diagnostic ``stop_hunt`` counterfactual.  Decisions are
made only after a completed 15-minute bar and before the original stop has
traded.  The primary target is paired policy value, not a hindsight class.
"""

from __future__ import annotations

import json
import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_klines, load_metrics  # noqa: E402
from load_trades import load_gf_on, metrics_subset  # noqa: E402

ART = ROOT / "artifacts"


@dataclass(frozen=True)
class ExtensionPolicy:
    checkpoint: float
    emergency: float
    target: float
    horizon_bars: int = 48

    @property
    def name(self) -> str:
        return (
            f"q{self.checkpoint:.2f}_E{self.emergency:.2f}"
            f"_T{self.target:.2f}_H{self.horizon_bars}"
        )


EXTENSION_POLICIES = tuple(
    ExtensionPolicy(checkpoint, emergency, target)
    for checkpoint in (0.65, 0.75, 0.85)
    for emergency in (1.25, 1.50)
    for target in (0.0, 0.5, 1.0)
)
POLICIES_BY_CHECKPOINT = {
    checkpoint: tuple(
        policy for policy in EXTENSION_POLICIES if policy.checkpoint == checkpoint
    )
    for checkpoint in (0.65, 0.75, 0.85)
}
EARLY_EXIT_RULES = tuple(
    (bars, current_max, mfe_max, taker_against)
    for bars in (2, 4, 8)
    for current_max in (-0.50, -0.25, 0.0, 0.25)
    for mfe_max in (0.25, 0.50, 1.0)
    for taker_against in (False, True)
)


def _directional_r(price: float, entry: float, risk: float, sign: float) -> float:
    return sign * (price - entry) / risk


def _first_touch_entry(
    times: pd.DatetimeIndex,
    lows: np.ndarray,
    highs: np.ndarray,
    entry_time: pd.Timestamp,
    entry_price: float,
    max_bars: int = 8,
) -> tuple[int | None, int | None]:
    """Infer the first 15m bar containing entry_price at/after signal time."""
    i0 = int(times.searchsorted(entry_time, side="left"))
    for i in range(i0, min(i0 + max_bars + 1, len(times))):
        if lows[i] <= entry_price <= highs[i]:
            return i, i - i0
    return None, None


def _race_policy(
    start: int,
    end: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry: float,
    risk: float,
    sign: float,
    emergency: float,
    target: float,
) -> tuple[float, float, str, int]:
    """First-passage policy return.

    Returns conservative and optimistic outcomes.  If both barriers trade in
    one 15m bar, conservative assumes emergency first and optimistic target
    first.  Gaps execute at the observed open.
    """
    emergency_px = entry - sign * emergency * risk
    target_px = entry + sign * target * risk
    for i in range(start, end + 1):
        open_r = _directional_r(opens[i], entry, risk, sign)
        if open_r <= -emergency:
            return open_r, open_r, "emergency_gap", i
        if open_r >= target:
            return open_r, open_r, "target_gap", i

        if sign > 0:
            hit_emergency = lows[i] <= emergency_px
            hit_target = highs[i] >= target_px
        else:
            hit_emergency = highs[i] >= emergency_px
            hit_target = lows[i] <= target_px

        if hit_emergency and hit_target:
            return -emergency, target, "same_bar_ambiguous", i
        if hit_emergency:
            return -emergency, -emergency, "emergency", i
        if hit_target:
            return target, target, "target", i

    timeout_r = _directional_r(closes[end], entry, risk, sign)
    return timeout_r, timeout_r, "timeout", end


def _race_after_stop_activation(
    activation_idx: int,
    end: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry: float,
    risk: float,
    sign: float,
    emergency: float,
    target: float,
) -> tuple[float, float, str, int]:
    """Race after the original stop activates an already-authorized extension.

    If the activation bar opens inside the original stop, its open and any
    favorable wick precede the unknown intrabar stop time and cannot be
    credited conservatively.  A farther adverse emergency is necessarily
    crossed after the original stop.  A close beyond target is also known to
    occur after activation because the close is the final bar event.
    """
    open_r = _directional_r(opens[activation_idx], entry, risk, sign)
    if open_r <= -1.0:
        return _race_policy(
            activation_idx,
            end,
            opens,
            highs,
            lows,
            closes,
            entry,
            risk,
            sign,
            emergency,
            target,
        )

    emergency_px = entry - sign * emergency * risk
    target_px = entry + sign * target * risk
    if sign > 0:
        hit_emergency = lows[activation_idx] <= emergency_px
        hit_target = highs[activation_idx] >= target_px
    else:
        hit_emergency = highs[activation_idx] >= emergency_px
        hit_target = lows[activation_idx] <= target_px
    close_r = _directional_r(closes[activation_idx], entry, risk, sign)

    if hit_emergency:
        optimistic = target if hit_target else -emergency
        outcome = (
            "activation_bar_ambiguous"
            if hit_target
            else "activation_bar_emergency"
        )
        return -emergency, optimistic, outcome, activation_idx
    if close_r >= target:
        return target, target, "activation_bar_target_at_close", activation_idx
    if activation_idx == end:
        optimistic = target if hit_target else close_r
        return close_r, optimistic, "activation_bar_timeout", activation_idx

    conservative, optimistic, outcome, outcome_idx = _race_policy(
        activation_idx + 1,
        end,
        opens,
        highs,
        lows,
        closes,
        entry,
        risk,
        sign,
        emergency,
        target,
    )
    if hit_target:
        optimistic = max(optimistic, target)
        outcome = f"activation_target_ambiguous_then_{outcome}"
    return conservative, optimistic, outcome, outcome_idx


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _risk_normalized_result(
    execution_r: float,
    emergency: float,
    continuation_r: float,
) -> tuple[float, float]:
    """Partial reduction sized to a nominal -1R emergency result."""
    if execution_r <= -emergency:
        return 1.0, continuation_r
    denominator = emergency + execution_r
    retained = (1 + execution_r) / denominator if denominator > 0 else 0.0
    retained = float(np.clip(retained, 0.0, 1.0))
    combined = (1 - retained) * execution_r + retained * continuation_r
    return retained, combined


def _last_bar_indices(
    bar_close_times: pd.DatetimeIndex,
    exit_time: pd.Timestamp,
) -> tuple[int, int]:
    """Last decision bar (< exit) and fully observed path bar (<= exit)."""
    last_decision = int(bar_close_times.searchsorted(exit_time, side="left") - 1)
    last_path = int(bar_close_times.searchsorted(exit_time, side="right") - 1)
    return last_decision, last_path


def _asof_position(times: pd.DatetimeIndex, when: pd.Timestamp) -> int | None:
    pos = int(times.searchsorted(when, side="right") - 1)
    return pos if pos >= 0 else None


def _oi_snapshot(
    metric_times: pd.DatetimeIndex,
    oi: np.ndarray,
    global_ls: np.ndarray,
    taker_ls: np.ndarray,
    decision_time: pd.Timestamp,
    fill_time: pd.Timestamp,
    publication_lag_minutes: int = 5,
) -> dict:
    """OI features available with an explicit publication lag."""
    available_at = decision_time - pd.Timedelta(minutes=publication_lag_minutes)
    j = _asof_position(metric_times, available_at)
    jf = _asof_position(metric_times, fill_time - pd.Timedelta(minutes=publication_lag_minutes))
    j1 = _asof_position(metric_times, available_at - pd.Timedelta(hours=1))
    if j is None:
        return {}

    def log_change(values: np.ndarray, a: int | None, b: int | None) -> float:
        if a is None or b is None or values[a] <= 0 or values[b] <= 0:
            return np.nan
        return float(np.log(values[a] / values[b]))

    return {
        "oi_delta_fill": log_change(oi, j, jf),
        "oi_delta_1h": log_change(oi, j, j1),
        "global_ls_delta_1h": log_change(global_ls, j, j1),
        "taker_ls_now": float(taker_ls[j]) if np.isfinite(taker_ls[j]) else np.nan,
        "oi_age_minutes": float((available_at - metric_times[j]).total_seconds() / 60),
    }


def _base_features(
    t: pd.Series,
    fill_idx: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    taker_buy: np.ndarray,
) -> dict:
    """Features using bars strictly before the inferred fill bar."""
    i = fill_idx - 1
    if i < 16:
        return {}
    sign = float(t["sign"])

    def directed_return(n: int) -> float:
        j = max(0, i - n)
        return sign * (closes[i] / closes[j] - 1)

    def range_pct(n: int) -> float:
        j = max(0, i - n + 1)
        return float((np.nanmax(highs[j : i + 1]) - np.nanmin(lows[j : i + 1])) / closes[i])

    j1 = max(0, i - 3)
    vv = float(np.nansum(volumes[j1 : i + 1]))
    tb = float(np.nansum(taker_buy[j1 : i + 1]))
    atr = float(np.nanmean(highs[max(0, i - 13) : i + 1] - lows[max(0, i - 13) : i + 1]))
    return {
        "stop_pct": float(t["sl_usd"] / t["entry_price"]),
        "stop_atr_pre": float(t["sl_usd"] / atr) if atr > 0 else np.nan,
        "mom_1h_pre": directed_return(4),
        "mom_4h_pre": directed_return(16),
        "range_4h_pre": range_pct(16),
        "range_24h_pre": range_pct(96),
        "range_72h_pre": range_pct(288),
        "taker_imb_1h_pre": sign * (2 * tb / vv - 1) if vv > 0 else np.nan,
    }


def _checkpoint_features(
    t: pd.Series,
    fill_idx: int,
    decision_idx: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    taker_buy: np.ndarray,
) -> dict:
    """Path features through the completed decision bar."""
    start = fill_idx + 1
    if decision_idx < start:
        return {}
    entry = float(t["entry_price"])
    risk = float(t["sl_usd"])
    sign = float(t["sign"])
    sl = slice(start, decision_idx + 1)
    close_r = _directional_r(closes[decision_idx], entry, risk, sign)
    if sign > 0:
        mfe = (np.nanmax(highs[sl]) - entry) / risk
        mae = (np.nanmin(lows[sl]) - entry) / risk
    else:
        mfe = (entry - np.nanmin(lows[sl])) / risk
        mae = (entry - np.nanmax(highs[sl])) / risk
    vv = float(np.nansum(volumes[sl]))
    tb = float(np.nansum(taker_buy[sl]))
    signed_path = sign * np.diff(closes[start - 1 : decision_idx + 1])
    path_eff = (
        abs(float(np.nansum(signed_path))) / float(np.nansum(np.abs(signed_path)))
        if np.nansum(np.abs(signed_path)) > 0
        else 0.0
    )
    reference_volume = volumes[max(0, fill_idx - 96) : fill_idx]
    vol_ref = (
        float(np.nanmedian(reference_volume)) if len(reference_volume) else np.nan
    )
    return {
        "current_r": float(close_r),
        "mfe_sofar": float(mfe),
        "mae_sofar": float(mae),
        "bars_since_fill": int(decision_idx - fill_idx),
        "path_efficiency": path_eff,
        "taker_imb_since_fill": sign * (2 * tb / vv - 1) if vv > 0 else np.nan,
        "volume_vs_pre24h": float(np.nanmean(volumes[sl]) / vol_ref) if vol_ref > 0 else np.nan,
        "last_bar_body_r": float(sign * (closes[decision_idx] - opens[decision_idx]) / risk),
        "last_bar_range_r": float((highs[decision_idx] - lows[decision_idx]) / risk),
    }


def build_snapshots(
    trades: pd.DataFrame,
    klines: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = pd.DatetimeIndex(klines["open_time"])
    bar_close_times = times + pd.Timedelta(minutes=15)
    opens = klines["open"].to_numpy(float)
    highs = klines["high"].to_numpy(float)
    lows = klines["low"].to_numpy(float)
    closes = klines["close"].to_numpy(float)
    volumes = klines["volume"].to_numpy(float)
    taker_buy = klines["taker_buy_volume"].to_numpy(float)

    metric_times = pd.DatetimeIndex(metrics["create_time"])
    oi = metrics["oi"].to_numpy(float)
    global_ls = metrics["global_ls"].to_numpy(float)
    taker_ls = metrics["taker_ls"].to_numpy(float)

    trade_rows: list[dict] = []
    adverse_rows: list[dict] = []
    timed_rows: list[dict] = []
    for _, t in trades.sort_values("entry_time").iterrows():
        fill_idx, fill_lag = _first_touch_entry(
            times, lows, highs, t["entry_time"], float(t["entry_price"])
        )
        if fill_idx is None:
            continue
        # 15M should fill in its timestamp bar.  1H remains a proxy because
        # entry_time appears to be signal-time in most rows.
        fill_quality = "exact_15m" if t["timeframe"] == "15M" and fill_lag == 0 else "proxy"
        fill_time = times[fill_idx]
        exit_idx = int(times.searchsorted(t["exit_time_parsed"], side="right") - 1)
        exit_idx = max(fill_idx, min(exit_idx, len(times) - 1))
        entry = float(t["entry_price"])
        risk = float(t["sl_usd"])
        sign = float(t["sign"])
        stop = float(t["stop_price"])
        canonical_price_r = _directional_r(
            float(t["exit_price"]), entry, risk, sign
        )
        reconciliation_error = float(t["r"] - canonical_price_r)
        r_reconciled = abs(reconciliation_error) <= 0.10
        base = {
            "trade_id": t["trade_id"],
            "entry_time": t["entry_time"],
            "fill_time_proxy": fill_time,
            "decision_time": fill_time,
            "label_end_time": t["exit_time_parsed"],
            "exit_time": t["exit_time_parsed"],
            "fill_lag_bars": fill_lag,
            "fill_quality": fill_quality,
            "timeframe": t["timeframe"],
            "setup": t["setup"],
            "direction": t["direction"],
            "r": float(t["r"]),
            "canonical_price_r": canonical_price_r,
            "r_reconciliation_error": reconciliation_error,
            "r_reconciled": r_reconciled,
            "tail_gt5": bool(t["r"] > 5),
        }
        base.update(_base_features(t, fill_idx, highs, lows, closes, volumes, taker_buy))
        base.update(
            _oi_snapshot(
                metric_times,
                oi,
                global_ls,
                taker_ls,
                fill_time,
                fill_time,
                publication_lag_minutes=5,
            )
        )
        trade_rows.append(base)

        # Timed checkpoints use only fully completed bars after the fill bar.
        for bars in (2, 4, 8):
            decision_idx = fill_idx + bars
            if decision_idx + 1 >= len(times):
                continue
            decision_time = times[decision_idx] + pd.Timedelta(minutes=15)
            if decision_time >= t["exit_time_parsed"]:
                continue
            path = slice(fill_idx + 1, decision_idx + 1)
            if sign > 0:
                original_stop_traded = bool(np.nanmin(lows[path]) <= stop)
            else:
                original_stop_traded = bool(np.nanmax(highs[path]) >= stop)
            if original_stop_traded:
                continue
            exit_r_next_open = _directional_r(opens[decision_idx + 1], entry, risk, sign)
            row = {
                **base,
                "checkpoint_bars": bars,
                "decision_time": decision_time,
                "action_r": exit_r_next_open,
                "delta_early_exit": (
                    exit_r_next_open - float(t["r"])
                    if r_reconciled
                    else np.nan
                ),
            }
            row.update(
                _checkpoint_features(
                    t, fill_idx, decision_idx, opens, highs, lows, closes, volumes, taker_buy
                )
            )
            row.update(
                _oi_snapshot(
                    metric_times,
                    oi,
                    global_ls,
                    taker_ls,
                    decision_time,
                    fill_time,
                    publication_lag_minutes=5,
                )
            )
            timed_rows.append(row)

        # Adverse checkpoint: completed close below q, but original stop has
        # not traded in any bar since the inferred fill.
        last_decision_idx, last_path_idx = _last_bar_indices(
            bar_close_times, t["exit_time_parsed"]
        )
        for q in (0.65, 0.75, 0.85):
            decision_idx = None
            max_scan = min(last_decision_idx, fill_idx + 48)
            for i in range(fill_idx + 1, max_scan + 1):
                if sign > 0:
                    stop_traded = lows[i] <= stop
                else:
                    stop_traded = highs[i] >= stop
                if stop_traded:
                    break
                close_r = _directional_r(closes[i], entry, risk, sign)
                if close_r <= -q:
                    decision_idx = i
                    break
            if decision_idx is None or decision_idx + 1 >= len(times):
                continue
            decision_time = times[decision_idx] + pd.Timedelta(minutes=15)
            row = {
                **base,
                "checkpoint": q,
                "decision_time": decision_time,
            }
            row.update(
                _checkpoint_features(
                    t, fill_idx, decision_idx, opens, highs, lows, closes, volumes, taker_buy
                )
            )
            row.update(
                _oi_snapshot(
                    metric_times,
                    oi,
                    global_ls,
                    taker_ls,
                    decision_time,
                    fill_time,
                    publication_lag_minutes=5,
                )
            )
            # The extension changes nothing unless the original -1R stop is
            # subsequently touched before the canonical exit.  This preserves
            # the existing trailing/structure exit on trades that recover
            # without needing rescue.
            original_stop_idx = None
            activation_source = "none"
            for i in range(decision_idx + 1, exit_idx + 1):
                if times[i] >= t["exit_time_parsed"]:
                    break
                open_r = _directional_r(opens[i], entry, risk, sign)
                touched_at_open = open_r <= -1.0
                touched_in_complete_bar = i <= last_path_idx and (
                    lows[i] <= stop if sign > 0 else highs[i] >= stop
                )
                touched = touched_at_open or touched_in_complete_bar
                if touched:
                    original_stop_idx = i
                    activation_source = "ohlc_before_canonical_exit"
                    break
            stop_like_canonical_exit = -1.25 <= canonical_price_r <= -0.90
            if (
                original_stop_idx is None
                and stop_like_canonical_exit
                and exit_idx > decision_idx
            ):
                # The ledger exit near -1R establishes activation, but the
                # containing 15m candle mixes pre/post-exit prices.  The
                # activation-bar helper therefore uses conservative bounds.
                original_stop_idx = exit_idx
                activation_source = "canonical_stop_exit_proxy"
            for p in POLICIES_BY_CHECKPOINT[q]:
                execution_r = _directional_r(
                    opens[decision_idx + 1], entry, risk, sign
                )
                if original_stop_idx is None:
                    conservative = float(t["r"])
                    optimistic = float(t["r"])
                    outcome = "no_stop_divergence"
                    outcome_idx = exit_idx
                else:
                    start = original_stop_idx
                    end = min(start + p.horizon_bars - 1, len(times) - 1)
                    conservative, optimistic, outcome, outcome_idx = (
                        _race_after_stop_activation(
                            start,
                            end,
                            opens,
                            highs,
                            lows,
                            closes,
                            entry,
                            risk,
                            sign,
                            p.emergency,
                            p.target,
                        )
                    )
                # Fraction retained after partial reduction so worst-case
                # combined trade result is no worse than -1R.
                retained, risk_normalized_value = _risk_normalized_result(
                    execution_r, p.emergency, conservative
                )
                _, risk_normalized_opt_value = _risk_normalized_result(
                    execution_r, p.emergency, optimistic
                )
                risk_normalized = (
                    risk_normalized_value
                    if r_reconciled
                    else np.nan
                )
                risk_normalized_opt = (
                    risk_normalized_opt_value
                    if r_reconciled
                    else np.nan
                )
                economic_available = original_stop_idx is None or r_reconciled
                full_r = conservative if economic_available else np.nan
                full_opt_r = optimistic if economic_available else np.nan
                row[f"{p.name}__full_r"] = full_r
                row[f"{p.name}__full_opt_r"] = full_opt_r
                row[f"{p.name}__risknorm_r"] = risk_normalized
                row[f"{p.name}__risknorm_opt_r"] = risk_normalized_opt
                row[f"{p.name}__delta_full"] = (
                    full_r - float(t["r"]) if economic_available else np.nan
                )
                row[f"{p.name}__delta_risknorm"] = (
                    risk_normalized - float(t["r"]) if r_reconciled else np.nan
                )
                row[f"{p.name}__outcome"] = outcome
                row[f"{p.name}__retained_fraction"] = retained
                row[f"{p.name}__execution_r"] = execution_r
                row[f"{p.name}__bars_to_outcome"] = outcome_idx - decision_idx
                row[f"{p.name}__original_stop_hit"] = original_stop_idx is not None
                row[f"{p.name}__activation_source"] = activation_source
                row[f"{p.name}__label_end_time"] = max(
                    t["exit_time_parsed"], bar_close_times[outcome_idx]
                )
            adverse_rows.append(row)

    return pd.DataFrame(trade_rows), pd.DataFrame(adverse_rows), pd.DataFrame(timed_rows)


NUMERIC_FEATURES = [
    "stop_pct",
    "stop_atr_pre",
    "mom_1h_pre",
    "mom_4h_pre",
    "range_4h_pre",
    "range_24h_pre",
    "range_72h_pre",
    "taker_imb_1h_pre",
    "current_r",
    "mfe_sofar",
    "mae_sofar",
    "bars_since_fill",
    "path_efficiency",
    "taker_imb_since_fill",
    "volume_vs_pre24h",
    "last_bar_body_r",
    "last_bar_range_r",
    "oi_delta_fill",
    "oi_delta_1h",
    "global_ls_delta_1h",
    "taker_ls_now",
    "oi_age_minutes",
]
CATEGORICAL_FEATURES = ["setup", "direction"]


def _make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )


def _make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _make_preprocessor(numeric, categorical)),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def _binary_auc_scores(
    data: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    target: str,
    numeric: list[str],
    categorical: list[str],
) -> tuple[float, float]:
    features = numeric + categorical
    logistic = Pipeline(
        [
            ("prep", _make_preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    logistic.fit(data.loc[train, features], data.loc[train, target])
    logistic_probability = logistic.predict_proba(data.loc[test, features])[:, 1]

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(data.loc[train, numeric])
    x_test = imputer.transform(data.loc[test, numeric])
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=1,
        random_state=42,
    )
    gradient_boosting.fit(x_train, data.loc[train, target])
    gradient_probability = gradient_boosting.predict_proba(x_test)[:, 1]
    return (
        float(roc_auc_score(data.loc[test, target], logistic_probability)),
        float(roc_auc_score(data.loc[test, target], gradient_probability)),
    )


def walk_forward_policy(
    frame: pd.DataFrame,
    delta_col: str,
    label: str,
    min_train: int = 60,
    label_end_col: str = "label_end_time",
) -> tuple[pd.DataFrame, dict]:
    """Expanding-year policy selection using only prior-year thresholds.

    A model predicts paired action value.  The action threshold is selected on
    the final 25% of the historical training window; the test year is never
    used to define a score quantile.
    """
    data = frame.dropna(subset=[delta_col, "decision_time", label_end_col]).copy()
    data = data.sort_values("decision_time")
    numeric = [c for c in NUMERIC_FEATURES if c in data.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in data.columns]
    predictions = []
    fold_rows = []
    for year in (2023, 2024, 2025, 2026):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        train = data[
            (data["decision_time"] < cutoff) & (data[label_end_col] < cutoff)
        ].copy()
        test = data[
            (data["decision_time"] >= cutoff) & (data["decision_time"] < end)
        ].copy()
        if len(train) < min_train or len(test) == 0:
            continue
        split = min(max(int(len(train) * 0.75), 20), len(train) - 15)
        if split < 20:
            continue
        validation_start = train.iloc[split]["decision_time"]
        inner_fit = train[
            (train["decision_time"] < validation_start)
            & (train[label_end_col] < validation_start)
        ]
        inner_val = train[train["decision_time"] >= validation_start]
        if len(inner_val) < 15:
            continue
        model = _make_model(numeric, categorical)
        model.fit(inner_fit[numeric + categorical], inner_fit[delta_col])
        val_score = model.predict(inner_val[numeric + categorical])
        candidate_thresholds = sorted(
            set(
                [0.0, 0.05, 0.10, 0.20, 0.30]
                + [
                    float(np.quantile(val_score, q))
                    for q in (0.50, 0.60, 0.70, 0.80, 0.90)
                ]
            )
        )
        best: tuple[float, float] | None = None
        for threshold in candidate_thresholds:
            selected = val_score >= threshold
            n = int(selected.sum())
            if n < 5:
                continue
            group = inner_val.loc[selected]
            total_delta = float(group[delta_col].sum())
            # No action on a >5R baseline trade in threshold selection.
            if bool(group["tail_gt5"].any()):
                continue
            score = total_delta
            if best is None or score > best[0]:
                best = (score, threshold)
        if best is None or best[0] <= 0:
            threshold = np.inf
        else:
            threshold = float(best[1])

        # Keep the model that produced validation scores.  Re-fitting on the
        # full train window would change score calibration while reusing a
        # threshold selected for the inner-fit model.
        test["predicted_delta"] = model.predict(test[numeric + categorical])
        test["act"] = test["predicted_delta"] >= threshold
        predictions.append(test)
        selected_test = test[test["act"]]
        fold_rows.append(
            {
                "label": label,
                "year": year,
                "train_n": len(train),
                "test_n": len(test),
                "threshold": threshold,
                "actions": len(selected_test),
                "delta_r": float(selected_test[delta_col].sum()),
                "baseline_r_acted": float(selected_test["r"].sum()),
                "tail_actions": int(selected_test["tail_gt5"].sum()),
                "max_forgone_winner": (
                    float(selected_test["r"].max()) if len(selected_test) else np.nan
                ),
            }
        )
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    if len(pred):
        acted = pred[pred["act"]]
        summary = {
            "label": label,
            "oos_rows": int(len(pred)),
            "actions": int(len(acted)),
            "delta_r": float(acted[delta_col].sum()),
            "tail_actions": int(acted["tail_gt5"].sum()),
            "positive_folds": int((folds["delta_r"] > 0).sum()),
            "active_folds": int((folds["actions"] > 0).sum()),
        }
    else:
        summary = {
            "label": label,
            "oos_rows": 0,
            "actions": 0,
            "delta_r": 0.0,
            "tail_actions": 0,
            "positive_folds": 0,
            "active_folds": 0,
        }
    return folds, summary


def fixed_policy_summary(
    adverse: pd.DataFrame,
    policy: ExtensionPolicy,
    timeframe: str,
) -> dict:
    d = adverse[
        (adverse["checkpoint"] == policy.checkpoint)
        & (adverse["timeframe"] == timeframe)
    ].copy()
    col = f"{policy.name}__delta_risknorm"
    full_col = f"{policy.name}__delta_full"
    outcome_col = f"{policy.name}__outcome"
    if not len(d):
        return {}
    economic = d[full_col].notna()
    risknorm_economic = d[col].notna()
    return {
        "policy": policy.name,
        "timeframe": timeframe,
        "decision_n": int(len(d)),
        "economic_n": int(economic.sum()),
        "risknorm_economic_n": int(risknorm_economic.sum()),
        "delta_always_risknorm": float(d.loc[risknorm_economic, col].sum()),
        "delta_always_full": float(d.loc[economic, full_col].sum()),
        "oracle_positive_delta": float(d.loc[d[col] > 0, col].sum()),
        "beneficial_actions": int((d[col] > 0).sum()),
        "harm_actions": int((d[col] < 0).sum()),
        "neutral_actions": int((d[col] == 0).sum()),
        "same_bar_ambiguous": int((d[outcome_col] == "same_bar_ambiguous").sum()),
        "original_stop_hit": int(
            d[f"{policy.name}__original_stop_hit"].sum()
        ),
        "reconciled_stop_hit": int(
            (
                d[f"{policy.name}__original_stop_hit"]
                & d["r_reconciled"]
            ).sum()
        ),
        "target_rate": float(d[outcome_col].isin(["target", "target_gap"]).mean()),
        "emergency_rate": float(
            d[outcome_col].isin(["emergency", "emergency_gap"]).mean()
        ),
        "median_retained_fraction": float(
            d[f"{policy.name}__retained_fraction"].median()
        ),
    }


def extension_predictability(adverse: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic OOS AUC for beneficial extension using causal snapshots."""
    rows = []
    for tf in ("15M", "1H"):
        for q in (0.65, 0.75, 0.85):
            p = ExtensionPolicy(q, 1.25, 0.5)
            d = adverse[
                (adverse["timeframe"] == tf) & (adverse["checkpoint"] == q)
            ].copy()
            delta_col = f"{p.name}__delta_full"
            decision_n = len(d)
            d = d.dropna(subset=[delta_col]).copy()
            d["beneficial"] = (d[delta_col] > 1e-9).astype(int)
            cutoff = pd.Timestamp("2023-01-01", tz="UTC")
            label_end_col = f"{p.name}__label_end_time"
            train = (d["decision_time"] < cutoff) & (
                d[label_end_col] < cutoff
            )
            test = d["decision_time"] >= cutoff
            numeric = [c for c in NUMERIC_FEATURES if c in d.columns]
            categorical = [c for c in CATEGORICAL_FEATURES if c in d.columns]
            row = {
                "timeframe": tf,
                "checkpoint": q,
                "decision_n": decision_n,
                "economic_n": len(d),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "beneficial_n": int(d["beneficial"].sum()),
                "delta_always_full": float(d[delta_col].sum()),
                "oracle_positive_delta": float(d.loc[d["beneficial"] == 1, delta_col].sum()),
            }
            if (
                train.sum() >= 20
                and test.sum() >= 10
                and d.loc[train, "beneficial"].nunique() == 2
                and d.loc[test, "beneficial"].nunique() == 2
            ):
                (
                    row["auc_logistic"],
                    row["auc_gradient_boosting"],
                ) = _binary_auc_scores(
                    d,
                    train,
                    test,
                    "beneficial",
                    numeric,
                    categorical,
                )
            else:
                row["auc_logistic"] = np.nan
                row["auc_gradient_boosting"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def early_path_predictability(timed: pd.DataFrame) -> pd.DataFrame:
    """OOS diagnostic: eventual loss predictability from causal checkpoints."""
    rows = []
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    for tf in ("15M", "1H"):
        for bars in (2, 4, 8):
            data = timed[
                (timed["timeframe"] == tf)
                & (timed["checkpoint_bars"] == bars)
            ].copy()
            train = (data["decision_time"] < cutoff) & (
                data["label_end_time"] < cutoff
            )
            test = data["decision_time"] >= cutoff
            numeric = [c for c in NUMERIC_FEATURES if c in data.columns]
            categorical = [c for c in CATEGORICAL_FEATURES if c in data.columns]
            for target_name, target in (
                ("negative", data["r"] < 0),
                ("toxic", data["r"] <= -0.5),
            ):
                data["target"] = target.astype(int)
                row = {
                    "timeframe": tf,
                    "checkpoint_bars": bars,
                    "target": target_name,
                    "train_n": int(train.sum()),
                    "test_n": int(test.sum()),
                    "test_rate": float(data.loc[test, "target"].mean()),
                }
                if (
                    train.sum() >= 40
                    and test.sum() >= 40
                    and data.loc[train, "target"].nunique() == 2
                    and data.loc[test, "target"].nunique() == 2
                ):
                    (
                        row["auc_logistic"],
                        row["auc_gradient_boosting"],
                    ) = _binary_auc_scores(
                        data,
                        train,
                        test,
                        "target",
                        numeric,
                        categorical,
                    )
                else:
                    row["auc_logistic"] = np.nan
                    row["auc_gradient_boosting"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def simple_early_exit_grid(timed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oos = timed[
        timed["decision_time"] >= pd.Timestamp("2023-01-01", tz="UTC")
    ]
    for tf in ("15M", "1H"):
        for bars, current_max, mfe_max, require_taker_against in EARLY_EXIT_RULES:
            d = oos[
                (oos["timeframe"] == tf)
                & (oos["checkpoint_bars"] == bars)
            ].dropna(subset=["delta_early_exit"])
            if not len(d):
                continue
            mask = (d["current_r"] <= current_max) & (d["mfe_sofar"] <= mfe_max)
            if require_taker_against:
                mask &= d["taker_imb_since_fill"] < 0
            acted = d[mask]
            rows.append(
                {
                    "timeframe": tf,
                    "checkpoint_bars": bars,
                    "current_max": current_max,
                    "mfe_max": mfe_max,
                    "taker_against": require_taker_against,
                    "actions": int(len(acted)),
                    "delta_r": float(acted["delta_early_exit"].sum()),
                    "baseline_r_acted": float(acted["r"].sum()),
                    "tail_actions": int(acted["tail_gt5"].sum()),
                    "max_forgone_winner": (
                        float(acted["r"].max()) if len(acted) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def giveback_policy_details(
    trades: pd.DataFrame, klines: pd.DataFrame
) -> pd.DataFrame:
    """Arm at +1R, then exit next open after a causal close retracement."""
    times = pd.DatetimeIndex(klines["open_time"])
    bar_close_times = times + pd.Timedelta(minutes=15)
    opens = klines["open"].to_numpy(float)
    highs = klines["high"].to_numpy(float)
    lows = klines["low"].to_numpy(float)
    closes = klines["close"].to_numpy(float)
    rows: list[dict] = []
    for threshold in (0.0, 0.25, 0.50):
        for tf in ("15M", "1H"):
            for _, t in trades[trades["timeframe"] == tf].iterrows():
                canonical_price_r = _directional_r(
                    float(t["exit_price"]),
                    float(t["entry_price"]),
                    float(t["sl_usd"]),
                    float(t["sign"]),
                )
                if abs(float(t["r"]) - canonical_price_r) > 0.10:
                    continue
                fill_idx, _ = _first_touch_entry(
                    times, lows, highs, t["entry_time"], float(t["entry_price"])
                )
                if fill_idx is None:
                    continue
                last_decision_idx, _ = _last_bar_indices(
                    bar_close_times,
                    t["exit_time_parsed"],
                )
                if last_decision_idx <= fill_idx:
                    continue
                entry = float(t["entry_price"])
                risk = float(t["sl_usd"])
                sign = float(t["sign"])
                armed_idx = None
                for i in range(fill_idx + 1, last_decision_idx + 1):
                    reached = (
                        (highs[i] - entry) / risk >= 1
                        if sign > 0
                        else (entry - lows[i]) / risk >= 1
                    )
                    if reached:
                        armed_idx = i
                        break
                if armed_idx is None:
                    continue
                trigger_idx = None
                # Do not use arm-bar ordering.  Earliest decision is a later
                # completed bar; execution is next open.
                for i in range(armed_idx + 1, last_decision_idx + 1):
                    close_r = _directional_r(closes[i], entry, risk, sign)
                    if close_r <= threshold:
                        trigger_idx = i
                        break
                if trigger_idx is None or trigger_idx + 1 >= len(times):
                    continue
                action_r = _directional_r(opens[trigger_idx + 1], entry, risk, sign)
                rows.append(
                    {
                        "trade_id": t["trade_id"],
                        "entry_time": t["entry_time"],
                        "timeframe": tf,
                        "arm_r": 1.0,
                        "close_threshold_r": threshold,
                        "r": float(t["r"]),
                        "action_r": action_r,
                        "delta_r": action_r - float(t["r"]),
                        "tail_gt5": bool(t["r"] > 5),
                    }
                )
    return pd.DataFrame(rows)


def summarize_giveback_policies(details: pd.DataFrame) -> pd.DataFrame:
    if not len(details):
        return pd.DataFrame()
    return (
        details.groupby(["timeframe", "arm_r", "close_threshold_r"])
        .agg(
            actions=("r", "size"),
            delta_r=("delta_r", "sum"),
            tail_actions=("tail_gt5", "sum"),
            max_forgone_winner=("r", "max"),
        )
        .reset_index()
    )


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    trades = metrics_subset(load_gf_on())
    klines = load_klines()
    metrics = load_metrics()
    trade_frame, adverse, timed = build_snapshots(trades, klines, metrics)

    input_paths = {
        "trades": ROOT / "data" / "gf_on.json",
        "klines": ROOT / "data" / "klines" / "ETHUSDT_15m.parquet",
        "metrics": ROOT / "data" / "metrics" / "ETHUSDT_metrics.parquet",
    }
    kline_diff = klines["open_time"].diff()
    metric_diff = metrics["create_time"].diff()
    manifest = {
        "commands": [
            "python3 research/scripts/analyze_causal_policies.py",
            "python3 research/scripts/validate_causal_candidates.py",
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {
            name: _sha256(path) for name, path in input_paths.items()
        },
        "script_sha256": {
            path.name: _sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("validate_causal_candidates.py"),
                Path(__file__).with_name("features.py"),
                Path(__file__).with_name("load_trades.py"),
            )
        },
        "coverage": {
            "klines_min": klines["open_time"].min(),
            "klines_max": klines["open_time"].max(),
            "kline_gaps_over_15m": int(
                (kline_diff > pd.Timedelta(minutes=15)).sum()
            ),
            "metrics_min": metrics["create_time"].min(),
            "metrics_max": metrics["create_time"].max(),
            "metric_gaps_over_5m": int(
                (metric_diff > pd.Timedelta(minutes=5)).sum()
            ),
        },
        "random_seed": 42,
        "r_reconciliation_tolerance": 0.10,
    }
    with open(ART / "causal_run_manifest.json", "w") as fh:
        json.dump(_json_safe(manifest), fh, indent=2, default=str, allow_nan=False)

    trade_frame.to_parquet(ART / "causal_trade_features.parquet", index=False)
    adverse.to_parquet(ART / "causal_adverse_checkpoints.parquet", index=False)
    timed.to_parquet(ART / "causal_timed_checkpoints.parquet", index=False)

    fixed_rows = [
        fixed_policy_summary(adverse, p, tf)
        for p in EXTENSION_POLICIES
        for tf in ("15M", "1H")
    ]
    fixed = pd.DataFrame([r for r in fixed_rows if r])
    fixed.to_csv(ART / "causal_extension_fixed_policies.csv", index=False)
    predictability = extension_predictability(adverse)
    predictability.to_csv(ART / "causal_extension_predictability.csv", index=False)
    early_predictability = early_path_predictability(timed)
    early_predictability.to_csv(
        ART / "causal_early_path_predictability.csv", index=False
    )

    wf_folds = []
    wf_summaries = []
    # Only a small, preregisterable family: checkpoint 0.75, target 0 or 0.5,
    # emergency 1.25/1.50, each timeframe separately.
    for tf in ("15M", "1H"):
        for emergency in (1.25, 1.50):
            for target in (0.0, 0.5):
                p = ExtensionPolicy(0.75, emergency, target)
                d = adverse[
                    (adverse["checkpoint"] == 0.75) & (adverse["timeframe"] == tf)
                ].copy()
                for mode, col in (
                    ("full", f"{p.name}__delta_full"),
                    ("risknorm", f"{p.name}__delta_risknorm"),
                ):
                    folds, summary = walk_forward_policy(
                        d,
                        col,
                        f"extend_{mode}|{tf}|{p.name}",
                        min_train=35,
                        label_end_col=f"{p.name}__label_end_time",
                    )
                    wf_folds.append(folds)
                    wf_summaries.append(summary)

    for tf in ("15M", "1H"):
        d = trade_frame[trade_frame["timeframe"] == tf].copy()
        d["delta_skip"] = -d["r"]
        folds, summary = walk_forward_policy(
            d,
            "delta_skip",
            f"pre_entry_skip|{tf}",
            min_train=100,
        )
        wf_folds.append(folds)
        wf_summaries.append(summary)

    for tf in ("15M", "1H"):
        for bars in (2, 4, 8):
            d = timed[
                (timed["timeframe"] == tf)
                & (timed["checkpoint_bars"] == bars)
            ].copy()
            folds, summary = walk_forward_policy(
                d,
                "delta_early_exit",
                f"early_exit|{tf}|{bars}bars",
                min_train=50,
            )
            wf_folds.append(folds)
            wf_summaries.append(summary)

    folds_all = pd.concat([x for x in wf_folds if len(x)], ignore_index=True)
    folds_all.to_csv(ART / "causal_policy_walkforward_folds.csv", index=False)
    summaries = pd.DataFrame(wf_summaries)
    summaries.to_csv(ART / "causal_policy_walkforward_summary.csv", index=False)

    early_grid = simple_early_exit_grid(timed)
    early_grid.to_csv(ART / "causal_early_exit_grid.csv", index=False)
    giveback_details = giveback_policy_details(trades, klines)
    giveback_details.to_csv(ART / "causal_giveback_details.csv", index=False)
    giveback = summarize_giveback_policies(giveback_details)
    giveback.to_csv(ART / "causal_giveback_policies.csv", index=False)

    data_quality = {
        "trade_rows": int(len(trade_frame)),
        "adverse_rows": int(len(adverse)),
        "timed_rows": int(len(timed)),
        "fill_quality": trade_frame["fill_quality"].value_counts().to_dict(),
        "r_reconciled_within_0_10": (
            trade_frame.groupby("timeframe")["r_reconciled"].sum().to_dict()
        ),
        "fill_lag_by_timeframe": (
            trade_frame.groupby("timeframe")["fill_lag_bars"].describe().to_dict()
        ),
        "warning": (
            "1H entry_time is signal-time for many rows; 1H path results are proxy-only. "
            "15M same-bar order remains unresolved. Synthetic price-R is not fully "
            "reconciled with exported r; policy economics exclude stop-divergence "
            "rows whose absolute reconciliation error exceeds 0.10R."
        ),
    }
    result = {
        "data_quality": data_quality,
        "fixed_best_risknorm": (
            fixed.sort_values("delta_always_risknorm", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ),
        "extension_predictability": predictability.to_dict(orient="records"),
        "early_path_predictability": early_predictability.to_dict(orient="records"),
        "walkforward": summaries.to_dict(orient="records"),
        "early_grid_best_no_tail": (
            early_grid[
                (early_grid["tail_actions"] == 0) & (early_grid["actions"] >= 5)
            ]
            .sort_values("delta_r", ascending=False)
            .head(15)
            .to_dict(orient="records")
        ),
        "giveback": giveback.to_dict(orient="records"),
    }
    result = _json_safe(result)
    with open(ART / "causal_policy_research.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str, allow_nan=False)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
