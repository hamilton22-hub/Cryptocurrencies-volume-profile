import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_causal_policies import (
    ExtensionPolicy,
    build_snapshots,
    _first_touch_entry,
    _last_bar_indices,
    _race_after_stop_activation,
    _race_policy,
    _risk_normalized_result,
)
from validate_causal_candidates import _monthly_sign_flip


class FirstTouchEntryTest(unittest.TestCase):
    def test_finds_first_bar_containing_entry(self):
        times = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
        lows = np.array([90.0, 95.0, 99.0, 101.0])
        highs = np.array([95.0, 99.0, 102.0, 103.0])

        idx, lag = _first_touch_entry(times, lows, highs, times[0], 100.0)

        self.assertEqual(idx, 2)
        self.assertEqual(lag, 2)


class RacePolicyTest(unittest.TestCase):
    def setUp(self):
        self.entry = 100.0
        self.risk = 10.0

    def test_target_before_emergency_long(self):
        result = _race_policy(
            0,
            1,
            np.array([94.0, 98.0]),
            np.array([99.0, 106.0]),
            np.array([92.0, 96.0]),
            np.array([97.0, 104.0]),
            self.entry,
            self.risk,
            1.0,
            emergency=1.25,
            target=0.5,
        )

        self.assertEqual(result[0], 0.5)
        self.assertEqual(result[2], "target")

    def test_emergency_before_target_short(self):
        result = _race_policy(
            0,
            0,
            np.array([106.0]),
            np.array([113.0]),
            np.array([104.0]),
            np.array([111.0]),
            self.entry,
            self.risk,
            -1.0,
            emergency=1.25,
            target=0.5,
        )

        self.assertEqual(result[0], -1.25)
        self.assertEqual(result[2], "emergency")

    def test_same_bar_uses_conservative_emergency_and_optimistic_target(self):
        result = _race_policy(
            0,
            0,
            np.array([95.0]),
            np.array([106.0]),
            np.array([87.0]),
            np.array([100.0]),
            self.entry,
            self.risk,
            1.0,
            emergency=1.25,
            target=0.5,
        )

        self.assertEqual(result[0], -1.25)
        self.assertEqual(result[1], 0.5)
        self.assertEqual(result[2], "same_bar_ambiguous")

    def test_gap_executes_at_observed_open(self):
        result = _race_policy(
            0,
            0,
            np.array([85.0]),
            np.array([90.0]),
            np.array([84.0]),
            np.array([88.0]),
            self.entry,
            self.risk,
            1.0,
            emergency=1.25,
            target=0.5,
        )

        self.assertEqual(result[0], -1.5)
        self.assertEqual(result[2], "emergency_gap")


class ActivationRaceTest(unittest.TestCase):
    def test_does_not_credit_target_before_intrabar_stop_activation(self):
        result = _race_after_stop_activation(
            0,
            1,
            np.array([106.0, 95.0]),
            np.array([107.0, 96.0]),
            np.array([90.0, 87.0]),
            np.array([95.0, 89.0]),
            entry=100.0,
            risk=10.0,
            sign=1.0,
            emergency=1.25,
            target=0.5,
        )

        self.assertEqual(result[0], -1.25)
        self.assertEqual(result[1], 0.5)
        self.assertIn("activation_target_ambiguous", result[2])

    def test_credits_target_at_activation_bar_close(self):
        result = _race_after_stop_activation(
            0,
            1,
            np.array([100.0, 105.0]),
            np.array([107.0, 106.0]),
            np.array([90.0, 100.0]),
            np.array([106.0, 105.0]),
            entry=100.0,
            risk=10.0,
            sign=1.0,
            emergency=1.25,
            target=0.5,
        )

        self.assertEqual(result[0], 0.5)
        self.assertEqual(result[2], "activation_bar_target_at_close")


class CausalClockTest(unittest.TestCase):
    def test_exit_at_bar_close_is_not_a_new_decision_but_is_observed_path(self):
        closes = pd.date_range(
            "2025-01-01 00:15", periods=3, freq="15min", tz="UTC"
        )

        decision_idx, path_idx = _last_bar_indices(closes, closes[1])

        self.assertEqual(decision_idx, 0)
        self.assertEqual(path_idx, 1)

    @staticmethod
    def _market_fixture(exit_time, exit_price, canonical_r):
        times = pd.date_range(
            "2025-01-01", periods=30, freq="15min", tz="UTC"
        )
        klines = pd.DataFrame(
            {
                "open_time": times,
                "open": np.full(30, 100.0),
                "high": np.full(30, 101.0),
                "low": np.full(30, 99.0),
                "close": np.full(30, 100.0),
                "volume": np.full(30, 10.0),
                "taker_buy_volume": np.full(30, 5.0),
            }
        )
        klines.loc[17, ["open", "high", "low", "close"]] = [
            95.0,
            96.0,
            91.0,
            92.0,
        ]
        metric_times = pd.date_range(
            times[0], periods=90, freq="5min", tz="UTC"
        )
        metrics = pd.DataFrame(
            {
                "create_time": metric_times,
                "oi": np.linspace(1000.0, 1010.0, len(metric_times)),
                "global_ls": np.full(len(metric_times), 1.0),
                "taker_ls": np.full(len(metric_times), 1.0),
            }
        )
        trades = pd.DataFrame(
            [
                {
                    "trade_id": "fixture",
                    "entry_time": times[16],
                    "exit_time_parsed": exit_time,
                    "entry_price": 100.0,
                    "exit_price": exit_price,
                    "sl_usd": 10.0,
                    "sign": 1.0,
                    "stop_price": 90.0,
                    "timeframe": "15M",
                    "setup": "TBX0",
                    "direction": "LONG",
                    "r": canonical_r,
                }
            ]
        )
        return trades, klines, metrics

    def test_checkpoint_after_canonical_exit_is_excluded(self):
        times = pd.date_range(
            "2025-01-01", periods=30, freq="15min", tz="UTC"
        )
        fixture = self._market_fixture(
            exit_time=times[17] + pd.Timedelta(minutes=5),
            exit_price=92.0,
            canonical_r=-0.8,
        )

        _, adverse, _ = build_snapshots(*fixture)

        self.assertTrue(adverse.empty)

    def test_no_stop_after_checkpoint_has_zero_full_policy_delta(self):
        times = pd.date_range(
            "2025-01-01", periods=30, freq="15min", tz="UTC"
        )
        fixture = self._market_fixture(
            exit_time=times[20] + pd.Timedelta(minutes=15),
            exit_price=100.0,
            canonical_r=0.0,
        )

        _, adverse, _ = build_snapshots(*fixture)
        policy = ExtensionPolicy(0.65, 1.25, 0.5)
        row = adverse.loc[adverse["checkpoint"] == 0.65].iloc[0]

        self.assertFalse(row[f"{policy.name}__original_stop_hit"])
        self.assertEqual(row[f"{policy.name}__delta_full"], 0.0)


class RiskNormalizationTest(unittest.TestCase):
    def test_non_gap_emergency_combines_to_negative_one_r(self):
        retained, combined = _risk_normalized_result(
            execution_r=-0.80,
            emergency=1.25,
            continuation_r=-1.25,
        )

        self.assertAlmostEqual(retained, 0.20 / 0.45)
        self.assertAlmostEqual(combined, -1.0)

    def test_gap_through_emergency_preserves_realized_slippage(self):
        retained, combined = _risk_normalized_result(
            execution_r=-1.50,
            emergency=1.25,
            continuation_r=-1.50,
        )

        self.assertEqual(retained, 1.0)
        self.assertEqual(combined, -1.50)


class SignFlipTest(unittest.TestCase):
    def test_uses_plus_one_monte_carlo_correction(self):
        result = _monthly_sign_flip(
            values=np.array([1.0]),
            months=np.array(["2025-01"]),
            masks=np.array([[1.0]]),
            observed_threshold=100.0,
            iterations=9,
        )

        self.assertEqual(result["family_p"], 0.1)

    def test_rejects_non_finite_inputs(self):
        with self.assertRaises(ValueError):
            _monthly_sign_flip(
                values=np.array([np.nan]),
                months=np.array(["2025-01"]),
                masks=np.array([[1.0]]),
                observed_threshold=1.0,
                iterations=9,
            )


if __name__ == "__main__":
    unittest.main()
