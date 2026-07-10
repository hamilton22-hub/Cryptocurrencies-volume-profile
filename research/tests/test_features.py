import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from features import node_exhaustion_mask


class NodeExhaustionTest(unittest.TestCase):
    def test_ignores_prior_trade_until_its_exit_is_known(self):
        base = pd.Timestamp("2025-01-01", tz="UTC")
        trades = pd.DataFrame(
            [
                {
                    "entry_time": base,
                    "exit_time_parsed": base + pd.Timedelta(hours=4),
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl_usd": 2.0,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
                {
                    "entry_time": base + pd.Timedelta(hours=1),
                    "exit_time_parsed": base + pd.Timedelta(hours=1, minutes=30),
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl_usd": 2.0,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
                {
                    "entry_time": base + pd.Timedelta(hours=2),
                    "exit_time_parsed": base + pd.Timedelta(hours=2, minutes=30),
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl_usd": 2.0,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
                {
                    "entry_time": base + pd.Timedelta(hours=3),
                    "exit_time_parsed": base + pd.Timedelta(hours=3, minutes=30),
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl_usd": 2.0,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
            ]
        )

        blocked = node_exhaustion_mask(trades, n_fails=3)

        self.assertFalse(blocked.iloc[-1])

    def test_blocked_trade_does_not_enter_counterfactual_history(self):
        base = pd.Timestamp("2025-01-01", tz="UTC")
        trades = pd.DataFrame(
            [
                {
                    "entry_time": base,
                    "exit_time_parsed": base + pd.Timedelta(minutes=30),
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "sl_usd": 1.0,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
                {
                    "entry_time": base + pd.Timedelta(hours=1),
                    "exit_time_parsed": base + pd.Timedelta(hours=1, minutes=30),
                    "direction": "LONG",
                    "entry_price": 101.0,
                    "sl_usd": 2.0,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
                {
                    "entry_time": base + pd.Timedelta(hours=2),
                    "exit_time_parsed": base + pd.Timedelta(hours=2, minutes=30),
                    "direction": "LONG",
                    "entry_price": 101.0,
                    "sl_usd": 0.4,
                    "r": -1.0,
                    "in_metrics": True,
                    "gf_blocked": False,
                },
            ]
        )

        blocked = node_exhaustion_mask(trades, n_fails=1)

        self.assertTrue(blocked.iloc[1])
        self.assertFalse(blocked.iloc[2])


if __name__ == "__main__":
    unittest.main()
