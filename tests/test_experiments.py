from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.experiments import StrategySelectorConfig, apply_strategy_selector, get_selected_strategy


class StrategySelectorTests(unittest.TestCase):
    def test_selector_marks_lowest_cvar_plus_cost_periodic_strategy(self) -> None:
        summary = pd.DataFrame(
            {
                "cvar_95_daily": [2.0, 1.2, 0.8, 3.0],
                "total_transaction_cost": [0.1, 0.4, 1.0, 0.0],
                "rehedge_count": [20, 10, 5, 0],
            },
            index=["every_1_day", "every_2_days", "every_5_days", "no_hedge"],
        )

        annotated = apply_strategy_selector(summary)

        self.assertTrue(bool(annotated.loc["every_2_days", "is_auto_selected"]))
        self.assertFalse(bool(annotated.loc["no_hedge", "is_auto_selected"]))
        self.assertAlmostEqual(float(annotated.loc["every_2_days", "selector_objective"]), 1.6)
        self.assertEqual(int(annotated.loc["every_2_days", "selector_rank"]), 1)

    def test_selector_uses_positive_cvar_loss(self) -> None:
        summary = pd.DataFrame(
            {
                "cvar_95_daily": [0.25, 0.75],
                "total_transaction_cost": [0.2, 0.1],
                "rehedge_count": [12, 6],
            },
            index=["every_1_day", "every_3_days"],
        )

        annotated = apply_strategy_selector(summary)

        self.assertAlmostEqual(float(annotated.loc["every_1_day", "selector_cvar_loss"]), 0.25)
        self.assertAlmostEqual(float(annotated.loc["every_3_days", "selector_cvar_loss"]), 0.75)

    def test_get_selected_strategy_returns_single_winner(self) -> None:
        summary = pd.DataFrame(
            {
                "cvar_99_daily": [1.0, 0.7],
                "total_transaction_cost": [0.2, 0.1],
                "rehedge_count": [15, 8],
            },
            index=["every_1_day", "every_2_days"],
        )

        annotated = apply_strategy_selector(
            summary,
            StrategySelectorConfig(cvar_metric="cvar_99_daily"),
        )
        selected = get_selected_strategy(annotated)

        self.assertEqual(selected.name, "every_2_days")
        self.assertAlmostEqual(float(selected["selector_objective"]), 0.8)
    
    def test_vol_mismatch_returns_aligned_frames(self) -> None:
        from src.experiments import VolMismatchConfig, run_vol_mismatch_experiment

        idx = pd.date_range("2023-01-02", periods=45, freq="B")
        prices = pd.DataFrame(
            {"spot": [100.0 + i * 0.2 for i in range(len(idx))]},
            index=idx,
        )
        cfg = VolMismatchConfig(
            ticker="SPY",
            start="2023-01-01",
            end="2023-03-01",
            maturity_days=21,
            fixed_volatility=0.20,
            realized_vol_window=10,
            rate=0.02,
            transaction_cost_bps=0.0,
            rehedge_every=1,
        )
        with patch("src.experiments.download_price_history", return_value=prices):
            fixed, real, comparison = run_vol_mismatch_experiment(cfg)
        self.assertEqual(len(fixed), len(real))
        self.assertIn("fixed_vol", comparison.index)
        self.assertIn("realized_vol", comparison.index)
        self.assertIn("hedge_error_rmse", comparison.columns)


if __name__ == "__main__":
    unittest.main()
