from __future__ import annotations

import unittest

import pandas as pd

from src.experiments import StrategySelectorConfig, apply_strategy_selector, get_selected_strategy


class StrategySelectorTests(unittest.TestCase):
    def test_selector_marks_lowest_cvar_plus_cost_periodic_strategy(self) -> None:
        summary = pd.DataFrame(
            {
                "cvar_95_daily": [-2.0, -1.2, -0.8, -3.0],
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

    def test_selector_converts_negative_cvar_to_positive_loss(self) -> None:
        summary = pd.DataFrame(
            {
                "cvar_95_daily": [0.25, -0.75],
                "total_transaction_cost": [0.2, 0.1],
                "rehedge_count": [12, 6],
            },
            index=["every_1_day", "every_3_days"],
        )

        annotated = apply_strategy_selector(summary)

        self.assertAlmostEqual(float(annotated.loc["every_1_day", "selector_cvar_loss"]), 0.0)
        self.assertAlmostEqual(float(annotated.loc["every_3_days", "selector_cvar_loss"]), 0.75)

    def test_get_selected_strategy_returns_single_winner(self) -> None:
        summary = pd.DataFrame(
            {
                "cvar_99_daily": [-1.0, -0.7],
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


if __name__ == "__main__":
    unittest.main()
