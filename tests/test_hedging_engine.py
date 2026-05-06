from __future__ import annotations

import unittest

import pandas as pd

from src.hedging_engine import HedgeConfig, run_delta_hedge
from src.pnl_attribution import add_bkl_variance_approximation, add_greek_pnl_attribution, summarize_results


class HedgingEngineTests(unittest.TestCase):
    def test_run_delta_hedge_handles_named_datetime_index(self) -> None:
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        idx.name = "Date"  # mimic yfinance naming
        option_data = pd.DataFrame(
            {
                "spot": [100.0, 101.0, 99.5],
                "strike": [100.0, 100.0, 100.0],
                "tau": [21 / 252, 20 / 252, 19 / 252],
                "rate": [0.02, 0.02, 0.02],
                "volatility": [0.20, 0.20, 0.20],
            },
            index=idx,
        )

        result = run_delta_hedge(option_data, HedgeConfig(rehedge_every=1))
        self.assertFalse(result.empty)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(len(result), len(option_data))

    def test_default_hedge_uses_positive_replication_delta(self) -> None:
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        option_data = pd.DataFrame(
            {
                "spot": [100.0, 101.0],
                "strike": [100.0, 100.0],
                "tau": [2 / 252, 1 / 252],
                "rate": [0.02, 0.02],
                "volatility": [0.20, 0.20],
            },
            index=idx,
        )

        result = run_delta_hedge(option_data, HedgeConfig(rehedge_every=1))
        self.assertGreater(result["stock_shares"].iloc[0], 0.0)
        self.assertAlmostEqual(result["hedge_error"].iloc[0], 0.0, places=10)

    def test_continuous_ideal_has_zero_mark_to_market_error(self) -> None:
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        option_data = pd.DataFrame(
            {
                "spot": [100.0, 101.0, 102.0],
                "strike": [100.0, 100.0, 100.0],
                "tau": [3 / 252, 2 / 252, 1 / 252],
                "rate": [0.02, 0.02, 0.02],
                "volatility": [0.20, 0.20, 0.20],
            },
            index=idx,
        )

        result = run_delta_hedge(option_data, HedgeConfig(is_continuous_ideal=True))
        self.assertTrue((result["hedge_error"].abs() < 1e-10).all())

    def test_put_option_uses_put_payoff_and_delta(self) -> None:
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        option_data = pd.DataFrame(
            {
                "spot": [100.0, 98.0],
                "strike": [100.0, 100.0],
                "tau": [1 / 252, 0.0],
                "rate": [0.02, 0.02],
                "volatility": [0.20, 0.20],
                "option_type": ["put", "put"],
            },
            index=idx,
        )

        result = run_delta_hedge(option_data, HedgeConfig(is_continuous_ideal=True))
        self.assertLess(result["delta"].iloc[0], 0.0)
        self.assertEqual(result["terminal_option_payoff"].iloc[-1], 2.0)
        self.assertAlmostEqual(result["hedge_error"].iloc[-1], 0.0, places=10)

    def test_bkl_columns_present_and_non_negative(self) -> None:
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        option_data = pd.DataFrame(
            {
                "spot": [100.0, 101.0, 102.0],
                "strike": [100.0, 100.0, 100.0],
                "tau": [3 / 252, 2 / 252, 1 / 252],
                "rate": [0.02, 0.02, 0.02],
                "volatility": [0.20, 0.20, 0.20],
            },
            index=idx,
        )
        result = run_delta_hedge(option_data, HedgeConfig(rehedge_every=1))
        result = add_greek_pnl_attribution(result)
        result = add_bkl_variance_approximation(result)
        self.assertIn("bkl_var_step", result.columns)
        self.assertIn("bkl_var_cumulative", result.columns)
        self.assertTrue((result["bkl_var_cumulative"] >= 0).all())

    def test_summary_reports_positive_loss_cvar(self) -> None:
        results = pd.DataFrame(
            {
                "hedge_error": [0.0, -1.0, 1.5],
                "transaction_cost": [0.0, 0.0, 0.0],
                "total_pnl": [0.0, -2.0, 1.0],
                "portfolio_value": [1.0, -1.0, 0.0],
                "did_rehedge": [True, True, True],
                "pnl_attribution_residual": [0.0, 0.0, 0.0],
            }
        )

        summary = summarize_results(results)
        self.assertGreaterEqual(summary["var_95_daily"], 0.0)
        self.assertGreaterEqual(summary["cvar_95_daily"], 0.0)


if __name__ == "__main__":
    unittest.main()
