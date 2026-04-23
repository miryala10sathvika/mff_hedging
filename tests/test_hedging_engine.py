from __future__ import annotations

import unittest

import pandas as pd

from src.hedging_engine import HedgeConfig, run_delta_hedge


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


if __name__ == "__main__":
    unittest.main()
