from __future__ import annotations

import unittest

import pandas as pd

from src.data_loader import _rank_contract_candidates, select_option_contract


class DataLoaderTests(unittest.TestCase):
    def test_select_option_contract_falls_back_when_open_interest_is_empty(self) -> None:
        chain = pd.DataFrame(
            {
                "contractSymbol": ["A", "B", "C"],
                "strike": [95.0, 100.0, 105.0],
                "openInterest": [0, 0, 0],
                "impliedVolatility": [0.2, 0.2, 0.2],
            }
        )

        selected = select_option_contract(
            chain=chain,
            underlying_spot=101.0,
            min_open_interest=1,
            min_implied_volatility=0.05,
        )

        self.assertEqual(selected["contractSymbol"], "B")

    def test_rank_contract_candidates_prefers_near_atm_meaningful_delta(self) -> None:
        chain = pd.DataFrame(
            {
                "contractSymbol": ["LOW", "ATM", "HIGH"],
                "strike": [650.0, 720.0, 900.0],
                "openInterest": [100, 100, 100],
                "impliedVolatility": [0.2, 0.2, 0.2],
            }
        )

        ranked = _rank_contract_candidates(
            chain=chain,
            underlying_spot=720.0,
            selected_expiration="2027-01-15",
            option_type="call",
            strike=None,
            max_strike_distance_pct=0.5,
            min_open_interest=1,
            min_implied_volatility=0.05,
            fallback_volatility=0.20,
            rate=0.04,
            target_abs_delta=0.50,
            min_abs_delta=0.25,
            max_abs_delta=0.75,
        )

        self.assertEqual(ranked.iloc[0]["contractSymbol"], "ATM")
        self.assertGreaterEqual(float(ranked.iloc[0]["estimated_abs_delta"]), 0.25)

    def test_rank_contract_candidates_keeps_near_atm_when_chain_iv_is_bad(self) -> None:
        chain = pd.DataFrame(
            {
                "contractSymbol": ["ATM", "FAR"],
                "strike": [720.0, 780.0],
                "openInterest": [0, 100],
                "impliedVolatility": [0.00001, 0.08],
            }
        )

        ranked = _rank_contract_candidates(
            chain=chain,
            underlying_spot=720.0,
            selected_expiration="2027-01-15",
            option_type="call",
            strike=None,
            max_strike_distance_pct=0.5,
            min_open_interest=1,
            min_implied_volatility=0.05,
            fallback_volatility=0.20,
            rate=0.04,
            target_abs_delta=0.50,
            min_abs_delta=0.25,
            max_abs_delta=0.75,
        )

        self.assertEqual(ranked.iloc[0]["contractSymbol"], "ATM")


if __name__ == "__main__":
    unittest.main()
