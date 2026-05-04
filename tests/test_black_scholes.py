from __future__ import annotations

import math
import unittest

from src.black_scholes import call_price, put_price
from src.greeks import call_delta, call_greeks, call_theta, gamma, put_delta, put_theta, vega


class BlackScholesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spot = 100.0
        self.strike = 100.0
        self.tau = 21.0 / 252.0
        self.rate = 0.02
        self.vol = 0.20

    def test_call_put_prices_are_positive(self) -> None:
        call = call_price(self.spot, self.strike, self.tau, self.rate, self.vol)
        put = put_price(self.spot, self.strike, self.tau, self.rate, self.vol)
        self.assertGreater(call, 0.0)
        self.assertGreater(put, 0.0)

    def test_put_call_parity(self) -> None:
        call = call_price(self.spot, self.strike, self.tau, self.rate, self.vol)
        put = put_price(self.spot, self.strike, self.tau, self.rate, self.vol)
        lhs = call - put
        rhs = self.spot - self.strike * math.exp(-self.rate * self.tau)
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_delta_bounds(self) -> None:
        self.assertGreaterEqual(call_delta(self.spot, self.strike, self.tau, self.rate, self.vol), 0.0)
        self.assertLessEqual(call_delta(self.spot, self.strike, self.tau, self.rate, self.vol), 1.0)
        self.assertGreaterEqual(put_delta(self.spot, self.strike, self.tau, self.rate, self.vol), -1.0)
        self.assertLessEqual(put_delta(self.spot, self.strike, self.tau, self.rate, self.vol), 0.0)

    def test_gamma_and_vega_positive(self) -> None:
        self.assertGreater(gamma(self.spot, self.strike, self.tau, self.rate, self.vol), 0.0)
        self.assertGreater(vega(self.spot, self.strike, self.tau, self.rate, self.vol), 0.0)

    def test_call_greeks_bundle_contains_all_fields(self) -> None:
        greeks = call_greeks(self.spot, self.strike, self.tau, self.rate, self.vol)
        self.assertEqual(set(greeks.keys()), {"delta", "gamma", "theta", "vega", "rho"})

    def test_theta_matches_finite_difference_time_decay(self) -> None:
        epsilon = 1e-5
        call_decay = (
            call_price(self.spot, self.strike, self.tau - epsilon, self.rate, self.vol)
            - call_price(self.spot, self.strike, self.tau, self.rate, self.vol)
        ) / epsilon
        put_decay = (
            put_price(self.spot, self.strike, self.tau - epsilon, self.rate, self.vol)
            - put_price(self.spot, self.strike, self.tau, self.rate, self.vol)
        ) / epsilon

        self.assertAlmostEqual(
            call_theta(self.spot, self.strike, self.tau, self.rate, self.vol),
            call_decay,
            places=3,
        )
        self.assertAlmostEqual(
            put_theta(self.spot, self.strike, self.tau, self.rate, self.vol),
            put_decay,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
