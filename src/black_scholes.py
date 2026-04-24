from __future__ import annotations

import math


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value**2) / math.sqrt(2.0 * math.pi)


def _validate_inputs(spot: float, strike: float, time_to_maturity: float, volatility: float) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if time_to_maturity < 0:
        raise ValueError("time_to_maturity cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")


def d1(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    if time_to_maturity == 0 or volatility == 0:
        return math.inf if spot > strike else -math.inf
    numerator = math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
    denominator = volatility * math.sqrt(time_to_maturity)
    return numerator / denominator


def d2(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    if time_to_maturity == 0 or volatility == 0:
        return d1(spot, strike, time_to_maturity, rate, volatility, dividend_yield)
    return d1(spot, strike, time_to_maturity, rate, volatility, dividend_yield) - volatility * math.sqrt(time_to_maturity)


def call_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    if time_to_maturity == 0:
        return max(spot - strike, 0.0)
    if volatility == 0:
        forward_intrinsic = spot * math.exp(-dividend_yield * time_to_maturity) - strike * math.exp(-rate * time_to_maturity)
        return max(forward_intrinsic, 0.0)

    d1_value = d1(spot, strike, time_to_maturity, rate, volatility, dividend_yield)
    d2_value = d2(spot, strike, time_to_maturity, rate, volatility, dividend_yield)
    discounted_spot = spot * math.exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike * math.exp(-rate * time_to_maturity)
    return discounted_spot * norm_cdf(d1_value) - discounted_strike * norm_cdf(d2_value)


def put_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    _validate_inputs(spot, strike, time_to_maturity, volatility)
    if time_to_maturity == 0:
        return max(strike - spot, 0.0)
    if volatility == 0:
        forward_intrinsic = strike * math.exp(-rate * time_to_maturity) - spot * math.exp(-dividend_yield * time_to_maturity)
        return max(forward_intrinsic, 0.0)

    d1_value = d1(spot, strike, time_to_maturity, rate, volatility, dividend_yield)
    d2_value = d2(spot, strike, time_to_maturity, rate, volatility, dividend_yield)
    discounted_spot = spot * math.exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike * math.exp(-rate * time_to_maturity)
    return discounted_strike * norm_cdf(-d2_value) - discounted_spot * norm_cdf(-d1_value)
