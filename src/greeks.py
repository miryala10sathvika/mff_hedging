from __future__ import annotations

import math

from src.black_scholes import d1, d2, norm_cdf, norm_pdf


def call_delta(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0:
        return 1.0 if spot > strike else 0.0
    return norm_cdf(d1(spot, strike, time_to_maturity, rate, volatility))


def put_delta(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    return call_delta(spot, strike, time_to_maturity, rate, volatility) - 1.0


def gamma(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0 or volatility == 0:
        return 0.0
    d1_value = d1(spot, strike, time_to_maturity, rate, volatility)
    denominator = spot * volatility * math.sqrt(time_to_maturity)
    return norm_pdf(d1_value) / denominator


def vega(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0:
        return 0.0
    d1_value = d1(spot, strike, time_to_maturity, rate, volatility)
    return spot * norm_pdf(d1_value) * math.sqrt(time_to_maturity)


def call_theta(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0:
        return 0.0
    d1_value = d1(spot, strike, time_to_maturity, rate, volatility)
    d2_value = d2(spot, strike, time_to_maturity, rate, volatility)
    first_term = -(spot * norm_pdf(d1_value) * volatility) / (2.0 * math.sqrt(time_to_maturity))
    second_term = -rate * strike * math.exp(-rate * time_to_maturity) * norm_cdf(d2_value)
    return first_term + second_term


def put_theta(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0:
        return 0.0
    d1_value = d1(spot, strike, time_to_maturity, rate, volatility)
    d2_value = d2(spot, strike, time_to_maturity, rate, volatility)
    first_term = -(spot * norm_pdf(d1_value) * volatility) / (2.0 * math.sqrt(time_to_maturity))
    second_term = rate * strike * math.exp(-rate * time_to_maturity) * norm_cdf(-d2_value)
    return first_term + second_term


def call_rho(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0:
        return 0.0
    d2_value = d2(spot, strike, time_to_maturity, rate, volatility)
    return strike * time_to_maturity * math.exp(-rate * time_to_maturity) * norm_cdf(d2_value)


def put_rho(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> float:
    if time_to_maturity == 0:
        return 0.0
    d2_value = d2(spot, strike, time_to_maturity, rate, volatility)
    return -strike * time_to_maturity * math.exp(-rate * time_to_maturity) * norm_cdf(-d2_value)


def call_greeks(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> dict[str, float]:
    return {
        "delta": call_delta(spot, strike, time_to_maturity, rate, volatility),
        "gamma": gamma(spot, strike, time_to_maturity, rate, volatility),
        "theta": call_theta(spot, strike, time_to_maturity, rate, volatility),
        "vega": vega(spot, strike, time_to_maturity, rate, volatility),
        "rho": call_rho(spot, strike, time_to_maturity, rate, volatility),
    }


def put_greeks(spot: float, strike: float, time_to_maturity: float, rate: float, volatility: float) -> dict[str, float]:
    return {
        "delta": put_delta(spot, strike, time_to_maturity, rate, volatility),
        "gamma": gamma(spot, strike, time_to_maturity, rate, volatility),
        "theta": put_theta(spot, strike, time_to_maturity, rate, volatility),
        "vega": vega(spot, strike, time_to_maturity, rate, volatility),
        "rho": put_rho(spot, strike, time_to_maturity, rate, volatility),
    }
