from __future__ import annotations

import math

import pandas as pd

from src.greeks import call_delta, call_rho, call_theta, gamma, vega


def add_greek_pnl_attribution(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        raise ValueError("results cannot be empty")

    frame = results.copy()
    frame["spot_change"] = frame["spot"].diff().fillna(0.0)
    frame["vol_change"] = frame["volatility"].diff().fillna(0.0)
    frame["rate_change"] = frame["rate"].diff().fillna(0.0)
    frame["dt"] = frame["tau"].shift(1).sub(frame["tau"]).fillna(0.0)
    
    if "dividend_yield" not in frame.columns:
        frame["dividend_yield"] = 0.0

    delta_terms = []
    gamma_terms = []
    theta_terms = []
    vega_terms = []
    rho_terms = []

    for i, row in enumerate(frame.itertuples()):
        if i == 0:
            delta_terms.append(0.0)
            gamma_terms.append(0.0)
            theta_terms.append(0.0)
            vega_terms.append(0.0)
            rho_terms.append(0.0)
            continue

        prev = frame.iloc[i - 1]
        delta_value = call_delta(prev["spot"], prev["strike"], prev["tau"], prev["rate"], prev["volatility"], prev["dividend_yield"])
        gamma_value = gamma(prev["spot"], prev["strike"], prev["tau"], prev["rate"], prev["volatility"], prev["dividend_yield"])
        theta_value = call_theta(prev["spot"], prev["strike"], prev["tau"], prev["rate"], prev["volatility"], prev["dividend_yield"])
        vega_value = vega(prev["spot"], prev["strike"], prev["tau"], prev["rate"], prev["volatility"], prev["dividend_yield"])
        rho_value = call_rho(prev["spot"], prev["strike"], prev["tau"], prev["rate"], prev["volatility"], prev["dividend_yield"])

        ds = frame.iloc[i]["spot_change"]
        dvol = frame.iloc[i]["vol_change"]
        drate = frame.iloc[i]["rate_change"]
        dt = frame.iloc[i]["dt"]

        delta_terms.append(delta_value * ds)
        gamma_terms.append(0.5 * gamma_value * ds**2)
        theta_terms.append(theta_value * dt)
        vega_terms.append(vega_value * dvol)
        rho_terms.append(rho_value * drate)

    frame["delta_pnl_approx"] = delta_terms
    frame["gamma_pnl_approx"] = gamma_terms
    frame["theta_pnl_approx"] = theta_terms
    frame["vega_pnl_approx"] = vega_terms
    frame["rho_pnl_approx"] = rho_terms
    frame["greek_pnl_approx_total"] = (
        frame["delta_pnl_approx"]
        + frame["gamma_pnl_approx"]
        + frame["theta_pnl_approx"]
        + frame["vega_pnl_approx"]
        + frame["rho_pnl_approx"]
    )
    frame["pnl_attribution_residual"] = frame["option_pnl"] - frame["greek_pnl_approx_total"]
    return frame


def summarize_results(results: pd.DataFrame) -> pd.Series:
    final_hedge_error = float(results["hedge_error"].iloc[-1])
    total_transaction_cost = float(results["transaction_cost"].sum())
    total_pnl = float(results["total_pnl"].sum())
    hedge_error_rmse = math.sqrt(float((results["hedge_error"] ** 2).mean()))
    hedge_error_mae = float(results["hedge_error"].abs().mean())

    # VaR and CVaR on daily P&L or hedge error change
    pnl_series = results["total_pnl"].dropna()
    if not pnl_series.empty:
        var_95 = float(pnl_series.quantile(0.05))
        cvar_95 = float(pnl_series[pnl_series <= var_95].mean())
        var_99 = float(pnl_series.quantile(0.01))
        cvar_99 = float(pnl_series[pnl_series <= var_99].mean())
    else:
        var_95 = cvar_95 = var_99 = cvar_99 = 0.0

    return pd.Series(
        {
            "final_hedge_error": final_hedge_error,
            "hedge_error_abs": abs(final_hedge_error),
            "hedge_error_rmse": hedge_error_rmse,
            "hedge_error_mae": hedge_error_mae,
            "max_abs_hedge_error": float(results["hedge_error"].abs().max()),
            "var_95_daily": var_95,
            "cvar_95_daily": cvar_95,
            "var_99_daily": var_99,
            "cvar_99_daily": cvar_99,
            "total_transaction_cost": total_transaction_cost,
            "mean_daily_pnl": float(results["total_pnl"].mean()),
            "std_daily_pnl": float(results["total_pnl"].std(ddof=0)),
            "total_pnl": total_pnl,
            "portfolio_value_final": float(results["portfolio_value"].iloc[-1]),
            "rehedge_count": int(results["did_rehedge"].sum()),
            "mean_attr_residual": float(results["pnl_attribution_residual"].mean()),
        }
    )
