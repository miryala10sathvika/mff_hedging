from __future__ import annotations

from dataclasses import dataclass

import math
import pandas as pd

from src.black_scholes import call_price
from src.greeks import call_delta, call_greeks


@dataclass(frozen=True)
class HedgeConfig:
    rehedge_every: int = 1
    transaction_cost_bps: float = 0.0
    option_position: int = 1
    stock_position_sign: int = -1
    trading_days_per_year: int = 252
    is_static_hedge: bool = False
    is_no_hedge: bool = False
    is_continuous_ideal: bool = False


def _transaction_cost(spot: float, shares_traded: float, transaction_cost_bps: float) -> float:
    return abs(shares_traded) * spot * (transaction_cost_bps / 10000.0)


def run_delta_hedge(option_data: pd.DataFrame, config: HedgeConfig) -> pd.DataFrame:
    required_columns = {"spot", "strike", "tau", "rate", "volatility"}
    missing = required_columns - set(option_data.columns)
    if missing:
        raise ValueError(f"option_data missing columns: {sorted(missing)}")
    if option_data.empty:
        raise ValueError("option_data cannot be empty")
    if config.rehedge_every < 1 and not config.is_continuous_ideal:
        raise ValueError("rehedge_every must be >= 1")

    index_name = option_data.index.name
    frame = option_data.copy().reset_index()
    if "date" not in frame.columns:
        if index_name and index_name in frame.columns:
            frame = frame.rename(columns={index_name: "date"})
        elif "index" in frame.columns:
            frame = frame.rename(columns={"index": "date"})
        else:
            frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    
    if "dividend_yield" not in frame.columns:
        frame["dividend_yield"] = 0.0

    if "option_price" not in frame.columns:
        frame["option_price"] = frame.apply(
            lambda row: call_price(row["spot"], row["strike"], row["tau"], row["rate"], row["volatility"], row["dividend_yield"]),
            axis=1,
        )
    if "delta" not in frame.columns:
        frame["delta"] = frame.apply(
            lambda row: call_delta(row["spot"], row["strike"], row["tau"], row["rate"], row["volatility"], row["dividend_yield"]),
            axis=1,
        )
    greek_columns = {"gamma", "theta", "vega", "rho"}
    if not greek_columns.issubset(frame.columns):
        greek_frame = frame.apply(
            lambda row: pd.Series(
                call_greeks(row["spot"], row["strike"], row["tau"], row["rate"], row["volatility"], row["dividend_yield"])
            ),
            axis=1,
        )
        for column in greek_columns | {"delta"}:
            if column not in frame.columns:
                frame[column] = greek_frame[column]

    rows: list[dict] = []
    current_stock_shares = 0.0
    cash_account = 0.0
    previous_option_price = None
    previous_portfolio_value = None
    previous_delta = None

    for i, row in frame.iterrows():
        target_stock_shares = current_stock_shares
        did_rehedge = False
        
        # Decide if we rehedge
        should_rehedge = False
        if config.is_no_hedge:
            should_rehedge = False
        elif config.is_continuous_ideal:
            should_rehedge = True
        elif config.is_static_hedge:
            should_rehedge = (i == 0)
        elif i == 0 or i % config.rehedge_every == 0:
            should_rehedge = True

        # Process dividend from previous holding (approx continuously collected per day: S * q * dt)
        if i > 0:
            dt = frame.loc[i - 1, "tau"] - row["tau"]
            if dt > 0:
                dividend_cash = current_stock_shares * row["spot"] * row["dividend_yield"] * dt
                cash_account += dividend_cash

        # Process rate accrual on cash
        if i > 0:
            dt = frame.loc[i - 1, "tau"] - row["tau"]
            if dt > 0:
                cash_account *= math.exp(row["rate"] * dt) if row["rate"] > 0 else 1.0

        if should_rehedge:
            target_stock_shares = config.stock_position_sign * config.option_position * row["delta"]
            shares_traded = target_stock_shares - current_stock_shares
            # No transaction cost for continuous ideal
            trading_cost = 0.0 if config.is_continuous_ideal else _transaction_cost(row["spot"], shares_traded, config.transaction_cost_bps)
            cash_account -= shares_traded * row["spot"]
            cash_account -= trading_cost
            current_stock_shares = target_stock_shares
            did_rehedge = True
        else:
            shares_traded = 0.0
            trading_cost = 0.0

        stock_value = current_stock_shares * row["spot"]
        option_value = config.option_position * row["option_price"]
        
        if config.is_continuous_ideal:
            portfolio_value = option_value
            # For theoretical continuous replication portfolio, the value of the stock + cash matches exactly option value (if self-financing)
            # We override the actual portfolio tracking since continuous hedge has exactly 0 hedging error
            cash_account = portfolio_value - stock_value
        else:
            portfolio_value = option_value + stock_value + cash_account

        option_pnl = 0.0 if previous_option_price is None else config.option_position * (row["option_price"] - previous_option_price)
        stock_pnl = 0.0 if previous_portfolio_value is None else current_stock_shares * (row["spot"] - frame.loc[i - 1, "spot"])
        total_pnl = 0.0 if previous_portfolio_value is None else portfolio_value - previous_portfolio_value

        rows.append(
            {
                "date": row["date"],
                "spot": row["spot"],
                "strike": row["strike"],
                "tau": row["tau"],
                "rate": row["rate"],
                "volatility": row["volatility"],
                "option_price": row["option_price"],
                "delta": row["delta"],
                "gamma": row["gamma"],
                "theta": row["theta"],
                "vega": row["vega"],
                "rho": row["rho"],
                "target_stock_shares": target_stock_shares,
                "stock_shares": current_stock_shares,
                "shares_traded": shares_traded,
                "transaction_cost": trading_cost,
                "cash_account": cash_account,
                "stock_value": stock_value,
                "option_value": option_value,
                "portfolio_value": portfolio_value,
                "option_pnl": option_pnl,
                "stock_pnl": stock_pnl,
                "total_pnl": total_pnl,
                "delta_change": 0.0 if previous_delta is None else row["delta"] - previous_delta,
                "did_rehedge": did_rehedge,
            }
        )

        previous_option_price = row["option_price"]
        previous_portfolio_value = portfolio_value
        previous_delta = row["delta"]

    result = pd.DataFrame(rows).set_index("date")
    terminal_payoff = max(float(result["spot"].iloc[-1] - result["strike"].iloc[-1]), 0.0)
    result["terminal_option_payoff"] = 0.0
    result.iloc[-1, result.columns.get_loc("terminal_option_payoff")] = terminal_payoff
    result["hedge_error"] = result["portfolio_value"]
    result.iloc[-1, result.columns.get_loc("hedge_error")] = (
        result["stock_value"].iloc[-1] + result["cash_account"].iloc[-1] + terminal_payoff * config.option_position
    )
    result["hedge_error_change"] = result["hedge_error"].diff().fillna(0.0)
    return result
