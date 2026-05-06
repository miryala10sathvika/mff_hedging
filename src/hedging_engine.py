from __future__ import annotations

from dataclasses import dataclass

import math
import pandas as pd

from src.black_scholes import call_price, put_price
from src.greeks import call_delta, call_greeks, put_delta, put_greeks


@dataclass(frozen=True)
class HedgeConfig:
    rehedge_every: int = 1
    transaction_cost_bps: float = 0.0
    option_position: int = 1
    stock_position_sign: int = 1
    trading_days_per_year: int = 252
    is_static_hedge: bool = False
    is_no_hedge: bool = False
    is_continuous_ideal: bool = False


def _transaction_cost(spot: float, shares_traded: float, transaction_cost_bps: float) -> float:
    return abs(shares_traded) * spot * (transaction_cost_bps / 10000.0)


def _option_type(row: pd.Series) -> str:
    option_type = str(row.get("option_type", "call")).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def _price(row: pd.Series) -> float:
    args = (row["spot"], row["strike"], row["tau"], row["rate"], row["volatility"], row["dividend_yield"])
    return call_price(*args) if _option_type(row) == "call" else put_price(*args)


def _delta(row: pd.Series) -> float:
    args = (row["spot"], row["strike"], row["tau"], row["rate"], row["volatility"], row["dividend_yield"])
    return call_delta(*args) if _option_type(row) == "call" else put_delta(*args)


def _greeks(row: pd.Series) -> dict[str, float]:
    args = (row["spot"], row["strike"], row["tau"], row["rate"], row["volatility"], row["dividend_yield"])
    return call_greeks(*args) if _option_type(row) == "call" else put_greeks(*args)


def _payoff(spot: float, strike: float, option_type: str) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    if option_type == "put":
        return max(strike - spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


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
    if "option_type" not in frame.columns:
        frame["option_type"] = "call"
    frame["option_type"] = frame["option_type"].astype(str).str.lower()
    invalid_option_types = set(frame["option_type"]) - {"call", "put"}
    if invalid_option_types:
        raise ValueError(f"option_type must be 'call' or 'put'; got {sorted(invalid_option_types)}")

    if "option_price" not in frame.columns:
        frame["option_price"] = frame.apply(_price, axis=1)
    if "delta" not in frame.columns:
        frame["delta"] = frame.apply(_delta, axis=1)
    greek_columns = {"gamma", "theta", "vega", "rho"}
    if not greek_columns.issubset(frame.columns):
        greek_frame = frame.apply(lambda row: pd.Series(_greeks(row)), axis=1)
        for column in greek_columns | {"delta"}:
            if column not in frame.columns:
                frame[column] = greek_frame[column]

    rows: list[dict] = []
    current_stock_shares = 0.0
    cash_account = float(config.option_position * frame.loc[0, "option_price"])
    previous_option_price = None
    previous_portfolio_value = None
    previous_delta = None

    for i, row in frame.iterrows():
        target_stock_shares = current_stock_shares
        shares_before_rehedge = current_stock_shares
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

        # Process dividend from previous holding (continuous-yield approximation).
        if i > 0:
            previous_row = frame.loc[i - 1]
            dt = previous_row["tau"] - row["tau"]
            if dt > 0:
                dividend_cash = current_stock_shares * previous_row["spot"] * previous_row["dividend_yield"] * dt
                cash_account += dividend_cash

        # Process rate accrual on cash
        if i > 0:
            previous_row = frame.loc[i - 1]
            dt = previous_row["tau"] - row["tau"]
            if dt > 0:
                cash_account *= math.exp(previous_row["rate"] * dt)

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
            cash_account = portfolio_value - stock_value
        else:
            portfolio_value = stock_value + cash_account

        mark_to_market_hedge_error = portfolio_value - option_value

        option_pnl = 0.0 if previous_option_price is None else config.option_position * (row["option_price"] - previous_option_price)
        stock_pnl = 0.0 if previous_portfolio_value is None else shares_before_rehedge * (row["spot"] - frame.loc[i - 1, "spot"])
        total_pnl = 0.0 if previous_portfolio_value is None else portfolio_value - previous_portfolio_value

        rows.append(
            {
                "date": row["date"],
                "spot": row["spot"],
                "strike": row["strike"],
                "tau": row["tau"],
                "rate": row["rate"],
                "dividend_yield": row["dividend_yield"],
                "option_type": row["option_type"],
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
                "mark_to_market_hedge_error": mark_to_market_hedge_error,
                "option_pnl": option_pnl,
                "stock_pnl": stock_pnl,
                "total_pnl": total_pnl,
                "delta_change": 0.0 if previous_delta is None else row["delta"] - previous_delta,
                "did_rehedge": did_rehedge,
                "rehedge_every": config.rehedge_every,
                "is_static_hedge": config.is_static_hedge,
                "is_no_hedge": config.is_no_hedge,
                "is_continuous_ideal": config.is_continuous_ideal,
            }
        )

        previous_option_price = row["option_price"]
        previous_portfolio_value = portfolio_value
        previous_delta = row["delta"]

    result = pd.DataFrame(rows).set_index("date")
    result["terminal_option_payoff"] = result.apply(
        lambda row: config.option_position * _payoff(float(row["spot"]), float(row["strike"]), str(row["option_type"])),
        axis=1,
    )
    result["hedge_error"] = result["mark_to_market_hedge_error"]
    final_tau = float(result["tau"].iloc[-1])
    if final_tau <= 1e-12:
        result.iloc[-1, result.columns.get_loc("hedge_error")] = (
            result["portfolio_value"].iloc[-1] - result["terminal_option_payoff"].iloc[-1]
        )
    result["hedge_error_change"] = result["hedge_error"].diff().fillna(0.0)
    return result
