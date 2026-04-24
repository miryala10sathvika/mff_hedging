from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


@dataclass(frozen=True)
class MarketDataConfig:
    ticker: str = "SPY"
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    auto_adjust: bool = True


@dataclass(frozen=True)
class YahooOptionContractConfig:
    ticker: str = "SPY"
    expiration: str | None = None
    option_type: str = "call"
    strike: float | None = None
    max_strike_distance_pct: float = 0.1
    min_open_interest: int = 1
    rate: float = 0.02
    history_period: str = "max"


def download_price_history(config: MarketDataConfig) -> pd.DataFrame:
    data = yf.download(
        tickers=config.ticker,
        start=config.start,
        end=config.end,
        auto_adjust=config.auto_adjust,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No price data returned for {config.ticker}")
    return _extract_close_history(data, "spot")


def get_option_expirations(ticker: str) -> tuple[str, ...]:
    return yf.Ticker(ticker).options


def add_realized_volatility(
    prices: pd.DataFrame,
    window: int = 21,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    data = prices.copy()
    data["log_return"] = np.log(data["spot"] / data["spot"].shift(1))
    rolling_std = data["log_return"].rolling(window=window).std()
    data["realized_vol"] = rolling_std * annualization_factor**0.5
    return data


def _normalize_ohlc_history(data: pd.DataFrame, close_name: str) -> pd.DataFrame:
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        level0 = [str(value) for value in data.columns.get_level_values(0)]
        level1 = [str(value) for value in data.columns.get_level_values(1)]
        if close_name in level0:
            data = data.xs(close_name, axis=1, level=0, drop_level=True).copy()
        elif close_name in level1:
            data = data.xs(close_name, axis=1, level=1, drop_level=True).copy()
        else:
            data = data.droplevel(-1, axis=1)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=close_name)
    return data


def _extract_close_history(data: pd.DataFrame, output_name: str) -> pd.DataFrame:
    if data.empty:
        return data

    normalized = _normalize_ohlc_history(data, "Close").copy()
    normalized.index = pd.to_datetime(normalized.index)

    candidate_columns = ["Close", "close", "Adj Close", "adjclose", output_name]
    for column in candidate_columns:
        if column in normalized.columns:
            history = normalized[[column]].rename(columns={column: output_name}).dropna().copy()
            history.index = pd.to_datetime(history.index)
            return history

    if normalized.shape[1] == 1:
        only_column = normalized.columns[0]
        history = normalized[[only_column]].rename(columns={only_column: output_name}).dropna().copy()
        history.index = pd.to_datetime(history.index)
        return history

    available = [str(column) for column in normalized.columns]
    raise ValueError(f"Could not identify close column. Available columns: {available}")


def fetch_current_option_chain(
    ticker: str,
    expiration: str | None = None,
    option_type: str = "call",
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    ticker_obj = yf.Ticker(ticker)
    try:
        expirations = ticker_obj.options
    except Exception as exc:
        raise RuntimeError(
            "Yahoo Finance option lookup failed. This is usually a network or DNS issue in the current environment."
        ) from exc
    if not expirations:
        raise ValueError(f"No option expirations returned by yfinance for {ticker}")

    selected_expiration = expiration or expirations[0]
    if selected_expiration not in expirations:
        raise ValueError(f"Expiration {selected_expiration} not available. Choices: {expirations}")

    option_chain = ticker_obj.option_chain(selected_expiration)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    chain = option_chain.calls if option_type == "call" else option_chain.puts
    if chain is None or chain.empty:
        raise ValueError(f"No {option_type} chain returned for {ticker} {selected_expiration}")

    return chain.copy(), option_chain.underlying or {}, selected_expiration


def select_option_contract(
    chain: pd.DataFrame,
    underlying_spot: float,
    strike: float | None = None,
    max_strike_distance_pct: float = 0.5, # increased default
    min_open_interest: int = 1,
) -> pd.Series:
    contracts = chain.copy()
    if "openInterest" in contracts.columns:
        contracts = contracts[contracts["openInterest"].fillna(0) >= min_open_interest]
    if contracts.empty:
        raise ValueError("No contracts left after open interest filter")

    target_strike = strike if strike is not None else underlying_spot
    contracts["strike_distance"] = (contracts["strike"] - target_strike).abs()
    contracts["distance_pct"] = contracts["strike_distance"] / max(underlying_spot, 1e-12)
    contracts = contracts[contracts["distance_pct"] <= max_strike_distance_pct]
    if contracts.empty:
        # Fallback to the single closest strike rather than raising an error
        contracts = chain.copy()
        if "openInterest" in contracts.columns:
            contracts = contracts[contracts["openInterest"].fillna(0) >= min_open_interest]
        contracts["strike_distance"] = (contracts["strike"] - target_strike).abs()

    sort_columns = ["strike_distance"]
    ascending = [True]
    if "openInterest" in contracts.columns:
        sort_columns.append("openInterest")
        ascending.append(False)
    contracts = contracts.sort_values(sort_columns, ascending=ascending)
    return contracts.iloc[0]


def download_option_history(contract_symbol: str, period: str = "max") -> pd.DataFrame:
    data = yf.download(contract_symbol, period=period, auto_adjust=False, progress=False)
    if data.empty:
        data = yf.Ticker(contract_symbol).history(period=period, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No option history returned for contract {contract_symbol}")
    try:
        return _extract_close_history(data, "option_price")
    except ValueError as exc:
        raise ValueError(f"Failed to parse option history for {contract_symbol}: {exc}") from exc


def build_yahoo_observed_option_frame(config: YahooOptionContractConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    chain, underlying, selected_expiration = fetch_current_option_chain(
        ticker=config.ticker,
        expiration=config.expiration,
        option_type=config.option_type,
    )

    underlying_spot = float(underlying.get("regularMarketPrice") or underlying.get("previousClose") or np.nan)
    if np.isnan(underlying_spot):
        spot_history = download_price_history(MarketDataConfig(ticker=config.ticker, start="2024-01-01", end="2026-12-31"))
        underlying_spot = float(spot_history["spot"].iloc[-1])

    contract = select_option_contract(
        chain=chain,
        underlying_spot=underlying_spot,
        strike=config.strike,
        max_strike_distance_pct=config.max_strike_distance_pct,
        min_open_interest=config.min_open_interest,
    )

    option_history = download_option_history(str(contract["contractSymbol"]), period=config.history_period)
    start = option_history.index.min().strftime("%Y-%m-%d")
    end = (option_history.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spot_history = download_price_history(MarketDataConfig(ticker=config.ticker, start=start, end=end))

    merged = spot_history.join(option_history, how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between underlying and option history")

    expiration_ts = pd.Timestamp(selected_expiration).normalize()
    if expiration_ts.tz is not None:
        expiration_ts = expiration_ts.tz_convert(None)
    merged["strike"] = float(contract["strike"])
    merged["rate"] = config.rate
    merged["volatility"] = float(contract["impliedVolatility"])
    merged_index = pd.DatetimeIndex(pd.to_datetime(merged.index)).normalize()
    if merged_index.tz is not None:
        merged_index = merged_index.tz_convert(None)
    tau_days = (expiration_ts - merged_index).to_numpy(dtype="timedelta64[ns]") / np.timedelta64(1, "D")
    merged["tau"] = pd.Series(tau_days, index=merged.index, dtype="float64") / CALENDAR_DAYS_PER_YEAR
    merged["bid"] = float(contract["bid"]) if pd.notna(contract.get("bid")) else np.nan
    merged["ask"] = float(contract["ask"]) if pd.notna(contract.get("ask")) else np.nan
    merged["last_trade_date"] = pd.to_datetime(contract.get("lastTradeDate")) if pd.notna(contract.get("lastTradeDate")) else pd.NaT
    merged["tau"] = merged["tau"].clip(lower=0.0)
    merged = merged.loc[merged["tau"] > 0].copy()
    if merged.empty:
        raise ValueError("Merged option frame has no observations before expiry")

    metadata = {
        "ticker": config.ticker,
        "option_type": config.option_type,
        "expiration": selected_expiration,
        "contract_symbol": str(contract["contractSymbol"]),
        "selected_strike": float(contract["strike"]),
        "current_underlying_spot": underlying_spot,
        "current_chain_iv": float(contract["impliedVolatility"]),
    }
    return merged, metadata


def build_option_frame(
    prices: pd.DataFrame,
    maturity_days: int = 21,
    strike_mode: str = "atm_initial",
    fixed_volatility: float = 0.20,
    rate: float = 0.02,
    use_realized_vol: bool = False,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices cannot be empty")

    data = prices.copy()
    if strike_mode != "atm_initial":
        raise ValueError("Only strike_mode='atm_initial' is currently supported")

    strike = float(data["spot"].iloc[0])
    n_obs = len(data)
    tau = [(maturity_days - i) / TRADING_DAYS_PER_YEAR for i in range(n_obs)]
    data["tau"] = pd.Series(tau, index=data.index).clip(lower=0.0)
    data["strike"] = strike
    data["rate"] = rate

    if use_realized_vol:
        if "realized_vol" not in data.columns:
            raise ValueError("realized_vol column missing; run add_realized_volatility first")
        data["volatility"] = data["realized_vol"].fillna(fixed_volatility)
    else:
        data["volatility"] = fixed_volatility

    return data.loc[data["tau"] > 0].copy()
