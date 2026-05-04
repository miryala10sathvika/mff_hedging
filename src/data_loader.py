from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.greeks import call_delta, put_delta


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
    min_days_to_expiration: int = 21
    max_strike_distance_pct: float = 0.1
    min_open_interest: int = 1
    min_implied_volatility: float = 0.05
    fallback_volatility: float = 0.20
    realized_vol_window: int = 21
    rate: float = 0.02
    history_period: str = "max"
    min_history_observations: int = 20
    max_expirations_to_scan: int = 12
    candidates_per_expiration: int = 5
    target_abs_delta: float = 0.50
    min_abs_delta: float = 0.25
    max_abs_delta: float = 0.75


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
    normalized.index = _to_naive_daily_index(normalized.index)

    candidate_columns = ["Close", "close", "Adj Close", "adjclose", output_name]
    for column in candidate_columns:
        if column in normalized.columns:
            history = normalized[[column]].rename(columns={column: output_name}).dropna().copy()
            history.index = _to_naive_daily_index(history.index)
            return history.groupby(level=0).last()

    if normalized.shape[1] == 1:
        only_column = normalized.columns[0]
        history = normalized[[only_column]].rename(columns={only_column: output_name}).dropna().copy()
        history.index = _to_naive_daily_index(history.index)
        return history.groupby(level=0).last()

    available = [str(column) for column in normalized.columns]
    raise ValueError(f"Could not identify close column. Available columns: {available}")


def _to_naive_daily_index(index: pd.Index) -> pd.DatetimeIndex:
    date_index = pd.DatetimeIndex(pd.to_datetime(index))
    if date_index.tz is not None:
        date_index = date_index.tz_convert(None)
    return date_index.normalize()


def _first_finite_float(values: list[Any]) -> float:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return float("nan")


def _resolve_underlying_spot(ticker_obj: yf.Ticker, underlying: dict[str, Any], ticker: str) -> float:
    spot = _first_finite_float(
        [
            underlying.get("regularMarketPrice"),
            underlying.get("currentPrice"),
            underlying.get("lastPrice"),
            underlying.get("previousClose"),
            underlying.get("regularMarketPreviousClose"),
        ]
    )
    if np.isfinite(spot):
        return spot

    try:
        fast_info = ticker_obj.fast_info
        spot = _first_finite_float(
            [
                fast_info.get("last_price"),
                fast_info.get("regular_market_price"),
                fast_info.get("previous_close"),
                fast_info.get("regular_market_previous_close"),
            ]
        )
    except Exception:
        spot = float("nan")
    if np.isfinite(spot):
        return spot

    recent_history = download_price_history(MarketDataConfig(ticker=ticker, start="2024-01-01", end="2026-12-31"))
    return float(recent_history["spot"].iloc[-1])


def fetch_current_option_chain(
    ticker: str,
    expiration: str | None = None,
    option_type: str = "call",
    min_days_to_expiration: int = 0,
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

    if expiration:
        selected_expiration = expiration
    else:
        today = pd.Timestamp.today().normalize()
        eligible_expirations = [
            exp
            for exp in expirations
            if (pd.Timestamp(exp).normalize() - today).days >= min_days_to_expiration
        ]
        selected_expiration = eligible_expirations[0] if eligible_expirations else expirations[-1]
    if selected_expiration not in expirations:
        raise ValueError(f"Expiration {selected_expiration} not available. Choices: {expirations}")

    option_chain = ticker_obj.option_chain(selected_expiration)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    chain = option_chain.calls if option_type == "call" else option_chain.puts
    if chain is None or chain.empty:
        raise ValueError(f"No {option_type} chain returned for {ticker} {selected_expiration}")

    return chain.copy(), getattr(option_chain, "underlying", None) or {}, selected_expiration


def _eligible_expirations(
    expirations: tuple[str, ...],
    min_days_to_expiration: int,
    max_expirations_to_scan: int,
) -> list[str]:
    today = pd.Timestamp.today().normalize()
    eligible = [
        expiration
        for expiration in expirations
        if (pd.Timestamp(expiration).normalize() - today).days >= min_days_to_expiration
    ]
    if not eligible:
        eligible = list(expirations)
    return eligible[:max_expirations_to_scan]


def _estimated_delta(
    option_type: str,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    volatility: float,
) -> float:
    if option_type == "call":
        return call_delta(spot, strike, tau, rate, volatility)
    if option_type == "put":
        return put_delta(spot, strike, tau, rate, volatility)
    raise ValueError("option_type must be 'call' or 'put'")


def _rank_contract_candidates(
    chain: pd.DataFrame,
    underlying_spot: float,
    selected_expiration: str,
    option_type: str,
    strike: float | None,
    max_strike_distance_pct: float,
    min_open_interest: int,
    min_implied_volatility: float,
    fallback_volatility: float,
    rate: float,
    target_abs_delta: float,
    min_abs_delta: float,
    max_abs_delta: float,
) -> pd.DataFrame:
    contracts = chain.copy()
    if contracts.empty:
        return contracts

    target_strike = strike if strike is not None else underlying_spot
    contracts["strike_distance"] = (contracts["strike"] - target_strike).abs()
    contracts["distance_pct"] = contracts["strike_distance"] / max(underlying_spot, 1e-12)
    distance_filtered = contracts[contracts["distance_pct"] <= max_strike_distance_pct].copy()
    if not distance_filtered.empty:
        contracts = distance_filtered

    expiration_ts = pd.Timestamp(selected_expiration).normalize()
    tau = max((expiration_ts - pd.Timestamp.today().normalize()).days / CALENDAR_DAYS_PER_YEAR, 1.0 / CALENDAR_DAYS_PER_YEAR)

    estimated_deltas = []
    for row in contracts.itertuples():
        volatility = _first_finite_float([getattr(row, "impliedVolatility", float("nan")), fallback_volatility])
        volatility = max(volatility, min_implied_volatility, 1e-6)
        estimated_deltas.append(
            _estimated_delta(
                option_type=option_type,
                spot=underlying_spot,
                strike=float(row.strike),
                tau=tau,
                rate=rate,
                volatility=volatility,
            )
        )

    contracts["estimated_delta"] = estimated_deltas
    contracts["estimated_abs_delta"] = contracts["estimated_delta"].abs()
    contracts["delta_in_range"] = contracts["estimated_abs_delta"].between(min_abs_delta, max_abs_delta)
    if contracts["delta_in_range"].any():
        contracts = contracts[contracts["delta_in_range"]].copy()

    contracts["delta_penalty"] = (contracts["estimated_abs_delta"] - target_abs_delta).abs()
    sort_columns = ["delta_penalty", "strike_distance"]
    ascending = [True, True]
    if "openInterest" in contracts.columns:
        sort_columns.append("openInterest")
        ascending.append(False)
    return contracts.sort_values(sort_columns, ascending=ascending)


def select_option_contract_with_history(
    ticker_obj: yf.Ticker,
    ticker: str,
    expirations: tuple[str, ...],
    underlying_spot: float,
    option_type: str = "call",
    strike: float | None = None,
    min_days_to_expiration: int = 21,
    max_strike_distance_pct: float = 0.1,
    min_open_interest: int = 1,
    min_implied_volatility: float = 0.05,
    fallback_volatility: float = 0.20,
    rate: float = 0.02,
    history_period: str = "max",
    min_history_observations: int = 20,
    max_expirations_to_scan: int = 12,
    candidates_per_expiration: int = 5,
    target_abs_delta: float = 0.50,
    min_abs_delta: float = 0.25,
    max_abs_delta: float = 0.75,
) -> tuple[pd.Series, str, pd.DataFrame]:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    attempts: list[dict[str, Any]] = []
    fallback: tuple[pd.Series, str, pd.DataFrame, int] | None = None
    for selected_expiration in _eligible_expirations(
        expirations,
        min_days_to_expiration=min_days_to_expiration,
        max_expirations_to_scan=max_expirations_to_scan,
    ):
        option_chain = ticker_obj.option_chain(selected_expiration)
        chain = option_chain.calls if option_type == "call" else option_chain.puts
        if chain is None or chain.empty:
            continue

        candidates = _rank_contract_candidates(
            chain=chain,
            underlying_spot=underlying_spot,
            selected_expiration=selected_expiration,
            option_type=option_type,
            strike=strike,
            max_strike_distance_pct=max_strike_distance_pct,
            min_open_interest=min_open_interest,
            min_implied_volatility=min_implied_volatility,
            fallback_volatility=fallback_volatility,
            rate=rate,
            target_abs_delta=target_abs_delta,
            min_abs_delta=min_abs_delta,
            max_abs_delta=max_abs_delta,
        ).head(candidates_per_expiration)

        for _, contract in candidates.iterrows():
            contract_symbol = str(contract["contractSymbol"])
            try:
                option_history = download_option_history(contract_symbol, period=history_period)
            except Exception as exc:
                attempts.append(
                    {
                        "expiration": selected_expiration,
                        "contract_symbol": contract_symbol,
                        "error": str(exc),
                    }
                )
                continue

            history_observations = int(len(option_history))
            attempts.append(
                {
                    "expiration": selected_expiration,
                    "contract_symbol": contract_symbol,
                    "history_observations": history_observations,
                    "estimated_abs_delta": float(contract.get("estimated_abs_delta", float("nan"))),
                }
            )
            if fallback is None or history_observations > fallback[3]:
                fallback = (contract, selected_expiration, option_history, history_observations)
            if history_observations >= min_history_observations:
                return contract, selected_expiration, option_history

    if fallback is not None:
        return fallback[0], fallback[1], fallback[2]
    raise ValueError(f"No option contract with usable history found for {ticker}. Attempts: {attempts}")


def select_option_contract(
    chain: pd.DataFrame,
    underlying_spot: float,
    strike: float | None = None,
    max_strike_distance_pct: float = 0.5, # increased default
    min_open_interest: int = 1,
    min_implied_volatility: float = 0.0,
) -> pd.Series:
    contracts = chain.copy()
    if "openInterest" in contracts.columns:
        open_interest_contracts = contracts[contracts["openInterest"].fillna(0) >= min_open_interest]
        if not open_interest_contracts.empty:
            contracts = open_interest_contracts
    if contracts.empty:
        raise ValueError("No contracts available for selection")
    selection_universe = contracts.copy()

    target_strike = strike if strike is not None else underlying_spot
    contracts["strike_distance"] = (contracts["strike"] - target_strike).abs()
    contracts["distance_pct"] = contracts["strike_distance"] / max(underlying_spot, 1e-12)
    contracts = contracts[contracts["distance_pct"] <= max_strike_distance_pct]
    if contracts.empty:
        # Fallback to the single closest strike rather than raising an error
        contracts = selection_universe.copy()
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
    ticker_obj = yf.Ticker(config.ticker)
    try:
        expirations = ticker_obj.options
    except Exception as exc:
        raise RuntimeError(
            "Yahoo Finance option lookup failed. This is usually a network or DNS issue in the current environment."
        ) from exc
    if not expirations:
        raise ValueError(f"No option expirations returned by yfinance for {config.ticker}")

    first_expiration = config.expiration or _eligible_expirations(
        expirations,
        min_days_to_expiration=config.min_days_to_expiration,
        max_expirations_to_scan=1,
    )[0]
    if first_expiration not in expirations:
        raise ValueError(f"Expiration {first_expiration} not available. Choices: {expirations}")

    first_chain = ticker_obj.option_chain(first_expiration)
    underlying_spot = _resolve_underlying_spot(
        ticker_obj,
        getattr(first_chain, "underlying", None) or {},
        config.ticker,
    )

    if config.expiration is None:
        contract, selected_expiration, option_history = select_option_contract_with_history(
            ticker_obj=ticker_obj,
            ticker=config.ticker,
            expirations=expirations,
            underlying_spot=underlying_spot,
            option_type=config.option_type,
            strike=config.strike,
            min_days_to_expiration=config.min_days_to_expiration,
            max_strike_distance_pct=config.max_strike_distance_pct,
            min_open_interest=config.min_open_interest,
            min_implied_volatility=config.min_implied_volatility,
            fallback_volatility=config.fallback_volatility,
            rate=config.rate,
            history_period=config.history_period,
            min_history_observations=config.min_history_observations,
            max_expirations_to_scan=config.max_expirations_to_scan,
            candidates_per_expiration=config.candidates_per_expiration,
            target_abs_delta=config.target_abs_delta,
            min_abs_delta=config.min_abs_delta,
            max_abs_delta=config.max_abs_delta,
        )
    else:
        chain = first_chain.calls if config.option_type == "call" else first_chain.puts
        if chain is None or chain.empty:
            raise ValueError(f"No {config.option_type} chain returned for {config.ticker} {first_expiration}")
        selected_expiration = first_expiration
        contract = select_option_contract(
            chain=chain,
            underlying_spot=underlying_spot,
            strike=config.strike,
            max_strike_distance_pct=config.max_strike_distance_pct,
            min_open_interest=config.min_open_interest,
            min_implied_volatility=config.min_implied_volatility,
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
    merged["option_type"] = config.option_type
    merged["dividend_yield"] = 0.0
    chain_iv = _first_finite_float([contract.get("impliedVolatility"), float("nan")])
    if np.isfinite(chain_iv) and chain_iv >= config.min_implied_volatility:
        merged["volatility"] = chain_iv
        volatility_source = "current_chain_iv"
    else:
        log_returns = np.log(merged["spot"] / merged["spot"].shift(1))
        realized_vol = log_returns.rolling(window=config.realized_vol_window, min_periods=2).std() * TRADING_DAYS_PER_YEAR**0.5
        merged["volatility"] = realized_vol.bfill().ffill().fillna(config.fallback_volatility)
        merged["volatility"] = merged["volatility"].clip(lower=config.min_implied_volatility)
        volatility_source = "realized_vol_fallback"
    merged_index = _to_naive_daily_index(merged.index)
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
        "min_days_to_expiration": config.min_days_to_expiration,
        "contract_symbol": str(contract["contractSymbol"]),
        "selected_strike": float(contract["strike"]),
        "current_underlying_spot": underlying_spot,
        "current_chain_iv": chain_iv,
        "volatility_source": volatility_source,
        "history_observations": int(len(merged)),
        "estimated_abs_delta": float(contract.get("estimated_abs_delta", float("nan"))),
        "selection_method": "history_aware_auto" if config.expiration is None else "single_expiration",
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
