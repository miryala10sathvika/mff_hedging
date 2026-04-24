from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import (
    MarketDataConfig,
    YahooOptionContractConfig,
    add_realized_volatility,
    build_yahoo_observed_option_frame,
    build_option_frame,
    download_price_history,
)
from src.hedging_engine import HedgeConfig, run_delta_hedge
from src.pnl_attribution import add_greek_pnl_attribution, summarize_results


@dataclass(frozen=True)
class ExperimentConfig:
    ticker: str = "SPY"
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    maturity_days: int = 21
    fixed_volatility: float = 0.20
    rate: float = 0.02
    use_realized_vol: bool = False
    realized_vol_window: int = 21
    transaction_cost_bps: float = 0.0
    rehedge_frequencies: tuple[int, ...] = (1, 2, 5)
    include_no_hedge: bool = True
    include_static_hedge: bool = True
    include_continuous_ideal: bool = True


@dataclass(frozen=True)
class YahooOptionExperimentConfig:
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
    transaction_cost_bps: float = 0.0
    rehedge_frequencies: tuple[int, ...] = (1, 2, 5)
    include_no_hedge: bool = True
    include_static_hedge: bool = True
    include_continuous_ideal: bool = True


def run_experiment(config: ExperimentConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    market_config = MarketDataConfig(ticker=config.ticker, start=config.start, end=config.end)
    prices = download_price_history(market_config)

    if config.use_realized_vol:
        prices = add_realized_volatility(prices, window=config.realized_vol_window)

    option_frame = build_option_frame(
        prices=prices,
        maturity_days=config.maturity_days,
        fixed_volatility=config.fixed_volatility,
        rate=config.rate,
        use_realized_vol=config.use_realized_vol,
    )

    paths: dict[str, pd.DataFrame] = {}
    summaries = []

    for frequency in config.rehedge_frequencies:
        label = f"every_{frequency}_day" if frequency == 1 else f"every_{frequency}_days"
        hedge_config = HedgeConfig(
            rehedge_every=frequency,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result

        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summaries.append(summary)

    if getattr(config, "include_no_hedge", False):
        label = "no_hedge"
        hedge_config = HedgeConfig(
            is_no_hedge=True,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result
        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summaries.append(summary)

    if getattr(config, "include_static_hedge", False):
        label = "static_hedge"
        hedge_config = HedgeConfig(
            is_static_hedge=True,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result
        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summaries.append(summary)

    if getattr(config, "include_continuous_ideal", False):
        label = "continuous_ideal"
        hedge_config = HedgeConfig(
            is_continuous_ideal=True,
            transaction_cost_bps=0.0,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result
        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).set_index("strategy").sort_index()
    return paths, summary_df


def run_yahoo_option_experiment(
    config: YahooOptionExperimentConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str | float]]:
    option_frame, metadata = build_yahoo_observed_option_frame(
        YahooOptionContractConfig(
            ticker=config.ticker,
            expiration=config.expiration,
            option_type=config.option_type,
            strike=config.strike,
            min_days_to_expiration=config.min_days_to_expiration,
            max_strike_distance_pct=config.max_strike_distance_pct,
            min_open_interest=config.min_open_interest,
            min_implied_volatility=config.min_implied_volatility,
            fallback_volatility=config.fallback_volatility,
            realized_vol_window=config.realized_vol_window,
            rate=config.rate,
            history_period=config.history_period,
        )
    )

    paths: dict[str, pd.DataFrame] = {}
    summaries = []

    for frequency in config.rehedge_frequencies:
        label = f"every_{frequency}_day" if frequency == 1 else f"every_{frequency}_days"
        hedge_config = HedgeConfig(
            rehedge_every=frequency,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result

        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summary["contract_symbol"] = metadata["contract_symbol"]
        summary["expiration"] = metadata["expiration"]
        summary["selected_strike"] = metadata["selected_strike"]
        summaries.append(summary)

    if getattr(config, "include_no_hedge", False):
        label = "no_hedge"
        hedge_config = HedgeConfig(
            is_no_hedge=True,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result
        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summary["contract_symbol"] = metadata["contract_symbol"]
        summary["expiration"] = metadata["expiration"]
        summary["selected_strike"] = metadata["selected_strike"]
        summaries.append(summary)

    if getattr(config, "include_static_hedge", False):
        label = "static_hedge"
        hedge_config = HedgeConfig(
            is_static_hedge=True,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result
        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summary["contract_symbol"] = metadata["contract_symbol"]
        summary["expiration"] = metadata["expiration"]
        summary["selected_strike"] = metadata["selected_strike"]
        summaries.append(summary)

    if getattr(config, "include_continuous_ideal", False):
        label = "continuous_ideal"
        hedge_config = HedgeConfig(
            is_continuous_ideal=True,
            transaction_cost_bps=0.0,
        )
        result = run_delta_hedge(option_frame, hedge_config)
        result = add_greek_pnl_attribution(result)
        paths[label] = result
        summary = summarize_results(result).to_dict()
        summary["strategy"] = label
        summary["contract_symbol"] = metadata["contract_symbol"]
        summary["expiration"] = metadata["expiration"]
        summary["selected_strike"] = metadata["selected_strike"]
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).set_index("strategy").sort_index()
    return paths, summary_df, metadata


def save_plots(results: dict[str, pd.DataFrame], output_dir: str = "outputs/figures") -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    plt.figure(figsize=(10, 6))
    for label, frame in results.items():
        plt.plot(frame.index, frame["portfolio_value"], label=label)
    plt.title("Delta-Hedged Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    portfolio_path = output_path / "portfolio_value_paths.png"
    plt.savefig(portfolio_path, dpi=150)
    plt.close()
    saved_files.append(portfolio_path)

    plt.figure(figsize=(10, 6))
    labels = list(results.keys())
    values = [frame["hedge_error"].iloc[-1] for frame in results.values()]
    plt.bar(labels, values)
    plt.title("Final Hedge Error by Rehedge Frequency")
    plt.xlabel("Strategy")
    plt.ylabel("Final Hedge Error")
    plt.tight_layout()
    error_path = output_path / "final_hedge_error.png"
    plt.savefig(error_path, dpi=150)
    plt.close()
    saved_files.append(error_path)

    return saved_files


def save_summary_table(summary: pd.DataFrame, output_path: str = "outputs/tables/summary.csv") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path)
    return path
