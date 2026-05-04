from __future__ import annotations

from dataclasses import dataclass, field
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
class StrategySelectorConfig:
    enabled: bool = True
    cvar_metric: str = "cvar_95_daily"
    cvar_weight: float = 1.0
    transaction_cost_weight: float = 1.0


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
    strategy_selector: StrategySelectorConfig = field(default_factory=StrategySelectorConfig)


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
    min_history_observations: int = 20
    max_expirations_to_scan: int = 12
    candidates_per_expiration: int = 5
    target_abs_delta: float = 0.50
    min_abs_delta: float = 0.25
    max_abs_delta: float = 0.75
    transaction_cost_bps: float = 0.0
    rehedge_frequencies: tuple[int, ...] = (1, 2, 5)
    include_no_hedge: bool = True
    include_static_hedge: bool = True
    include_continuous_ideal: bool = True
    strategy_selector: StrategySelectorConfig = field(default_factory=StrategySelectorConfig)


def apply_strategy_selector(
    summary: pd.DataFrame,
    selector_config: StrategySelectorConfig | None = None,
) -> pd.DataFrame:
    if summary.empty:
        raise ValueError("summary cannot be empty")

    selector = selector_config or StrategySelectorConfig()
    if not selector.enabled:
        return summary.copy()

    required_columns = {selector.cvar_metric, "total_transaction_cost", "rehedge_count"}
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"summary is missing required selector columns: {sorted(missing_columns)}")

    annotated = summary.copy()
    annotated["selector_cvar_loss"] = pd.NA
    annotated["selector_objective"] = pd.NA
    annotated["selector_rank"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["is_auto_selected"] = False

    periodic_mask = annotated.index.to_series().str.startswith("every_")
    periodic = annotated.loc[periodic_mask].copy()
    if periodic.empty:
        raise ValueError("strategy selector requires at least one periodic rehedging strategy")

    periodic["selector_cvar_loss"] = -periodic[selector.cvar_metric].clip(upper=0.0)
    periodic["selector_objective"] = (
        selector.cvar_weight * periodic["selector_cvar_loss"]
        + selector.transaction_cost_weight * periodic["total_transaction_cost"]
    )
    periodic = periodic.sort_values(
        by=["selector_objective", "selector_cvar_loss", "total_transaction_cost", "rehedge_count"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    periodic["selector_rank"] = pd.Series(range(1, len(periodic) + 1), index=periodic.index, dtype="Int64")

    selected_strategy = periodic.index[0]
    periodic["is_auto_selected"] = periodic.index == selected_strategy

    annotated.loc[periodic.index, "selector_cvar_loss"] = periodic["selector_cvar_loss"]
    annotated.loc[periodic.index, "selector_objective"] = periodic["selector_objective"]
    annotated.loc[periodic.index, "selector_rank"] = periodic["selector_rank"]
    annotated.loc[periodic.index, "is_auto_selected"] = periodic["is_auto_selected"]
    return annotated


def get_selected_strategy(summary: pd.DataFrame) -> pd.Series:
    if "is_auto_selected" not in summary.columns:
        raise ValueError("summary does not contain strategy-selector output")

    selected = summary.loc[summary["is_auto_selected"].fillna(False)]
    if selected.empty:
        raise ValueError("summary does not mark any auto-selected strategy")
    if len(selected) > 1:
        raise ValueError("summary marks more than one auto-selected strategy")
    return selected.iloc[0]


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
    summary_df = apply_strategy_selector(summary_df, config.strategy_selector)
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
            min_history_observations=config.min_history_observations,
            max_expirations_to_scan=config.max_expirations_to_scan,
            candidates_per_expiration=config.candidates_per_expiration,
            target_abs_delta=config.target_abs_delta,
            min_abs_delta=config.min_abs_delta,
            max_abs_delta=config.max_abs_delta,
        )
    )

    paths: dict[str, pd.DataFrame] = {}
    summaries = []
    metadata_summary_fields = [
        "contract_symbol",
        "expiration",
        "selected_strike",
        "history_observations",
        "estimated_abs_delta",
        "volatility_source",
        "selection_method",
    ]

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
        for field in metadata_summary_fields:
            summary[field] = metadata.get(field)
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
        for field in metadata_summary_fields:
            summary[field] = metadata.get(field)
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
        for field in metadata_summary_fields:
            summary[field] = metadata.get(field)
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
        for field in metadata_summary_fields:
            summary[field] = metadata.get(field)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).set_index("strategy").sort_index()
    summary_df = apply_strategy_selector(summary_df, config.strategy_selector)
    return paths, summary_df, metadata


def save_plots(results: dict[str, pd.DataFrame], output_dir: str = "outputs/figures") -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    representative_label, representative_frame = next(iter(results.items()))
    practical_results = {
        label: frame for label, frame in results.items() if label != "continuous_ideal"
    }
    comparison_results = practical_results or results

    fig, spot_axis = plt.subplots(figsize=(10, 6))
    option_axis = spot_axis.twinx()
    spot_axis.plot(representative_frame.index, representative_frame["spot"], color="tab:blue", label="SPY spot")
    spot_axis.axhline(
        float(representative_frame["strike"].iloc[0]),
        color="tab:gray",
        linestyle="--",
        linewidth=1.0,
        label="strike",
    )
    option_axis.plot(
        representative_frame.index,
        representative_frame["option_price"],
        color="tab:orange",
        label="option price",
    )
    spot_axis.set_title("Underlying Spot, Strike, and Option Price")
    spot_axis.set_xlabel("Date")
    spot_axis.set_ylabel("Underlying / Strike")
    option_axis.set_ylabel("Option Price")
    lines, labels = spot_axis.get_legend_handles_labels()
    option_lines, option_labels = option_axis.get_legend_handles_labels()
    spot_axis.legend(lines + option_lines, labels + option_labels, loc="best")
    fig.tight_layout()
    market_path = output_path / "market_inputs.png"
    fig.savefig(market_path, dpi=150)
    plt.close(fig)
    saved_files.append(market_path)

    plt.figure(figsize=(10, 6))
    for label, frame in comparison_results.items():
        plt.plot(frame.index, frame["portfolio_value"], label=label)
    plt.plot(
        representative_frame.index,
        representative_frame["option_value"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="observed option value",
    )
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
    for label, frame in comparison_results.items():
        plt.plot(frame.index, frame["hedge_error"], label=label)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Practical Strategy Hedge Error Through Time")
    plt.xlabel("Date")
    plt.ylabel("Hedge Error")
    plt.legend()
    plt.tight_layout()
    hedge_error_path = output_path / "hedge_error_paths.png"
    plt.savefig(hedge_error_path, dpi=150)
    plt.close()
    saved_files.append(hedge_error_path)

    plt.figure(figsize=(10, 6))
    for label, frame in comparison_results.items():
        plt.plot(frame.index, frame["stock_shares"], label=label)
    plt.title("Stock Hedge Position Per Option")
    plt.xlabel("Date")
    plt.ylabel("Shares Held")
    plt.legend()
    plt.tight_layout()
    shares_path = output_path / "stock_shares_paths.png"
    plt.savefig(shares_path, dpi=150)
    plt.close()
    saved_files.append(shares_path)

    plt.figure(figsize=(10, 6))
    plt.plot(representative_frame.index, representative_frame["delta"], label="delta")
    plt.title(f"Option Delta Path ({representative_label})")
    plt.xlabel("Date")
    plt.ylabel("Delta")
    plt.legend()
    plt.tight_layout()
    delta_path = output_path / "option_delta_path.png"
    plt.savefig(delta_path, dpi=150)
    plt.close()
    saved_files.append(delta_path)

    plt.figure(figsize=(10, 6))
    for label, frame in comparison_results.items():
        plt.plot(frame.index, frame["transaction_cost"].cumsum(), label=label)
    plt.title("Practical Strategy Cumulative Transaction Cost")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Cost")
    plt.legend()
    plt.tight_layout()
    transaction_cost_path = output_path / "cumulative_transaction_cost.png"
    plt.savefig(transaction_cost_path, dpi=150)
    plt.close()
    saved_files.append(transaction_cost_path)

    plt.figure(figsize=(10, 6))
    for label, frame in comparison_results.items():
        plt.plot(frame.index, frame["total_pnl"], label=label)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Practical Strategy Daily Portfolio PnL")
    plt.xlabel("Date")
    plt.ylabel("Daily PnL")
    plt.legend()
    plt.tight_layout()
    pnl_path = output_path / "daily_pnl_paths.png"
    plt.savefig(pnl_path, dpi=150)
    plt.close()
    saved_files.append(pnl_path)

    attribution_label = next((label for label in results if label.startswith("every_")), representative_label)
    attribution_frame = results[attribution_label]
    attribution_columns = [
        "delta_pnl_approx",
        "gamma_pnl_approx",
        "theta_pnl_approx",
        "vega_pnl_approx",
        "rho_pnl_approx",
        "pnl_attribution_residual",
    ]
    available_attribution_columns = [
        column for column in attribution_columns if column in attribution_frame.columns
    ]
    if available_attribution_columns:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        greek_columns = [
            column for column in available_attribution_columns if column != "pnl_attribution_residual"
        ]
        for column in greek_columns:
            axes[0].plot(attribution_frame.index, attribution_frame[column].cumsum(), label=column)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_title(f"Cumulative Greek Approximation Terms ({attribution_label})")
        axes[0].set_ylabel("Cumulative Greek PnL")
        axes[0].legend()

        if "pnl_attribution_residual" in available_attribution_columns:
            axes[1].plot(
                attribution_frame.index,
                attribution_frame["pnl_attribution_residual"].cumsum(),
                color="tab:brown",
                label="residual",
            )
            axes[1].legend()
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title("Cumulative Residual")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Residual")
        fig.tight_layout()
        attribution_path = output_path / "greek_pnl_attribution_cumulative.png"
        fig.savefig(attribution_path, dpi=150)
        plt.close(fig)
        saved_files.append(attribution_path)

    plt.figure(figsize=(10, 6))
    labels = list(comparison_results.keys())
    values = [frame["hedge_error"].iloc[-1] for frame in comparison_results.values()]
    plt.bar(labels, values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Final Hedge Error by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Final Hedge Error")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    error_path = output_path / "final_hedge_error.png"
    plt.savefig(error_path, dpi=150)
    plt.close()
    saved_files.append(error_path)

    return saved_files


def save_summary_plots(summary: pd.DataFrame, output_dir: str = "outputs/figures") -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    plotted_summary = summary.drop(index="continuous_ideal", errors="ignore")
    if plotted_summary.empty:
        plotted_summary = summary

    metrics = [
        ("hedge_error_abs", "Absolute Final Hedge Error"),
        ("hedge_error_rmse", "Hedge Error RMSE"),
        ("total_transaction_cost", "Total Transaction Cost"),
        ("cvar_95_daily", "Daily CVaR 95%"),
    ]
    available_metrics = [(column, title) for column, title in metrics if column in summary.columns]
    if not available_metrics:
        return saved_files

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_list = list(axes.flat)
    for axis, (column, title) in zip(axes_list, available_metrics):
        values = pd.to_numeric(plotted_summary[column], errors="coerce")
        if "is_auto_selected" in plotted_summary.columns:
            selected_flags = plotted_summary["is_auto_selected"].fillna(False)
        else:
            selected_flags = pd.Series(False, index=plotted_summary.index)
        colors = ["tab:green" if bool(value) else "tab:blue" for value in selected_flags]
        axis.bar(plotted_summary.index, values, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Strategy")
        axis.set_ylabel(column)
        axis.tick_params(axis="x", rotation=25)

    for axis in axes_list[len(available_metrics):]:
        axis.axis("off")

    fig.tight_layout()
    summary_path = output_path / "summary_metrics.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    saved_files.append(summary_path)
    return saved_files


def save_summary_table(summary: pd.DataFrame, output_path: str = "outputs/tables/summary.csv") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path)
    return path
