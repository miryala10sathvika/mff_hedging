from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

from src.experiments import (
    YahooOptionExperimentConfig,
    get_selected_strategies,
    run_yahoo_option_experiment,
    save_plots,
    save_summary_plots,
    save_summary_table,
)


def _contract_results(results: dict[str, object], contract_symbol: str) -> dict[str, object]:
    prefix = f"{contract_symbol}::"
    return {
        label.removeprefix(prefix): frame
        for label, frame in results.items()
        if label.startswith(prefix)
    }


def main() -> None:
    (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

    config = YahooOptionExperimentConfig(
        ticker="SPY",
        expiration=None,
        option_type="call",
        strike=None,
        min_days_to_expiration=30,
        max_strike_distance_pct=0.1,
        min_open_interest=1,
        min_implied_volatility=0.05,
        min_history_observations=30,
        max_expirations_to_scan=16,
        candidates_per_expiration=6,
        target_abs_delta=0.50,
        min_abs_delta=0.25,
        max_abs_delta=0.75,
        max_contracts=5,
        fixed_contracts_path="data/yahoo_option_contracts.csv",
        refresh_contract_selection=False,
        rate=0.04,  # more realistic dynamic rate average
        history_period="max",
        transaction_cost_bps=5.0,
        rehedge_frequencies=(1, 2, 5),
    )

    results, summary, metadata = run_yahoo_option_experiment(config)
    selected_strategies = get_selected_strategies(summary)
    figure_paths = []
    for contract_metadata in metadata:
        contract_symbol = str(contract_metadata["contract_symbol"])
        contract_output_dir = f"outputs/figures/yahoo_option/{contract_symbol}"
        per_contract_results = _contract_results(results, contract_symbol)
        per_contract_summary = summary.xs(contract_symbol, level="contract_symbol")
        figure_paths.extend(save_plots(per_contract_results, output_dir=contract_output_dir))
        figure_paths.extend(save_summary_plots(per_contract_summary, output_dir=contract_output_dir))

    table_path = save_summary_table(summary, output_path="outputs/tables/yahoo_option_summary.csv")

    print("Yahoo option run complete.")
    print("\nSelected contracts:")
    for contract_metadata in metadata:
        print(
            f"- {contract_metadata['contract_symbol']} "
            f"exp={contract_metadata['expiration']} "
            f"strike={contract_metadata['selected_strike']} "
            f"history={contract_metadata['history_observations']} "
            f"vol_source={contract_metadata['volatility_source']} "
            f"selection={contract_metadata['selection_method']}"
        )

    print("\nAuto-selected rehedge frequencies:")
    for index, selected_strategy in selected_strategies.iterrows():
        contract_symbol, strategy = index
        print(
            f"- {contract_symbol}: {strategy} "
            f"objective={float(selected_strategy['selector_objective']):.6f} "
            f"cvar_loss={float(selected_strategy['selector_cvar_loss']):.6f} "
            f"tc={float(selected_strategy['total_transaction_cost']):.6f}"
        )

    print("\nSummary:")
    print(summary.round(6).to_string())

    print("\nSaved figures:")
    for path in figure_paths:
        print(f"- {path}")

    print(f"\nSaved table:\n- {table_path}")


if __name__ == "__main__":
    main()
