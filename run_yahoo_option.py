from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

from src.experiments import (
    YahooOptionExperimentConfig,
    get_selected_strategy,
    run_yahoo_option_experiment,
    save_plots,
    save_summary_table,
)


def main() -> None:
    (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

    config = YahooOptionExperimentConfig(
        ticker="SPY",
        expiration="2026-12-18",
        option_type="call",
        strike=600.0,
        rate=0.04,  # more realistic dynamic rate average
        history_period="max",
        transaction_cost_bps=5.0,
        rehedge_frequencies=(1, 2, 3), # tighter discrete hedges
    )

    results, summary, metadata = run_yahoo_option_experiment(config)
    selected_strategy = get_selected_strategy(summary)
    figure_paths = save_plots(results, output_dir="outputs/figures/yahoo_option")
    table_path = save_summary_table(summary, output_path="outputs/tables/yahoo_option_summary.csv")

    print("Yahoo option run complete.")
    print("\nSelected contract:")
    for key, value in metadata.items():
        print(f"- {key}: {value}")

    print(
        "\nAuto-selected rehedge frequency:"
        f"\n- strategy: {selected_strategy.name}"
        f"\n- objective: {float(selected_strategy['selector_objective']):.6f}"
        f"\n- cvar_loss: {float(selected_strategy['selector_cvar_loss']):.6f}"
        f"\n- total_transaction_cost: {float(selected_strategy['total_transaction_cost']):.6f}"
    )

    print("\nSummary:")
    print(summary.round(6).to_string())

    print("\nSaved figures:")
    for path in figure_paths:
        print(f"- {path}")

    print(f"\nSaved table:\n- {table_path}")


if __name__ == "__main__":
    main()
