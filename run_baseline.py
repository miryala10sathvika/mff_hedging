from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

from src.experiments import (
    ExperimentConfig,
    get_selected_strategy,
    run_experiment,
    save_plots,
    save_summary_plots,
    save_summary_table,
)


def main() -> None:
    (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        ticker="SPY",
        start="2020-01-01",
        end="2024-12-31",
        maturity_days=21,
        fixed_volatility=0.20,
        rate=0.02,
        use_realized_vol=False,
        transaction_cost_bps=5.0,
        rehedge_frequencies=(1, 2, 5),
    )

    results, summary = run_experiment(config)
    selected_strategy = get_selected_strategy(summary)
    figure_paths = save_plots(results)
    figure_paths.extend(save_summary_plots(summary))
    table_path = save_summary_table(summary)

    print("Baseline run complete.")
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
