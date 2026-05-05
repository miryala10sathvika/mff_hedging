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
    VolMismatchConfig,
    run_vol_mismatch_experiment,
    save_vol_mismatch_plots,
)


def main() -> None:
    (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

    # ===============================
    # Baseline Experiment
    # ===============================
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

    # ===============================
    # Volatility Mismatch Experiment
    # ===============================
    vol_cfg = VolMismatchConfig(
        ticker="SPY",
        start="2020-01-01",
        end="2024-12-31",
        maturity_days=21,
        fixed_volatility=0.20,
        realized_vol_window=21,
        rate=0.02,
        transaction_cost_bps=5.0,
        rehedge_every=1,
    )

    fixed_result, real_result, vol_comparison = run_vol_mismatch_experiment(vol_cfg)

    save_vol_mismatch_plots(fixed_result, real_result)
    vol_table_path = save_summary_table(
        vol_comparison,
        "outputs/tables/vol_mismatch_summary.csv",
    )

    print("\nVolatility mismatch comparison:")
    print(vol_comparison.round(6).to_string())

    print("\nSaved volatility mismatch table:")
    print(f"- {vol_table_path}")


if __name__ == "__main__":
    main()