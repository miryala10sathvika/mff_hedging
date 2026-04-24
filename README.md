# Dynamic Hedging and Hedging Error Attribution

This project implements a dynamic delta-hedging backtest for European options. It supports Black-Scholes pricing and Greeks, discrete rebalancing, transaction costs, baseline hedge policies, Greek P&L attribution, and Yahoo Finance option-data experiments.

## What This Code Does

The project has two main workflows:

1. **Baseline synthetic option experiment**
   - Downloads historical underlying prices from Yahoo Finance.
   - Builds a synthetic European option using Black-Scholes.
   - Runs no hedge, static hedge, periodic delta hedge, and continuous ideal benchmark strategies.

2. **Live Yahoo option experiment**
   - Reads the current Yahoo option chain.
   - Selects an available option contract.
   - Downloads historical option prices for that contract.
   - Merges option history with underlying history.
   - Runs the same hedge strategy comparison using observed option prices.

## Repository Structure

```text
mff_hedging/
  src/
    black_scholes.py      # Black-Scholes prices and normal PDF/CDF helpers
    greeks.py             # Delta, gamma, theta, vega, rho
    data_loader.py        # Yahoo price/option data loading and option-frame builders
    hedging_engine.py     # Self-financing delta hedge engine
    pnl_attribution.py    # Greek P&L attribution and summary metrics
    experiments.py        # End-to-end experiment orchestration and plot/table saving
    utils.py              # Utility module placeholder
  tests/
    test_black_scholes.py
    test_hedging_engine.py
  notebooks/
    hedging_analysis.ipynb
  outputs/
    figures/
    tables/
  run_baseline.py         # Baseline synthetic option run
  run_yahoo_option.py     # Live/current Yahoo option run
  change_report.md        # Report of the recent code audit and fixes
  requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `pip` is not available but `uv` is installed, you can use:

```bash
uv venv env
source env/bin/activate
uv pip install -r requirements.txt
```

The current repo already uses `env/bin/python` in the examples below.

## Run Tests

Use the project virtual environment:

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python -m unittest discover -s tests -v
```

Expected result:

```text
Ran 9 tests
OK
```

Compile-check the source:

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python -m compileall src tests run_baseline.py run_yahoo_option.py
```

## Run the Baseline Experiment

The baseline experiment uses historical underlying prices and synthetic Black-Scholes option values.

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python run_baseline.py
```

Default baseline settings are in `run_baseline.py`:

```python
ExperimentConfig(
    ticker="SPY",
    start="2020-01-01",
    end="2024-12-31",
    maturity_days=21,
    fixed_volatility=0.20,
    rate=0.02,
    transaction_cost_bps=5.0,
    rehedge_frequencies=(1, 2, 5),
)
```

Outputs:

```text
outputs/figures/portfolio_value_paths.png
outputs/figures/final_hedge_error.png
outputs/tables/summary.csv
```

## Run the Yahoo Option Experiment

The Yahoo option experiment uses the current Yahoo option chain and historical prices for the selected option contract.

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python run_yahoo_option.py
```

This requires internet access. If Yahoo Finance is unreachable, you may see DNS or network errors from `yfinance`.

Default Yahoo settings are in `run_yahoo_option.py`:

```python
YahooOptionExperimentConfig(
    ticker="SPY",
    expiration=None,
    option_type="call",
    strike=None,
    min_days_to_expiration=30,
    max_strike_distance_pct=0.1,
    min_open_interest=1,
    min_implied_volatility=0.05,
    fallback_volatility=0.20,
    realized_vol_window=21,
    rate=0.04,
    history_period="max",
    transaction_cost_bps=5.0,
    rehedge_frequencies=(1, 2, 5),
)
```

Important behavior:

- `expiration=None` means the loader chooses a current Yahoo expiration.
- `min_days_to_expiration=30` avoids same-day or very near-expiry contracts.
- `strike=None` selects a strike near the current underlying spot.
- If Yahoo returns an unusable implied volatility, the loader falls back to rolling realized volatility.

Outputs:

```text
outputs/figures/yahoo_option/portfolio_value_paths.png
outputs/figures/yahoo_option/final_hedge_error.png
outputs/tables/yahoo_option_summary.csv
```

## Strategy Definitions

The experiment compares these strategies:

| Strategy | Meaning |
| --- | --- |
| `no_hedge` | Hold only the initial option premium in cash; no stock hedge |
| `static_hedge` | Set the initial delta hedge once and never rebalance |
| `every_1_day` | Rebalance the stock hedge every observation |
| `every_2_days` | Rebalance every 2 observations |
| `every_5_days` | Rebalance every 5 observations |
| `continuous_ideal` | Theoretical zero-error Black-Scholes replication reference |

## Hedging Convention

The hedging engine tracks a self-financing replication portfolio:

```text
portfolio_value = stock_value + cash_account
hedge_error = portfolio_value - option_value
```

At expiry:

```text
hedge_error = portfolio_value - option_payoff
```

By default, one option is replicated using positive delta stock exposure:

```text
stock_shares = option_position * delta
```

Transaction costs are charged when the hedge stock position changes:

```text
transaction_cost = abs(shares_traded) * spot * transaction_cost_bps / 10000
```

## Summary Metrics

The summary table includes:

- final hedge error
- absolute hedge error
- hedge error RMSE and MAE
- max absolute hedge error
- VaR 95% and 99%
- CVaR 95% and 99%
- total transaction cost
- mean and standard deviation of daily P&L
- total P&L
- final portfolio value
- number of rehedges
- mean Greek attribution residual

## Using the Modules Directly

Example: compute Black-Scholes prices and Greeks.

```python
from src.black_scholes import call_price, put_price
from src.greeks import call_greeks

spot = 100.0
strike = 100.0
tau = 21 / 252
rate = 0.02
volatility = 0.20

call = call_price(spot, strike, tau, rate, volatility)
put = put_price(spot, strike, tau, rate, volatility)
greeks = call_greeks(spot, strike, tau, rate, volatility)
```

Example: run an experiment from Python.

```python
from src.experiments import ExperimentConfig, run_experiment

config = ExperimentConfig(
    ticker="SPY",
    start="2020-01-01",
    end="2024-12-31",
    transaction_cost_bps=5.0,
    rehedge_frequencies=(1, 2, 5),
)

paths, summary = run_experiment(config)
print(summary)
```

## Common Changes

Change the underlying:

```python
ticker="AAPL"
```

Use realized volatility in the baseline run:

```python
use_realized_vol=True
realized_vol_window=21
```

Change transaction costs:

```python
transaction_cost_bps=1.0
```

Run put-option hedging:

```python
option_type="put"
```

Change rebalancing intervals:

```python
rehedge_frequencies=(1, 3, 5, 10)
```

## Notes and Limitations

- Yahoo Finance option data can be noisy, incomplete, delayed, or unavailable.
- Historical option prices from Yahoo are used when available, but historical implied volatility is not provided as a clean time series.
- For Yahoo option experiments, current chain IV is used if plausible; otherwise, rolling realized volatility is used as a fallback.
- The `continuous_ideal` strategy is a theoretical reference, not a tradable strategy.
- The code includes VaR and CVaR metrics, but it does not yet implement a formal optimizer that automatically chooses the best rebalancing frequency.
- The explicit Bertsimas-Kogan-Lo discrete hedging variance approximation mentioned in the proposal is not yet implemented.

## Useful Project Documents

- `Project Proposal.pdf`: original proposal
- `change_report.md`: detailed report of the formula audit and code changes
- `project_kickoff.md`: project scope and practical implementation plan
- `resources.md`: reference resources
