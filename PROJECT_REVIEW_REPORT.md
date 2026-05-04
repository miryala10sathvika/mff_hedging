# Project Review Report

## Scope

This review covered the project code, Black-Scholes formulas, hedging accounting, PnL attribution, tests, generated summary tables, generated plots, and the analysis notebook.

Reviewed files include:

- `src/black_scholes.py`
- `src/greeks.py`
- `src/hedging_engine.py`
- `src/data_loader.py`
- `src/pnl_attribution.py`
- `src/experiments.py`
- `run_baseline.py`
- `run_yahoo_option.py`
- `tests/test_black_scholes.py`
- `tests/test_hedging_engine.py`
- `tests/test_experiments.py`
- `notebooks/hedging_analysis.ipynb`
- `outputs/tables/summary.csv`
- `outputs/tables/yahoo_option_summary.csv`
- `outputs/figures/portfolio_value_paths.png`
- `outputs/figures/final_hedge_error.png`
- `outputs/figures/yahoo_option/portfolio_value_paths.png`
- `outputs/figures/yahoo_option/final_hedge_error.png`

## Verification Run

Commands run:

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall src tests run_baseline.py run_yahoo_option.py
```

Results:

- Unit tests: 14 tests passed.
- Compile check: passed for source, tests, and runner scripts.
- Plot files: expanded baseline and Yahoo diagnostic PNG files were generated.

## Project Summary

The project implements a dynamic delta-hedging backtest for European options. It has two main workflows:

1. Baseline synthetic experiment:
   - Downloads SPY underlying prices.
   - Builds a synthetic Black-Scholes option.
   - Compares no hedge, static hedge, periodic delta hedges, and a continuous ideal benchmark.

2. Yahoo option experiment:
   - Reads a Yahoo option chain.
   - Selects a listed option contract.
   - Downloads historical option and underlying prices.
   - Runs the same hedge strategy comparison using observed option prices.

The central hedging convention is:

```text
portfolio_value = stock_value + cash_account
hedge_error = portfolio_value - option_value
terminal_hedge_error = portfolio_value - option_payoff
```

The default convention is a replication portfolio for one long option, so the stock hedge uses positive delta exposure:

```text
stock_shares = option_position * delta
```

## Formula Review

### Black-Scholes

The implemented Black-Scholes formulas are standard for European options with continuous dividend yield:

```text
d1 = [ln(S / K) + (r - q + 0.5 sigma^2) tau] / [sigma sqrt(tau)]
d2 = d1 - sigma sqrt(tau)
call = S exp(-q tau) N(d1) - K exp(-r tau) N(d2)
put = K exp(-r tau) N(-d2) - S exp(-q tau) N(-d1)
```

The price implementation handles expiry and zero-volatility cases separately. Put-call parity is tested and passes.

### Greeks

The implemented Greeks are consistent with the Black-Scholes model:

- Call delta: `exp(-q tau) N(d1)`
- Put delta: `exp(-q tau) (N(d1) - 1)`
- Gamma: `exp(-q tau) n(d1) / (S sigma sqrt(tau))`
- Vega: `S exp(-q tau) n(d1) sqrt(tau)`
- Call and put theta include carry and discounting terms.
- Call and put rho use discounted strike sensitivity.

The formulas are annualized in the usual Black-Scholes convention. Vega and rho are per unit change in volatility and rate, not per 1 percent point.

## Hedging Engine Review

The hedging engine is coherent for a self-financing replication portfolio:

- Initial cash is set to the initial option value.
- Stock purchases reduce cash.
- Stock sales increase cash.
- Transaction costs are charged on absolute shares traded.
- Cash accrues at the previous interval rate.
- Dividend yield is credited to the stock holding with a continuous-yield approximation.
- Static hedge rebalances only at the first observation.
- No hedge keeps the initial premium in cash and does not trade stock.
- Continuous ideal forces portfolio value to option value as a theoretical zero-error benchmark.

The terminal payoff override only applies when the last observation has `tau <= 1e-12`. The generated baseline output does not reach `tau = 0`, so its final hedge error is still mark-to-market error rather than expiry payoff error.

## Data Pipeline Review

The Yahoo data loader includes useful protections:

- Handles yfinance multi-index columns.
- Normalizes timezone-aware indexes to naive daily dates.
- Filters contracts by open interest and implied volatility.
- Falls back to realized volatility if chain IV is unusable.
- Joins option and underlying histories on overlapping dates.

One important behavior: the synthetic baseline builder uses one option life, not a rolling backtest. With `maturity_days=21`, it keeps only the first 21 observations from the downloaded price history. That is why the baseline plot covers January 2020 even though the configured date range is 2020-01-01 to 2024-12-31.

## Generated Output Summary

### Baseline Synthetic Run

Output table: `outputs/tables/summary.csv`

Key results:

| Strategy | Final hedge error | RMSE | Transaction cost | Rehedges | Auto-selected |
| --- | ---: | ---: | ---: | ---: | --- |
| continuous_ideal | 0.000000 | 0.000000 | 0.000000 | 21 | False |
| every_1_day | 2.384936 | 1.669493 | 0.294954 | 21 | False |
| every_2_days | 2.678498 | 1.712462 | 0.242122 | 11 | False |
| every_5_days | 3.602051 | 2.098950 | 0.170054 | 5 | True |
| no_hedge | 6.616837 | 2.305122 | 0.000000 | 0 | False |
| static_hedge | 4.791579 | 2.342743 | 0.077639 | 1 | False |

Interpretation:

- Daily rehedging has the lowest final error and RMSE among the practical hedges.
- The selector chose `every_5_days` because its objective combines CVaR loss and transaction cost, not final hedge error alone.
- `continuous_ideal` is a benchmark, not a tradable strategy.

Generated plots:

- `outputs/figures/portfolio_value_paths.png`
- `outputs/figures/hedge_error_paths.png`
- `outputs/figures/stock_shares_paths.png`
- `outputs/figures/option_delta_path.png`
- `outputs/figures/cumulative_transaction_cost.png`
- `outputs/figures/daily_pnl_paths.png`
- `outputs/figures/greek_pnl_attribution_cumulative.png`
- `outputs/figures/final_hedge_error.png`
- `outputs/figures/summary_metrics.png`

The baseline portfolio plot shows the hedged strategies tracking one another during the short synthetic option window. The no-hedge strategy stays nearly flat because it is only the premium cash account. The final hedge-error plot correctly shows increasing error from frequent hedging to no hedge, with continuous ideal at zero.

### Yahoo Option Run

Output table: `outputs/tables/yahoo_option_summary.csv`

Current generated table uses:

- Contract: `SPY260618C00725000`
- Expiration: `2026-06-18`
- Selected strike: `725.0`
- History observations: `309`
- Estimated absolute delta at selection: `0.478821`
- Volatility source: `realized_vol_fallback`

Key results:

| Strategy | Final hedge error | RMSE | Transaction cost | Rehedges | Auto-selected |
| --- | ---: | ---: | ---: | ---: | --- |
| continuous_ideal | 0.000000 | 0.000000 | 0.000000 | 309 | False |
| every_1_day | 38.939223 | 29.511160 | 2.119477 | 309 | False |
| every_2_days | 38.688234 | 29.311731 | 1.616635 | 155 | False |
| every_5_days | 29.967164 | 22.085719 | 1.027174 | 62 | True |
| no_hedge | -8.582266 | 6.296970 | 0.000000 | 0 | False |
| static_hedge | 14.029135 | 9.225977 | 0.042862 | 1 | False |

Interpretation:

- The latest Yahoo run is materially stronger than the earlier sample because it uses a near-ATM option with meaningful delta and 309 observations.
- The selected chain IV was unusably low, so the loader used realized-volatility fallback for Greeks and hedging.
- The selector chose `every_5_days` because it had the lowest CVaR-plus-transaction-cost objective.
- The large residuals and hedge errors show model mismatch between Black-Scholes Greeks and observed market option prices; this is expected to some degree when using one realized-volatility input against live option prices.

Generated plots:

- `outputs/figures/yahoo_option/portfolio_value_paths.png`
- `outputs/figures/yahoo_option/hedge_error_paths.png`
- `outputs/figures/yahoo_option/stock_shares_paths.png`
- `outputs/figures/yahoo_option/option_delta_path.png`
- `outputs/figures/yahoo_option/cumulative_transaction_cost.png`
- `outputs/figures/yahoo_option/daily_pnl_paths.png`
- `outputs/figures/yahoo_option/greek_pnl_attribution_cumulative.png`
- `outputs/figures/yahoo_option/final_hedge_error.png`
- `outputs/figures/yahoo_option/summary_metrics.png`

The Yahoo portfolio plot shows the selected option sample and the replication portfolios over the available overlapping dates. The hedge-error, hedge-shares, transaction-cost, daily-PnL, and Greek-attribution plots provide the intermediate diagnostics needed to explain why strategies differ.

## Test Coverage

Current tests cover:

- Black-Scholes price positivity.
- Put-call parity.
- Delta bounds.
- Gamma and vega positivity.
- Greek bundle keys.
- Datetime-index handling in the hedging engine.
- Positive delta replication convention.
- Continuous ideal zero hedge error.
- Put option delta and payoff handling.
- Open-interest fallback during option-contract selection.
- Strategy selector ranking and objective calculations.

Important gaps:

- Only one narrow `data_loader.py` contract-selection fallback test.
- No mocked yfinance end-to-end tests.
- No tests for `save_plots`.
- No tests for end-to-end runner scripts.
- Only theta currently has a finite-difference Greek test.
- No tests for transaction-cost accounting details.
- No tests for dividend-yield cashflow behavior.
- No tests for `option_position != 1`.

## Issues And Risks Found

1. The baseline experiment date range looks broad, but `build_option_frame` keeps only the first `maturity_days` observations. This is valid for a single option-life simulation, but it is not a multi-year rolling hedging backtest.

2. `HedgeConfig.trading_days_per_year` is defined but unused. Time scaling is controlled by the input `tau` values instead.

3. PnL attribution does not multiply Greek approximation terms by `option_position`. It is correct only for the default one-option case.

4. `summarize_results` computes VaR/CVaR from `total_pnl`, including the first zero PnL row. This is acceptable for a simple report, but it slightly affects small samples.

5. The root debug scripts named `test_*.py` are not part of the `tests/` suite. They are manual scripts and may confuse readers because they look like tests.

6. The Yahoo option experiment depends on live option-chain availability. Auto-selection is more robust than fixed contracts, but selected contracts and plots will change from run to run.

Fixed in the latest update:

- `run_yahoo_option.py` now uses `expiration=None` and `strike=None`.
- `select_option_contract` now falls back gracefully when Yahoo reports no usable open interest.
- The Yahoo auto-selection path now scans candidate expirations/contracts, prefers meaningful delta, requires option-history observations, and falls back from bad chain IV to realized volatility.
- The final hedge-error plot title now says "Final Hedge Error by Strategy".
- Additional diagnostic plots are generated for portfolio value, hedge error, stock shares, option delta, transaction costs, daily PnL, Greek attribution, and summary metrics.
- Added tests for theta finite-difference behavior and open-interest fallback.

## Recommended Next Steps

1. Decide whether the baseline should remain a single 21-day option-life example or become a rolling backtest across the whole 2020-2024 period.

2. Add mocked yfinance tests for option-chain selection, fallback strike selection, timezone normalization, and option/underlying history joins.

3. Add finite-difference tests for delta, gamma, vega, and rho.

4. Fix Greek attribution scaling for non-unit `option_position`.

5. Move manual root-level `test_*.py` scripts into a `scripts/` or `debug/` directory to avoid confusion with automated tests.
