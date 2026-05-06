# Dynamic Hedging Code Review and Change Report

Date: 2026-04-24

## Summary

This report documents the review and correction of the dynamic hedging implementation, Yahoo option-data pipeline, and proposal-requirement coverage for the project.

The main issue found was in the hedging engine accounting. The original implementation mixed the option value directly into the tracked portfolio and did not book the initial option premium into cash. As a result, the reported `hedge_error` was closer to a portfolio value than a true replication error. The corrected implementation now tracks a self-financing replication portfolio:

```text
replication_portfolio_value = stock_position_value + cash_account
mark_to_market_hedge_error = replication_portfolio_value - option_value
terminal_hedge_error = replication_portfolio_value - option_payoff
```

This makes no hedge, static hedge, periodic delta hedging, and the continuous ideal benchmark comparable on the same basis.

## Files Changed

- `src/hedging_engine.py`   
- `src/data_loader.py`
- `src/experiments.py`
- `src/pnl_attribution.py`
- `run_yahoo_option.py`
- `tests/test_hedging_engine.py`
- `test_chain.py`
- `test_chain2.py`
- `test_find_good_option.py`
- `test_merge.py`
- `outputs/figures/yahoo_option/final_hedge_error.pdf`
- `outputs/figures/yahoo_option/portfolio_value_paths.pdf`
- `outputs/tables/yahoo_option_summary.csv`

## Formula and Accounting Corrections

### Black-Scholes pricing and Greeks

The Black-Scholes formulas in `src/black_scholes.py` and `src/greeks.py` were reviewed. The core pricing, delta, gamma, vega, theta, and rho formulas are consistent with standard Black-Scholes with continuous dividend yield:

```text
d1 = [ln(S/K) + (r - q + 0.5 sigma^2) tau] / [sigma sqrt(tau)]
d2 = d1 - sigma sqrt(tau)
call = S e^(-q tau) N(d1) - K e^(-r tau) N(d2)
put = K e^(-r tau) N(-d2) - S e^(-q tau) N(-d1)
```

No correction was required in the Black-Scholes pricing module.

### Hedging portfolio convention

The previous default hedge sign was `stock_position_sign = -1`, meaning a long call was paired with a short delta stock position. That convention can be valid for hedging the P&L of a long option position, but the rest of the engine was trying to measure replication error. For replication of one long option, the stock holding should be positive delta:

```text
stock_shares = option_position * delta
```

The default was changed to `stock_position_sign = 1`.

### Initial cash account

The original engine started with:

```text
cash_account = 0
```

This omitted the financing from the option premium. The corrected engine starts with:

```text
cash_account = option_position * initial_option_price
```

Then, when the stock hedge is purchased, cash is reduced by the trade amount and transaction costs. This makes the initial replication portfolio value equal to the option value, before transaction costs.

### Hedge error definition

The original implementation used:

```text
portfolio_value = option_value + stock_value + cash_account
hedge_error = portfolio_value
```

This was not a true hedge error. The corrected implementation uses:

```text
portfolio_value = stock_value + cash_account
hedge_error = portfolio_value - option_value
```

At expiry, if `tau == 0`, the error is measured against the option payoff:

```text
hedge_error = portfolio_value - terminal_option_payoff
```

### Continuous ideal benchmark

The previous `continuous_ideal` branch forced the portfolio value to the option price but then still produced an inconsistent terminal hedge error. The corrected version treats the continuous ideal as an exact theoretical replication reference with zero mark-to-market hedge error.

### Dividend and cash accrual

The dividend and interest accrual were adjusted to use the previous interval consistently:

```text
dt = previous_tau - current_tau
cash_account *= exp(previous_rate * dt)
dividend_cash = previous_stock_shares * previous_spot * previous_dividend_yield * dt
```

This avoids mixing current-day market data into the previous holding period.

### Put option support

The hedging engine and attribution code previously assumed calls. The implementation now supports:

- call pricing and Greeks
- put pricing and Greeks
- call payoff
- put payoff
- call and put Greek attribution

The option type is passed through the Yahoo loader using the `option_type` column.

## Yahoo Data Loader Corrections

### Contract selection

`run_yahoo_option.py` previously hard-coded:

```text
expiration = "2026-05-15"
strike = 700.0
```

This was brittle because the contract could be unavailable, stale, far from current market conditions, or not ideal for a dynamic hedging run.

The updated runner now selects from the current Yahoo chain:

```text
expiration = None
strike = None
min_days_to_expiration = 30
max_strike_distance_pct = 0.1
min_open_interest = 1
```

This avoids same-day options and picks a currently available contract with enough remaining life.

### Timezone-safe date handling

Yahoo history can return timezone-aware indexes or multi-index columns. The loader now normalizes dates using a helper that converts indexes to naive daily dates. This prevents failed joins between underlying history and option history.

### Underlying spot fallback

The loader now resolves the underlying spot using multiple possible Yahoo fields:

- `regularMarketPrice`
- `currentPrice`
- `lastPrice`
- `previousClose`
- `regularMarketPreviousClose`
- `fast_info`
- recent downloaded close history as a final fallback

### Volatility sanity check

Yahoo sometimes reports unusable implied volatility values, such as `0.00001`. The loader now applies:

```text
min_implied_volatility = 0.05
fallback_volatility = 0.20
realized_vol_window = 21
```

If the selected contract has a plausible chain IV, that IV is used. Otherwise, the loader falls back to rolling realized volatility from the underlying price history, with a floor.

The metadata now records the selected volatility source:

```text
volatility_source = "current_chain_iv" or "realized_vol_fallback"
```

## Test and Script Corrections

Root-level Yahoo debug scripts named `test_*.py` were being imported by `unittest discover`, causing live Yahoo network calls during test discovery. These scripts were wrapped in:

```python
if __name__ == "__main__":
    main()
```

This keeps them usable as manual debug scripts without breaking automated tests.

Additional tests were added to `tests/test_hedging_engine.py` for:

- positive delta replication by default
- zero hedge error for the continuous ideal benchmark
- put option delta and payoff handling

## Verification Results

The test suite was run with the project virtual environment:

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 9 tests
OK
```

The code was also compiled successfully:

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python -m compileall src tests run_yahoo_option.py
```

The live Yahoo option experiment was run successfully:

```bash
MPLCONFIGDIR=.cache/matplotlib env/bin/python run_yahoo_option.py
```

Final verified live contract:

```text
ticker: SPY
option_type: call
expiration: 2026-05-29
min_days_to_expiration: 30
contract_symbol: SPY260529C00760000
selected_strike: 760.0
current_underlying_spot: 708.45
current_chain_iv: 0.062509375
volatility_source: current_chain_iv
```

Outputs were regenerated:

- `outputs/figures/yahoo_option/portfolio_value_paths.pdf`
- `outputs/figures/yahoo_option/final_hedge_error.pdf`
- `outputs/tables/yahoo_option_summary.csv`

## Proposal Requirement Coverage

| Proposal requirement | Current status |
| --- | --- |
| Black-Scholes European option pricing | Satisfied |
| Greeks: Delta, Gamma, Vega, Theta, Rho | Satisfied |
| Dynamic delta hedging | Satisfied |
| Discrete rebalancing intervals | Satisfied |
| Transaction costs in basis points | Satisfied |
| No-hedge baseline | Satisfied |
| Static delta hedge baseline | Satisfied |
| Periodic Black-Scholes delta hedge baseline | Satisfied |
| Continuous-time ideal theoretical reference | Satisfied as an idealized zero-error benchmark |
| Historical underlying data from Yahoo | Satisfied |
| Live/current Yahoo option-chain selection | Satisfied |
| Real option price history from Yahoo option contracts | Satisfied |
| Greek-based P&L attribution | Mostly satisfied |
| VaR and CVaR summary metrics | Satisfied |
| Optimization of rebalancing frequency | Satisfied with an automatic periodic-strategy selector based on downside CVaR plus transaction cost |
| Bertsimas-Kogan-Lo discrete hedging variance approximation | Satisfied |

## Remaining Work

The project now satisfies the main engineering and backtesting requirements in the proposal. The remaining academic/reporting gaps are:

1. Expand the report notebook or final writeup to explain why live Yahoo option data can be noisy, especially around implied volatility and thinly traded contracts.
2. Optionally add more robust empirical experiments across multiple expirations and moneyness levels.

## Conclusion

The formulas in the Black-Scholes and Greeks modules were correct, but the hedging engine accounting needed correction. After the changes, hedge error is now measured as a true replication error, the continuous ideal benchmark behaves consistently, the Yahoo data path works with live options data, and the implementation is much closer to the stated project proposal.
