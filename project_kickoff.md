# Project Kickoff

## Recommended scope

Build a **discrete-time delta hedging backtest for a European call option** under Black-Scholes, then compare realized hedge error across:

- rebalancing frequencies: daily, every 2 days, weekly
- volatility assumptions: fixed implied vol vs realized-vol proxy
- transaction cost settings: zero cost vs proportional cost

If your team has **OptionMetrics / IvyDB access through the university**, use it for the empirical extension.
If not, do **not** block the project on data access. Use:

- historical underlying prices from `yfinance`
- synthetic option prices and Greeks recomputed each day with Black-Scholes

That is a valid course-project design and much safer for a one-week timeline.

## Practical recommendation on data

Use this priority order:

1. `OptionMetrics IvyDB` if your team already has access
2. `yfinance` underlying prices + synthetic BS option pricing
3. pure GBM simulation if data access fails

Do not make the whole project depend on getting IvyDB access this week unless access is already confirmed.

## Minimal deliverable

By the deadline, you should be able to show:

1. Black-Scholes pricing and delta/Gamma/Theta/Vega functions
2. discrete delta-hedging engine
3. hedge P&L decomposition
4. comparison of hedge error distribution across rebalance frequencies
5. effect of transaction costs
6. benchmark discussion against the Bertsimas-Kogan-Lo intuition that hedge error variance grows with discrete hedging interval

## Stretch goals

- implied-vol surface fitting
- empirical backtest on real historical option panel data
- CVaR optimization
- gamma hedge with another option

These are optional unless your proposal explicitly requires them.

## Suggested repo structure

Keep the implementation small and modular:

```text
mff_hedging/
  data/
  notebooks/
  src/
    black_scholes.py
    greeks.py
    data_loader.py
    hedging_engine.py
    pnl_attribution.py
    experiments.py
    utils.py
  outputs/
    figures/
    tables/
  project_kickoff.md
  team_week_plan.md
  resources.md
```

## Core pipeline

### 1. Data

For each date:

- underlying close `S_t`
- risk-free rate `r_t` or constant proxy
- implied vol assumption `sigma_t`
- strike `K`
- time to maturity `tau_t`

### 2. Pricing and Greeks

Implement for a European call:

- price
- delta
- gamma
- theta
- vega

### 3. Hedging engine

At each rebalance date:

- compute option value and delta
- hold `-delta` shares of underlying against one long option or the opposite sign depending on portfolio convention
- carry remaining cash in risk-free account
- rebalance at chosen interval
- subtract transaction costs when changing hedge position

### 4. Outputs

Generate:

- final hedge error histogram
- mean / std / RMSE of hedge error
- hedge error by rebalance frequency
- cumulative P&L path
- P&L attribution table

## Recommended experiment design

### Baseline

- underlying: `SPY` or `AAPL`
- one maturity: 1 month or 3 months
- one moneyness level: ATM
- Black-Scholes with constant vol
- no transaction costs

### Main comparisons

- daily vs 2-day vs weekly rehedging
- zero vs small proportional cost
- fixed vol vs rolling realized-vol estimate

### Optional empirical extension

If you have IvyDB:

- track one European-style index option or a stable option panel
- compare vendor Greeks with your BS Greeks
- compare realized hedging error to your synthetic benchmark

## What to ignore this week

Avoid these unless already working:

- reinforcement learning
- Heston calibration
- full implied-vol surface modeling across many maturities and strikes
- intraday hedging
- overly broad universe selection

## Immediate next steps

1. Confirm today whether IvyDB access is real and usable.
2. Lock the project to **European call, single underlying, discrete delta hedging**.
3. Convert one reference notebook into reusable Python modules under `src/`.
4. Run one baseline experiment end to end before adding extra features.
5. Freeze figures and writing by Day 6.

