# Team Week Plan

## Team setup

Assume 4 people total:

- `Person A`: modeling / pricing lead
- `Person B`: data + backtest lead
- `Person C`: experiments + figures lead
- `Person D`: writeup + integration lead

If you want, map these to your actual teammate names.

## Work split

### Person A: pricing + Greeks

Own:

- `src/black_scholes.py`
- `src/greeks.py`
- unit checks against known BS values

Deliverables:

- correct European call price formula
- delta, gamma, theta, vega
- small validation table

Notes:

- use `numpy` and `scipy.stats.norm`
- do not overengineer

### Person B: data + hedge engine

Own:

- `src/data_loader.py`
- `src/hedging_engine.py`

Deliverables:

- pull underlying price history from `yfinance`
- construct contract inputs over time
- run discrete hedge with cash account
- add transaction cost toggle

Notes:

- start with synthetic option values if historical option panel data is unavailable
- get one complete run working before generalizing

### Person C: experiments + visuals

Own:

- `src/experiments.py`
- `outputs/figures/`
- `outputs/tables/`

Deliverables:

- daily vs 2-day vs weekly hedge comparison
- hedge error histograms / boxplots
- summary stats table
- transaction cost sensitivity chart

Notes:

- standardize figure titles, labels, and filenames early

### Person D: theory + paper + final integration

Own:

- writeup
- methods section
- literature summary
- final notebook or script that reproduces headline results

Deliverables:

- short Bertsimas-Kogan-Lo summary
- short Carr-Wu or Greek attribution summary
- project report slides / paper draft
- final result narrative

Notes:

- this person should also keep the repo clean and coordinate merges

## Day-by-day plan

### Day 1

- confirm dataset access
- lock scope
- assign files and ownership
- Person A builds BS pricing + Greeks
- Person B builds initial data pull
- Person D starts theory summary and report outline

Checkpoint:

- one-page scope statement
- one shared repo structure

### Day 2

- Person B builds first working hedging loop
- Person A validates Greeks against textbook formulas
- Person C sets up experiment templates and plotting style
- Person D writes methods and literature sections

Checkpoint:

- one end-to-end baseline run exists, even if ugly

### Day 3

- compare rebalance frequencies
- add transaction costs
- clean outputs into reusable figures and tables
- D starts assembling results section

Checkpoint:

- baseline charts generated from code, not manually

### Day 4

- empirical extension if IvyDB exists
- otherwise strengthen synthetic / yfinance backtest
- add P&L attribution
- review whether results support proposal claims

Checkpoint:

- final experiment list frozen

### Day 5

- rerun all figures cleanly
- finalize summary statistics
- draft interpretation and discussion
- identify missing robustness checks

Checkpoint:

- near-final report draft

### Day 6

- full integration day
- fix bugs
- improve plots
- polish writeup and slides
- rehearse presentation

Checkpoint:

- code and report both complete

### Day 7

- buffer day only
- rerun headline results
- final formatting
- final submission package

## Team rules for this week

- each person owns files; avoid editing the same file at the same time
- merge daily, not at the very end
- keep one `main` branch stable
- save every final figure from code
- no result should exist only in a notebook cell output

## Risk control

### Biggest risks

- waiting too long for option data access
- spending too much time on implied-vol surfaces
- writing code only in notebooks
- adding too many extensions before the baseline works

### Mitigation

- baseline must work with `yfinance` + synthetic BS pricing
- keep one reproducible script for main results
- treat IvyDB as an enhancement, not the foundation, unless already available

## What your final report should claim

Aim for a claim like:

> Under Black-Scholes-style assumptions, discrete delta hedging reduces exposure substantially, but hedge error remains and increases with less frequent rebalancing and with transaction costs.

That is defendable in one week.

