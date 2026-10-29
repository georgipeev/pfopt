# ETF Portfolio Optimizer

A Python-based portfolio optimizer for a low-cost, ETF-focused portfolio. It estimates expected returns from historical
data, blends those estimates with configurable priors and alpha overlays, estimates risk from historical returns, and
evaluates portfolios across a regularized mean-variance frontier.

> **Note:** This README documents the project based on the currently available `main.py` and parameter module. Some
> implementation details live in modules under `data/`, `estimates/`, and `optim/`; details that cannot be verified
> without those modules are called out below.

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Expected Returns](#expected-returns)
    - [Alpha Overlays](#alpha-overlays)
- [Risk Estimation](#risk-estimation)
- [Portfolio Optimization and Selection](#portfolio-optimization-and-selection)
- [ETF Universe](#etf-universe)
    - [Overlapping Exposures](#overlapping-exposures)
- [Current Parameter Snapshot](#current-parameter-snapshot)
- [Taxable-Account Rebalancing Considerations](#taxable-account-rebalancing-considerations)
- [Project Structure](#project-structure)
- [Setup and Running](#setup-and-running)
- [Limitations and Validation](#limitations-and-validation)

## Overview

The optimizer is designed to:

- Work with a configurable universe of ETFs.
- Combine sample return estimates, prior expected returns, and alpha signals.
- Estimate covariance using Ledoit–Wolf shrinkage, optionally blending in a second historical period.
- Sweep a range of regularization values to evaluate candidate portfolios.
- Select a portfolio whose Sharpe ratio is close to the best observed Sharpe ratio, rather than necessarily selecting
  the maximum-Sharpe point.
- Report portfolio weights and dollar allocations using a configurable AUM value.

This is a research tool, not investment, tax, or legal advice. Results depend on the quality of inputs and the
assumptions encoded in the model.

## Workflow

At a high level, the main workflow is:

1. Define the ETF universe and load the selected parameter module.
2. Fetch adjusted price history and the market-based risk-free rate.
3. Compute returns and annualized sample estimates.
4. Blend sample expected returns with priors, then apply alpha overlays.
5. Estimate and, when configured, blend covariance matrices.
6. Evaluate minimum-variance and regularized mean-variance portfolios across a lambda grid.
7. Calculate portfolio metrics and choose a portfolio using the configured Sharpe threshold.
8. Print portfolio weights, dollar allocations, and diagnostics.

Exact function signatures and the precise optimization objective are implemented in the imported project modules and
should be verified there.

## Expected Returns

The parameter module sets `RETURN_SHRINKAGE_FACTOR`. The expected-return blend used by the main program is:

```text
expected_return = (1 - s) * sample_return + s * prior_return + alpha
```

where:

- `s` is `RETURN_SHRINKAGE_FACTOR`.
- `sample_return` is the annualized historical return estimate.
- `prior_return` is the configured prior return.
- `alpha` is the configured alpha overlay.

For example, `RETURN_SHRINKAGE_FACTOR = 0.7` means 30% weight on the sample estimate and 70% weight on the prior
estimate, before adding alpha. This can reduce sensitivity to the chosen historical window, but it also means results
depend substantially on the prior assumptions.

### Alpha Overlays

Alpha values are configured by asset-class / geography / category keys. The parameter module currently divides raw alpha
values by `ALPHA_SHRINKAGE` before applying them. With `ALPHA_SHRINKAGE = 5`, a raw alpha of `0.025` becomes an
effective annual alpha of `0.005` (0.5 percentage points).

Alpha is a subjective forecast adjustment, not a return guaranteed by the model. Keep the sign, units, and
classification keys consistent, and inspect the output to confirm that each intended alpha is actually being applied to
the expected ETFs.

## Risk Estimation

The covariance estimate uses Ledoit–Wolf shrinkage on historical returns. The configuration can also blend this estimate
with a covariance matrix calculated from a separate historical period.

The current parameter file configures a 30% blend from the January 2016–December 2020 period with the remaining 70%
coming from the more recent period. This is intended to make the risk estimate less dependent on a single recent market
regime. It does not guarantee that the resulting covariance matrix will predict future risk.

The program prints eigenvalue diagnostics; these are useful for checking numerical properties of the covariance
estimate.

## Portfolio Optimization and Selection

The configured lambda sweep currently spans 0 through 500, with 50 steps. `GAMMA` is set to 10. The main program also
sets `OPT_SHARPE_FRACTION = 0.95` and refines the grid around the maximum-Sharpe region.

The selected portfolio is intended to be the smallest-lambda candidate whose Sharpe ratio is at least 95% of the maximum
observed Sharpe ratio. This “lambda pullback” favors a more regularized candidate when its Sharpe is close to the best
observed value.

The exact mathematical objective, constraints, and role of `GAMMA` should be confirmed in `optim/mean_variance.py`; they
are not fully specified by the main program and parameter file alone. The 95% threshold is a selection heuristic, not a
guarantee of out-of-sample performance.

## ETF Universe

The ETF symbols defined in the current `main.py` are:

`VTI`, `VXUS`, `BND`, `BNDX`, `VOO`, `VUG`, `VWO`, `VEA`, `IAU`, `VNQ`, `IEF`, `TLT`, `EMB`, `XLC`, `XLY`, `XLU`, `XLE`,
`XLB`, `XLRE`, `MTUM`, `QUAL`, `DBE`, `VGT`, `SOXX`, `XLF`, `IMTM`, and `IQLT`.

The universe includes broad-market funds, international and emerging-market equities, bonds, gold, REITs, sectors, style
factors, energy commodities, technology, semiconductors, and financials.

### Overlapping Exposures

Several ETFs overlap substantially. For example, VTI and VOO both hold broad US equities; VUG, VGT, and SOXX can add
overlapping growth and technology exposure; and VXUS overlaps with more specific international funds such as VEA, VWO,
IMTM, and IQLT. Consequently, an allocation split across multiple ETF tickers does not necessarily represent a similarly
diversified set of underlying holdings.

Review look-through exposures as well as ticker weights, particularly when the optimizer assigns large weights to sector
or factor funds.

## Current Parameter Snapshot

The parameter module named `parameters_202609.py` currently includes these settings:

| Parameter                   |           Configured value |
|-----------------------------|---------------------------:|
| `PRICE_START`               |               June 1, 2021 |
| `RETURN_SHRINKAGE_FACTOR`   |                        0.7 |
| Historical covariance blend |                        30% |
| Historical blend period     | January 2016–December 2020 |
| `GAMMA`                     |                         10 |
| Lambda range                |                      0–500 |
| Lambda steps                |                         50 |
| `OPT_SHARPE_FRACTION`       |                       0.95 |
| `ALPHA_SHRINKAGE`           |                          5 |

The AUM setting is configurable and is used for dollar-allocation output. Avoid committing personal portfolio values to
a public repository if you do not want them disclosed.

Parameter values are project-specific and may change; inspect the parameter file before running the optimizer.

## Taxable-Account Rebalancing Considerations

The optimizer's target weights are mathematical outputs, not automatic instructions to sell. In a taxable account, the
tax impact of moving from current weights to target weights can be material.

A practical workflow may include:

- Directing new contributions toward underweight positions before selling existing holdings.
- Considering tax-loss harvesting where appropriate.
- Reviewing tax lots and distinguishing short-term from long-term gains.
- Selling overweight positions only after comparing the portfolio benefit with the tax cost.
- Keeping portfolio constraints, liquidity needs, and investment objectives in view.

The code's target allocations do not replace a tax-lot-aware rebalancing plan.

## Project Structure

The main program imports functionality from these areas:

```text
.
├── main.py
├── parameters_202609.py
├── data/
│   ├── market_data.py
│   └── universe.py
├── estimates/
│   ├── returns.py
│   └── covariance.py
└── optim/
    └── mean_variance.py
```

This is an inferred outline of the modules referenced by the main program, not a guarantee that it lists every file in
the repository.

## Setup and Running

The exact dependencies should be taken from the project's dependency file or import statements. A typical local setup
is:

```bash
python -m venv .venv
```

Activate the virtual environment, install the project's dependencies, then run:

```bash
python main.py
```

Before relying on the output, verify that:

1. Market data loaded successfully and covers the intended dates.
2. The risk-free rate is current and expressed in the expected units.
3. Priors and alpha values use annualized decimal returns (for example, `0.01` means 1%).
4. The ETF classification rules match the intended alpha keys.
5. Covariance diagnostics are sensible.
6. Weights and dollar allocations are feasible and sum as expected.
7. Results are not being driven by a single asset, a narrow date window, or an unintended classification.

## Limitations and Validation

- Historical returns and covariance are estimates, not reliable forecasts of future outcomes.
- Expected returns are particularly uncertain; shrinkage and priors make assumptions more explicit but do not eliminate
  estimation error.
- Alpha overlays are subjective and should be tested for sensitivity.
- A covariance blend across market regimes can alter risk estimates but cannot ensure robustness to future regimes.
- Correlated and overlapping ETFs can create concentrated underlying exposures that are not obvious from ticker-level
  weights.
- The maximum observed Sharpe ratio can be noisy and sample-dependent.
- The exact objective, constraints, prior mapping, asset classification hierarchy, data source behavior, and dependency
  list require inspection of the relevant imported modules.

For a robust evaluation, compare results across several start dates, prior/alpha assumptions, covariance settings, and
simpler baseline portfolios. Treat small differences between candidate portfolios cautiously.