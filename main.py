"""
Optimal portfolio construction.

1. Define the trading UNIVERSE
2. Prepare market data
2.1. Get the risk-free rate
2.2. Get recent prices for UNIVERSE from yfinance starting at PRICE_START (typically 4-5 years) and compute mean returns
3. Modeling
3.1. Use the sample returns from 2.2. to compute the covariance matrix
3.2. If using covariance blending (COV_BLEND set and weight>0), get prices for the blending period, compute their
covariance matrix, and blend it with he covariance matrix from 3.1. using the COV_BLEND["weight"] value
3.3. Blend the sample returns from 2.2. with the return priors using the RETURN_SHRINKAGE_FACTOR (Bayesian approach) and
add the ALPHA* returns
4. Portfolio optimization
4.1. Compute the minimum variance anchor portfolio
4.2. Compute the efficient frontier using the regularization parameter GAMMA for different values of lambda (risk
aversion parameter). The frontier will be calculated for values of lambda within the bounds defined in LAMBDA, with more
detail around the maximum Sharpe portfolio.
5. Output
5.1. Select the minimum lambda portfolio that has a Sharpe of at least 95% (OPT_SHARPE_FRACTION) of the maximum Sharpe
portfolio and print it

*ALPHA return values should be chosen per the following table:
Strong positive	    +0.75% to +1.0%
Moderate positive	+0.5%
Mild positive	    +0.25%
Neutral	            0%
Mild negative	    −0.25%
Moderate negative	−0.5%
Strong negative	    −0.75% to −1.0%

Rebalancing.

Construct a new optimal portfolio. Then trade towards it:
1. New cash -> underweights: Use essentially all new investment cash to move toward target.
2. Tax-loss positions -> consider selling: If an ETF is at a loss, this can be an unusually cheap way to rebalance.
3. Long-term-gain positions -> sell if materially overweight: Once lots have crossed one year, selling becomes much
more attractive.
4. Short-term-gain positions -> generally leave alone: Unless the deviation is large enough that the investment risk
of staying overweight exceeds the tax cost.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from data.market_data import fetch_prices, fetch_risk_free_rate
from data.universe import *
from estimates.covariance import ledoit_wolf_covariance
from estimates.returns import annualize_returns, compute_log_returns
from optim.mean_variance import min_variance_portfolio, efficient_frontier_and_metrics

pd.set_option("display.float_format", "{:,.5f}".format)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 500)

# Universe of assets to include in the portfolio.
UNIVERSE = [
    # Bogleheads canon
    ETF.VTI,
    ETF.VXUS,
    ETF.BND,
    ETF.BNDX,
    # Asset classes
    ETF.VOO,
    ETF.VUG,
    ETF.VWO,
    ETF.VEA,
    ETF.IAU,
    ETF.VNQ,
    ETF.IEF,
    ETF.TLT,
    ETF.EMB,
    # Sectors
    ETF.XLC,
    ETF.XLY,
    ETF.XLU,
    ETF.XLE,
    ETF.XLB,
    ETF.XLRE,
    # Styles
    ETF.MTUM,
    ETF.QUAL,
    # Added in 2026-09
    ETF.DBE,
    ETF.VGT,
    ETF.SOXX,
    ETF.XLF,
    ETF.IMTM,
    ETF.IQLT,
]

OPT_SHARPE_FRACTION = 0.95

# Import the parameters for the current date
from parameters_202609 import *

# Check if there are any alpha signals that are not represented by any ETFs in the current universe
unrepresented = unrepresented_in_universe(UNIVERSE, ALPHA.keys())
if any(unrepresented):
    print("\n".join(["currently unrepresented alpha signals:"] + list(map(str, unrepresented))))

# Market data
universe = {i.ticker: i for i in UNIVERSE}
tickers = list(universe.keys())
risk_free_rate = fetch_risk_free_rate(start=date.today() - timedelta(days=90))
prices = fetch_prices(tickers, start=PRICE_START)
returns = compute_log_returns(prices)
cov = ledoit_wolf_covariance(returns)
assert all(cov.index.values == cov.columns)

# Covariance blending, if specified
if COV_BLEND and COV_BLEND["weight"] > 0:
    cov_blend_prices = fetch_prices(tickers, start=COV_BLEND["start"], end=COV_BLEND["end"])
    cov_blend_returns = compute_log_returns(cov_blend_prices)
    cov_blend = ledoit_wolf_covariance(cov_blend_returns)
    assert all(cov_blend.index.values == cov_blend.columns)
    assert all(cov.columns == cov_blend.columns)
    cov = cov * (1 - COV_BLEND["weight"]) + cov_blend * COV_BLEND["weight"]
    assert all(cov.index.values == cov.columns)
# Sanity check the covariance matrix
eigvals = np.linalg.eigvalsh(cov)
# The min eigenvalue should be positive and small, not negative and large
print("\n* cov eigenvalues:", eigvals)
print("* min cov eigenvalue:", eigvals.min())

# Combine the sample returns with the prior return and the alpha
mu = annualize_returns(returns)
assert cov.index.equals(mu.index)
ret_df = pd.DataFrame(
    {"desc": [universe[t].desc for t in cov.index], "return_sample": mu, "return_prior": 0., "alpha": 0., "mean": 0.,
     "stdev": np.diag(cov) ** 0.5}, index=cov.index)
for ticker in tickers:
    return_prior = universe[ticker].return_prior()
    alpha = universe[ticker].accumulate_matched_returns(ALPHA)
    mu[ticker] = mu[ticker] * (1 - RETURN_SHRINKAGE_FACTOR) + return_prior * RETURN_SHRINKAGE_FACTOR + alpha
    ret_df.loc[ticker, ["return_prior", "alpha", "mean"]] = return_prior, alpha, mu[ticker]
print(f"\n* universe return stats:\n{ret_df}")

# Min variance anchor portfolio
w0 = min_variance_portfolio(cov)

# Efficient frontier for different values of lambda
lambda_grid = np.linspace(LAMBDA["min"], LAMBDA["max"], LAMBDA["steps"] + 1)
frontier, metrics = efficient_frontier_and_metrics(
    mu=mu,
    cov=cov,
    w_anchor=w0,
    gamma_reg=GAMMA,
    lambda_grid=lambda_grid,
    returns=returns,
    rf=risk_free_rate,
)
# Compute more detail around the max Sharpe portfolio
max_sharpe_lambda = metrics["sharpe"].idxmax()
lambda_detail = (LAMBDA["max"] - LAMBDA["min"]) / 20
lambda_grid = sorted(np.append(lambda_grid,
                               np.linspace(max_sharpe_lambda - lambda_detail, max_sharpe_lambda + lambda_detail,
                                           LAMBDA["steps"] + 1)))
# Recalculate the frontier with the additional detail around the max Sharpe portfolio
frontier, metrics = efficient_frontier_and_metrics(
    mu=mu,
    cov=cov,
    w_anchor=w0,
    gamma_reg=GAMMA,
    lambda_grid=lambda_grid,
    returns=returns,
    rf=risk_free_rate,
)
print(f"\n* frontier portfolio weights:\n{frontier}")
print(f"* frontier portfolio metrics:\n{metrics}")

# Choose the optimal portfolio as the min lambda portfolio with a sharpe=0.95*max_sharpe
max_sharpe = metrics["sharpe"].max()
opt_sharpe = max_sharpe * OPT_SHARPE_FRACTION
opt_lambda = metrics[metrics["sharpe"] >= opt_sharpe].index[0]
print(f"\n* max sharpe: {max_sharpe:.6f} at lambda={metrics["sharpe"].idxmax():.2f}")
print(f"* optimal portfolio: sharpe={metrics.loc[opt_lambda, 'sharpe']:.6f} at lambda={opt_lambda:.2f}")
print(f"* optimal portfolio:")
pfo = frontier.loc[opt_lambda].abs()
pfo = pfo[pfo > 0.01]
print(f"{'ticker':6} | {'description':>35} | {'alpha':>8} | {'weight':>8} | $value")
print("-" * (6 + 35 + 8 + 8 + 10 + 3 * 4))
for ticker, weight in pfo.items():
    print(f"{ticker:6} | {universe[ticker].desc:>35} | {ret_df.loc[ticker, 'alpha']:>8.6}"
          f" | {weight:>8.2%} | ${AUM * weight:>,.2f}")
