import numpy as np
import pandas as pd

from analytics.drawdown import (
    cumulative_returns,
    expected_drawdown,
    max_drawdown,
    portfolio_returns,
)


def portfolio_expected_return(
    weights: pd.Series,
    mu: pd.Series,
) -> float:
    """
    Annualized expected portfolio return.
    """
    return float(weights @ mu)


def portfolio_volatility(
    weights: pd.Series,
    cov: pd.DataFrame,
) -> float:
    """
    Annualized portfolio volatility.
    """
    return float(np.sqrt(weights @ cov @ weights))


def portfolio_sharpe(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio.
    """
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return np.nan
    return (portfolio_expected_return(weights, mu) - rf) / vol


def portfolio_summary(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    returns: pd.DataFrame,
    rf: float = 0.0,
) -> dict:
    """
    Compute key portfolio metrics.
    """
    port_rets = portfolio_returns(returns, weights)
    cum_rets = cumulative_returns(port_rets)

    return {
        "expected_return": portfolio_expected_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov, rf),
        "max_drawdown": max_drawdown(cum_rets),
        "expected_drawdown": expected_drawdown(cum_rets),
    }
