import pandas as pd


def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Compute portfolio return time series.
    """
    return returns @ weights


def cumulative_returns(
    portfolio_returns: pd.Series,
) -> pd.Series:
    """
    Compute cumulative returns.
    """
    return (1 + portfolio_returns).cumprod()


def drawdowns(
    cumulative_returns: pd.Series,
) -> pd.Series:
    """
    Compute drawdown series.
    """
    running_max = cumulative_returns.cummax()
    return cumulative_returns / running_max - 1


def max_drawdown(
    cumulative_returns: pd.Series,
) -> float:
    """
    Maximum drawdown.
    """
    return drawdowns(cumulative_returns).min()


def expected_drawdown(
    cumulative_returns: pd.Series,
    quantile: float = 0.1,
) -> float:
    """
    Expected drawdown: mean of worst drawdowns.
    """
    dd = drawdowns(cumulative_returns)
    cutoff = dd.quantile(quantile)
    return dd[dd <= cutoff].mean()
