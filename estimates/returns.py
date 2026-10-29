import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns


def annualize_returns(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Annualized mean returns.
    """
    return returns.mean() * periods_per_year
