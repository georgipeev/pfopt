import pandas as pd
from sklearn.covariance import LedoitWolf


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compute annualized Ledoit-Wolf shrinkage covariance matrix.
    """
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * periods_per_year
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
