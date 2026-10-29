import cvxpy as cp
import numpy as np
import pandas as pd

from analytics.metrics import portfolio_summary


def min_variance_portfolio(
        cov: pd.DataFrame,
        long_only: bool = True,
) -> pd.Series:
    """
    Compute minimum variance portfolio (anchor portfolio).
    """
    n = cov.shape[0]
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov.values))
    constraints = [cp.sum(w) == 1]

    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return pd.Series(w.value, index=cov.index)


def regularized_mean_variance(
        mu: pd.Series,
        cov: pd.DataFrame,
        w_anchor: pd.Series,
        lambda_ret: float,
        gamma_reg: float,
        long_only: bool = True,
) -> pd.Series:
    """
    Regularized mean-variance optimization.
    """
    n = len(mu)
    w = cp.Variable(n)

    risk = cp.quad_form(w, cov.values)
    ret = mu.values @ w
    reg = cp.sum_squares(w - w_anchor.values)

    objective = cp.Minimize(risk - lambda_ret * ret + gamma_reg * reg)

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return pd.Series(w.value, index=mu.index)


def efficient_frontier(
        mu: pd.Series,
        cov: pd.DataFrame,
        w_anchor: pd.Series,
        gamma_reg: float,
        lambda_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Generate efficient frontier by sweeping lambda.
    """
    weights = []

    for lam in lambda_grid:
        w = regularized_mean_variance(
            mu=mu,
            cov=cov,
            w_anchor=w_anchor,
            lambda_ret=lam,
            gamma_reg=gamma_reg,
        )
        weights.append(w)

    return pd.DataFrame(weights, index=lambda_grid)


def efficient_frontier_and_metrics(
        mu: pd.Series,
        cov: pd.DataFrame,
        w_anchor: pd.Series,
        gamma_reg: float,
        lambda_grid: np.ndarray,
        returns: pd.DataFrame,
        rf: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience method to compute efficient frontier and the portfolio metrics on it.
    """
    frontier = efficient_frontier(mu=mu, cov=cov, w_anchor=w_anchor, gamma_reg=gamma_reg, lambda_grid=lambda_grid)
    # Compute portfolio metrics
    metrics = pd.DataFrame(
        [portfolio_summary(weights=w, mu=mu, cov=cov, returns=returns, rf=rf) | {"lambda": lam}
         for lam, w in frontier.iterrows()]).set_index("lambda")
    return frontier, metrics
