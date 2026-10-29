from datetime import date

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    start: date,
    end: date | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for ETFs / mutual funds.
    """
    if end is None:
        end = date.today().isoformat()

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="any")
    return prices


def fetch_risk_free_rate(start: date, end: date | None = None) -> float:
    """
    Fetch average 3M T-bill yield over the period.
    """
    irx = yf.download("^IRX", start=start, end=end, progress=False)["Close"]
    rf_annual = irx.mean().values[0] / 100.0
    return rf_annual
