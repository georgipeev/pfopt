from datetime import date

from data.universe import *

# Price data for this model should include 4-5 years.
PRICE_START = date(2021, 6, 1)

# Weight of the return prior in the final returns; values of lambda_mu towards 0 favor the sample returns, values
# towards 1 favor the return prior
RETURN_SHRINKAGE_FACTOR = 0.7

# Covariance blending: if COV_BLEND is not None, use the period and weight specified in the dictionary.
TRUMP_START = date(2016, 1, 1)
TRUMP_END = date(2020, 12, 31)
COV_BLEND = {"start": TRUMP_START, "end": TRUMP_END, "weight": 0.3}

# Gamma controls regularization and affects portfolio stability. It quantifies the trust in the min variance anchor
# portfolio w0. A good range of values is [5, 20].
GAMMA = 10

# Lambda controls risk aversion / return aggressiveness, larger lambda = higher expected return / risk.
LAMBDA = {"min": 0, "max": 500, "steps": 50}

# Alpha (annualized)
ALPHA = {
    (COMMODITY, GLOBAL, None): +0.035,
    (COMMODITY, GLOBAL, GOLD): +0.015,
    (COMMODITY, GLOBAL, ENERGY): -0.075,

    (EQUITY, US, BROAD): +0.025,
    (EQUITY, US, CONSUMER_DISCRETIONARY): +0.035,
    (EQUITY, US, ENERGY): +0.04,
    (EQUITY, US, MATERIALS): -0.035,
    (EQUITY, US, REAL_ESTATE): -0.03,
    (EQUITY, US, COMMUNICATION_SERVICES): +0.025,
    (EQUITY, US, UTILITIES): -0.025,
    (EQUITY, US, INFORMATION_TECHNOLOGY): +0.025,
    (EQUITY, US, SEMICONDUCTORS): -0.05,
    (EQUITY, US, DIVERSIFIED_FINANCIALS): +0.04,
    (EQUITY, US, QUALITY): +0.025,
    (EQUITY, INTL, MOMENTUM): +0.035,
    (EQUITY, INTL, QUALITY): +0.025,

    (REIT, None, None): -0.03,

    (BOND, US, TREASURY): -0.02,
    (BOND, EM, CREDIT): -0.03,
}

# Shrinkage factor for the excess annualized alpha return values above
ALPHA_SHRINKAGE = 5

for mask in ALPHA:
    ALPHA[mask] /= ALPHA_SHRINKAGE

# Current value plus cash to add
AUM = 500_000 + 100_000
