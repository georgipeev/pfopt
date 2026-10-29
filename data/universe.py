from enum import StrEnum

# Asset classes
EQUITY = "equity"
COMMODITY = "commodity"
BOND = "bond"
REIT = "reit"
# Regions
US = "us"
INTL = "intl"
EM = "em"
GLOBAL = "global"
# Sectors / styles
BROAD = "broad"
AGG = "agg"
GROWTH = "growth"
GOLD = "gold"
REAL_ESTATE = "real_estate"
TREASURY = "treasury"
TIPS = "tips"
CREDIT = "credit"
COMMUNICATION_SERVICES = "communication_services"
CONSUMER_DISCRETIONARY = "consumer_discretionary"
UTILITIES = "utilities"
ENERGY = "energy"
MATERIALS = "materials"
MOMENTUM = "momentum"
QUALITY = "quality"
# NEW
INFORMATION_TECHNOLOGY = "information_technology"
SEMICONDUCTORS = "semiconductors"
DIVERSIFIED_FINANCIALS = "diversified_financials"


class ETF(StrEnum):
    ticker: str
    asset_class: str
    region: str
    sector: str
    desc: str

    def __new__(cls, ticker, asset_class, region, sector, desc):
        etf = str.__new__(cls, ticker)
        etf._value_ = ticker
        etf.ticker = ticker
        etf.asset_class = asset_class
        etf.region = region
        etf.sector = sector
        etf.desc = desc
        return etf

    # Bogleheads canon
    VTI = ("VTI", EQUITY, US, BROAD, "US total equity")
    VXUS = ("VXUS", EQUITY, INTL, BROAD, "Intl ex-US equity")
    BND = ("BND", BOND, US, AGG, "US aggregate bonds")
    BNDX = ("BNDX", BOND, INTL, AGG, "Intl bonds")
    TIP = ("TIP", BOND, US, TIPS, "TIPS bonds")
    # Asset classes
    VOO = ("VOO", EQUITY, US, BROAD, "US large cap")
    VUG = ("VUG", EQUITY, US, GROWTH, "Long-term growth")
    VWO = ("VWO", EQUITY, EM, BROAD, "EM Equity")
    IEMG = ("IEMG", EQUITY, EM, BROAD, "EM Equity")
    VEA = ("VEA", EQUITY, INTL, BROAD, "Developed ex-US")
    IAU = ("IAU", COMMODITY, GLOBAL, GOLD, "Gold")
    GLD = ("GLD", COMMODITY, GLOBAL, GOLD, "Gold")
    VNQ = ("VNQ", REIT, US, REAL_ESTATE, "REITs (broad)")
    VNQI = ("VNQI", REIT, US, REAL_ESTATE, "REITs (broad)")
    IEF = ("IEF", BOND, US, TREASURY, "US Treasury bonds (intermed)")
    TLT = ("TLT", BOND, US, TREASURY, "US Treasury bonds (long)")
    EMB = ("EMB", BOND, EM, CREDIT, "EM credit")
    # Sectors
    XLC = ("XLC", EQUITY, US, COMMUNICATION_SERVICES, "Communication services")
    XLY = ("XLY", EQUITY, US, CONSUMER_DISCRETIONARY, "Consumer discretionary")
    XLU = ("XLU", EQUITY, US, UTILITIES, "Utilities")
    XLE = ("XLE", EQUITY, US, ENERGY, "Energy")
    XLB = ("XLB", EQUITY, US, MATERIALS, "Materials")
    XLRE = ("XLRE", EQUITY, US, REAL_ESTATE, "Real estate")
    # Styles
    MTUM = ("MTUM", EQUITY, US, MOMENTUM, "Momentum")
    QUAL = ("QUAL", EQUITY, US, QUALITY, "Quality / low leverage")

    # Added 202609
    DBE = ("DBE", COMMODITY, GLOBAL, ENERGY, "Futures on oil/gas")
    VGT = ("VGT", EQUITY, US, INFORMATION_TECHNOLOGY, "Vanguard Information Technology")
    SOXX = ("SOXX", EQUITY, US, SEMICONDUCTORS, "iShares Semiconductor")
    XLF = ("XLF", EQUITY, US, DIVERSIFIED_FINANCIALS, "State Street Financial Sector")
    IMTM = ("IMTM", EQUITY, INTL, MOMENTUM, "iShares MSCI Intl Momentum Factor")
    IQLT = ("IQLT", EQUITY, INTL, QUALITY, "iShares MSCI Intl Quality Factor")

    def return_prior(self) -> float:
        return self.accumulate_matched_returns(ETF_PRIOR_RETURNS)

    def matches(self, mask: tuple[str, str, str]):
        return ((mask[0] is None or mask[0] == self.asset_class) and
                (mask[1] is None or mask[1] == self.region) and
                (mask[2] is None or mask[2] == self.sector))

    def accumulate_matched_returns(self, returns: dict[tuple[str, str, str], float]) -> float:
        return sum([ret for mask, ret in returns.items() if self.matches(mask)])


# Maps (asset_class, region, sector) to prior return value, values for all matched masks are added
ETF_PRIOR_RETURNS = {
    (EQUITY, US, None): 0.045,
    (EQUITY, INTL, None): 0.045,
    (EQUITY, EM, None): 0.055,
    (COMMODITY, GLOBAL, GOLD): 0.00,
    (BOND, None, None): 0.005,
    # TIPS match item above, so adjust the 0.5% to a final return prior of 0.25%
    (BOND, US, TIPS): -0.0025,
    (REIT, None, None): 0.015,
}


def unrepresented_in_universe(universe: list[ETF], masks: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """
    Returns masks that are not represented in the given universe.
    """
    return [mask for mask in masks if all(not etf.matches(mask) for etf in universe)]
