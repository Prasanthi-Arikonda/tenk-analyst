"""Exact numbers from SEC's XBRL company facts.

A full-year number has the frame 'CY<year>'. If SEC has no value, we return None.
We never estimate a number.
"""
import csv
import functools
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

FILINGS = Path(__file__).resolve().parent.parent / "data" / "filings.csv"

# metric -> the XBRL tags to try, in order
METRIC_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "operating_income": ["OperatingIncomeLoss"],
}


@functools.lru_cache(maxsize=1)
def ciks() -> dict:
    """ticker -> 10-digit CIK, read from data/filings.csv."""
    with open(FILINGS, encoding="utf-8") as f:
        return {r["ticker"]: r["cik"].zfill(10) for r in csv.DictReader(f)}


@functools.lru_cache(maxsize=16)  # the file is large and does not change, so keep it in memory
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    # SEC requires a User-Agent with a name and email. It comes from .env, never from the code.
    r = httpx.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60)
    r.raise_for_status()
    return r.json()


def annual_value(ticker: str, metric: str, year: int):
    """Return (value in dollars, tag) for a full-year amount, or (None, None) if SEC has none."""
    facts = company_facts(ciks()[ticker])["facts"].get("us-gaap", {})
    for tag in METRIC_TAGS[metric]:
        for values in facts.get(tag, {}).get("units", {}).values():
            for v in values:
                if v.get("frame") == f"CY{year}":
                    return v["val"], tag
    return None, None
