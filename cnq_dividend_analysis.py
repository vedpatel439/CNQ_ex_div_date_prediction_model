
"""
CNQ Dividend Sustainability & Valuation Analysis
Author: Ved Patel
Description:
End-to-end financial analysis program evaluating Canadian Natural Resources (CNQ)
dividend sustainability using free cash flow coverage, payout ratios, valuation
metrics, and oil price sensitivity.

Data Sources (when executed in an internet-enabled environment):
- Yahoo Finance (yfinance)
- Public financial statements

This file is designed to be resume- and GitHub-ready.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
TICKER = "CNQ.TO"
OIL_PROXY = "CL=F"
START_DATE = "2015-01-01"
END_DATE = None

# -----------------------------
# Data Retrieval (placeholder)
# -----------------------------
def load_market_data():
    """
    Load price and dividend data using yfinance.
    Requires internet access when executed.
    """
    import yfinance as yf

    stock = yf.Ticker(TICKER)
    price_data = stock.history(start=START_DATE, end=END_DATE)
    dividends = stock.dividends

    return price_data, dividends


def load_financials():
    """
    Load financial statement data.
    """
    import yfinance as yf

    stock = yf.Ticker(TICKER)
    cashflow = stock.cashflow.T
    income = stock.financials.T

    return cashflow, income

# -----------------------------
# Analysis Functions
# -----------------------------
def dividend_cagr(dividends):
    """
    Calculate dividend compound annual growth rate.
    """
    start = dividends.iloc[0]
    end = dividends.iloc[-1]
    years = (dividends.index[-1] - dividends.index[0]).days / 365
    return (end / start) ** (1 / years) - 1


def fcf_dividend_coverage(cashflow, dividends):
    """
    Calculate free cash flow dividend coverage ratio.
    """
    fcf = cashflow["Free Cash Flow"]
    total_dividends = dividends.resample("Y").sum()
    coverage = fcf / total_dividends
    return coverage


def payout_ratio(net_income, dividends):
    """
    Calculate earnings payout ratio.
    """
    total_dividends = dividends.resample("Y").sum()
    return total_dividends / net_income


def oil_price_sensitivity():
    """
    Placeholder for oil price sensitivity modeling.
    """
    scenarios = {
        "Bear": 55,
        "Base": 70,
        "Bull": 90
    }
    return scenarios

# -----------------------------
# Visualization
# -----------------------------
def plot_dividends(dividends):
    plt.figure()
    dividends.plot(title="CNQ Dividend History")
    plt.xlabel("Date")
    plt.ylabel("Dividend")
    plt.show()


def plot_coverage(coverage):
    plt.figure()
    coverage.plot(title="FCF Dividend Coverage Ratio")
    plt.axhline(1, linestyle="--")
    plt.ylabel("Coverage Ratio")
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    price_data, dividends = load_market_data()
    cashflow, income = load_financials()

    cagr = dividend_cagr(dividends)
    coverage = fcf_dividend_coverage(cashflow, dividends)

    print(f"Dividend CAGR: {cagr:.2%}")
    print("FCF Dividend Coverage:")
    print(coverage)

    plot_dividends(dividends)
    plot_coverage(coverage)


if __name__ == "__main__":
    main()
