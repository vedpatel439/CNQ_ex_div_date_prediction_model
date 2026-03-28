import pandas as pd
import numpy as np
from scipy import stats

prices = pd.read_csv(
    r"C:\Users\patel\PycharmProjects\PythonProject7\cnq_prices.csv",
    index_col="Date",
    parse_dates=True
)

dividends = pd.read_csv(
    r"C:\Users\patel\PycharmProjects\PythonProject7\cnq_dividends.csv",
    index_col="Date",
    parse_dates=True
)

prices.index = pd.to_datetime(prices.index, utc=True).normalize()
dividends.index = pd.to_datetime(dividends.index, utc=True).normalize()
prices["Return"] = prices["Close"].pct_change()

print("Data loaded")

window = 10
factor_data = []

for date in dividends.index:
    if date in prices.index:
        loc = prices.index.get_loc(date)

        # Need at least 20 days before and 2 days after
        if loc < 20 or loc + 2 >= len(prices):
            continue

        # Get the event window returns
        start = loc - window
        end = loc + window + 1
        window_returns = prices["Return"].iloc[start:end].values

        if len(window_returns) != (window * 2 + 1):
            continue

        # Factor 1: Momentum - cumulative return over 20 days before event
        momentum = prices["Return"].iloc[loc - 20:loc].sum()

        # Factor 2: Volatility - std of returns over 20 days before event
        volatility = prices["Return"].iloc[loc - 20:loc].std()

        # Factor 3: Dividend size relative to stock price
        div_amount = dividends.iloc[dividends.index.get_loc(date), 0]
        stock_price = prices["Close"].iloc[loc]
        div_yield = div_amount / stock_price

        # Factor 4: Volume ratio - average volume 5 days before vs 20 day average
        vol_5day = prices["Volume"].iloc[loc - 5:loc].mean()
        vol_20day = prices["Volume"].iloc[loc - 20:loc].mean()
        volume_ratio = vol_5day / vol_20day

        # Day +1 return - what we're trying to predict
        day1_return = window_returns[window + 1]

        factor_data.append({
            "date": date,
            "momentum": momentum,
            "volatility": volatility,
            "div_yield": div_yield,
            "volume_ratio": volume_ratio,
            "day1_return": day1_return
        })

factors_df = pd.DataFrame(factor_data)
print(f"Events with full factor data: {len(factors_df)}")
print(factors_df.head())

# Split each factor into high and low groups and compare Day +1 returns
factors_to_test = ["momentum", "volatility", "div_yield", "volume_ratio"]

print("Factor Analysis — Does the factor predict Day +1 return?\n")

for factor in factors_to_test:
    median = factors_df[factor].median()
    high = factors_df[factors_df[factor] >= median]["day1_return"]
    low = factors_df[factors_df[factor] < median]["day1_return"]

    t_stat, p_value = stats.ttest_ind(high, low)

    print(f"{factor.upper()}")
    print(f"  High group avg: {high.mean():.4f} ({len(high)} events)")
    print(f"  Low group avg:  {low.mean():.4f} ({len(low)} events)")
    print(f"  p-value: {p_value:.4f}")
    print()