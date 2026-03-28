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
event_returns = []

for date in dividends.index:
    if date in prices.index:
        loc = prices.index.get_loc(date)
        start = max(0, loc - window)
        end = min(len(prices), loc + window + 1)
        window_returns = prices["Return"].iloc[start:end].values
        if len(window_returns) == (window * 2 + 1):
            event_returns.append(window_returns)

event_matrix = np.array(event_returns)

# Extract Day 0 and Day +1 returns across all events
day0_returns = event_matrix[:, window]
day1_returns = event_matrix[:, window + 1]

# Run t-tests
t0, p0 = stats.ttest_1samp(day0_returns, 0)
t1, p1 = stats.ttest_1samp(day1_returns, 0)

print(f"Day 0  — Average: {day0_returns.mean():.4f}, t-stat: {t0:.3f}, p-value: {p0:.4f}")
print(f"Day +1 — Average: {day1_returns.mean():.4f}, t-stat: {t1:.3f}, p-value: {p1:.4f}")

# Consistency check
pct_positive_d0 = (day0_returns > 0).mean() * 100
pct_positive_d1 = (day1_returns > 0).mean() * 100

print(f"\nDay 0  — % positive: {pct_positive_d0:.1f}%")
print(f"Day +1 — % positive: {pct_positive_d1:.1f}%")

# Split events in half - first half trains, second half tests
split = len(event_matrix) // 2
train = event_matrix[:split, window + 1]
test = event_matrix[split:, window + 1]

t_train, p_train = stats.ttest_1samp(train, 0)
t_test, p_test = stats.ttest_1samp(test, 0)

print(f"\nOut-of-sample test (Day +1):")
print(f"Train — Average: {train.mean():.4f}, p-value: {p_train:.4f}")
print(f"Test  — Average: {test.mean():.4f}, p-value: {p_test:.4f}")