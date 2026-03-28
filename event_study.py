import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

print("Data loaded Succesfully")
print(f"Price rows: {len(prices)}")
print(f"Dividend events: {len(dividends)}")

# Calculate daily returns
prices["Return"] = prices["Close"].pct_change()

# Clean up the dividend index - convert to date only (no timestamp)
dividends.index = pd.to_datetime(dividends.index, utc=True).normalize()
prices.index = pd.to_datetime(prices.index, utc=True).normalize()

print("\nFirst few returns:")
print(prices.head())

# Define our event window - 10 days before and after each dividend
window = 10

# For each dividend date, collect the returns in the window around it
event_returns = []

for date in dividends.index:
    if date in prices.index:
        loc = prices.index.get_loc(date)
        start = max(0, loc - window)
        end = min(len(prices), loc + window + 1)
        window_returns = prices["Return"].iloc[start:end].values
        if len(window_returns) == (window * 2 + 1):
            event_returns.append(window_returns)

print(f"Events captured: {len(event_returns)}")

import numpy as np

# Convert to numpy array - rows are events, columns are days in window
event_matrix = np.array(event_returns)

# Calculate average return for each day in the window across all events
avg_returns = event_matrix.mean(axis=0)

# Create day labels: 10 days before to 10 days after
days = list(range(-window, window + 1))

print("Average returns around ex-dividend date:")
for day, ret in zip(days, avg_returns):
    print(f"Day {day:+3d}: {ret:.4f}")


plt.figure(figsize=(12, 6))
plt.bar(days, avg_returns, color=["green" if r > 0 else "red" for r in avg_returns])
plt.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Ex-Dividend Date")
plt.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
plt.xlabel("Days Relative to Ex-Dividend Date")
plt.ylabel("Average Return")
plt.title("CNQ.TO Average Returns Around Ex-Dividend Dates (2016-2026)")
plt.legend()
plt.tight_layout()
plt.show()