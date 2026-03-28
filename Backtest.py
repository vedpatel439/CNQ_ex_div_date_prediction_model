import pandas as pd
import numpy as np

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

position_size = 10000  # dollars per trade
trades = []

for date in dividends.index:
    if date in prices.index:
        loc = prices.index.get_loc(date)

        if loc + 1 >= len(prices):
            continue

        # Buy at Day 0 close, sell at Day 1 close
        day0_price = prices["Close"].iloc[loc]
        day1_price = prices["Close"].iloc[loc + 1]

        # Calculate P&L
        shares = position_size / day0_price
        pnl = (day1_price - day0_price) * shares
        pnl_pct = (day1_price - day0_price) / day0_price

        trades.append({
            "date": date,
            "day0_price": day0_price,
            "day1_price": day1_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })

trades_df = pd.DataFrame(trades)

# Summary stats
total_pnl = trades_df["pnl"].sum()
avg_pnl = trades_df["pnl"].mean()
win_rate = (trades_df["pnl"] > 0).mean() * 100
avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean()
avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean()

print(f"Total trades: {len(trades_df)}")
print(f"Win rate: {win_rate:.1f}%")
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Average P&L per trade: ${avg_pnl:.2f}")
print(f"Average win: ${avg_win:.2f}")
print(f"Average loss: ${avg_loss:.2f}")
print(f"Profit factor: {abs(avg_win / avg_loss):.2f}")

import matplotlib.pyplot as plt

trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
trades_df["date"] = pd.to_datetime(trades_df["date"], utc=True).dt.normalize()

plt.figure(figsize=(12, 6))
plt.plot(trades_df["date"], trades_df["cumulative_pnl"],
         color="green", linewidth=2)
plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
plt.fill_between(trades_df["date"], trades_df["cumulative_pnl"],
                 where=trades_df["cumulative_pnl"] >= 0,
                 color="green", alpha=0.1)
plt.fill_between(trades_df["date"], trades_df["cumulative_pnl"],
                 where=trades_df["cumulative_pnl"] < 0,
                 color="red", alpha=0.1)
plt.xlabel("Date")
plt.ylabel("Cumulative P&L ($)")
plt.title("CNQ.TO Dividend Capture Strategy — Cumulative P&L (2001-2026)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()