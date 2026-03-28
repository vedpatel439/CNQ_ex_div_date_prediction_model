import yfinance as yf
import pandas as pd

cnq = yf.Ticker("CNQ.TO")
prices = cnq.history(period = "32y")

dividends = cnq.dividends

prices = prices [["Close", "Volume"]]

print("\n--- PRICE DATA ---")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Total trading days: {len(prices)}")
print(prices.head(10))

print ("\n--- DIVIDEND DATA ---")
print (f"Total dividend events: {len(dividends)}")
print (dividends.tail(10))

# Save to CSV so we don't have to re-download every time
prices.to_csv("cnq_prices.csv")
dividends.to_csv("cnq_dividends.csv")
print("\nData saved successfully.")