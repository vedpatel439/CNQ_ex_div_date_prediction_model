import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
POSITION_SIZE = 10_000   # $ long CNQ
WINDOW        = 25       # rolling beta lookback (days)
HEDGE_TICKERS = ["USO", "UCO", "CL=F"]
HEDGE_LABELS  = {"USO": "USO", "UCO": "UCO (2x)", "CL=F": "WTI Futures"}

# ── Download recent price data ────────────────────────────────────────────────
print("Downloading price data...")
tickers = ["CNQ"] + HEDGE_TICKERS
raw = yf.download(tickers, period="6mo", auto_adjust=True, progress=False)["Close"]
raw.index = raw.index.tz_localize(None)
raw.columns = [c if c != "CL=F" else "WTI" for c in raw.columns]

# Map ticker → column name for hedge loop
col_map = {"USO": "USO", "UCO": "UCO", "CL=F": "WTI"}

data    = raw.dropna()
returns = data.pct_change().dropna()

if len(returns) < WINDOW:
    raise ValueError(f"Not enough data: need {WINDOW} days, got {len(returns)}")

# ── Compute rolling beta (last value = today's estimate) ─────────────────────
print(f"\nUsing last {WINDOW} trading days of data.")
print(f"Most recent date in dataset: {data.index[-1].date()}\n")

results = {}
for ticker in HEDGE_TICKERS:
    col = col_map[ticker]
    cov = returns["CNQ"].iloc[-WINDOW:].cov(returns[col].iloc[-WINDOW:])
    var = returns[col].iloc[-WINDOW:].var()
    beta = cov / var

    # Current prices
    cnq_price   = data["CNQ"].iloc[-1]
    hedge_price = data[col].iloc[-1]

    # Hedge shares to short
    hedge_shares = (POSITION_SIZE * beta) / hedge_price

    # WTI trend filter check (MA20 vs MA50 spread)
    ma20 = data["WTI"].rolling(20).mean().iloc[-1]
    ma50 = data["WTI"].rolling(50).mean().iloc[-1]
    spread_pct = abs(ma20 - ma50) / ma50 * 100
    regime = "RANGE-BOUND ✓" if spread_pct < 3.0 else "TRENDING ✗"

    results[ticker] = {
        "label":        HEDGE_LABELS[ticker],
        "beta":         beta,
        "cnq_price":    cnq_price,
        "hedge_price":  hedge_price,
        "hedge_shares": hedge_shares,
        "notional":     hedge_shares * hedge_price,
    }

# ── Print results ─────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  CNQ HEDGE RATIOS  —  {date.today()}")
print("=" * 55)
print(f"  Long CNQ:       ${POSITION_SIZE:,.0f} position")
print(f"  CNQ price:      ${results['USO']['cnq_price']:.2f}")
print(f"  CNQ shares:     {POSITION_SIZE / results['USO']['cnq_price']:.2f}")
print(f"\n  WTI regime:     {regime}  ({spread_pct:.2f}% MA spread)")
print("-" * 55)

for ticker in HEDGE_TICKERS:
    r = results[ticker]
    print(f"\n  Hedge: {r['label']}")
    print(f"    Beta (25d):       {r['beta']:.4f}")
    print(f"    {r['label']} price:     ${r['hedge_price']:.2f}")
    print(f"    Shares to SHORT:  {r['hedge_shares']:.2f}")
    print(f"    Hedge notional:   ${r['notional']:,.2f}")

print("\n" + "=" * 55)
print("  Interpretation:")
print("  Short the hedge notional against your $10k CNQ long.")
print("  Re-run this script on each ex-div date for fresh beta.")
print("=" * 55)