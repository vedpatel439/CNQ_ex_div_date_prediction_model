import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

print("Downloading data...")

# ── Data ──────────────────────────────────────────────────────────────────────
cnq = yf.Ticker("CNQ").history(period="10y")["Close"]
uso = yf.Ticker("USO").history(period="10y")["Close"]
uco = yf.Ticker("UCO").history(period="10y")["Close"]
wti = yf.Ticker("CL=F").history(period="10y")["Close"]

data = pd.DataFrame({"CNQ": cnq, "USO": uso, "UCO": uco, "WTI": wti}).dropna()
data.index = data.index.tz_localize(None)
returns = data.pct_change().dropna()

# ── Dividends ─────────────────────────────────────────────────────────────────
dividends = pd.read_csv(
    r"C:\Users\patel\PycharmProjects\PythonProject7\cnq_dividends.csv",
    index_col="Date", parse_dates=True
)
dividends.index = pd.to_datetime(dividends.index, utc=True).normalize()
dividends.index = dividends.index.tz_localize(None)

print(f"Data loaded: {len(data)} rows | Dividend events: {len(dividends)}")

# ── WTI regime filter ─────────────────────────────────────────────────────────
data["WTI_MA20"]    = data["WTI"].rolling(20).mean()
data["WTI_MA50"]    = data["WTI"].rolling(50).mean()
data["WTI_trend"]   = abs(data["WTI_MA20"] - data["WTI_MA50"]) / data["WTI_MA50"]
data["range_bound"] = data["WTI_trend"] < 0.03

# ── Window sizes to test ──────────────────────────────────────────────────────
WINDOWS       = list(range(5, 181, 5))   # 5, 10, 15, ... 180
HEDGE_COLS    = ["USO", "UCO", "WTI"]
POSITION_SIZE = 10_000

# ── Run backtest for every window × hedge instrument ─────────────────────────
# results[hedge][window] = {"total", "win_rate", "avg", "pf", "sharpe", "n"}

all_results = {h: {} for h in HEDGE_COLS}

for window in WINDOWS:
    # Pre-compute rolling betas for all three hedges
    betas = {}
    for h in HEDGE_COLS:
        betas[h] = (
            returns["CNQ"].rolling(window).cov(returns[h]) /
            returns[h].rolling(window).var()
        )

    pnls = {h: [] for h in HEDGE_COLS}

    for date in dividends.index:
        if date not in data.index:
            continue
        loc = data.index.get_loc(date)
        if loc + 1 >= len(data) or loc < window:
            continue

        cnq0 = data["CNQ"].iloc[loc]
        cnq1 = data["CNQ"].iloc[loc + 1]
        div  = dividends.loc[date].iloc[0] if date in dividends.index else 0

        shares_cnq = POSITION_SIZE / cnq0
        pnl_long   = (cnq1 - cnq0 + div) * shares_cnq

        for h in HEDGE_COLS:
            h0   = data[h].iloc[loc]
            h1   = data[h].iloc[loc + 1]
            beta = betas[h].iloc[loc]

            hedge_shares = (POSITION_SIZE * beta) / h0
            pnl_hedge    = pnl_long - (h1 - h0) * hedge_shares
            pnls[h].append(pnl_hedge)

    for h in HEDGE_COLS:
        arr = np.array(pnls[h])
        if len(arr) < 3:
            continue
        wins   = arr[arr > 0]
        losses = arr[arr < 0]
        pf     = abs(wins.mean() / losses.mean()) if len(losses) > 0 and len(wins) > 0 else np.nan
        sharpe = arr.mean() / arr.std() * np.sqrt(len(arr)) if arr.std() > 0 else np.nan
        all_results[h][window] = {
            "total":    arr.sum(),
            "win_rate": (arr > 0).mean() * 100,
            "avg":      arr.mean(),
            "pf":       pf,
            "sharpe":   sharpe,
            "n":        len(arr),
        }

print(f"Tested {len(WINDOWS)} window sizes × {len(HEDGE_COLS)} hedges\n")

# ── Print best window per hedge ───────────────────────────────────────────────
print("=" * 60)
print("  BEST WINDOW PER HEDGE  (ranked by Total P&L)")
print("=" * 60)
for h in HEDGE_COLS:
    res = all_results[h]
    best_w = max(res, key=lambda w: res[w]["total"])
    r = res[best_w]
    print(f"\n  {h}")
    print(f"    Best window:   {best_w} days")
    print(f"    Total P&L:     ${r['total']:,.2f}")
    print(f"    Win rate:      {r['win_rate']:.1f}%")
    print(f"    Avg trade:     ${r['avg']:.2f}")
    print(f"    Profit factor: {r['pf']:.2f}")
    print(f"    Sharpe:        {r['sharpe']:.2f}")
print("=" * 60)

# ── Also print by Sharpe ──────────────────────────────────────────────────────
print("\n  BEST WINDOW PER HEDGE  (ranked by Sharpe)")
print("=" * 60)
for h in HEDGE_COLS:
    res = all_results[h]
    best_w = max(res, key=lambda w: res[w]["sharpe"] if not np.isnan(res[w]["sharpe"]) else -999)
    r = res[best_w]
    print(f"\n  {h}")
    print(f"    Best window:   {best_w} days")
    print(f"    Sharpe:        {r['sharpe']:.2f}")
    print(f"    Total P&L:     ${r['total']:,.2f}")
    print(f"    Win rate:      {r['win_rate']:.1f}%")
print("=" * 60)

# ── Plotting ──────────────────────────────────────────────────────────────────
plt.style.use("dark_background")

PALETTE = {
    "USO":  "#ffcc80",
    "UCO":  "#ef9a9a",
    "WTI":  "#ce93d8",
    "grid": "#263238",
    "text": "#eceff1",
    "sub":  "#90a4ae",
    "bg":   "#0d1117",
    "ax":   "#161b22",
}
dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")

METRICS = [
    ("total",    "Total P&L ($)",    dollar_fmt),
    ("win_rate", "Win Rate (%)",      None),
    ("pf",       "Profit Factor",     None),
    ("sharpe",   "Sharpe Ratio",      None),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=PALETTE["bg"])
fig.suptitle("CNQ Dividend Capture — Rolling Beta Window Sensitivity (5→180 days)",
             fontsize=16, color=PALETTE["text"], fontweight="bold", y=0.99)

for ax, (metric, ylabel, fmt) in zip(axes.flat, METRICS):
    ax.set_facecolor(PALETTE["ax"])
    ax.set_title(ylabel, color=PALETTE["text"], fontsize=11, pad=8)
    ax.tick_params(colors=PALETTE["sub"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.6)
    ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    for h in HEDGE_COLS:
        res = all_results[h]
        ws  = sorted(res.keys())
        ys  = [res[w][metric] for w in ws]
        ax.plot(ws, ys, label=h, color=PALETTE[h], linewidth=2, alpha=0.9)

        # Mark best window
        best_w = max(res, key=lambda w: res[w][metric] if not np.isnan(res[w][metric]) else -999)
        best_y = res[best_w][metric]
        ax.scatter([best_w], [best_y], color=PALETTE[h], s=80, zorder=5)
        ax.annotate(f"{best_w}d", xy=(best_w, best_y),
                    xytext=(6, 4), textcoords="offset points",
                    color=PALETTE[h], fontsize=8)

    if fmt:
        ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Window (days)", color=PALETTE["sub"], fontsize=9)
    ax.legend(facecolor="#1c2128", edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(r"C:\Users\patel\PycharmProjects\PythonProject7\cnq_window_sensitivity.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nChart saved → cnq_window_sensitivity.png")
plt.show()