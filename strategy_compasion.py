import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

print("Downloading data...")

# NYSE listed CNQ in USD
cnq = yf.Ticker("CNQ").history(period="10y")["Close"]
uso = yf.Ticker("USO").history(period="10y")["Close"]
uco = yf.Ticker("UCO").history(period="10y")["Close"]  # 2x leveraged crude
wti = yf.Ticker("CL=F").history(period="10y")["Close"]

data = pd.DataFrame({
    "CNQ": cnq,
    "USO": uso,
    "UCO": uco,
    "WTI": wti
}).dropna()

data.index = data.index.tz_localize(None)

# Calculate returns
returns = data.pct_change().dropna()

# Load dividend dates
dividends = pd.read_csv(
    r"C:\Users\patel\PycharmProjects\PythonProject7\cnq_dividends.csv",
    index_col="Date",
    parse_dates=True
)
dividends.index = pd.to_datetime(dividends.index, utc=True).normalize()
dividends.index = dividends.index.tz_localize(None)

print(f"Data loaded: {len(data)} rows")
print(f"Dividend events: {len(dividends)}")

# WTI trend filter using moving average spread
data["WTI_MA20"] = data["WTI"].rolling(20).mean()
data["WTI_MA50"] = data["WTI"].rolling(50).mean()
data["WTI_trend"] = abs(data["WTI_MA20"] - data["WTI_MA50"]) / data["WTI_MA50"]
data["range_bound"] = data["WTI_trend"] < 0.03

print(f"Range-bound days: {data['range_bound'].sum()}")
print(f"Trending days: {(~data['range_bound']).sum()}")
print(f"% range-bound: {data['range_bound'].mean()*100:.1f}%")

position_size = 10000
WINDOW = 25  # rolling beta window (days) — optimized from sensitivity analysis

results = {
    "unfiltered":  [],
    "filtered":    [],
    "hedged_uso":  [],
    "hedged_wti":  [],
    "hedged_uco":  [],
}

# Rolling hedge ratios for each instrument
rolling_beta_uso = (
    returns["CNQ"].rolling(WINDOW).cov(returns["USO"]) /
    returns["USO"].rolling(WINDOW).var()
)
rolling_beta_wti = (
    returns["CNQ"].rolling(WINDOW).cov(returns["WTI"]) /
    returns["WTI"].rolling(WINDOW).var()
)
rolling_beta_uco = (
    returns["CNQ"].rolling(WINDOW).cov(returns["UCO"]) /
    returns["UCO"].rolling(WINDOW).var()
)

for date in dividends.index:
    if date not in data.index:
        continue

    loc = data.index.get_loc(date)

    if loc + 1 >= len(data) or loc < WINDOW:
        continue

    # Prices
    cnq_day0 = data["CNQ"].iloc[loc]
    cnq_day1 = data["CNQ"].iloc[loc + 1]
    uso_day0 = data["USO"].iloc[loc]
    uso_day1 = data["USO"].iloc[loc + 1]
    uco_day0 = data["UCO"].iloc[loc]
    uco_day1 = data["UCO"].iloc[loc + 1]
    wti_day0 = data["WTI"].iloc[loc]
    wti_day1 = data["WTI"].iloc[loc + 1]

    is_range_bound = data["range_bound"].iloc[loc]

    # Hedge ratios
    beta_uso = rolling_beta_uso.iloc[loc]
    beta_wti = rolling_beta_wti.iloc[loc]
    beta_uco = rolling_beta_uco.iloc[loc]

    # Dividend for this date
    div_amount = dividends.loc[date].iloc[0] if date in dividends.index else 0

    # Base long P&L (price change + dividend)
    shares_cnq = position_size / cnq_day0
    pnl_long = (cnq_day1 - cnq_day0 + div_amount) * shares_cnq

    regime = "range" if is_range_bound else "trend"

    # Strategy 1 — Unfiltered long
    results["unfiltered"].append({"date": date, "pnl": pnl_long, "regime": regime})

    # Strategy 2 — Filtered long (range-bound only)
    if is_range_bound:
        results["filtered"].append({"date": date, "pnl": pnl_long, "regime": "range"})

    # Strategy 3 — Hedged with USO
    shares_uso = (position_size * beta_uso) / uso_day0
    pnl_hedged_uso = pnl_long - (uso_day1 - uso_day0) * shares_uso
    results["hedged_uso"].append({"date": date, "pnl": pnl_hedged_uso, "regime": regime})

    # Strategy 4 — Hedged with WTI futures
    shares_wti = (position_size * beta_wti) / wti_day0
    pnl_hedged_wti = pnl_long - (wti_day1 - wti_day0) * shares_wti
    results["hedged_wti"].append({"date": date, "pnl": pnl_hedged_wti, "regime": regime})

    # Strategy 5 — Hedged with UCO (2x levered crude)
    shares_uco = (position_size * beta_uco) / uco_day0
    pnl_hedged_uco = pnl_long - (uco_day1 - uco_day0) * shares_uco
    results["hedged_uco"].append({"date": date, "pnl": pnl_hedged_uco, "regime": regime})

# Convert to DataFrames
dfs = {k: pd.DataFrame(v) for k, v in results.items()}

# Print summary
summary_data = {}
STRAT_LABELS = {
    "unfiltered":  "Unfiltered",
    "filtered":    "Filtered (Range-Bound Only)",
    "hedged_uso":  "Hedged with USO",
    "hedged_wti":  "Hedged with WTI",
    "hedged_uco":  "Hedged with UCO",
}

for key, label in STRAT_LABELS.items():
    df = dfs[key]
    total    = df["pnl"].sum()
    win_rate = (df["pnl"] > 0).mean() * 100
    avg      = df["pnl"].mean()
    wins     = df[df["pnl"] > 0]["pnl"].mean()
    losses   = df[df["pnl"] < 0]["pnl"].mean()
    pf       = abs(wins / losses) if losses != 0 else float("inf")
    sharpe   = df["pnl"].mean() / df["pnl"].std() * np.sqrt(len(df)) if df["pnl"].std() > 0 else 0
    summary_data[key] = {
        "trades": len(df), "total": total,
        "win_rate": win_rate, "avg": avg, "pf": pf, "sharpe": sharpe
    }
    print(f"\n{label}")
    print(f"  Trades:         {len(df)}")
    print(f"  Total P&L:      ${total:.2f}")
    print(f"  Win rate:       {win_rate:.1f}%")
    print(f"  Avg per trade:  ${avg:.2f}")
    print(f"  Profit factor:  {pf:.2f}")
    print(f"  Sharpe ratio:   {sharpe:.2f}")


# ─── PLOTTING ────────────────────────────────────────────────────────────────

plt.style.use("dark_background")

PALETTE = {
    "unfiltered":  "#4fc3f7",
    "filtered":    "#a5d6a7",
    "hedged_uso":  "#ffcc80",
    "hedged_wti":  "#ce93d8",
    "hedged_uco":  "#ef9a9a",
    "zero":        "#546e7a",
    "grid":        "#263238",
    "text":        "#eceff1",
    "subtext":     "#90a4ae",
}

dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")

fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
fig.suptitle("CNQ Dividend Capture — Strategy Analysis (5 Strategies, 25d Window)",
             fontsize=18, color=PALETTE["text"], fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dist = fig.add_subplot(gs[1, 0])
ax_bar  = fig.add_subplot(gs[1, 1])

def style_ax(ax, title):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color=PALETTE["text"], fontsize=12, pad=10)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

PLOT_ORDER = [
    ("unfiltered",  "Unfiltered Long"),
    ("filtered",    "Filtered (Range-Bound)"),
    ("hedged_uso",  "Hedged — USO"),
    ("hedged_wti",  "Hedged — WTI Futures"),
    ("hedged_uco",  "Hedged — UCO (2x)"),
]

# ── 1. Cumulative P&L ────────────────────────────────────────────────────────
style_ax(ax_cum, "Cumulative P&L by Strategy")

for key, label in PLOT_ORDER:
    df_sorted  = dfs[key].sort_values("date")
    cumulative = df_sorted["pnl"].cumsum()
    color      = PALETTE[key]
    ax_cum.plot(df_sorted["date"], cumulative, label=label,
                color=color, linewidth=2.2, alpha=0.92)
    ax_cum.scatter([df_sorted["date"].iloc[-1]], [cumulative.iloc[-1]],
                   color=color, s=60, zorder=5)

ax_cum.axhline(0, color=PALETTE["zero"], linewidth=1, linestyle="--", alpha=0.6)
ax_cum.yaxis.set_major_formatter(dollar_fmt)
ax_cum.set_xlabel("Date",               color=PALETTE["subtext"], fontsize=9)
ax_cum.set_ylabel("Cumulative P&L ($)", color=PALETTE["subtext"], fontsize=9)
ax_cum.legend(facecolor="#1c2128", edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=10, loc="upper left")

# ── 2. P&L Distribution ──────────────────────────────────────────────────────
style_ax(ax_dist, "Per-Trade P&L Distribution")

all_pnls = pd.concat([dfs[k]["pnl"] for k in results])
bins = np.linspace(all_pnls.min() - 50, all_pnls.max() + 50, 40)

for key, label in PLOT_ORDER:
    short_label = label.split("—")[-1].strip() if "—" in label else label
    color = PALETTE[key]
    ax_dist.hist(dfs[key]["pnl"], bins=bins, alpha=0.25,
                 label=short_label, color=color, edgecolor="none")
    ax_dist.hist(dfs[key]["pnl"], bins=bins, histtype="step",
                 color=color, linewidth=1.6)

ax_dist.axvline(0, color=PALETTE["zero"], linewidth=1.2, linestyle="--", alpha=0.8)
ax_dist.xaxis.set_major_formatter(dollar_fmt)
ax_dist.set_xlabel("P&L per Trade ($)", color=PALETTE["subtext"], fontsize=9)
ax_dist.set_ylabel("Frequency",         color=PALETTE["subtext"], fontsize=9)
ax_dist.legend(facecolor="#1c2128", edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8)

# ── 3. Summary Metrics Bar Chart ─────────────────────────────────────────────
style_ax(ax_bar, "Strategy Metrics Comparison")

metrics      = ["Win Rate (%)", "Avg Trade ($)", "Profit Factor", "Sharpe Ratio"]
bar_keys     = [k for k, _ in PLOT_ORDER]
short_labels = ["Unfiltered", "Filtered", "USO", "WTI", "UCO"]
x     = np.arange(len(metrics))
width = 0.14

for i, (key, short) in enumerate(zip(bar_keys, short_labels)):
    s      = summary_data[key]
    values = [s["win_rate"], s["avg"], s["pf"], s["sharpe"]]
    bars   = ax_bar.bar(x + i * width, values, width, label=short,
                        color=PALETTE[key], alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
        va    = "bottom" if val >= 0 else "top"
        ax_bar.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.1f}", ha="center", va=va,
                    color=PALETTE["text"], fontsize=7)

ax_bar.set_xticks(x + width * 2)
ax_bar.set_xticklabels(metrics, color=PALETTE["subtext"], fontsize=9)
ax_bar.legend(facecolor="#1c2128", edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

plt.savefig(r"C:\Users\patel\PycharmProjects\PythonProject7\cnq_strategy_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nChart saved → cnq_strategy_analysis.png")
plt.show()