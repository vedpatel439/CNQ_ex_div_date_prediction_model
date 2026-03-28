import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

print("Downloading data...")

cnq = yf.Ticker("CNQ").history(period="10y")["Close"]
uso = yf.Ticker("USO").history(period="10y")["Close"]
uco = yf.Ticker("UCO").history(period="10y")["Close"]
wti = yf.Ticker("CL=F").history(period="10y")["Close"]

data = pd.DataFrame({
    "CNQ": cnq,
    "USO": uso,
    "UCO": uco,
    "WTI": wti
}).dropna()

returns = data.pct_change().dropna()

# Rolling hedge ratio
rolling_beta = (
    returns["CNQ"].rolling(60).cov(returns["WTI"]) /
    returns["WTI"].rolling(60).var()
)

# Rolling correlation (60-day)
rolling_corr = returns["CNQ"].rolling(60).corr(returns["USO"])

# VIX
vix = yf.Ticker("^VIX").history(period="10y")["Close"]
vix.index = vix.index.tz_localize(None)

rolling_beta.index = rolling_beta.index.tz_localize(None)
rolling_corr.index = rolling_corr.index.tz_localize(None)

combined = pd.DataFrame({
    "hedge_ratio": rolling_beta,
    "rolling_corr": rolling_corr,
    "vix": vix
}).dropna()

# --- Plot 1: Hedge Ratio vs VIX ---
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(combined.index, combined["hedge_ratio"],
         color="blue", linewidth=1, label="Hedge Ratio")
ax1.set_ylabel("Hedge Ratio", color="blue")
ax2 = ax1.twinx()
ax2.plot(combined.index, combined["vix"],
         color="red", linewidth=1, alpha=0.6, label="VIX")
ax2.set_ylabel("VIX", color="red")
plt.title("CNQ/USO Hedge Ratio vs VIX")
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.show()

corr = combined["hedge_ratio"].corr(combined["vix"])
print(f"Correlation between hedge ratio and VIX: {corr:.4f}")

# --- Plot 2: Rolling 60-day Correlation CNQ vs USO ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(combined.index, combined["rolling_corr"],
        color="purple", linewidth=1, label="60-day Rolling Correlation")
ax.axhline(combined["rolling_corr"].mean(), color="black",
           linestyle="--", linewidth=0.8, label=f"Mean: {combined['rolling_corr'].mean():.2f}")
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

# Shade high VIX periods (VIX > 25) to show correlation behavior in stress
high_vix = combined["vix"] > 25
ax.fill_between(combined.index, -1, 1,
                where=high_vix, alpha=0.15, color="red", label="VIX > 25")

ax.set_ylim(-1, 1)
ax.set_ylabel("Correlation")
ax.set_title("CNQ/USO 60-Day Rolling Correlation")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot 3: Scatter — CNQ vs USO Returns by VIX Regime ---
scatter_data = pd.DataFrame({
    "CNQ": returns["CNQ"],
    "USO": returns["USO"],
    "vix": vix
}).dropna()

low_vix = scatter_data["vix"] <= 20
mid_vix = (scatter_data["vix"] > 20) & (scatter_data["vix"] <= 30)
high_vix_scatter = scatter_data["vix"] > 30

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(scatter_data.loc[low_vix, "USO"],
           scatter_data.loc[low_vix, "CNQ"],
           alpha=0.3, s=8, color="green", label="VIX ≤ 20 (Low)")
ax.scatter(scatter_data.loc[mid_vix, "USO"],
           scatter_data.loc[mid_vix, "CNQ"],
           alpha=0.3, s=8, color="orange", label="VIX 20–30 (Mid)")
ax.scatter(scatter_data.loc[high_vix_scatter, "USO"],
           scatter_data.loc[high_vix_scatter, "CNQ"],
           alpha=0.4, s=8, color="red", label="VIX > 30 (High)")

# Fit a regression line for each regime
for mask, color in [(low_vix, "green"), (mid_vix, "orange"), (high_vix_scatter, "red")]:
    subset = scatter_data[mask]
    if len(subset) > 10:
        m, b = np.polyfit(subset["USO"], subset["CNQ"], 1)
        x_range = np.linspace(subset["USO"].min(), subset["USO"].max(), 100)
        ax.plot(x_range, m * x_range + b, color=color, linewidth=1.5)

ax.set_xlabel("USO Daily Return")
ax.set_ylabel("CNQ Daily Return")
ax.set_title("CNQ vs USO Returns by VIX Regime")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()

# Beta by regime
for label, mask in [("Low VIX (≤20)", low_vix), ("Mid VIX (20-30)", mid_vix), ("High VIX (>30)", high_vix_scatter)]:
    subset = scatter_data[mask]
    if len(subset) > 10:
        beta = np.cov(subset["CNQ"], subset["USO"])[0, 1] / np.var(subset["USO"])
        corr_val = subset["CNQ"].corr(subset["USO"])
        print(f"{label}: Beta = {beta:.3f}, Correlation = {corr_val:.3f}, N = {len(subset)}")