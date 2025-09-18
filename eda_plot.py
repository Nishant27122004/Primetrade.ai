import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

# Load the merged dataset
merged = pd.read_csv("D:/primetrade.ai/daily_with_sentiment.csv", parse_dates=["date"])

# --- Step 2: EDA ---

# 1. Time series plot of daily PnL vs sentiment
fig, ax1 = plt.subplots(figsize=(10,5))

ax1.set_title("Daily Total Closed PnL vs Sentiment Value")
ax1.plot(merged["date"], merged["total_closed_pnl"], color="tab:blue", marker="o", label="Total Closed PnL")
ax1.set_ylabel("Total Closed PnL", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(merged["date"], merged["sentiment_value"], color="tab:orange", marker="s", label="Sentiment Value")
ax2.set_ylabel("Sentiment Value", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.show()

# 2. Boxplot: PnL by sentiment classification
plt.figure(figsize=(8,5))
sns.boxplot(x="sentiment_class", y="total_closed_pnl", data=merged)
plt.title("Distribution of Daily PnL by Sentiment Classification")
plt.xticks(rotation=45)
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(8,6))
corr = merged[["total_trades","total_volume_usd","total_closed_pnl","win_rate","sentiment_value"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Trader Metrics and Sentiment")
plt.show()

# --- Step 3: Hypothesis Testing ---
# Compare PnL on Fear vs Greed days
fear_pnl = merged.loc[merged["sentiment_class"].str.contains("Fear", case=False, na=False), "total_closed_pnl"].dropna()
greed_pnl = merged.loc[merged["sentiment_class"].str.contains("Greed", case=False, na=False), "total_closed_pnl"].dropna()

# Run Mann-Whitney U test (non-parametric)
if len(fear_pnl) > 1 and len(greed_pnl) > 1:
    stat, p_val = mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
    test_result = {"test": "Mann-Whitney U", "statistic": stat, "p-value": p_val,
                   "mean_fear": fear_pnl.mean(), "mean_greed": greed_pnl.mean(),
                   "n_fear": len(fear_pnl), "n_greed": len(greed_pnl)}
else:
    test_result = {"error": "Not enough data for test", "n_fear": len(fear_pnl), "n_greed": len(greed_pnl)}

test_result