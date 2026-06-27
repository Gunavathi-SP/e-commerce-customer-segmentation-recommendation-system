"""
=============================================================
  Project 3 — E-Commerce Customer Behaviour & Recommendation
  Script 1: Exploratory Data Analysis (EDA)
=============================================================
Run:
    python ecom_eda.py
Output:
    ecom_eda_plots/  — folder with all saved PNG charts
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

OUT_DIR = "ecom_eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = ["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6"]

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("  E-COMMERCE SEGMENTATION — EDA")
print("=" * 55)

df_raw = pd.read_csv("ecommerce_segmentation.csv")
print(f"\nRaw shape : {df_raw.shape}")

df = df_raw.copy()

# Fix inconsistent labels
df["Country"]         = df["Country"].str.strip().replace({"IND": "India", "IN": "India"})
df["DiscountApplied"] = df["DiscountApplied"].str.strip().replace({"Y": "Yes", "N": "No"})

# Fill missing categoricals with mode
cat_cols = ["Category","PaymentType","ReturnStatus","ShippingType","Device","Browser","Country","DiscountApplied"]
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)

# Fill missing numeric
df["Rating"].fillna(df["Rating"].median(), inplace=True)

# Parse date
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
df["TransactionDate"].fillna(df["TransactionDate"].mode()[0], inplace=True)

# Derived columns
df["TotalSpend"]  = df["Price"] * df["Quantity"]
df["Month"]       = df["TransactionDate"].dt.month_name()
df["DayOfWeek"]   = df["TransactionDate"].dt.day_name()
df["IsDiscounted"]= (df["DiscountApplied"] == "Yes").astype(int)

print(f"Clean shape: {df.shape}")
print(f"Missing   : {df.isnull().sum().sum()}")
print(f"\nUnique Customers : {df['CustomerID'].nunique():,}")
print(f"Total Transactions: {len(df):,}")
print(f"Total Revenue     : ₹{df['TotalSpend'].sum():,.0f}")

# ─────────────────────────────────────────────────────────
# 2. PLOT 1 — Overview KPIs (bar summary)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("E-Commerce Overview", fontsize=15, fontweight="bold")

# Revenue by Category
cat_rev = df.groupby("Category")["TotalSpend"].sum().sort_values(ascending=False)
axes[0].bar(cat_rev.index, cat_rev.values, color=PALETTE[:len(cat_rev)], edgecolor="white")
axes[0].set_title("Revenue by Category")
axes[0].set_ylabel("Total Revenue (₹)")
for i, val in enumerate(cat_rev.values):
    axes[0].text(i, val + 5000, f"₹{val/1e6:.1f}M", ha="center", fontsize=9, fontweight="bold")

# Transactions by Payment Type
pay = df["PaymentType"].value_counts()
axes[1].bar(pay.index, pay.values, color=PALETTE[:len(pay)], edgecolor="white")
axes[1].set_title("Transactions by Payment Type")
axes[1].set_ylabel("Count")
for i, val in enumerate(pay.values):
    axes[1].text(i, val + 10, str(val), ha="center", fontsize=9, fontweight="bold")

# Device Usage
dev = df["Device"].value_counts()
axes[2].pie(dev.values, labels=dev.index, autopct="%1.1f%%",
            colors=PALETTE[:len(dev)],
            wedgeprops={"edgecolor":"white","linewidth":2})
axes[2].set_title("Device Usage")

plt.tight_layout()
save(fig, "01_overview.png")

# ─────────────────────────────────────────────────────────
# 3. PLOT 2 — Revenue & Transactions over Months
# ─────────────────────────────────────────────────────────
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

monthly = df.groupby("Month").agg(
    Revenue=("TotalSpend","sum"),
    Transactions=("TransactionID","count")
).reindex([m for m in month_order if m in df["Month"].unique()])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Monthly Trends", fontsize=15, fontweight="bold")

axes[0].plot(monthly.index, monthly["Revenue"], marker="o", color="#3498DB", lw=2.5)
axes[0].fill_between(range(len(monthly)), monthly["Revenue"].values, alpha=0.15, color="#3498DB")
axes[0].set_xticks(range(len(monthly)))
axes[0].set_xticklabels(monthly.index, rotation=45, ha="right")
axes[0].set_title("Monthly Revenue")
axes[0].set_ylabel("Revenue (₹)")

axes[1].bar(range(len(monthly)), monthly["Transactions"], color="#E74C3C", edgecolor="white")
axes[1].set_xticks(range(len(monthly)))
axes[1].set_xticklabels(monthly.index, rotation=45, ha="right")
axes[1].set_title("Monthly Transactions")
axes[1].set_ylabel("Count")

plt.tight_layout()
save(fig, "02_monthly_trends.png")

# ─────────────────────────────────────────────────────────
# 4. PLOT 3 — Price & Spend Distributions
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Spend & Price Distributions", fontsize=15, fontweight="bold")

for ax, col, color, label in zip(
    axes,
    ["Price","Quantity","TotalSpend"],
    ["#3498DB","#2ECC71","#E74C3C"],
    ["Price (₹)","Quantity","Total Spend (₹)"]
):
    sns.histplot(df[col], bins=30, ax=ax, color=color, edgecolor="white")
    ax.axvline(df[col].median(), color="black", linestyle="--", lw=1.5, label=f"Median={df[col].median():.0f}")
    ax.set_title(label)
    ax.set_xlabel("")
    ax.legend()

plt.tight_layout()
save(fig, "03_spend_distributions.png")

# ─────────────────────────────────────────────────────────
# 5. PLOT 4 — Rating Distribution & by Category
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Customer Ratings", fontsize=15, fontweight="bold")

sns.histplot(df["Rating"], bins=20, ax=axes[0], color="#9B59B6", edgecolor="white")
axes[0].set_title("Overall Rating Distribution")
axes[0].set_xlabel("Rating")

cat_rating = df.groupby("Category")["Rating"].mean().sort_values(ascending=False)
axes[1].barh(cat_rating.index, cat_rating.values,
             color=PALETTE[:len(cat_rating)], edgecolor="white")
for i, val in enumerate(cat_rating.values):
    axes[1].text(val + 0.02, i, f"{val:.2f}", va="center", fontsize=10)
axes[1].set_xlim(0, 6)
axes[1].set_title("Avg Rating by Category")
axes[1].set_xlabel("Average Rating")

plt.tight_layout()
save(fig, "04_ratings.png")

# ─────────────────────────────────────────────────────────
# 6. PLOT 5 — Return Rate Analysis
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Return Rate Analysis", fontsize=15, fontweight="bold")

# Overall return rate
ret = df["ReturnStatus"].value_counts()
axes[0].pie(ret.values, labels=ret.index, autopct="%1.1f%%",
            colors=["#E74C3C","#2ECC71"],
            wedgeprops={"edgecolor":"white","linewidth":2})
axes[0].set_title("Overall Return Rate")

# Return rate by Category
for ax, col in zip(axes[1:], ["Category","ShippingType"]):
    rate = (df.groupby(col)["ReturnStatus"]
              .apply(lambda x: (x == "Returned").mean() * 100)
              .sort_values(ascending=False))
    ax.bar(rate.index, rate.values,
           color=PALETTE[:len(rate)], edgecolor="white")
    ax.set_title(f"Return Rate by {col}")
    ax.set_ylabel("Return Rate (%)")
    for i, val in enumerate(rate.values):
        ax.text(i, val + 0.3, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
save(fig, "05_returns.png")

# ─────────────────────────────────────────────────────────
# 7. PLOT 6 — Discount Impact
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Discount Impact on Behaviour", fontsize=15, fontweight="bold")

for ax, metric, label in zip(
    axes,
    ["TotalSpend","Rating","Quantity"],
    ["Avg Spend (₹)","Avg Rating","Avg Quantity"]
):
    disc_grp = df.groupby("DiscountApplied")[metric].mean()
    bars = ax.bar(disc_grp.index, disc_grp.values,
                  color=["#E74C3C","#2ECC71"], edgecolor="white", width=0.4)
    for bar, val in zip(bars, disc_grp.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + disc_grp.max()*0.01,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"Discount vs {label}")
    ax.set_ylabel(label)
    ax.set_xlabel("Discount Applied")

plt.tight_layout()
save(fig, "06_discount_impact.png")

# ─────────────────────────────────────────────────────────
# 8. PLOT 7 — Top Products & Categories heatmap
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Product & Category Insights", fontsize=15, fontweight="bold")

# Top 5 products by revenue
top_prod = (df.groupby("Product")["TotalSpend"].sum()
              .sort_values(ascending=False).head(5))
axes[0].barh(top_prod.index[::-1], top_prod.values[::-1],
             color=PALETTE, edgecolor="white")
axes[0].set_title("Top 5 Products by Revenue")
axes[0].set_xlabel("Revenue (₹)")
for i, val in enumerate(top_prod.values[::-1]):
    axes[0].text(val + 1000, i, f"₹{val/1e3:.0f}K", va="center", fontsize=9)

# Category × Device heatmap
cat_dev = df.groupby(["Category","Device"]).size().unstack(fill_value=0)
sns.heatmap(cat_dev, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=axes[1])
axes[1].set_title("Transactions: Category × Device")
axes[1].set_xlabel("Device")
axes[1].set_ylabel("Category")

plt.tight_layout()
save(fig, "07_products_heatmap.png")

# ─────────────────────────────────────────────────────────
# 9. PLOT 8 — Customer-level RFM distributions
# ─────────────────────────────────────────────────────────
snapshot = df["TransactionDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency    =("TransactionDate", lambda x: (snapshot - x.max()).days),
    Frequency  =("TransactionID", "count"),
    Monetary   =("TotalSpend", "sum"),
    AvgRating  =("Rating", "mean"),
    ReturnRate =("ReturnStatus", lambda x: (x == "Returned").mean()),
    AvgSession =("SessionDuration", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Customer-Level RFM Distributions", fontsize=15, fontweight="bold")

for ax, col, color in zip(axes, ["Recency","Frequency","Monetary"],
                           ["#3498DB","#E74C3C","#2ECC71"]):
    sns.histplot(rfm[col], bins=25, ax=ax, color=color, edgecolor="white")
    ax.axvline(rfm[col].median(), color="black", linestyle="--",
               lw=1.5, label=f"Median={rfm[col].median():.0f}")
    ax.set_title(f"{col} Distribution")
    ax.set_xlabel(col)
    ax.legend()

plt.tight_layout()
save(fig, "08_rfm_distributions.png")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EDA COMPLETE — Saved 8 plots to:", OUT_DIR)
print("=" * 55)
print(f"\n  Customers    : {df['CustomerID'].nunique():,}")
print(f"  Transactions : {len(df):,}")
print(f"  Total Revenue: ₹{df['TotalSpend'].sum():,.0f}")
print(f"  Avg Rating   : {df['Rating'].mean():.2f}")
print(f"  Return Rate  : {(df['ReturnStatus']=='Returned').mean()*100:.1f}%")
print(f"\n  ➜  Run ecom_ml.py next")
