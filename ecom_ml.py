"""
=============================================================
  Project 3 — E-Commerce Customer Behaviour & Recommendation
  Script 2: Unsupervised ML — Customer Segmentation + Recommendations
=============================================================
Run:
    python ecom_ml.py
Output:
    ecom_ml_plots/  — clustering & evaluation charts
    ecom_models/    — saved model + segment profiles
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing    import StandardScaler
from sklearn.cluster          import KMeans
from sklearn.decomposition    import PCA
from sklearn.metrics          import silhouette_score, davies_bouldin_score
from sklearn.pipeline         import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUT_DIR   = "ecom_ml_plots"
MODEL_DIR = "ecom_models"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEGMENT_COLORS = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("  E-COMMERCE — SEGMENTATION ML PIPELINE")
print("=" * 55)

df = pd.read_csv("ecommerce_segmentation.csv")

df["Country"]         = df["Country"].str.strip().replace({"IND":"India","IN":"India"})
df["DiscountApplied"] = df["DiscountApplied"].str.strip().replace({"Y":"Yes","N":"No"})

cat_cols = ["Category","PaymentType","ReturnStatus","ShippingType",
            "Device","Browser","Country","DiscountApplied"]
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)

df["Rating"].fillna(df["Rating"].median(), inplace=True)
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
df["TransactionDate"].fillna(df["TransactionDate"].mode()[0], inplace=True)
df["TotalSpend"] = df["Price"] * df["Quantity"]

# ─────────────────────────────────────────────────────────
# 2. CUSTOMER-LEVEL FEATURE ENGINEERING (RFM + Behaviour)
# ─────────────────────────────────────────────────────────
print("\nBuilding customer-level features...")

snapshot = df["TransactionDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency       =("TransactionDate", lambda x: (snapshot - x.max()).days),
    Frequency     =("TransactionID",   "count"),
    Monetary      =("TotalSpend",      "sum"),
    AvgOrderValue =("TotalSpend",      "mean"),
    AvgRating     =("Rating",          "mean"),
    ReturnRate    =("ReturnStatus",    lambda x: (x == "Returned").mean()),
    AvgSession    =("SessionDuration", "mean"),
    DiscountUsage =("DiscountApplied", lambda x: (x == "Yes").mean()),
    UniqueCategories=("Category",      "nunique"),
    AvgQuantity   =("Quantity",        "mean"),
).reset_index()

print(f"  Customer features shape: {rfm.shape}")
print(rfm.describe().round(2))

# ─────────────────────────────────────────────────────────
# 3. FIND OPTIMAL K — Elbow + Silhouette
# ─────────────────────────────────────────────────────────
feature_cols = ["Recency","Frequency","Monetary","AvgOrderValue",
                "AvgRating","ReturnRate","AvgSession","DiscountUsage",
                "UniqueCategories","AvgQuantity"]

X = rfm[feature_cols].copy()

# Fill any NaNs (e.g. Recency if TransactionDate was all NaT)
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFinding optimal number of clusters...")
inertias, silhouettes, db_scores = [], [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    print(f"  K={k} | Inertia={km.inertia_:.0f} | Silhouette={silhouettes[-1]:.3f} | DB={db_scores[-1]:.3f}")

# ─────────────────────────────────────────────────────────
# 4. PLOT A — Elbow + Silhouette
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Finding Optimal Number of Clusters", fontsize=14, fontweight="bold")

axes[0].plot(K_range, inertias, marker="o", color="#3498DB", lw=2.5)
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")

axes[1].plot(K_range, silhouettes, marker="o", color="#2ECC71", lw=2.5)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")

axes[2].plot(K_range, db_scores, marker="o", color="#E74C3C", lw=2.5)
axes[2].set_title("Davies-Bouldin Score (lower = better)")
axes[2].set_xlabel("Number of Clusters (K)")
axes[2].set_ylabel("DB Score")

plt.tight_layout()
save(fig, "01_optimal_k.png")

# ─────────────────────────────────────────────────────────
# 5. TRAIN FINAL MODEL (K=4)
# ─────────────────────────────────────────────────────────
BEST_K = 4
print(f"\nTraining final KMeans with K={BEST_K}...")

kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
rfm["Segment"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, rfm["Segment"])
db  = davies_bouldin_score(X_scaled, rfm["Segment"])
print(f"  Silhouette Score : {sil:.4f}")
print(f"  Davies-Bouldin   : {db:.4f}")

# ─────────────────────────────────────────────────────────
# 6. LABEL SEGMENTS based on cluster profiles
# ─────────────────────────────────────────────────────────
profile = rfm.groupby("Segment")[feature_cols].mean()
print("\nCluster Profiles:")
print(profile.round(2))

# Auto-label: highest Monetary + low Recency = Champions
labels_map = {}
ranked = profile.copy()

for seg in range(BEST_K):
    r = ranked.loc[seg,"Recency"]
    m = ranked.loc[seg,"Monetary"]
    f = ranked.loc[seg,"Frequency"]
    ret = ranked.loc[seg,"ReturnRate"]

    if m == ranked["Monetary"].max():
        labels_map[seg] = "💎 Champions"
    elif r == ranked["Recency"].min():
        labels_map[seg] = "🔥 Loyal Customers"
    elif ret == ranked["ReturnRate"].max():
        labels_map[seg] = "⚠️  At-Risk Customers"
    else:
        labels_map[seg] = "💤 Dormant Customers"

# Fallback if duplicates
used = list(labels_map.values())
defaults = ["💎 Champions","🔥 Loyal Customers","⚠️  At-Risk Customers","💤 Dormant Customers"]
for seg in range(BEST_K):
    if seg not in labels_map:
        for d in defaults:
            if d not in used:
                labels_map[seg] = d
                used.append(d)
                break

rfm["SegmentLabel"] = rfm["Segment"].map(labels_map)
print("\nSegment Labels:")
print(rfm["SegmentLabel"].value_counts())

# ─────────────────────────────────────────────────────────
# 7. PLOT B — PCA 2D Scatter
# ─────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
rfm["PCA1"] = X_pca[:, 0]
rfm["PCA2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 7))
for (seg, label), color in zip(labels_map.items(), SEGMENT_COLORS):
    mask = rfm["Segment"] == seg
    ax.scatter(rfm.loc[mask,"PCA1"], rfm.loc[mask,"PCA2"],
               c=color, label=label, s=60, alpha=0.7, edgecolors="white", lw=0.5)

ax.set_title("Customer Segments — PCA Projection (2D)", fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(fontsize=11, framealpha=0.9)
save(fig, "02_pca_clusters.png")

# ─────────────────────────────────────────────────────────
# 8. PLOT C — Segment Size
# ─────────────────────────────────────────────────────────
seg_counts = rfm["SegmentLabel"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Segment Distribution", fontsize=14, fontweight="bold")

axes[0].pie(seg_counts.values, labels=seg_counts.index,
            autopct="%1.1f%%", colors=SEGMENT_COLORS[:len(seg_counts)],
            wedgeprops={"edgecolor":"white","linewidth":2})
axes[0].set_title("Customer Share by Segment")

axes[1].barh(seg_counts.index[::-1], seg_counts.values[::-1],
             color=SEGMENT_COLORS[:len(seg_counts)], edgecolor="white")
axes[1].set_title("Customer Count by Segment")
axes[1].set_xlabel("Number of Customers")
for i, val in enumerate(seg_counts.values[::-1]):
    axes[1].text(val + 2, i, str(val), va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
save(fig, "03_segment_distribution.png")

# ─────────────────────────────────────────────────────────
# 9. PLOT D — Radar / Spider Chart per Segment
# ─────────────────────────────────────────────────────────
radar_cols = ["Frequency","Monetary","AvgRating","AvgSession","DiscountUsage"]
radar_profile = rfm.groupby("SegmentLabel")[radar_cols].mean()

# Normalize 0-1 for radar
radar_norm = (radar_profile - radar_profile.min()) / (radar_profile.max() - radar_profile.min())
N = len(radar_cols)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(1, len(radar_profile), figsize=(18, 5),
                          subplot_kw=dict(polar=True))
fig.suptitle("Segment Radar Profiles", fontsize=14, fontweight="bold")

for ax, (seg_label, row), color in zip(axes, radar_norm.iterrows(), SEGMENT_COLORS):
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, color=color, lw=2)
    ax.fill(angles, vals, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_cols, size=8)
    ax.set_title(seg_label, size=10, fontweight="bold", pad=15)
    ax.set_ylim(0, 1)

plt.tight_layout()
save(fig, "04_radar_profiles.png")

# ─────────────────────────────────────────────────────────
# 10. PLOT E — RFM Box Plots per Segment
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("RFM Distribution by Segment", fontsize=14, fontweight="bold")

seg_palette = {label: color for label, color in zip(seg_counts.index, SEGMENT_COLORS)}

for ax, metric in zip(axes, ["Recency","Frequency","Monetary"]):
    order = rfm.groupby("SegmentLabel")[metric].median().sort_values().index
    sns.boxplot(data=rfm, x="SegmentLabel", y=metric, ax=ax,
                order=order, palette=seg_palette)
    ax.set_title(f"{metric} by Segment")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
save(fig, "05_rfm_by_segment.png")

# ─────────────────────────────────────────────────────────
# 11. PRODUCT RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────
print("\nBuilding recommendation engine...")

# Merge segment labels back into transaction data
df = df.merge(rfm[["CustomerID","SegmentLabel"]], on="CustomerID", how="left")

def get_recommendations(segment_label, top_n=5):
    """Return top N products for a segment based on revenue + rating score."""
    seg_df = df[df["SegmentLabel"] == segment_label]
    scored = (
        seg_df.groupby("Product")
        .agg(Revenue=("TotalSpend","sum"),
             Frequency=("TransactionID","count"),
             AvgRating=("Rating","mean"))
        .reset_index()
    )
    # Composite score: normalize revenue + rating
    scored["RevScore"]    = (scored["Revenue"]   - scored["Revenue"].min()) / (scored["Revenue"].max()   - scored["Revenue"].min() + 1e-9)
    scored["RatingScore"] = (scored["AvgRating"] - scored["AvgRating"].min()) / (scored["AvgRating"].max() - scored["AvgRating"].min() + 1e-9)
    scored["FinalScore"]  = 0.6 * scored["RevScore"] + 0.4 * scored["RatingScore"]
    return scored.sort_values("FinalScore", ascending=False).head(top_n)

print("\nTop Product Recommendations by Segment:")
recommendations = {}
for seg in rfm["SegmentLabel"].unique():
    recs = get_recommendations(seg)
    recommendations[seg] = recs
    print(f"\n  {seg}:")
    print(recs[["Product","Revenue","AvgRating","FinalScore"]].to_string(index=False))

# Save recommendations
with open(os.path.join(MODEL_DIR, "recommendations.pkl"), "wb") as f:
    pickle.dump(recommendations, f)

# ─────────────────────────────────────────────────────────
# 12. SAVE MODELS & DATA
# ─────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "kmeans.pkl"),    "wb") as f: pickle.dump(kmeans, f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"),    "wb") as f: pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "labels_map.pkl"),"wb") as f: pickle.dump(labels_map, f)

rfm.to_csv(os.path.join(MODEL_DIR, "rfm_segmented.csv"), index=False)

meta = {
    "feature_cols"  : feature_cols,
    "best_k"        : BEST_K,
    "silhouette"    : sil,
    "labels_map"    : labels_map,
}
with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f: pickle.dump(meta, f)

print("\n" + "=" * 55)
print("  SEGMENTATION COMPLETE")
print("=" * 55)
print(f"  Clusters         : {BEST_K}")
print(f"  Silhouette Score : {sil:.4f}")
print(f"  Davies-Bouldin   : {db:.4f}")
print(f"  Saved to         : {MODEL_DIR}/")
print("\n  ➜  Run: streamlit run ecom_app.py")
