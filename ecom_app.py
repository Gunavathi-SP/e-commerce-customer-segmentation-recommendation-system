"""
=============================================================
  Project 3 — E-Commerce Customer Behaviour & Recommendation
  Script 3: Streamlit Web App
=============================================================
Run:
    streamlit run ecom_app.py

Prerequisites:
    pip install streamlit plotly pandas scikit-learn
    Run ecom_ml.py first to generate ecom_models/
=============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .segment-card {
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .champion   { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .loyal      { background: linear-gradient(135deg, #3498db, #2980b9); }
    .atrisk     { background: linear-gradient(135deg, #f39c12, #e67e22); }
    .dormant    { background: linear-gradient(135deg, #95a5a6, #7f8c8d); }
    .rec-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .kpi-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛒 E-Commerce Customer Segmentation</h1>
    <p style="font-size:1.1rem; opacity:0.85;">
        RFM Analysis · KMeans Clustering · Product Recommendations · Live Segment Predictor
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
SEG_COLORS = {
    "💎 Champions"      : "#E74C3C",
    "🔥 Loyal Customers": "#3498DB",
    "⚠️  At-Risk Customers": "#F39C12",
    "💤 Dormant Customers" : "#95A5A6",
}

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
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
    return df

@st.cache_resource
def load_models():
    base = "ecom_models"
    try:
        with open(f"{base}/kmeans.pkl",    "rb") as f: kmeans     = pickle.load(f)
        with open(f"{base}/scaler.pkl",    "rb") as f: scaler     = pickle.load(f)
        with open(f"{base}/labels_map.pkl","rb") as f: labels_map = pickle.load(f)
        with open(f"{base}/meta.pkl",      "rb") as f: meta       = pickle.load(f)
        rfm = pd.read_csv(f"{base}/rfm_segmented.csv")
        with open(f"{base}/recommendations.pkl","rb") as f: recs  = pickle.load(f)
        return kmeans, scaler, labels_map, meta, rfm, recs
    except:
        return None, None, None, None, None, None

def build_rfm(df):
    snapshot = df["TransactionDate"].max() + pd.Timedelta(days=1)
    return df.groupby("CustomerID").agg(
        Recency         =("TransactionDate", lambda x: (snapshot - x.max()).days),
        Frequency       =("TransactionID",   "count"),
        Monetary        =("TotalSpend",      "sum"),
        AvgOrderValue   =("TotalSpend",      "mean"),
        AvgRating       =("Rating",          "mean"),
        ReturnRate      =("ReturnStatus",    lambda x: (x == "Returned").mean()),
        AvgSession      =("SessionDuration", "mean"),
        DiscountUsage   =("DiscountApplied", lambda x: (x == "Yes").mean()),
        UniqueCategories=("Category",        "nunique"),
        AvgQuantity     =("Quantity",        "mean"),
    ).reset_index()

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload ecommerce_segmentation.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "**Unsupervised ML Project**\n\n"
        "• RFM Feature Engineering\n"
        "• KMeans Clustering (K=4)\n"
        "• Segment Profiling\n"
        "• Product Recommendations\n"
        "• Live Customer Predictor"
    )

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
if uploaded:
    df = load_and_clean(uploaded)
else:
    default = "ecommerce_segmentation.csv"
    if os.path.exists(default):
        df = load_and_clean(default)
        st.info("📊 Using default `ecommerce_segmentation.csv`. Upload your own via sidebar.")
    else:
        st.warning("⬆️ Please upload `ecommerce_segmentation.csv`.")
        st.stop()

kmeans, scaler, labels_map, meta, rfm_seg, recs = load_models()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA & Insights",
    "🎯 Customer Segments",
    "🛍️ Recommendations",
    "🔮 Predict My Segment",
    "📋 Data Explorer",
])

# ═══════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════
with tab1:
    total_rev = df["TotalSpend"].sum()
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Transactions", f"{len(df):,}")
    k2.metric("Unique Customers",   f"{df['CustomerID'].nunique():,}")
    k3.metric("Total Revenue",      f"₹{total_rev/1e6:.2f}M")
    k4.metric("Avg Rating",         f"{df['Rating'].mean():.2f}")
    k5.metric("Return Rate",        f"{(df['ReturnStatus']=='Returned').mean()*100:.1f}%")

    st.markdown('<div class="section-header">Revenue & Transactions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        cat_rev = df.groupby("Category")["TotalSpend"].sum().reset_index()
        fig = px.pie(cat_rev, names="Category", values="TotalSpend",
                     title="Revenue by Category", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pay_cnt = df["PaymentType"].value_counts().reset_index()
        pay_cnt.columns = ["PaymentType","Count"]
        fig = px.bar(pay_cnt, x="PaymentType", y="Count",
                     color="PaymentType", title="Transactions by Payment Type",
                     text="Count", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Spend & Behaviour Analysis</div>', unsafe_allow_html=True)
    feat_sel = st.selectbox("Select feature to explore", ["Price","TotalSpend","Quantity","Rating","SessionDuration"])
    group_by = st.selectbox("Group by", ["Category","Device","Browser","ShippingType","ReturnStatus"])

    fig = px.box(df, x=group_by, y=feat_sel, color=group_by,
                 title=f"{feat_sel} by {group_by}",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Discount Impact</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, metric, label in zip(
        [col1, col2, col3],
        ["TotalSpend","Rating","Quantity"],
        ["Avg Spend (₹)","Avg Rating","Avg Quantity"]
    ):
        disc = df.groupby("DiscountApplied")[metric].mean().reset_index()
        fig = px.bar(disc, x="DiscountApplied", y=metric,
                     color="DiscountApplied", title=label,
                     color_discrete_map={"Yes":"#2ECC71","No":"#E74C3C"})
        fig.update_layout(showlegend=False)
        col.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Return Rate Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ret_cat = (df.groupby("Category")["ReturnStatus"]
                     .apply(lambda x: (x=="Returned").mean()*100)
                     .reset_index())
        ret_cat.columns = ["Category","ReturnRate"]
        fig = px.bar(ret_cat, x="Category", y="ReturnRate",
                     color="Category", title="Return Rate by Category (%)",
                     text=ret_cat["ReturnRate"].round(1).astype(str)+"%",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ret_ship = (df.groupby("ShippingType")["ReturnStatus"]
                      .apply(lambda x: (x=="Returned").mean()*100)
                      .reset_index())
        ret_ship.columns = ["ShippingType","ReturnRate"]
        fig = px.bar(ret_ship, x="ShippingType", y="ReturnRate",
                     color="ShippingType", title="Return Rate by Shipping Type (%)",
                     text=ret_ship["ReturnRate"].round(1).astype(str)+"%",
                     color_discrete_sequence=["#E74C3C","#3498DB"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — SEGMENTS
# ═══════════════════════════════════════════════════════
with tab2:
    if rfm_seg is None:
        st.warning("⚠️ Run `ecom_ml.py` first to generate segments.")
    else:
        # Segment cards
        seg_counts = rfm_seg["SegmentLabel"].value_counts()
        cols = st.columns(len(seg_counts))
        css_map = {
            "💎 Champions"         : "champion",
            "🔥 Loyal Customers"   : "loyal",
            "⚠️  At-Risk Customers" : "atrisk",
            "💤 Dormant Customers"  : "dormant",
        }
        for col, (seg, cnt) in zip(cols, seg_counts.items()):
            pct = cnt / len(rfm_seg) * 100
            css = css_map.get(seg, "loyal")
            col.markdown(f"""
            <div class="segment-card {css}">
                {seg}<br>
                <span style="font-size:1.8rem">{cnt}</span><br>
                <span style="font-size:0.85rem">{pct:.1f}% of customers</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Segment Visualisation</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if "PCA1" in rfm_seg.columns:
                fig = px.scatter(rfm_seg, x="PCA1", y="PCA2",
                                 color="SegmentLabel",
                                 color_discrete_map=SEG_COLORS,
                                 title="Customer Clusters — PCA 2D View",
                                 hover_data=["CustomerID","Monetary","Frequency","Recency"],
                                 opacity=0.75)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            seg_pie = rfm_seg["SegmentLabel"].value_counts().reset_index()
            seg_pie.columns = ["Segment","Count"]
            fig = px.pie(seg_pie, names="Segment", values="Count",
                         color="Segment", color_discrete_map=SEG_COLORS,
                         title="Segment Share", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">RFM Comparison by Segment</div>', unsafe_allow_html=True)
        rfm_metric = st.selectbox("Select metric", ["Monetary","Frequency","Recency","AvgRating","ReturnRate","AvgSession"])
        fig = px.box(rfm_seg, x="SegmentLabel", y=rfm_metric,
                     color="SegmentLabel", color_discrete_map=SEG_COLORS,
                     title=f"{rfm_metric} Distribution by Segment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Segment Profile Summary</div>', unsafe_allow_html=True)
        feature_cols = ["Recency","Frequency","Monetary","AvgOrderValue",
                        "AvgRating","ReturnRate","AvgSession","DiscountUsage"]
        profile = rfm_seg.groupby("SegmentLabel")[feature_cols].mean().round(2)
        st.dataframe(profile.style.background_gradient(cmap="RdYlGn_r", axis=0),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 3 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════
with tab3:
    if recs is None:
        st.warning("⚠️ Run `ecom_ml.py` first.")
    else:
        st.markdown("### 🛍️ Product Recommendations by Customer Segment")
        st.markdown("Each segment gets personalised product recommendations based on **revenue + average rating**.")

        seg_sel = st.selectbox("Select a Segment", list(recs.keys()))
        rec_df  = recs[seg_sel]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Top Products for {seg_sel}**")
            for i, row in rec_df.iterrows():
                stars = "⭐" * int(round(row["AvgRating"]))
                st.markdown(f"""
                <div class="rec-card">
                    <b>#{list(rec_df.index).index(i)+1} {row['Product']}</b><br>
                    Revenue: ₹{row['Revenue']:,.0f} &nbsp;|&nbsp; Rating: {row['AvgRating']:.1f} {stars}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            fig = px.bar(rec_df, x="Product", y="FinalScore",
                         color="AvgRating", color_continuous_scale="RdYlGn",
                         title=f"Recommendation Score — {seg_sel}",
                         text=rec_df["FinalScore"].round(2))
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_colorbar_title="Avg Rating")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 All Segments — Top Product")
        all_tops = []
        for seg, df_rec in recs.items():
            if not df_rec.empty:
                top = df_rec.iloc[0].copy()
                top["Segment"] = seg
                all_tops.append(top)

        top_df = pd.DataFrame(all_tops)[["Segment","Product","Revenue","AvgRating","FinalScore"]]
        st.dataframe(top_df.reset_index(drop=True), use_container_width=True)

        # Segment × Product heatmap
        st.markdown("### 🔥 Segment × Product Revenue Heatmap")
        if rfm_seg is not None:
            df_merged = df.merge(rfm_seg[["CustomerID","SegmentLabel"]], on="CustomerID", how="left")
            heat = df_merged.groupby(["SegmentLabel","Product"])["TotalSpend"].sum().unstack(fill_value=0)
            fig = px.imshow(heat, text_auto=".0f", aspect="auto",
                            color_continuous_scale="Blues",
                            title="Total Revenue: Segment × Product")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 4 — LIVE SEGMENT PREDICTOR
# ═══════════════════════════════════════════════════════
with tab4:
    if kmeans is None:
        st.warning("⚠️ Run `ecom_ml.py` first.")
    else:
        st.markdown("### 🔮 Predict Which Segment a Customer Belongs To")
        st.markdown("Enter customer behaviour details to find their segment and get product recommendations.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Purchase History**")
            recency   = st.slider("Days Since Last Purchase (Recency)", 1, 365, 30)
            frequency = st.slider("Number of Purchases (Frequency)", 1, 20, 5)
            monetary  = st.number_input("Total Amount Spent ₹ (Monetary)", 500, 500000, 15000, step=500)
            avg_order = st.number_input("Avg Order Value ₹", 100, 100000, 3000, step=100)

        with col2:
            st.markdown("**Engagement**")
            avg_rating    = st.slider("Average Rating Given", 1.0, 5.0, 3.5, step=0.1)
            return_rate   = st.slider("Return Rate (0 = never, 1 = always)", 0.0, 1.0, 0.2, step=0.05)
            avg_session   = st.slider("Avg Session Duration (min)", 1, 500, 200)
            discount_use  = st.slider("Discount Usage Rate (0-1)", 0.0, 1.0, 0.4, step=0.05)

        with col3:
            st.markdown("**Shopping Style**")
            unique_cats = st.slider("Unique Categories Purchased", 1, 5, 2)
            avg_qty     = st.slider("Avg Items per Order", 1.0, 4.0, 1.5, step=0.5)

        st.markdown("---")
        predict_btn = st.button("🔮 Find My Segment", use_container_width=True, type="primary")

        if predict_btn:
            input_data = np.array([[recency, frequency, monetary, avg_order,
                                    avg_rating, return_rate, avg_session,
                                    discount_use, unique_cats, avg_qty]])
            input_scaled  = scaler.transform(input_data)
            cluster_id    = kmeans.predict(input_scaled)[0]
            segment_label = labels_map[cluster_id]

            color = SEG_COLORS.get(segment_label, "#3498DB")

            st.markdown(f"""
            <div style="background:{color}; color:white; padding:1.5rem;
                        border-radius:12px; text-align:center; margin:1rem 0;">
                <h2 style="margin:0">You belong to: {segment_label}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Segment description
            desc_map = {
                "💎 Champions"         : "Your best customers! High spenders, frequent buyers, recent purchasers with great ratings.",
                "🔥 Loyal Customers"   : "Consistent buyers who trust your brand. Focus on upselling and loyalty rewards.",
                "⚠️  At-Risk Customers" : "Once active but showing disengagement. Re-engagement campaigns needed urgently.",
                "💤 Dormant Customers"  : "Low activity customers. Win-back offers and personalised nudges can help.",
            }
            st.info(desc_map.get(segment_label, ""))

            # Recommendations
            st.markdown("#### 🛍️ Recommended Products for You")
            if recs and segment_label in recs:
                rec_df = recs[segment_label]
                cols = st.columns(len(rec_df))
                for col, (_, row) in zip(cols, rec_df.iterrows()):
                    col.markdown(f"""
                    <div style="background:#f0f4f8; border-radius:8px; padding:0.8rem;
                                text-align:center; border-top: 4px solid {color};">
                        <b>{row['Product']}</b><br>
                        ⭐ {row['AvgRating']:.1f}<br>
                        <small>₹{row['Revenue']:,.0f} revenue</small>
                    </div>
                    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Transaction Data</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",     f"{df.shape[0]:,}")
    c2.metric("Columns",  df.shape[1])
    c3.metric("Customers",f"{df['CustomerID'].nunique():,}")
    c4.metric("Products", f"{df['Product'].nunique():,}")

    cat_f = st.selectbox("Filter by Category", ["All"] + df["Category"].dropna().unique().tolist())
    view  = df if cat_f == "All" else df[df["Category"] == cat_f]
    st.dataframe(view.head(300), use_container_width=True)

    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe(include="all").round(2), use_container_width=True)

    if rfm_seg is not None:
        st.markdown("**RFM Customer Table**")
        st.dataframe(rfm_seg.round(2), use_container_width=True)
