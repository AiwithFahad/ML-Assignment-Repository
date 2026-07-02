"""
E-Commerce Customer Behavior Analysis - Interactive Streamlit Dashboard
Author: Data Analytics Portfolio Project

Run with: streamlit run ui/app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Customer Behavior Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }
    .kpi-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        border: 1px solid #2d3748;
    }
    .kpi-value { font-size: 28px; font-weight: 700; color: #38bdf8; }
    .kpi-label { font-size: 13px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
    h1, h2, h3 { color: #f1f5f9; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, "dataset", "cleaned_ecommerce_data.csv")
    raw_path = os.path.join(base_dir, "dataset", "Ecommerce_Customer_Behavior.csv")

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    else:
        # Fallback: clean on the fly if notebooks haven't been run yet
        df = pd.read_csv(raw_path)
        df = df.rename(columns={
            "customer_age": "age", "customer_gender": "gender",
            "product_category": "category", "order_value_usd": "total_spend",
            "delivery_time_days": "delivery_days", "customer_rating": "rating",
            "returned": "is_returned",
        })
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["is_returned"] = df["is_returned"].map({"Yes": 1, "No": 0})
        df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 100],
                                  labels=["18-25", "26-35", "36-45", "46-55", "56+"])
        df["satisfaction_level"] = pd.cut(df["rating"], bins=[0, 2, 3.5, 5],
                                           labels=["Low", "Medium", "High"], include_lowest=True)
    return df


df_full = load_data()

# --------------------------------------------------------------------------------
# Sidebar Filters
# --------------------------------------------------------------------------------
st.sidebar.title("🛒 Dashboard Filters")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Overview", "📈 Interactive Visualizations", "💡 Business Insights", "🔍 Customer Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

genders = st.sidebar.multiselect("Gender", options=sorted(df_full["gender"].unique()),
                                  default=sorted(df_full["gender"].unique()))
categories = st.sidebar.multiselect("Product Category", options=sorted(df_full["category"].unique()),
                                     default=sorted(df_full["category"].unique()))
payment_methods = st.sidebar.multiselect("Payment Method", options=sorted(df_full["payment_method"].unique()),
                                          default=sorted(df_full["payment_method"].unique()))
satisfaction_opts = [s for s in ["Low", "Medium", "High"] if s in df_full["satisfaction_level"].astype(str).unique()]
satisfaction = st.sidebar.multiselect("Satisfaction Level", options=satisfaction_opts, default=satisfaction_opts)
returned_filter = st.sidebar.multiselect("Order Returned?", options=["No", "Yes"], default=["No", "Yes"])

age_min, age_max = int(df_full["age"].min()), int(df_full["age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

# Apply filters
returned_map = {"No": 0, "Yes": 1}
returned_vals = [returned_map[r] for r in returned_filter]

df = df_full[
    (df_full["gender"].isin(genders)) &
    (df_full["category"].isin(categories)) &
    (df_full["payment_method"].isin(payment_methods)) &
    (df_full["satisfaction_level"].astype(str).isin(satisfaction)) &
    (df_full["is_returned"].isin(returned_vals)) &
    (df_full["age"].between(age_range[0], age_range[1]))
]

if df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
    st.stop()


def kpi_card(col, label, value):
    col.markdown(
        f"""<div class="kpi-card"><div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div></div>""",
        unsafe_allow_html=True,
    )


# ==================================================================================
# PAGE: HOME
# ==================================================================================
if page == "🏠 Home":
    st.title("🛒 E-Commerce Customer Behavior Dashboard")
    st.caption("An interactive exploratory data analysis dashboard built with Streamlit & Plotly")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Orders", f"{len(df):,}")
    kpi_card(c2, "Avg. Spend", f"${df['total_spend'].mean():,.2f}")
    kpi_card(c3, "Avg. Rating", f"{df['rating'].mean():.2f} ⭐")
    kpi_card(c4, "Total Revenue", f"${df['total_spend'].sum():,.0f}")
    kpi_card(c5, "Categories", f"{df['category'].nunique()}")

    st.markdown("###")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="category", color="category", title="Orders by Product Category")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.pie(df, names="gender", title="Customer Gender Split", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)


# ==================================================================================
# PAGE: DATA OVERVIEW
# ==================================================================================
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Column Data Types")
        st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with col2:
        st.subheader("Missing Values per Column")
        st.dataframe(df.isnull().sum().rename("missing_count"), use_container_width=True)

    st.subheader("Summary Statistics (Numerical)")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Summary Statistics (Categorical)")
    st.dataframe(df.describe(include="object"), use_container_width=True)


# ==================================================================================
# PAGE: INTERACTIVE VISUALIZATIONS
# ==================================================================================
elif page == "📈 Interactive Visualizations":
    st.title("📈 Interactive Visualizations")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Relationships", "Categorical Breakdown", "Correlation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            num_col = st.selectbox("Select numerical feature", ["age", "total_spend", "delivery_days", "rating"], key="hist")
            fig = px.histogram(df, x=num_col, nbins=30, marginal="box", color_discrete_sequence=["#38bdf8"],
                                title=f"Distribution of {num_col}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            num_col2 = st.selectbox("Select numerical feature (box plot)", ["age", "total_spend", "delivery_days", "rating"], key="box")
            fig = px.box(df, y=num_col2, color="gender", title=f"Box Plot of {num_col2} by Gender")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", ["age", "delivery_days", "rating"], key="scatter_x")
            fig = px.scatter(df, x=x_axis, y="total_spend", color="category", opacity=0.6,
                              title=f"{x_axis} vs Total Spend", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df, x="delivery_days", y="rating", color="satisfaction_level",
                              title="Delivery Time vs Rating", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            cat_feature = st.selectbox("Select categorical feature", ["category", "payment_method", "gender", "satisfaction_level"], key="count")
            fig = px.histogram(df, x=cat_feature, color=cat_feature, title=f"Count Plot - {cat_feature}")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            pie_feature = st.selectbox("Select categorical feature (pie)", ["category", "payment_method", "gender", "satisfaction_level"], key="pie")
            fig = px.pie(df, names=pie_feature, title=f"Pie Chart - {pie_feature}", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(df.groupby("category", observed=True)["total_spend"].mean().reset_index(),
                     x="category", y="total_spend", color="category", title="Average Spend by Category")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = ["age", "total_spend", "delivery_days", "rating", "is_returned"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================================
# PAGE: BUSINESS INSIGHTS
# ==================================================================================
elif page == "💡 Business Insights":
    st.title("💡 Business Insights")
    st.markdown("---")

    revenue_by_cat = df.groupby("category", observed=True)["total_spend"].sum().sort_values(ascending=False)
    top_category = revenue_by_cat.index[0]
    avg_spend_by_payment = df.groupby("payment_method", observed=True)["total_spend"].mean().sort_values(ascending=False)
    top_payment = avg_spend_by_payment.index[0]
    discount_corr = df["total_spend"].corr(df["rating"])
    top_gender = df["gender"].value_counts().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Top Revenue Category", top_category)
    kpi_card(c2, "Highest Avg. Spend Payment", top_payment)
    kpi_card(c3, "Most Common Gender", top_gender)
    kpi_card(c4, "Spend ↔ Rating Corr.", f"{discount_corr:.2f}")

    st.markdown("###")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Category")
        fig = px.bar(revenue_by_cat.reset_index(), x="category", y="total_spend", color="category",
                     title="Total Revenue by Product Category")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**{top_category}** generates the highest total revenue. Prioritize inventory and marketing spend here.")

    with col2:
        st.subheader("Customer Satisfaction Summary")
        sat_counts = df["satisfaction_level"].value_counts()
        fig = px.pie(values=sat_counts.values, names=sat_counts.index, title="Satisfaction Level Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Return Rate by Category")
        return_rate = df.groupby("category", observed=True)["is_returned"].mean().sort_values(ascending=False)
        fig = px.bar(return_rate.reset_index(), x="category", y="is_returned", color="category",
                     title="Return Rate by Category")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Spend by Gender")
        spend_gender = df.groupby("gender", observed=True)["total_spend"].mean().sort_values(ascending=False)
        fig = px.bar(spend_gender.reset_index(), x="gender", y="total_spend", color="gender",
                     title="Average Spend by Gender")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customer Segments (Category x Age Group)")
    seg = df.groupby(["category", "age_group"], observed=True).agg(
        avg_spend=("total_spend", "mean"), avg_rating=("rating", "mean"), orders=("order_id", "count")
    ).reset_index()
    seg["priority_score"] = (seg["avg_spend"].rank(pct=True) * 0.5 +
                              seg["avg_rating"].rank(pct=True) * 0.3 +
                              seg["orders"].rank(pct=True) * 0.2)
    st.dataframe(seg.sort_values("priority_score", ascending=False).head(10), use_container_width=True)
    st.success("These category/age-group combinations represent the highest-priority marketing targets, blending spend, satisfaction, and order volume.")


# ==================================================================================
# PAGE: CUSTOMER EXPLORER
# ==================================================================================
elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer / Order Explorer")
    st.markdown("---")
    st.caption("Drill into the filtered dataset using the sidebar, or refine further below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        explore_gender = st.selectbox("Gender", ["All"] + sorted(df["gender"].unique().tolist()))
    with col2:
        explore_category = st.selectbox("Category", ["All"] + sorted(df["category"].unique().tolist()))
    with col3:
        explore_payment = st.selectbox("Payment Method", ["All"] + sorted(df["payment_method"].unique().tolist()))

    explore_df = df.copy()
    if explore_gender != "All":
        explore_df = explore_df[explore_df["gender"] == explore_gender]
    if explore_category != "All":
        explore_df = explore_df[explore_df["category"] == explore_category]
    if explore_payment != "All":
        explore_df = explore_df[explore_df["payment_method"] == explore_payment]

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Filtered Orders", f"{len(explore_df):,}")
    kpi_card(c2, "Avg Spend", f"${explore_df['total_spend'].mean():,.2f}" if len(explore_df) else "$0")
    kpi_card(c3, "Avg Rating", f"{explore_df['rating'].mean():.2f}" if len(explore_df) else "N/A")
    kpi_card(c4, "Return Rate", f"{explore_df['is_returned'].mean()*100:.1f}%" if len(explore_df) else "N/A")

    st.markdown("###")
    if len(explore_df):
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(explore_df, x="total_spend", nbins=25, title="Spend Distribution (Filtered)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(explore_df, x="satisfaction_level", y="total_spend", color="satisfaction_level",
                         category_orders={"satisfaction_level": ["Low", "Medium", "High"]},
                         title="Spend by Satisfaction Level (Filtered)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered Dataset")
    st.dataframe(explore_df, use_container_width=True)
    st.download_button("⬇️ Download Filtered Data as CSV", explore_df.to_csv(index=False).encode("utf-8"),
                        "filtered_ecommerce_data.csv", "text/csv")

# --------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly · E-Commerce Customer Behavior EDA Project")
