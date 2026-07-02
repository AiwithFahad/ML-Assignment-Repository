# 🛒 E-Commerce Customer Behavior Analysis — EDA Project

An end-to-end **Exploratory Data Analysis (EDA)** project on e-commerce order data, featuring six structured Jupyter notebooks, 20+ professional visualizations, statistical hypothesis testing, and an interactive **Streamlit dashboard**.

> This is a pure analytics/EDA project — **no machine learning models** are built. The focus is data cleaning, visualization, statistics, and business insight generation.

---

## 📌 Project Overview

The project analyzes 5,000 e-commerce order records to understand customer demographics, spending behavior, satisfaction, delivery performance, and return patterns — then translates those findings into concrete business recommendations through an interactive dashboard.

## 🎯 Problem Statement

E-commerce platforms generate large volumes of order data but often fail to translate it into actionable insight. This project answers:

- Which categories and customer segments drive the most revenue?
- Does demographic profile (age, gender) predict spending or satisfaction?
- How do delivery time and payment method relate to customer ratings and returns?
- Which segments should marketing prioritize?

## 🗂️ Dataset Description

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `customer_age` | Age of the purchasing customer |
| `customer_gender` | Gender (Male / Female / Other) |
| `product_category` | Product category purchased |
| `payment_method` | Payment method used |
| `order_value_usd` | Order value in USD |
| `delivery_time_days` | Delivery time in days |
| `customer_rating` | Customer satisfaction rating (1–5) |
| `returned` | Whether the order was returned (Yes/No) |
| `order_date` | Date of the order |

**Note:** Each row represents an individual **order/transaction** rather than a unique customer profile (there is no `customer_id` field), so "customer segment" analysis is performed at the order/demographic level.

After cleaning (`notebooks/02_Data_Cleaning.ipynb`), an enriched file `dataset/cleaned_ecommerce_data.csv` is produced with renamed columns and engineered features (`age_group`, `satisfaction_level`, `order_month`).

## 🛠️ Technologies Used

- Python 3
- Jupyter Notebook
- NumPy, Pandas
- Matplotlib, Seaborn, Plotly
- SciPy / Statsmodels (statistical testing)
- Streamlit (interactive dashboard)
- Git & GitHub

## 📁 Folder Structure

```
ECommerce_Customer_Behavior_Analysis/
│
├── dataset/
│   ├── Ecommerce_Customer_Behavior.csv      # raw data
│   └── cleaned_ecommerce_data.csv           # output of Notebook 02
│
├── notebooks/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Univariate_EDA.ipynb
│   ├── 04_Bivariate_EDA.ipynb
│   ├── 05_Multivariate_EDA.ipynb
│   └── 06_Business_Insights.ipynb
│
├── ui/
│   └── app.py                                # Streamlit dashboard
│
├── screenshots/                               # charts exported from notebooks
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ECommerce_Customer_Behavior_Analysis.git
cd ECommerce_Customer_Behavior_Analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🧹 Data Cleaning Process (Notebook 02)

1. Missing-value audit (none found) and duplicate check (none found).
2. Column renaming for readability (`order_value_usd` → `total_spend`, etc.).
3. Data-type correction (datetime parsing, categorical casting, binary encoding of returns).
4. Outlier detection via IQR method on all numerical columns.
5. Feature engineering: `age_group`, `satisfaction_level`, `order_month`.
6. Export of `cleaned_ecommerce_data.csv` for downstream analysis.

## 📊 Exploratory Data Analysis

- **Notebook 03 — Univariate Analysis:** Histograms, KDE, box, and violin plots for numerical features; count plots and pie charts for categorical features.
- **Notebook 04 — Bivariate Analysis:** Scatter plots, box plots, bar plots, and a correlation heatmap exploring pairwise relationships (age vs spend, category vs spend, payment vs spend, returns vs spend, etc.).
- **Notebook 05 — Multivariate Analysis:** Pair plots, full correlation matrices, GroupBy aggregations, and pivot-table heatmaps across 3+ dimensions simultaneously (e.g., category × age group × spend).

## 📈 Statistical Analysis

Includes mean, median, mode, standard deviation, variance, quartiles, Pearson correlation, one-way ANOVA, and IQR-based outlier detection — each interpreted in business terms within the notebooks.

## ❓ Business Questions Answered (Notebook 06)

1. Which product category generates the highest revenue?
2. Which gender purchases the most (by order volume)?
3. Does payment method correlate with spending behavior?
4. Which age group spends the most?
5. Which segment gives the highest satisfaction ratings?
6. Which customer segment is least satisfied?
7. Is satisfaction related to total spending?
8. Does delivery time affect satisfaction?
9. Which category/payment combination has the highest return rate?
10. Which customer segment should marketing target?

Each question is supported with a **visualization**, **statistical evidence**, and a **business recommendation**.

## 💡 Key Insights

- Revenue is concentrated in a handful of top product categories.
- Demographics (age, gender) are weak predictors of spend — behavioral and category signals matter more.
- Customer satisfaction is not strongly tied to order value; fulfillment factors (e.g., delivery time) are more influential.
- Specific category/payment-method combinations show elevated return rates and warrant quality-control review.
- A weighted scoring model (spend + satisfaction + volume) identifies the highest-priority segments for marketing investment.

## 🖥️ Streamlit Dashboard

The dashboard (`ui/app.py`) provides:

- **Sidebar filters:** Gender, Category, Payment Method, Satisfaction Level, Return Status, Age Range
- **🏠 Home:** KPI cards (orders, avg. spend, avg. rating, total revenue, category count) + quick visuals
- **📊 Data Overview:** Dataset preview, shape, dtypes, missing values, summary statistics
- **📈 Interactive Visualizations:** Tabbed Plotly charts — distributions, relationships, categorical breakdowns, correlation heatmap — all reactive to sidebar filters
- **💡 Business Insights:** Revenue by category, satisfaction summary, return-rate analysis, spend by gender, and a ranked top-segments table
- **🔍 Customer Explorer:** Drill-down filters with live KPIs, charts, and a CSV download of the filtered data

### Run the dashboard locally

```bash
streamlit run ui/app.py
```

## 🖼️ Screenshots

Chart images exported from the notebooks are saved automatically to the `screenshots/` folder during notebook execution (e.g., distribution plots, heatmaps, pair plots, business insight charts). Add dashboard screenshots here as well for the GitHub README gallery.

## 🚀 Future Improvements

- Incorporate a true `customer_id` (if available) to enable repeat-purchase / cohort / RTV analysis.
- Add time-series trend analysis on `order_month` for seasonality detection.
- Layer in a recommendation or churn-prediction model as a follow-up ML project.
- Add automated PDF/HTML report export from the dashboard.

## ☁️ Deployment Instructions (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select this repository and set the main file path to `ui/app.py`.
4. Deploy — Streamlit Cloud will automatically install `requirements.txt`.
5. Copy the generated public URL into this README once live.

**Live App:** _Add your deployed Streamlit URL here after deployment._

## ▶️ How to Run the Project

```bash
# Run notebooks in order
jupyter notebook notebooks/01_Data_Loading.ipynb
# ... through 06_Business_Insights.ipynb

# Launch the dashboard
streamlit run ui/app.py
```

## 📄 License

This project is released under the MIT License — feel free to use, modify, and share with attribution.
