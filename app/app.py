import streamlit as st
import pandas as pd
import joblib
import sys, os
from huggingface_hub import hf_hub_download  # NEW

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_next_month

st.set_page_config(page_title="Meditrak - Demand Forecasting", layout="wide")

st.title("Meditrak — AI Inventory Forecasting Module")
st.caption("Predictive demand forecasting for multi-store pharmacy inventory")

# --- NEW: Download model from Hugging Face (free hosting) ---
REPO_ID = "somanpatrasom/meditrak-demand-model"  # <-- change "yourname" to your HF username

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="demand_model.pkl",
        local_dir="models"
    )
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_columns"]

model, feature_columns = load_model()
# --- END NEW ---

# --- Load or generate forecast ---
@st.cache_data
def load_forecast():
    path = "data/predicted_inventory_next_month.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return predict_next_month()

inventory = load_forecast()

# --- Sidebar filters ---
st.sidebar.header("Filters")
store_filter = st.sidebar.selectbox("Select Store", ["All"] + sorted(inventory["store_id"].unique()))
category_filter = st.sidebar.selectbox("Select Category", ["All"] + sorted(inventory["category"].unique()))

filtered = inventory.copy()
if store_filter != "All":
    filtered = filtered[filtered["store_id"] == store_filter]
if category_filter != "All":
    filtered = filtered[filtered["category"] == category_filter]

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Predicted Units (Next 30 Days)", f"{filtered['predicted_monthly_demand'].sum():,.0f}")
col2.metric("Items Tracked", filtered["item_id"].nunique())
col3.metric("Stores Covered", filtered["store_id"].nunique())

st.divider()

# --- Reorder alert threshold ---
st.subheader("Recommended Replenishment List")
threshold = st.slider("Flag items predicted to sell above this many units next month", 0, 500, 100)

reorder_list = filtered[filtered["predicted_monthly_demand"] >= threshold].sort_values(
    "predicted_monthly_demand", ascending=False
)
st.dataframe(reorder_list, use_container_width=True)

st.download_button(
    "Download Replenishment List (CSV)",
    reorder_list.to_csv(index=False),
    file_name="replenishment_list.csv"
)

st.divider()

# --- Chart: Top items by predicted demand ---
st.subheader("Top 10 Medicines by Predicted Demand")
top10 = filtered.groupby("item_id")["predicted_monthly_demand"].sum().nlargest(10)
st.bar_chart(top10)

# --- Model performance section ---
st.divider()
st.subheader("Model Performance")
report_path = "models/performance_report.txt"
if os.path.exists(report_path):
    with open(report_path, "r", encoding = "utf-8") as f:
        st.text(f.read())
else:
    st.info("Run train_model.py first to generate the performance report.")

if os.path.exists("models/feature_importance.png"):
    st.subheader("What Drives Demand (Trend Analysis Model)")
    st.image("models/feature_importance.png")