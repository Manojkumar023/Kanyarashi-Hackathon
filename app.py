import streamlit as st
import numpy as np
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="California House Price Prediction",
    layout="wide"
)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Input House Features")

longitude = st.sidebar.slider("Longitude", -125, -113, -120)
latitude = st.sidebar.slider("Latitude", 32, 43, 36)
housing_median_age = st.sidebar.slider("Median Age of House", 1, 52, 20)
total_rooms = st.sidebar.slider("Total Rooms", 2, 40000, 1500)
total_bedrooms = st.sidebar.slider("Total Bedrooms", 1, 6500, 300)
population = st.sidebar.slider("Population", 3, 35000, 1500)
households = st.sidebar.slider("Households", 1, 6000, 500)
median_income = st.sidebar.slider("Median Income (in 10k USD)", 0.5, 15.0, 4.0)

# -------------------------
# Dummy ocean_proximity encoding
# (replace with your real encoding)
# -------------------------
ocean_proximity_encoded = 2   # INLAND placeholder

# -------------------------
# Load Saved Models
# -------------------------
try:
    models = joblib.load("my_models.pkl")
    lr_model = models.get("linear_regression")
    xgb_model = models.get("xgboost")
except Exception as e:
    lr_model = None
    xgb_model = None
    st.error(f"⚠️ Could not load models: {e}")

# -------------------------
# Create input array
# -------------------------
input_data = np.array([
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity_encoded
]).reshape(1, -1)

# -------------------------
# Prediction Button
# -------------------------
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("Calculate House Price", use_container_width=True):

        if xgb_model is None or lr_model is None:
            st.error("Models not loaded. Please upload correctly.")
        else:
            pred_lr = lr_model.predict(input_data)[0]
            pred_xgb = xgb_model.predict(input_data)[0]

            st.success(f"📌 Linear Regression Estimate: **${pred_lr:,.0f}**")
            st.success(f"🚀 XGBoost Estimate: **${pred_xgb:,.0f}**")

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <p style="text-align:center; margin-top:50px;">
        Made with <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </p>
    """,
    unsafe_allow_html=True
)
