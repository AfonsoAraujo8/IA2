import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("model.pkl")

# -----------------------------
# Page title
# -----------------------------

st.set_page_config(page_title="E-commerce Conversion Predictor")

st.title("🛒 E-commerce Conversion Predictor")

st.write(
    "Predict whether a customer is likely to complete a purchase."
)

# -----------------------------
# User Inputs
# -----------------------------

age = st.slider("Age", 18, 65, 30)

previous_purchases = st.slider("Previous Purchases", 0, 20, 3)

traffic_source = st.selectbox(
    "Traffic Source",
    ["Organic", "Ads", "Social Media", "Email"]
)

time_on_site = st.slider("Time on Site (minutes)", 1.0, 30.0, 8.0)

cart_value = st.slider("Cart Value (€)", 5.0, 500.0, 120.0)

viewed_reviews = st.selectbox(
    "Viewed Reviews",
    [0, 1]
)

pages_visited = st.slider("Pages Visited", 1, 30, 10)

device = st.selectbox(
    "Device",
    ["Mobile", "Desktop", "Tablet"]
)

added_to_cart = st.selectbox(
    "Added to Cart",
    [0, 1]
)

# -----------------------------
# Create DataFrame
# -----------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "PreviousPurchases": [previous_purchases],
    "TrafficSource": [traffic_source],
    "TimeOnSite": [time_on_site],
    "CartValue": [cart_value],
    "ViewedReviews": [viewed_reviews],
    "PagesVisited": [pages_visited],
    "Device": [device],
    "AddedToCart": [added_to_cart]
})

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Conversion"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Likely to Purchase")

    else:
        st.error("❌ Unlikely to Purchase")

    st.write(f"Conversion Probability: {probability:.2%}")

    st.write("Customer Data:")
    st.dataframe(input_data)