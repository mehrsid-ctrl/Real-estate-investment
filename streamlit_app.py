# STREAMLIT APP — REAL ESTATE ADVISOR
# -------------------------------
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

st.set_page_config(page_title="🏠 Real Estate Investment Advisor", layout="centered")
st.title("🏠 Real Estate Investment Advisor")

# -------------------------------
# Load models
# -------------------------------
try:
    reg = CatBoostRegressor()
    clf = CatBoostClassifier()
    reg.load_model("reg_model.cbm")
    clf.load_model("clf_model.cbm")
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# -------------------------------
# User input
# -------------------------------
st.subheader("Property Information")

price = st.number_input("Current Price (Lakhs)", min_value=1.0, max_value=100000.0, value=50.0)
size = st.number_input("Size (sq ft)", min_value=100, max_value=20000, value=1000)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=2)
hosp = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=1)
pt = st.slider("Transport Accessibility", min_value=1, max_value=10, value=5)

furn = st.selectbox("Furnished Status", ["Unfurnished", "Semi", "Fully", "Unknown"])
ptype = st.selectbox("Property Type", ["Apartment", "House", "Villa", "Unknown"])
face = st.selectbox("Facing", ["North", "South", "East", "West", "Unknown"])
owner = st.selectbox("Owner Type", ["Builder", "Agent", "Individual", "Unknown"])
av = st.selectbox("Availability Status", ["Available", "Sold", "Under Construction", "Unknown"])
state = st.text_input("State", "Unknown")
city = st.text_input("City", "Unknown")
locality = st.text_input("Locality", "Unknown")

# -------------------------------
# Prepare input dataframe
# -------------------------------
input_df = pd.DataFrame([{
    "Price_in_Lakhs": price,
    "Size_in_SqFt": size,
    "BHK": bhk,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hosp,
    "Public_Transport_Accessibility": pt,
    "Furnished_Status": furn,
    "Property_Type": ptype,
    "Facing": face,
    "Owner_Type": owner,
    "Availability_Status": av,
    "State": state,
    "City": city,
    "Locality": locality,
    # Derived features
    "Price_per_SqFt": (price*100000)/max(size,1),
    "Age_of_Property": 5,  # placeholder
    "Price_per_BHK": price / max(bhk, 1)
}])

# Convert categorical columns to string (for CatBoost)
cat_cols = ["Furnished_Status","Property_Type","Facing","Owner_Type",
            "Availability_Status","State","City","Locality"]
for col in cat_cols:
    input_df[col] = input_df[col].astype(str)

# -------------------------------
# Predictions
# -------------------------------
try:
    future_price = reg.predict(input_df)[0]
    good_investment = clf.predict(input_df)[0]

    st.subheader("Predictions")
    st.success(f"Estimated Price in 5 years: {future_price:.2f} Lakhs")
    st.info(f"Good Investment? {'YES ✅' if good_investment==1 else 'NO ❌'}")

except Exception as e:
    st.error(f"Prediction failed: {e}")
