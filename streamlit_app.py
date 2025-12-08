# streamlit_app.py
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

st.set_page_config(page_title="🏠 Real Estate Investment Advisor", layout="wide")
st.title("🏠 Real Estate Investment Advisor")

# -------------------------------
# Load models from repo folder
# -------------------------------
MODEL_PATH_REG = "models/reg_model.cbm"
MODEL_PATH_CLF = "models/clf_model.cbm"

try:
    reg = CatBoostRegressor()
    reg.load_model(MODEL_PATH_REG)
    clf = CatBoostClassifier()
    clf.load_model(MODEL_PATH_CLF)
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# -------------------------------
# User input
# -------------------------------
st.subheader("Property Information")

col1, col2, col3 = st.columns(3)
with col1:
    price = st.number_input("Current Price (Lakhs)", 1.0, 100000.0, 50.0)
    size = st.number_input("Size (sq ft)", 100, 20000, 1000)
    bhk = st.number_input("BHK", 1, 10, 2)
    schools = st.number_input("Nearby Schools", 0, 20, 2)
with col2:
    hosp = st.number_input("Nearby Hospitals", 0, 20, 1)
    pt = st.slider("Transport Accessibility", 1, 10, 5)
    furn = st.selectbox("Furnished Status", ["Unfurnished", "Semi", "Fully", "Unknown"])
    ptype = st.selectbox("Property Type", ["Apartment", "House", "Villa", "Unknown"])
with col3:
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
    "Price_per_SqFt": (price*100000)/size,
    "Age_of_Property": 5,  # placeholder
    "Price_per_BHK": price / max(bhk, 1)
}])

# Convert categorical columns to string
cat_cols = ["Furnished_Status","Property_Type","Facing","Owner_Type","Availability_Status",
            "State","City","Locality"]
for col in cat_cols:
    input_df[col] = input_df[col].astype(str)

# -------------------------------
# Show user input
# -------------------------------
st.subheader("Your Input")
st.dataframe(input_df)

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
