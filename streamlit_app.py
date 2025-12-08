# STREAMLIT APP — REAL ESTATE ADVISOR
# -------------------------------
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

st.title("🏠 Real Estate Investment Advisor")

# -------------------------------
# Load models
# -------------------------------
try:
    reg = CatBoostRegressor()
    reg.load_model("reg_model.cbm")
    
    clf = CatBoostClassifier()
    clf.load_model("clf_model.cbm")
    
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# -------------------------------
# Allowed categorical values (from training)
# -------------------------------
allowed_values = {
    "Furnished_Status": ["Unfurnished", "Semi", "Fully", "Unknown"],
    "Property_Type": ["Apartment", "House", "Villa", "Unknown"],
    "Facing": ["North", "South", "East", "West", "Unknown"],
    "Owner_Type": ["Builder", "Agent", "Individual", "Unknown"],
    "Availability_Status": ["Available", "Sold", "Under Construction", "Unknown"],
}

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

furn = st.selectbox("Furnished Status", allowed_values["Furnished_Status"])
ptype = st.selectbox("Property Type", allowed_values["Property_Type"])
face = st.selectbox("Facing", allowed_values["Facing"])
owner = st.selectbox("Owner Type", allowed_values["Owner_Type"])
av = st.selectbox("Availability Status", allowed_values["Availability_Status"])
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
    "Price_per_SqFt": (price*100000)/size,
    "Age_of_Property": 5,  # placeholder
    "Price_per_BHK": price / max(bhk, 1)
}])

# -------------------------------
# Safe categorical mapping
# -------------------------------
def safe_cat(value, allowed_list):
    return value if value in allowed_list else allowed_list[0]

for col, allowed in allowed_values.items():
    input_df[col] = input_df[col].apply(lambda x: safe_cat(x, allowed))
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
