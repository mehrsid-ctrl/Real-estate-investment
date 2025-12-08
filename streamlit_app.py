import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

# Load models
reg = CatBoostRegressor()
reg.load_model("reg_model.cbm")

clf = CatBoostClassifier()
clf.load_model("clf_model.cbm")

st.title("🏠 Real Estate Investment Predictor")

# Example inputs
bhk = st.number_input("BHK", 1, 10, 2)
size = st.number_input("Size (sq ft)", 100, 20000, 1000)
price = st.number_input("Price (Lakhs)", 1, 100000, 50)

if st.button("Predict"):
    # Build input row (update with all features)
    row = pd.DataFrame({
        "BHK": [bhk],
        "Size_in_SqFt": [size],
        "Price_in_Lakhs": [price],
        # add other numeric/categorical features here
    })

    future_price = reg.predict(row)[0]
    good_investment = clf.predict(row)[0]

    st.success(f"Estimated price in 5 years: {future_price:.2f} Lakhs")
    st.info(f"Good Investment? {'YES ✅' if good_investment==1 else 'NO ❌'}")
