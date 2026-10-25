import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Taxi Fare Predictor", layout="centered")


@st.cache_resource            # load the model only once, not on every click
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("New York Taxi Fare Predictor")
st.write("Enter the details of a taxi trip and the trained model will estimate the fare.")

col1, col2 = st.columns(2)
with col1:
    passengers = st.number_input("Passengers", min_value=1, max_value=6, value=1, step=1)
    distance = st.number_input("Distance (miles)", min_value=0.1, max_value=40.0, value=2.0, step=0.1)
    color = st.selectbox("Taxi colour", ["yellow", "green"])
with col2:
    payment = st.selectbox("Payment method", ["credit card", "cash"])
    pickup_borough = st.selectbox("Pickup borough", ["Manhattan", "Queens", "Brooklyn", "Bronx"])
    dropoff_borough = st.selectbox("Dropoff borough", ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"])

if st.button("Predict fare", type="primary"):
    # ONE row with the SAME column names that were used in training
    row = pd.DataFrame([{
        "passengers": passengers,
        "distance": distance,
        "color": color,
        "payment": payment,
        "pickup_borough": pickup_borough,
        "dropoff_borough": dropoff_borough,
    }])
    fare = model.predict(row)[0]
    st.success(f"Estimated fare: ${fare:,.2f}")

st.caption("Model chosen with 10-fold cross validation (RMSE). Predictions are estimates.")
