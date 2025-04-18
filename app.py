import streamlit as st
from pipeline import load_data, train_model, make_prediction

# Load & train
df = load_data()
model = train_model(df)
neighborhoods = sorted(df["neighborhood"].dropna().unique())

# App title
st.title("🏙️ Mexico City Apartment Price Predictor")

st.markdown("""
Enter apartment details below to predict the price (in USD).
""")

# Inputs
area = st.slider("Covered Area (in m²)", min_value=10, max_value=300, value=70)
lat = st.number_input("Latitude", value= -34.60)
lon = st.number_input("Longitude", value= -58.38)
neighborhood = st.selectbox("Neighborhood", neighborhoods)

# Prediction
if st.button("Predict Price"):
    price = make_prediction(area, lat, lon, neighborhood)
    st.success(f"💰 Estimated Price: ${price:,.2f} USD")
