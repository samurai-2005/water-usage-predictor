import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load Data
data = pd.read_csv("data.csv")

# Features and Target
X = data[['temperature', 'family_size', 'day_of_week']]
y = data['water_usage']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🚰 Water Usage Forecasting App")
st.markdown("Predict daily household water usage using temperature, family size, and weekday.")

temp = st.slider("🌡️ Temperature (°C)", 20, 45, 30)
family = st.number_input("👨‍👩‍👧‍👦 Family Size", min_value=1, max_value=10, value=4)
day = st.selectbox("📅 Day of the Week", {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
})

# Make prediction
if st.button("Predict Water Usage"):
    input_data = pd.DataFrame([[temp, family, day]], columns=['temperature', 'family_size', 'day_of_week'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Daily Water Usage: {prediction:.2f} Liters")

st.markdown("---")
st.subheader("Sample Data")
st.dataframe(data)
