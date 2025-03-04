import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessed_data.pkl", "rb") as file:
    _, _, _, _, scaler = pickle.load(file)

st.title("🚢 Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert inputs
sex = 1 if sex == "Female" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

# Predict button
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, fare, sibsp, parch, embarked_C, embarked_Q]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ The passenger is likely to survive!")
    else:
        st.error("❌ The passenger is unlikely to survive.")
