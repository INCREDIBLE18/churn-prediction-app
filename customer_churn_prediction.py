import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load artifacts
pipeline = joblib.load("churn_pipeline.pkl")
importance_df = joblib.load("feature_importance.pkl")
shap_values = joblib.load("shap_values.pkl")
X_test = joblib.load("X_test.pkl")
expected_value = joblib.load("shap_expected_value.pkl")

# Page title
st.title("Customer Churn Prediction (Enhanced)")
st.markdown("Enter customer details below:")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", value=50000.0)
products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", value=50000.0)

# Convert to binary values
has_card = 1 if has_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': has_card,
    'IsActiveMember': active_member,
    'EstimatedSalary': salary
}])

# Predict
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction:")
    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**. (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ This customer is **likely to stay**. (Confidence: {1 - probability:.2%})")

    st.subheader("Feature Importance (Overall):")
    st.bar_chart(importance_df.set_index('Feature'))

    # Get closest instance from test set
    input_array = pipeline.named_steps['preprocessor'].transform(input_df)
    test_array = pipeline.named_steps['preprocessor'].transform(X_test)
    distances = np.linalg.norm(test_array - input_array, axis=1)
    closest_idx = np.argmin(distances)

    # Check the size of X_test and shap_values
    st.write(f"Closest index: {closest_idx}")
    st.write(f"X_test size: {len(X_test)}")
    st.write(f"SHAP values size: {len(shap_values[1])}")

    # Ensure closest_idx is within bounds of SHAP values
    if closest_idx < len(shap_values[1]):
        shap.initjs()
        shap_value_to_plot = shap_values[1][closest_idx]  # Class 1: Churn
        expected_val = expected_value[1]
        instance_df = X_test.iloc[[closest_idx]]

        # SHAP Force Plot
        force_plot = shap.plots.force(expected_val, shap_value_to_plot, instance_df, matplotlib=False)
        st.subheader("Explanation (SHAP Force Plot):")
        st.components.v1.html(force_plot.html(), height=300)
    else:
        st.error(f"Error: The closest index {closest_idx} exceeds the number of SHAP values available.")
