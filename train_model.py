import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import shap
import joblib

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Set correct target column
target_column = "Exited"

# Drop irrelevant columns (optional but recommended)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Save feature names
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
expected_value = explainer.expected_value

# Save model and SHAP-related files
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_test_scaled, "X_test_transformed.pkl")
joblib.dump(shap_values, "shap_values.pkl")
joblib.dump(expected_value, "shap_expected_value.pkl")

print("✅ Model training and SHAP export complete.")
