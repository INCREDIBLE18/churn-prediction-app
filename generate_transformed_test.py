import joblib

# Load your pipeline and X_test (raw)
pipeline = joblib.load("churn_pipeline.pkl")
X_test = joblib.load("X_test.pkl")

# Transform the raw test data using the pipeline's preprocessor
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# Save the transformed data
joblib.dump(X_test_transformed, "X_test_transformed.pkl")

print("✅ Transformed test set saved as 'X_test_transformed.pkl'")
