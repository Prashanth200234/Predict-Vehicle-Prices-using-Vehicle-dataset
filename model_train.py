import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 📌 Load dataset
df = pd.read_csv("cleaned_vehicles.csv")

# 📌 Check for missing values
print("Missing values before cleaning:\n", df.isnull().sum())

# 📌 Fill missing numerical values with median
num_cols = ["mileage", "cylinders", "doors", "vehicle_age"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 📌 Fill missing categorical values with mode
cat_cols = ["make", "model", "engine", "fuel", "transmission", "trim", "body", "drivetrain"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 📌 Convert categorical columns to numeric using Label Encoding
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding

# 📌 Define Features (X) and Target (y)
X = df.drop(columns=["price"])  # Features (excluding price)
y = df["price"]  # Target (actual price)

# 📌 Standardize numerical features (DON'T scale 'price')
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])  # Scale only numerical features
joblib.dump(scaler, "scaler.pkl")  # Save scaler to use in Flask app

# 📌 Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Model Training Completed!")

# 📌 Model Evaluation
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# 📌 Save the trained model
joblib.dump(rf_model, "vehicle_price_model.pkl")
print("✅ Model Saved Successfully as `vehicle_price_model.pkl`")
