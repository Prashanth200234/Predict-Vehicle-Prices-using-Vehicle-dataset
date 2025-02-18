import pandas as pd
import numpy as np

# Load dataset (Replace 'vehicles.csv' with your actual dataset file)
df = pd.read_csv("dataset.csv")

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())
# Fill missing numerical values with the median
num_cols = ["price", "mileage", "year", "cylinders"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with the most frequent category
cat_cols = ["make", "model", "fuel", "transmission", "body", "drivetrain"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
from sklearn.preprocessing import LabelEncoder

# Convert categorical features to numerical using Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
# Create new feature: Vehicle Age
current_year = 2024
df["vehicle_age"] = current_year - df["year"]

# Drop unnecessary columns
df.drop(columns=["name", "description", "year", "exterior_color", "interior_color"], inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["price", "mileage", "cylinders", "vehicle_age"]] = scaler.fit_transform(df[["price", "mileage", "cylinders", "vehicle_age"]])

# Save cleaned data
df.to_csv("cleaned_vehicles.csv", index=False)
print("Data Cleaning Completed! ")
