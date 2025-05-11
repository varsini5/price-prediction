import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("shipping_price_prediction.csv")  # Ensure this file is available in your project

# Convert date columns to numerical values (days)
date_cols = ['ETD', 'ATD', 'ETA', 'ATA']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col] = (df[col] - df[col].min()).dt.days  # Convert dates to integer days

# Handle missing values
df.fillna(0, inplace=True)

# Feature Selection
X = df[['Week', 'Volume (CBM)', 'ETD', 'ATD', 'ETA', 'ATA']]
y = df['Achieved']  # Predicted price

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)  # Train with scaled data

# Save Models as .pkl Files
with open("random_forest.pkl", "wb") as rf_file:
    pickle.dump(rf_model, rf_file)

with open("gradient_boosting.pkl", "wb") as gb_file:
    pickle.dump(gb_model, gb_file)

with open("svr.pkl", "wb") as svr_file:
    pickle.dump((svr_model, scaler), svr_file)  # Save SVR model along with scaler

print("✅ Models trained and saved as pickle files successfully!")
