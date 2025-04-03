import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes  # Simulating house prices dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Load dataset (simulating house price prediction for SmartEstate-PriceML)
data = load_diabetes()
X, y = data.data, data.target  # Target is house price in our scenario

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset details
print(f"Number of Observations: {X.shape[0]}")
print(f"Number of Features: {X.shape[1]}")

# Initialize models
n_estimators = 100
rf = RandomForestRegressor(
    n_estimators=n_estimators, 
    random_state=42, 
    min_samples_leaf=2, 
    max_features='sqrt'
)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# Train models & measure training time
start_time_rf = time.time()
rf.fit(X_train, y_train)
rf_train_time = time.time() - start_time_rf

start_time_xgb = time.time()
xgb.fit(X_train, y_train)
xgb_train_time = time.time() - start_time_xgb

# Measure prediction time
start_time_rf = time.time()
y_pred_rf = rf.predict(X_test)
rf_pred_time = time.time() - start_time_rf

start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)
xgb_pred_time = time.time() - start_time_xgb

# Calculate MSE and R² score
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print performance metrics
print(f"SmartEstate-PriceML (Random Forest) - MSE: {mse_rf:.4f}, R² Score: {r2_rf:.4f}")
print(f"SmartEstate-PriceML (XGBoost) - MSE: {mse_xgb:.4f}, R² Score: {r2_xgb:.4f}")

# Print timing results
print(f"Random Forest: Training Time = {rf_train_time:.3f}s, Prediction Time = {rf_pred_time:.3f}s")
print(f"XGBoost: Training Time = {xgb_train_time:.3f}s, Prediction Time = {xgb_pred_time:.3f}s")

# Standard deviation of test data
std_y = np.std(y_test)
print(f"Standard Deviation of House Prices: {std_y:.4f}")

# Scatter plot of Actual vs. Predicted values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ideal regression line (y = x)
x_range = np.linspace(min(y_test), max(y_test), 100)

# Random Forest plot
axes[0].scatter(y_test, y_pred_rf, alpha=0.5, color='blue', label='Predictions')
axes[0].plot(x_range, x_range, color='red', linestyle='--', label='Ideal Line')
axes[0].fill_between(x_range, x_range - std_y, x_range + std_y, color='gray', alpha=0.2, label='±1 Std Dev')
axes[0].set_title('SmartEstate-PriceML (Random Forest): Actual vs Predicted Prices')
axes[0].set_xlabel('Actual Prices')
axes[0].set_ylabel('Predicted Prices')
axes[0].legend()

# XGBoost plot
axes[1].scatter(y_test, y_pred_xgb, alpha=0.5, color='green', label='Predictions')
axes[1].plot(x_range, x_range, color='red', linestyle='--', label='Ideal Line')
axes[1].fill_between(x_range, x_range - std_y, x_range + std_y, color='gray', alpha=0.2, label='±1 Std Dev')
axes[1].set_title('SmartEstate-PriceML (XGBoost): Actual vs Predicted Prices')
axes[1].set_xlabel('Actual Prices')
axes[1].set_ylabel('Predicted Prices')
axes[1].legend()

plt.tight_layout()
plt.show()
