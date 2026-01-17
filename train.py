import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib

# 1. Load the Cleaned Data
df = pd.read_csv('riyasewana_cleaned.csv')

print(f"Initial dataset size: {len(df)} rows")

# 1.5. CLEAN OUTLIERS (Crucial step!)
# Remove prices < 500k (rentals/parts/errors) and > 80M (extreme luxury outliers)
df = df[(df['Price'] > 500000) & (df['Price'] < 80000000)]
print(f"After removing outliers: {len(df)} rows")

# Handle any remaining missing values
print(f"\nMissing values before handling:\n{df.isnull().sum()}\n")

# Fill remaining NaN values in numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Drop rows with any remaining NaN values
df = df.dropna()

print(f"Data shape after cleaning: {df.shape}")
print(f"Missing values after handling:\n{df.isnull().sum()}\n")

# 1.6. FREQUENCY ENCODING FOR 'MODEL'
# Instead of dropping 'Model', we map it to how common it is.
# This helps the model distinguish a common "Vitz" from a rare "Land Cruiser".
if 'Model' in df.columns:
    model_counts = df['Model'].value_counts()
    df['Model_Freq'] = df['Model'].map(model_counts)
    print(f"Model frequency encoding applied. Range: {df['Model_Freq'].min()} to {df['Model_Freq'].max()}\n")
    # Drop the original text 'Model' column now that we have the number
    df = df.drop(columns=['Model'])
else:
    print("Warning: 'Model' column not found in dataset\n")

# 2. Split into Features (X) and Target (y)
X = df.drop(columns=['Price'])
y = df['Price']

# 2.5. Log-Transform the Target Variable (to handle wide price range)
# This compresses the range and makes errors relative (percentages) rather than absolute
y_log = np.log1p(y)  # log(1 + price) - handles zeros safely
print(f"Price range: Rs. {y.min():,.0f} to Rs. {y.max():,.0f}")
print(f"Log-transformed range: {y_log.min():.2f} to {y_log.max():.2f}\n")

# 3. Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for SHAP analysis (XAI)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print(f"Test data saved: X_test.csv ({X_test.shape[0]} rows) and y_test.csv\n")

# Also split the log-transformed target
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# --- BASELINE MODEL (Linear Regression) ---
# We use this just to show that our advanced model is better
lr = LinearRegression()
lr.fit(X_train, y_train_log)  # Train on log-transformed target
y_pred_lr_log = lr.predict(X_test)
y_pred_lr = np.expm1(y_pred_lr_log)  # Convert back to original scale

print("--- Linear Regression (Baseline) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):,.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_lr):.4f}")

# --- ADVANCED MODEL (LightGBM with GridSearchCV) ---
# Use GridSearchCV to find the best hyperparameters automatically
print("\n--- LightGBM with GridSearchCV (Optimizing...) ---")

# Define the parameter grid to search
# NOTE: LightGBM uses 'num_leaves' as the main complexity parameter
param_grid = {
    'n_estimators': [100, 500, 1000],    # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1],  # Step size
    'num_leaves': [20, 31, 50],          # Controls tree complexity (31 is default)
    'subsample': [0.7, 0.8, 0.9]         # Fraction of data to use for each tree
}

# Initialize LightGBM regressor
lgbm = LGBMRegressor(objective='regression', random_state=42, verbose=-1)

# Initialize GridSearchCV (3-fold cross-validation)
grid_search = GridSearchCV(
    estimator=lgbm, 
    param_grid=param_grid, 
    cv=3,  # 3-fold cross-validation
    scoring='r2',  # Optimize for R² score
    verbose=1,  # Show progress
    n_jobs=-1  # Use all CPU cores
)

# Fit the grid search on log-transformed target
grid_search.fit(X_train, y_train_log)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation R² Score: {grid_search.best_score_:.4f}")

# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred_lgbm_log = best_model.predict(X_test)
y_pred_lgbm = np.expm1(y_pred_lgbm_log)  # Convert back to original scale

print("\n--- LightGBM Best Model Results ---")
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print(f"RMSE: {rmse_lgbm:,.2f}")
print(f"R² Score: {r2_lgbm:.4f}")

# --- VISUALIZATION (For your Report) ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.5, color='green', label='Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Price (LKR)")
plt.ylabel("Predicted Price (LKR)")
plt.title(f"Actual vs Predicted Prices - LightGBM (R² = {r2_lgbm:.2f})")
plt.legend()
plt.grid(True)
plt.savefig('model_performance_plot.png')
print(f"\nPlot saved as 'model_performance_plot.png'")
plt.show()

# --- SAVE THE BEST MODEL ---
joblib.dump(best_model, 'lightgbm_best_model.pkl')
print(f"Best model saved as 'lightgbm_best_model.pkl'")