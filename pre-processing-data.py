import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_csv('riyasewana_raw_data.csv')

# --- CLEANING ---

# 2. Drop rows where Target (Price) is missing
# (We can't use these for training)
df = df.dropna(subset=['Price'])

# 3. Handle Missing Mileage
# Fill NaN with the median mileage of the dataset
median_mileage = df['Mileage_km'].median()
df['Mileage_km'] = df['Mileage_km'].fillna(median_mileage)

# 4. Fix Logical Errors (Engine CC)
# Cars usually have > 600cc. If < 100, it's likely a mistake or a bike misclassified.
# Let's replace unreasonable values with the median for that Model, or just global median
median_engine = df['Engine_cc'].median()
df.loc[df['Engine_cc'] < 600, 'Engine_cc'] = median_engine

# --- FEATURE ENGINEERING ---

# 5. Create 'Vehicle_Age'
# Assuming current year is 2026
df['Vehicle_Age'] = 2026 - df['YOM']

# 6. Process 'Options' (Extract Key Features)
# Check if string contains 'AIR CONDITION' and create a new column
df['Has_AC'] = df['Options'].astype(str).str.contains('AIR CONDITION', case=False).astype(int)
df['Has_PowerSteering'] = df['Options'].astype(str).str.contains('POWER STEERING', case=False).astype(int)
df['Has_PowerMirror'] = df['Options'].astype(str).str.contains('POWER MIRROR', case=False).astype(int)

# --- ENCODING ---

# 7. One-Hot Encoding for Categoricals
# Converts 'Transmission' -> 'Transmission_Automatic', 'Transmission_Manual'
# Converts 'Fuel_Type' -> 'Fuel_Type_Diesel', 'Fuel_Type_Petrol'
df = pd.get_dummies(df, columns=['Transmission', 'Fuel_Type', 'Make'], drop_first=True)

# 8. Drop unused columns
# We drop 'YOM' (since we have Age), 'Options' (processed), and identifiers
cols_to_drop = ['Title', 'Contact', 'Make_Detail', 'Options', 'YOM']
df = df.drop(columns=cols_to_drop)

# --- FINAL CHECK ---
print("New Data Shape:", df.shape)
print(df.head())

# Save clean data for Phase 3 (Modeling)
df.to_csv('riyasewana_cleaned.csv', index=False)