import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Step 1: Load the CSV file
file_path = r'C:\Users\HARSH\bitcoin predication\venv\bitcoin_historical_data.csv'  # Update with your correct file path

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at {file_path} was not found.")

data = pd.read_csv(file_path)

# Step 2: Select the columns to scale
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 3: Scale the selected columns between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Step 4: Convert scaled features back to a DataFrame with the same column names
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Print the first few rows of the scaled data
print(scaled_df.head())
