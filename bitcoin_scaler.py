from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

# Assuming `data` is your training dataset with relevant columns
data = pd.read_csv('bitcoin_historical_data.csv')  # Load your training data
scaler = MinMaxScaler()
scaler.fit(data[['Open', 'High', 'Low', 'Close', 'Volume']])  # Fit scaler to relevant columns
joblib.dump(scaler, 'bitcoin_scaler.pkl')  # Save as bitcoin_scaler.pkl
