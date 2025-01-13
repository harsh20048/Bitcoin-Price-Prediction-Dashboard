import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_bitcoin_data(start_date='2015-01-01', save_path='bitcoin_historical_data.csv'):
    """
    Download Bitcoin historical data using yfinance
    
    Parameters:
    start_date (str): Start date for historical data in YYYY-MM-DD format
    save_path (str): Path to save the CSV file
    """
    # Download Bitcoin data from Yahoo Finance
    btc = yf.Ticker("BTC-USD")
    df = btc.history(start=start_date)
    
    # Reset index to make Date a column
    df.reset_index(inplace=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Bitcoin historical data saved to {save_path}")
    print(f"Data shape: {df.shape}")
    return df

def create_dummy_sentiment_data(price_data_path='bitcoin_historical_data.csv', 
                              save_path='bitcoin_sentiment_data.csv'):
    """
    Create a dummy sentiment dataset matching the dates in price data
    You can replace this with real sentiment data later
    """
    # Read price data to get dates
    df_prices = pd.read_csv(price_data_path)
    dates = pd.to_datetime(df_prices['Date'])
    
    # Create dummy sentiment features
    df_sentiment = pd.DataFrame({
        'Date': dates,
        'price_momentum': 0,
        'volume_momentum': 0,
        'price_volatility': 0,
        'rsi': 50,
        'macd': 0,
        'macd_signal': 0
    })
    
    # Save to CSV
    df_sentiment.to_csv(save_path, index=False)
    print(f"Dummy sentiment data saved to {save_path}")
    return df_sentiment

if __name__ == "__main__":
    # Download price data
    price_data = download_bitcoin_data()
    
    # Create dummy sentiment data
    sentiment_data = create_dummy_sentiment_data()
    
    print("\nYou can now use these files with the HybridBitcoinPredictor class")