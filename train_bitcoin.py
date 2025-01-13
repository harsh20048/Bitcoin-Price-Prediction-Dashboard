import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.news_urls = [
            'https://cointelegraph.com/tags/bitcoin',
            'https://bitcoin.com/news/',
            'https://coindesk.com/tag/bitcoin/'
        ]
    
    def get_crypto_news(self):
        """Fetch cryptocurrency news from multiple sources"""
        articles = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in self.news_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article titles and text (adjust selectors based on website structure)
                titles = soup.find_all(['h1', 'h2', 'h3'])
                articles.extend([title.text.strip() for title in titles])
            except Exception as e:
                print(f"Error fetching news from {url}: {str(e)}")
        
        return articles
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment of given texts"""
        sentiments = []
        for text in texts:
            try:
                analysis = TextBlob(text)
                sentiments.append(analysis.sentiment.polarity)
            except:
                sentiments.append(0)
        
        # Return average sentiment and sentiment volatility
        return np.mean(sentiments), np.std(sentiments)

class HybridBitcoinPredictor:
    def __init__(self, lookback_days=60):
        """
        Initialize the hybrid prediction model
        
        Parameters:
        lookback_days (int): Number of previous days to use for prediction
        """
        self.lookback_days = lookback_days
        self.lstm_model = None
        self.arima_model = None
        self.scaler = MinMaxScaler()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def download_bitcoin_data(self, start_date='2020-01-01'):
        """Download Bitcoin historical data using yfinance"""
        btc = yf.download('BTC-USD', start=start_date)
        return btc
    
    def prepare_data(self, df):
        """Prepare data with technical indicators and sentiment analysis"""
        # Calculate technical indicators
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add volatility
        df['Volatility'] = df['Close'].rolling(window=30).std()
        
        # Get sentiment data
        articles = self.sentiment_analyzer.get_crypto_news()
        sentiment_score, sentiment_volatility = self.sentiment_analyzer.analyze_sentiment(articles)
        df['Sentiment'] = sentiment_score
        df['Sentiment_Volatility'] = sentiment_volatility
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Scale features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'SMA_30', 
                   'RSI', 'Volatility', 'Sentiment', 'Sentiment_Volatility']
        scaled_data = self.scaler.fit_transform(df[features])
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i])
            y.append(scaled_data[i, features.index('Close')])
        
        return np.array(X), np.array(y), df['Close']
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_arima(self, data):
        """Train ARIMA model"""
        try:
            model = ARIMA(data, order=(5,1,0))
            return model.fit()
        except Exception as e:
            print(f"ARIMA training error: {str(e)}")
            return None
    
    def train(self, epochs=50, batch_size=32):
        """Train both ARIMA and LSTM models"""
        # Download and prepare data
        df = self.download_bitcoin_data()
        X, y, close_prices = self.prepare_data(df)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train LSTM
        self.lstm_model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Train ARIMA
        self.arima_model = self.train_arima(close_prices[:train_size])
        
        # Make predictions
        lstm_pred = self.lstm_model.predict(X_test)
        arima_pred = self.arima_model.forecast(len(X_test))
        
        # Combine predictions (simple average)
        lstm_pred_unscaled = self.scaler.inverse_transform(
            np.concatenate([np.zeros((len(lstm_pred), 10)), lstm_pred, np.zeros((len(lstm_pred), 0))], axis=1)
        )[:, 3]
        hybrid_pred = (lstm_pred_unscaled + arima_pred) / 2
        
        # Plot results
        self.plot_results(history, close_prices[train_size:], hybrid_pred, lstm_pred_unscaled, arima_pred)
        
        return hybrid_pred
    
    def plot_results(self, history, actual, hybrid_pred, lstm_pred, arima_pred):
        """Plot training history and predictions"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot training history
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot predictions
        ax2.plot(actual, label='Actual Prices', color='blue')
        ax2.plot(hybrid_pred, label='Hybrid Prediction', color='red')
        ax2.plot(lstm_pred, label='LSTM Prediction', color='green', alpha=0.5)
        ax2.plot(arima_pred, label='ARIMA Prediction', color='orange', alpha=0.5)
        ax2.set_title('Bitcoin Price Predictions vs Actual')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('bitcoin_hybrid_prediction.png')
        plt.close()
    
    def predict_future(self, days=30):
        """Predict future prices using both models"""
        if not self.lstm_model or not self.arima_model:
            raise Exception("Models not trained yet. Call train() first.")
        
        # Get latest data
        df = self.download_bitcoin_data()
        X, _, _ = self.prepare_data(df)
        last_sequence = X[-1:]
        
        # Make predictions
        lstm_predictions = []
        arima_predictions = self.arima_model.forecast(days)
        current_sequence = last_sequence[0]
        
        for _ in range(days):
            # LSTM prediction
            next_pred = self.lstm_model.predict(current_sequence.reshape(1, self.lookback_days, -1))
            lstm_predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1, 3] = next_pred[0, 0]
        
        # Unscale LSTM predictions
        lstm_predictions = self.scaler.inverse_transform(
            np.concatenate([np.zeros((len(lstm_predictions), 10)), 
                          np.array(lstm_predictions).reshape(-1, 1), 
                          np.zeros((len(lstm_predictions), 0))], axis=1)
        )[:, 3]
        
        # Combine predictions
        hybrid_predictions = (lstm_predictions + arima_predictions) / 2
        
        return hybrid_predictions

def main():
    try:
        # Initialize and train model
        predictor = HybridBitcoinPredictor(lookback_days=60)
        print("Training models...")
        predictor.train(epochs=50, batch_size=32)
        
        # Make future predictions
        print("\nPredicting future prices...")
        future_prices = predictor.predict_future(days=30)
        
        print("\nPrediction completed successfully!")
        print("Check 'bitcoin_hybrid_prediction.png' for visualization")
        print("\nPredicted prices for next 30 days:")
        for i, price in enumerate(future_prices, 1):
            print(f"Day {i}: ${price:,.2f}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()