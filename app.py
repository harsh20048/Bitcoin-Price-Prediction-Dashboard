import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
import json
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

class SentimentAnalyzer:
    def __init__(self):
        self.news_urls = [
            'https://cointelegraph.com/tags/bitcoin',
            'https://bitcoin.com/news/',
            'https://coindesk.com/tag/bitcoin/'
        ]
    
    def get_crypto_news(self):
        articles = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in self.news_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                titles = soup.find_all(['h1', 'h2', 'h3'])
                articles.extend([title.text.strip() for title in titles])
            except Exception as e:
                logging.error(f"Error fetching news from {url}: {str(e)}")
        
        return articles
    
    def analyze_sentiment(self, texts):
        sentiments = []
        for text in texts:
            try:
                analysis = TextBlob(text)
                sentiments.append(analysis.sentiment.polarity)
            except Exception as e:
                logging.error(f"Error analyzing sentiment: {str(e)}")
                sentiments.append(0)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_std = np.std(sentiments) if sentiments else 0
        
        confidence = max(0, min(100, (1 - sentiment_std) * 100))
        
        if avg_sentiment > 0.2:
            mood = "Bullish"
        elif avg_sentiment < -0.2:
            mood = "Bearish"
        else:
            mood = "Neutral"
        
        return {
            'score': avg_sentiment,
            'confidence': confidence,
            'mood': mood,
            'std': sentiment_std
        }

class HybridBitcoinPredictor:
    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days
        self.lstm_model = None
        self.arima_model = None
        self.scaler = MinMaxScaler()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def get_historical_data(self, days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download('BTC-USD', start=start_date, end=end_date)
        return df
    
    def get_technical_analysis(self):
        try:
            df = self.get_historical_data(self.lookback_days)
            
            # Calculate moving average
            df['SMA_7'] = df['Close'].rolling(window=7).mean()
            df['SMA_30'] = df['Close'].rolling(window=30).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            
            # Identify support and resistance levels
            support_levels = [df['Low'].min(), df['Low'].quantile(0.25), df['Low'].quantile(0.75)]
            resistance_levels = [df['High'].max(), df['High'].quantile(0.25), df['High'].quantile(0.75)]
            
            return {
                'moving_average': {
                    'short_term': df['SMA_7'].iloc[-1],
                    'long_term': df['SMA_30'].iloc[-1]
                },
                'rsi': df['RSI'].iloc[-1],
                'macd': df['MACD'].iloc[-1],
                'support': [round(level, 2) for level in support_levels],
                'resistance': [round(level, 2) for level in resistance_levels]
            }
        except Exception as e:
            logging.error(f"Error fetching technical analysis data: {str(e)}")
            return {
                'moving_average': '--',
                'rsi': '--',
                'macd': '--',
                'support': ['--', '--', '--'],
                'resistance': ['--', '--', '--']
            }
    
    def predict_future(self, future_date):
        df = self.get_historical_data()
        
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(
            self.sentiment_analyzer.get_crypto_news()
        )
        
        last_price = df['Close'].iloc[-1]
        days_to_future = (future_date - datetime.now().date()).days
        
        sma_trend = df['SMA_7'].iloc[-1] > df['SMA_30'].iloc[-1]
        sentiment_factor = 1 + (sentiment_data['score'] * 0.1)
        
        if sma_trend:
            predicted_price = last_price * (1 + 0.02 * days_to_future) * sentiment_factor
        else:
            predicted_price = last_price * (1 - 0.01 * days_to_future) * sentiment_factor
        
        technical_confidence = 70
        sentiment_confidence = sentiment_data['confidence']
        overall_confidence = (technical_confidence + sentiment_confidence) / 2
        
        return {
            'predicted_price': predicted_price,
            'confidence': overall_confidence,
            'sentiment': sentiment_data,
            'last_price': last_price
        }

predictor = HybridBitcoinPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/current-price')
@cache.cached(timeout=60)  # Cache for 60 seconds
def get_current_price():
    try:
        btc = yf.Ticker("BTC-USD")
        current_price = btc.history(period='1d')['Close'].iloc[-1]
        return jsonify({
            'price': current_price,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error fetching current price: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-data')
def get_historical_data():
    try:
        days = int(request.args.get('days', 30))
        df = predictor.get_historical_data(days)
        data = df['Close'].to_dict()
        return jsonify({
            'prices': [{'date': str(k), 'price': v} for k, v in data.items()]
        })
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        future_date = datetime.strptime(data['future_date'], '%Y-%m-%d').date()
        
        if future_date < datetime.now().date():
            return jsonify({'error': 'Cannot predict for past dates'}), 400
        
        prediction = predictor.predict_future(future_date)
        
        return jsonify({
            'predicted_price': prediction['predicted_price'],
            'confidence': prediction['confidence'],
            'sentiment': prediction['sentiment'],
            'last_price': prediction['last_price']
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment')
def get_sentiment():
    try:
        articles = predictor.sentiment_analyzer.get_crypto_news()
        sentiment_data = predictor.sentiment_analyzer.analyze_sentiment(articles)
        return jsonify(sentiment_data)
    except Exception as e:
        logging.error(f"Error fetching sentiment data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/technical-analysis')
def get_technical_analysis():
    try:
        technical_data = predictor.get_technical_analysis()
        return jsonify(technical_data)
    except Exception as e:
        logging.error(f"Error fetching technical analysis data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)