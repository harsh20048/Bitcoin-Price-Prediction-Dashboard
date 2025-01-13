// main.js

// Function to fetch the current Bitcoin price
async function fetchCurrentPrice() {
    try {
        const response = await fetch('/api/current-price');
        const data = await response.json();
        document.getElementById('current-price').textContent = `$${data.price.toFixed(2)}`;
        document.getElementById('price-updated').textContent = `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
    } catch (error) {
        console.error('Error fetching current price:', error);
        document.getElementById('current-price').textContent = 'Error fetching price';
    }
}

// Function to fetch historical price data
async function fetchHistoricalData() {
    try {
        const response = await fetch('/api/historical-data?days=30');
        const data = await response.json();
        updatePriceChart(data.prices);
    } catch (error) {
        console.error('Error fetching historical data:', error);
    }
}

// Function to update price chart using Chart.js
function updatePriceChart(prices) {
    const ctx = document.getElementById('price-Chart').getContext('2d');
    const labels = prices.map(price => price.date);
    const data = prices.map(price => price.price);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Bitcoin Price',
                data: data,
                borderColor: 'rgba(75, 192, 192, 1)',
                fill: false,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Function to handle price prediction
async function predictPrice() {
    const futureDateInput = document.getElementById('future-date').value;
    const futureDate = new Date(futureDateInput);

    if (futureDate < new Date()) {
        alert('Please select a future date.');
        return;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ future_date: futureDateInput })
        });
        const prediction = await response.json();
        displayPrediction(prediction);
    } catch (error) {
        console.error('Error predicting price:', error);
    }
}

// Function to display prediction results
function displayPrediction(prediction) {
    document.getElementById('predicted-price').textContent = `$${prediction.predicted_price.toFixed(2)}`;
    document.getElementById('confidence-score').textContent = `Confidence: ${prediction.confidence.toFixed(0)}%`;
    document.getElementById('sentiment-score').textContent = `Market Sentiment: ${prediction.sentiment.mood}`;
}

// Function to fetch and display technical analysis data
async function fetchTechnicalAnalysis() {
    try {
        const response = await fetch('/api/technical-analysis');
        const technicalData = await response.json();
        displayTechnicalAnalysis(technicalData);
    } catch (error) {
        console.error('Error fetching technical analysis data:', error);
        document.getElementById('technical-indicators').innerHTML = '<li>Error fetching data</li>';
    }
}

// Function to display technical analysis data
function displayTechnicalAnalysis(data) {
    const technicalIndicators = document.getElementById('technical-indicators');
    technicalIndicators.innerHTML = `
        <li>Short-term Moving Average: ${data.moving_average.short_term.toFixed(2)}</li>
        <li>Long-term Moving Average: ${data.moving_average.long_term.toFixed(2)}</li>
        <li>RSI: ${data.rsi.toFixed(2)}</li>
        <li>MACD: ${data.macd.toFixed(2)}</li>
        <li>Support Levels: ${data.support.map(level => `$${level}`).join(', ')}</li>
        <li>Resistance Levels: ${data.resistance.map(level => `$${level}`).join(', ')}</li>
    `;
}

// Function to fetch and display market sentiment data
async function fetchMarketSentiment() {
    try {
        const response = await fetch('/api/sentiment');
        const sentimentData = await response.json();
        displayMarketSentiment(sentimentData);
    } catch (error) {
        console.error('Error fetching sentiment data:', error);
        document.getElementById('sentiment-indicators').innerHTML = '<li>Error fetching data</li>';
        document.getElementById('sentiment-analysis-text').textContent = 'Error fetching sentiment data';
    }
}

// Function to display market sentiment data
function displayMarketSentiment(data) {
    const sentimentIndicators = document.getElementById('sentiment-indicators');
    sentimentIndicators.innerHTML = `
        <li>Sentiment Score: ${data.score.toFixed(2)}</li>
        <li>Confidence: ${data.confidence.toFixed(0)}%</li>
        <li>Mood: ${data.mood}</li>
    `;

    document.getElementById('sentiment-analysis-text').textContent = `Overall sentiment analysis: The current market sentiment is ${data.mood.toLowerCase()} with a score of ${data.score.toFixed(2)} and a confidence of ${data.confidence.toFixed(0)}%.`;
}

document.addEventListener('DOMContentLoaded', () => {
    fetchCurrentPrice();
    fetchHistoricalData();
    fetchTechnicalAnalysis();
    fetchMarketSentiment();

    document.getElementById('predict-btn').addEventListener('click', predictPrice);
});