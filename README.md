# Stock-Price-Prediction-Using-ML
This project implements a hybrid deep learning model for stock price prediction using historical data. The approach combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to capture short-term and medium-term temporal trends in financial time series data.

# Problem Statement
Financial markets are highly volatile, non-linear, and influenced by numerous unpredictable factors. Traditional statistical models such as ARIMA and GARCH often struggle to accurately forecast stock prices due to their limitations in handling non-stationarity and complex dependencies in time series data. This project addresses these challenges by developing a hybrid deep learning model that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The model is designed to learn both spatial and temporal patterns in historical stock data to enhance the accuracy and robustness of stock price predictions.

# Dataset
- Source: Yahoo Finance
- Stock: Reliance Industries Limited (RIL)
- Features: Open, High, Low, Close, Volume
- Time Period: 10 years
- Train/Test Split: 9.5 years training / 6 months testing
