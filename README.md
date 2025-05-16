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

# Approach
To address the challenges of stock price prediction, we designed a multi-scale hybrid deep learning model that leverages the strengths of both Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks.

- The model processes stock data using three temporal windows: 1-day, 3-day, and 5-day sequences.
- Each window is independently fed into a CNN-LSTM pipeline:
    - CNN layers extract short-term spatial patterns.
    - LSTM layers capture temporal dependencies and trends.
-  Outputs from each branch are concatenated and passed through fully connected layers to make the final prediction.

This approach captures both short-term and medium-term trends, reduces overfitting through multi-branch learning and leverages the CNN's strength and LSTM's ability to recognize patterns and model sequences respectively.

# Results
1. Train-Test Split
   
![image](https://github.com/user-attachments/assets/0c2bedc8-ca9e-42cd-a055-d4abfab4e1a1)

2. Actual Stock Price
   
![image](https://github.com/user-attachments/assets/65063236-91e7-496d-94db-c95b2c695c19)

3. Predicted Stock Price

![image](https://github.com/user-attachments/assets/0d95a38c-c7b3-4b32-951f-0b19d96477e7)

4. Actual v/s Predicted Stock Price
   
![image](https://github.com/user-attachments/assets/e1c294d6-ca6e-46f1-94e8-c30533a20fda)

| Evaluation Metric |     Result      |
| ----------------  | --------------- | 
|       MAE         |      20.26      | 
|       RMSE        |      26.52      | 
|   R-Square Score  |      0.9428     |
|       MAPE        |      1.42%      |




