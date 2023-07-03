# Stock Market Prediction and Forecasting Using LSTM

This repository contains a project for stock market prediction and forecasting using Long Short-Term Memory (LSTM) neural networks. LSTM is a type of recurrent neural network (RNN) that is well-suited for sequential data like time series.

## Data

The data used for this project is from the AAPL (Apple Inc.) stock market. It includes historical stock price information such as open, close, high, low, volume, and adjusted prices. The data is stored in a CSV file, and we use pandas to load and preprocess it.

## Exploratory Data Analysis

We begin by performing exploratory data analysis on the stock price data to gain insights into its behavior over the years. Various visualizations are created to understand the trends, patterns, and fluctuations in the stock price.

## Data Preprocessing

LSTM models are sensitive to the scale of the data, so we apply MinMax scaling to normalize the stock prices between 0 and 1. Additionally, we split the data into training and testing sets based on date, ensuring that the model is trained on past data and tested on future data.

## LSTM Model

We build a sequential LSTM model using TensorFlow's Keras API. The model consists of multiple LSTM layers with different units, followed by a dense output layer. Mean squared error is used as the loss function, and the Adam optimizer is used for training.

## Training and Evaluation

The model is trained on the training data and evaluated on the testing data. The training and validation loss are monitored to avoid overfitting. After training the model, we can use it to make predictions and forecast future stock prices.

## Forecasting

In addition to predicting the stock prices for the testing period, we extend the predictions to forecast the stock prices for the next 30 days. This allows us to gain insights into the potential future performance of the stock.

## Results

The performance of the LSTM model on the testing data is analyzed, and visualizations are created to compare the predicted stock prices with the actual prices. We can also calculate metrics such as root mean squared error (RMSE) to quantify the model's accuracy.

## How to Use the Code

To run the code in this repository, you will need Python installed along with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow

You can install the required libraries using `pip`:

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Next, you can clone this repository and run the Python script to perform stock market prediction and forecasting using LSTM.

```
git clone https://github.com/your_username/stock-market-prediction.git
cd stock-market-prediction
```

Please note that the data used in this project is specific to AAPL stock. If you want to apply the same approach to other stocks, you need to replace the CSV file with the respective stock's data.

Feel free to explore the code and modify it according to your requirements. Happy predicting!
