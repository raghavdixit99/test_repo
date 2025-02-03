
# Preprocessing Script for Advanced TA-Only Strategy
import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """
    Calculate technical indicators required for the strategy.
    """
    data = df.copy()

    # Calculate Moving Averages
    data["EMA_Short"] = data["Close"].ewm(span=10).mean()
    data["EMA_Long"] = data["Close"].ewm(span=50).mean()

    # Calculate RSI
    delta = data["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    rolling_mean = data["Close"].rolling(window=20).mean()
    rolling_std = data["Close"].rolling(window=20).std()
    data["BB_Lower"] = rolling_mean - (2 * rolling_std)
    data["BB_Upper"] = rolling_mean + (2 * rolling_std)
    data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]

    # Calculate ATR
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    data["ATR"] = tr.rolling(window=14).mean()

    # Calculate Supertrend
    atr = data["ATR"]
    data["Supertrend_Upper"] = data["Close"] + 3 * atr
    data["Supertrend_Lower"] = data["Close"] - 3 * atr
    data["Supertrend"] = np.where(data["Close"] > data["Supertrend_Lower"], 1, -1)

    # Calculate Volatility Breakout
    data["Volatility_Breakout"] = np.where(data["High"] - data["Low"] > 1.5 * atr, 1, 0)

    return data

if __name__ == "__main__":
    # Load raw data
    train_data = pd.read_csv("train_data_final.csv")
    backtest_data = pd.read_csv("backtest_data_classified.csv")

    # Calculate indicators
    train_data_with_indicators = calculate_technical_indicators(train_data)
    backtest_data_with_indicators = calculate_technical_indicators(backtest_data)

    # Save processed data
    train_data_with_indicators.to_csv("train_data_with_indicators.csv", index=False)
    backtest_data_with_indicators.to_csv("backtest_data_with_indicators.csv", index=False)

    print("Preprocessing complete. Files saved as 'train_data_with_indicators.csv' and 'backtest_data_with_indicators.csv'.")
