
# Production-Ready Script for Optimized TA-Only Strategy
import pandas as pd
import numpy as np

# Step 1: Generate Signals with ADX and Technical Indicators
def generate_signals_with_adx(data):
    """
    Generate signals with ADX filtering and technical indicators.
    """
    data = data.copy()
    data["Signal"] = "Hold"  # Default signal

    # Masks for market conditions
    trending_mask = data["Final_Condition"] == "Trending"
    ranging_mask = data["Final_Condition"] == "Ranging"
    choppy_mask = data["Final_Condition"] == "Choppy"

    # ADX for trend strength confirmation
    adx_trending = data["ADX"] > 25

    # Trending Market Signals with ADX and Momentum Confirmation
    data.loc[trending_mask & adx_trending & (data["EMA_Short"] > data["EMA_Long"]) & (data["RSI"] > 50), "Signal"] = "Buy"
    data.loc[trending_mask & adx_trending & (data["EMA_Short"] < data["EMA_Long"]) & (data["RSI"] < 50), "Signal"] = "Sell"

    # Ranging Market Signals with Bollinger Bands and MACD Confirmation
    data.loc[ranging_mask & (data["Close"] < data["BB_Lower"]) & (data["MACD"] > data["Signal_Line"]), "Signal"] = "Buy"
    data.loc[ranging_mask & (data["Close"] > data["BB_Upper"]) & (data["MACD"] < data["Signal_Line"]), "Signal"] = "Sell"

    # No trades for Choppy Market
    data.loc[choppy_mask, "Signal"] = "Hold"

    return data

# Step 2: Execute Strategy with Stop-Loss and Take-Profit
def execute_strategy_with_stop_loss(data, slippage=0.0002, transaction_cost=0.001, atr_multiplier=2):
    """
    Execute trades with stop-loss and take-profit levels.
    """
    data = data.copy()
    data["Position"] = 0  # 1 for long, -1 for short, 0 for neutral
    data["Strategy_Returns"] = 0.0

    stop_loss, take_profit = None, None  # Initialize stop-loss and take-profit levels

    for i in range(1, len(data)):
        previous_signal = data.loc[i - 1, "Signal"]
        current_signal = data.loc[i, "Signal"]
        atr = data.loc[i, "ATR"]

        if previous_signal == "Buy":
            if stop_loss is None:
                stop_loss = data.loc[i - 1, "Close"] - atr_multiplier * atr
                take_profit = data.loc[i - 1, "Close"] + atr_multiplier * atr

            if data.loc[i, "Close"] <= stop_loss or data.loc[i, "Close"] >= take_profit:
                gross_return = data.loc[i, "Close"] / data.loc[i - 1, "Close"] - 1
                net_return = gross_return - slippage - transaction_cost
                data.loc[i, "Strategy_Returns"] = net_return
                stop_loss, take_profit = None, None  # Reset levels

        elif previous_signal == "Short":
            if stop_loss is None:
                stop_loss = data.loc[i - 1, "Close"] + atr_multiplier * atr
                take_profit = data.loc[i - 1, "Close"] - atr_multiplier * atr

            if data.loc[i, "Close"] >= stop_loss or data.loc[i, "Close"] <= take_profit:
                gross_return = -data.loc[i, "Close"] / data.loc[i - 1, "Close"] + 1
                net_return = gross_return - slippage - transaction_cost
                data.loc[i, "Strategy_Returns"] = net_return
                stop_loss, take_profit = None, None  # Reset levels

        if current_signal == "Buy":
            data.loc[i, "Position"] = 1
        elif current_signal == "Sell":
            data.loc[i, "Position"] = 0
            stop_loss, take_profit = None, None
        elif current_signal == "Short":
            data.loc[i, "Position"] = -1
        elif current_signal == "Cover":
            data.loc[i, "Position"] = 0
            stop_loss, take_profit = None, None
        else:
            data.loc[i, "Position"] = data.loc[i - 1, "Position"]

    return data

# Step 3: Calculate Performance Metrics
def calculate_performance_metrics(data):
    """
    Calculate key performance metrics for the strategy.
    """
    strategy_returns = data["Strategy_Returns"]

    cumulative_return = (1 + strategy_returns).prod() - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    max_drawdown = drawdown.min()
    num_trades = data["Signal"].isin(["Buy", "Sell"]).sum()

    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trades": num_trades,
    }

# Main Execution Flow
if __name__ == "__main__":
    # Load your dataset (example: backtesting data)
    data = pd.read_csv("backtest_data_classified.csv")

    # Apply optimized parameters
    data["EMA_Short"] = data["Close"].ewm(span=10).mean()
    data["EMA_Long"] = data["Close"].ewm(span=50).mean()
    data["BB_Lower"] = data["Close"] - 0.05 * data["Close"].rolling(20).std()
    data["BB_Upper"] = data["Close"] + 0.05 * data["Close"].rolling(20).std()

    # Generate signals and execute strategy
    data_with_signals = generate_signals_with_adx(data)
    data_with_strategy = execute_strategy_with_stop_loss(data_with_signals, atr_multiplier=2.5)

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(data_with_strategy)
    print("Performance Metrics:", performance_metrics)
