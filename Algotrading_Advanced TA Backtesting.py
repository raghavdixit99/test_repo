```python
# Training Script for Advanced TA-Only Strategy
import pandas as pd
import numpy as np

# Function definitions
def generate_signals_advanced(data):
    """ Generate signals using advanced technical indicators.
    
    This function analyzes the market conditions and generates trading signals
    based on advanced technical indicators such as Supertrend, RSI, and moving averages.
    """
    data = data.copy()
    data["Signal"] = "Hold"  # Default signal

    # Masks for market conditions
    trending_mask = data["Final_Condition"] == "Trending"
    ranging_mask = data["Final_Condition"] == "Ranging"
    choppy_mask = data["Final_Condition"] == "Choppy"

    # Advanced Trending Signals with Supertrend, RSI, and Moving Averages
    data.loc[
        trending_mask & (data["Supertrend"] == 1) & (data["RSI"] > 55) & 
        (data["Moving_Average_50"] > data["Moving_Average_200"]), "Signal"] = "Buy"
    data.loc[
        trending_mask & (data["Supertrend"] == -1) & (data["RSI"] < 45) & 
        (data["Moving_Average_50"] < data["Moving_Average_200"]), "Signal"] = "Sell"

    # Advanced Ranging Signals with Bollinger Bands and Volatility Breakouts
    data.loc[
        ranging_mask & (data["Close"] < data["BB_Lower"]) & (data["Volatility_Breakout"] == 1), "Signal"] = "Buy"
    data.loc[
        ranging_mask & (data["Close"] > data["BB_Upper"]) & (data["Volatility_Breakout"] == 1), "Signal"] = "Sell"

    # No trades for Choppy Market unless breakout confirmed
    data.loc[choppy_mask & (data["Volatility_Breakout"] == 1), "Signal"] = "Buy"

    return data

def execute_strategy_with_stop_loss(df, slippage=0.0002, transaction_cost=0.001, atr_multiplier=2.5, trailing_stop=True):
    """ Execute trades with stop-loss, take-profit, and optional trailing stop levels.
    
    This function manages trade execution by setting stop-loss and take-profit levels
    based on the Average True Range (ATR). It also includes an optional trailing stop feature
    to lock in profits as the price moves favorably.
    """
    data = df.copy()
    data["Position"] = 0
    data["Strategy_Returns"] = 0.0
    stop_loss, take_profit, trailing_stop_level = None, None, None

    for i in range(1, len(data)):
        previous_signal = data.loc[i - 1, "Signal"]
        current_signal = data.loc[i, "Signal"]
        atr = data.loc[i, "ATR"]

        if previous_signal == "Buy":
            if stop_loss is None:
                stop_loss = data.loc[i - 1, "Close"] - atr_multiplier * atr
                take_profit = data.loc[i - 1, "Close"] + atr_multiplier * atr
                trailing_stop_level = stop_loss if trailing_stop else None

            if trailing_stop and data.loc[i, "Close"] > take_profit * 0.5:
                trailing_stop_level = max(trailing_stop_level, data.loc[i, "Close"] - atr)

            if data.loc[i, "Close"] <= stop_loss or (trailing_stop and data.loc[i, "Close"] <= trailing_stop_level):
                gross_return = data.loc[i, "Close"] / data.loc[i - 1, "Close"] - 1
                net_return = gross_return - slippage - transaction_cost
                data.loc[i, "Strategy_Returns"] = net_return
                stop_loss, take_profit, trailing_stop_level = None, None, None

        if current_signal == "Buy":
            data.loc[i, "Position"] = 1
        elif current_signal == "Sell":
            data.loc[i, "Position"] = 0
            stop_loss, take_profit, trailing_stop_level = None, None, None
        else:
            data.loc[i, "Position"] = data.loc[i - 1, "Position"]

    return data

def calculate_performance_metrics(data):
    """ Calculate key performance metrics for the strategy.
    
    This function computes various performance metrics including cumulative return,
    Sharpe Ratio, Sortino Ratio, maximum drawdown, win rate, and the number of trades.
    """
    strategy_returns = data["Strategy_Returns"]

    cumulative_return = (1 + strategy_returns).prod() - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

    downside_deviation = strategy_returns[strategy_returns < 0].std()
    sortino_ratio = strategy_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0

    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    max_drawdown = drawdown.min()

    num_trades = data["Signal"].isin(["Buy", "Sell"]).sum()
    win_rate = (strategy_returns > 0).sum() / num_trades if num_trades > 0 else 0

    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Number of Trades": num_trades,
    }

# Main Execution Flow
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv("backtest_data_with_indicators.csv")

    # Generate signals and execute strategy
    data_with_signals = generate_signals_advanced(data)
    data_with_strategy = execute_strategy_with_stop_loss(data_with_signals, atr_multiplier=2.5, trailing_stop=True)

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(data_with_strategy)
    print("Performance Metrics:", performance_metrics)
```