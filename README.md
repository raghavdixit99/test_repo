# test_repo

## Overview

This repository contains a backtesting framework for an advanced technical analysis (TA) strategy. The strategy is designed to generate trading signals based on various market conditions and technical indicators. It now includes a new trailing stop feature and additional performance metrics to enhance trading performance evaluation.

## Features

- **Advanced Signal Generation**: Utilizes a combination of technical indicators such as Supertrend, RSI, Bollinger Bands, and Volatility Breakouts to generate buy and sell signals based on market conditions (Trending, Ranging, Choppy).
- **Trailing Stop Feature**: The strategy now supports a trailing stop mechanism, which adjusts the stop-loss level as the trade becomes profitable, potentially locking in gains.
- **Performance Metrics**: The strategy calculates key performance metrics, including:
  - Cumulative Return
  - Sharpe Ratio
  - Sortino Ratio (new)
  - Max Drawdown
  - Win Rate (new)
  - Number of Trades

## Usage

1. **Load Dataset**: The strategy requires a dataset with pre-calculated technical indicators. Load your dataset using:
   ```python
   data = pd.read_csv("backtest_data_with_indicators.csv")
   ```

2. **Generate Signals**: Use the `generate_signals_advanced` function to generate trading signals based on the loaded data.
   ```python
   data_with_signals = generate_signals_advanced(data)
   ```

3. **Execute Strategy**: Execute the strategy with the option to enable the trailing stop feature.
   ```python
   data_with_strategy = execute_strategy_with_stop_loss(data_with_signals, atr_multiplier=2.5, trailing_stop=True)
   ```

4. **Calculate Performance Metrics**: Evaluate the strategy's performance using the `calculate_performance_metrics` function.
   ```python
   performance_metrics = calculate_performance_metrics(data_with_strategy)
   print("Performance Metrics:", performance_metrics)
   ```

## Code Structure

- **generate_signals_advanced(data)**: Generates buy/sell signals based on advanced technical indicators.
- **execute_strategy_with_stop_loss(df, slippage=0.0002, transaction_cost=0.001, atr_multiplier=2.5, trailing_stop=True)**: Executes trades with stop-loss, take-profit, and optional trailing stop.
- **calculate_performance_metrics(data)**: Computes performance metrics for the strategy.

## Main Execution Flow

The main script loads the dataset, generates signals, executes the strategy, and calculates performance metrics. The results are printed to the console for analysis.

```python
if __name__ == "__main__":
    data = pd.read_csv("backtest_data_with_indicators.csv")
    data_with_signals = generate_signals_advanced(data)
    data_with_strategy = execute_strategy_with_stop_loss(data_with_signals, atr_multiplier=2.5, trailing_stop=True)
    performance_metrics = calculate_performance_metrics(data_with_strategy)
    print("Performance Metrics:", performance_metrics)
```

This documentation provides a comprehensive overview of the strategy's features and usage, reflecting the latest code updates.