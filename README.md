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

---

# Documentation: Advanced TA-Only Strategy Training Script

## Overview

This script implements an advanced technical analysis (TA) trading strategy using indicators such as Supertrend, RSI, Bollinger Bands, and volatility breakouts. It is designed to classify market conditions into trending, ranging, and choppy states and adjust trading signals accordingly. Additionally, it includes a stop-loss mechanism, a trailing stop feature to manage risk, and performance metrics to evaluate strategy effectiveness.

## Requirements

### Dependencies

- Python 3.x
- pandas
- numpy

### Data Requirements

The script expects a CSV file containing historical market data with the following columns:

- `Date`: Date of the data point
- `Close`: Closing price of the asset
- `Supertrend`: Supertrend indicator value
- `RSI`: Relative Strength Index value
- `Final_Condition`: Market condition classification (Trending, Ranging, Choppy)
- `BB_Lower`, `BB_Upper`: Bollinger Band lower and upper bounds
- `Volatility_Breakout`: Binary indicator for breakout confirmation
- `ATR`: Average True Range, used for stop-loss calculations
- `Moving_Average_50`, `Moving_Average_200`: 50-period and 200-period moving averages for trend confirmation

## Functionality

### `generate_signals_advanced(data)`

This function processes the dataset and assigns buy, sell, or hold signals based on market conditions and technical indicators.

- **Trending Market**:
  - Buy when `Supertrend` is 1, `RSI` > 55, and `Moving_Average_50` > `Moving_Average_200`
  - Sell when `Supertrend` is -1, `RSI` < 45, and `Moving_Average_50` < `Moving_Average_200`
- **Ranging Market**:
  - Buy when `Close` < `BB_Lower` and `Volatility_Breakout` is 1
  - Sell when `Close` > `BB_Upper` and `Volatility_Breakout` is 1
- **Choppy Market**:
  - No trades unless a breakout occurs, then issue a buy signal.

### `execute_strategy_with_stop_loss(df, slippage=0.0002, transaction_cost=0.001, atr_multiplier=2.5, trailing_stop=True)`

This function executes trades based on the generated signals while implementing stop-loss and take-profit levels using ATR-based calculations. It also includes a trailing stop feature.

- Stop-loss is set at `entry_price - atr_multiplier * ATR`
- Take-profit is set at `entry_price + atr_multiplier * ATR`
- Trailing stop adjusts the stop-loss level as the price moves favorably
- Returns are adjusted for slippage and transaction costs.

### `calculate_performance_metrics(data)`

This function calculates key performance metrics:

- **Cumulative Return**: Overall strategy profit/loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Risk-adjusted return measure considering downside risk
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Proportion of profitable trades
- **Number of Trades**: Total buy and sell signals executed

## Execution Flow

1. Load the dataset with the required indicators.
2. Generate trading signals using `generate_signals_advanced`.
3. Execute the strategy with `execute_strategy_with_stop_loss`.
4. Calculate performance metrics using `calculate_performance_metrics`.

Ensure `backtest_data_with_indicators.csv` is in the same directory.

## Best Practices

- Regularly update the dataset with the latest market data for accurate backtesting.
- Ensure that input data has all required indicators for accurate signal generation.

## License

This script is open-source and can be modified for research and trading development purposes.