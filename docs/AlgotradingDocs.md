# Documentation: Advanced TA-Only Strategy Training Script

## Overview
This script implements an advanced technical analysis (TA) trading strategy using indicators such as Supertrend, RSI, Bollinger Bands, and volatility breakouts. It is designed to classify market conditions into trending, ranging, and choppy states and adjust trading signals accordingly. Additionally, it includes a stop-loss mechanism to manage risk and performance metrics to evaluate strategy effectiveness.

## Requirements
### Dependencies
The script requires the following Python libraries:
- `pandas`
- `numpy`

Ensure these are installed using:
```bash
pip install pandas numpy
```

### Input Data
The script expects a CSV file containing historical market data with the following columns:
- `Final_Condition`: Market state (Trending, Ranging, Choppy)
- `Supertrend`: Indicator signal (1 for uptrend, -1 for downtrend)
- `RSI`: Relative Strength Index value
- `Close`: Closing price of the asset
- `BB_Lower`, `BB_Upper`: Bollinger Band lower and upper bounds
- `Volatility_Breakout`: Binary indicator for breakout confirmation
- `ATR`: Average True Range, used for stop-loss calculations
- `Moving_Average_50`, `Moving_Average_200`: 50-day and 200-day moving averages for trend confirmation

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
This function executes trades based on the generated signals while implementing stop-loss and take-profit levels using ATR-based calculations. It also supports a trailing stop mechanism.

- Stop-loss is set at `entry_price - atr_multiplier * ATR`
- Take-profit is set at `entry_price + atr_multiplier * ATR`
- Trailing stop adjusts the stop-loss level as the price moves in favor
- Returns are adjusted for slippage and transaction costs.

### `calculate_performance_metrics(data)`
This function calculates key performance metrics:
- **Cumulative Return**: Overall strategy profit/loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Proportion of profitable trades
- **Number of Trades**: Total buy and sell signals executed

## Execution Flow
1. **Load dataset**: Reads historical data from a CSV file (`backtest_data_with_indicators.csv`).
2. **Generate trading signals**: Calls `generate_signals_advanced()`.
3. **Execute strategy with stop-loss management**: Calls `execute_strategy_with_stop_loss()`.
4. **Calculate performance metrics**: Calls `calculate_performance_metrics()` and prints results.

## Usage
Run the script using:
```bash
python advanced_ta_strategy.py
```
Ensure `backtest_data_with_indicators.csv` is in the same directory.

## Notes
- Adjust the `atr_multiplier` parameter to fine-tune stop-loss levels.
- Ensure that input data has all required indicators for accurate signal generation.

## License
This script is open-source and can be modified for research and trading development purposes.