#!/usr/bin/env python3
"""
Trading Strategy Simulator
==========================

A comprehensive backtesting framework for evaluating trading strategies
on minute-by-minute S&P 500 stock data.

Strategies implemented:
1. Moving Average Crossover (Momentum)
2. Bollinger Bands (Mean Reversion)
3. Breakout Strategy (Price Action)

Usage:
    python trading_simulator.py                    # Use default config
    python trading_simulator.py --preset quick_test # Use preset configuration
    python trading_simulator.py --symbols AAPL,MSFT # Test specific symbols
    python trading_simulator.py --max-rows 10000   # Limit data rows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import argparse
import sys
import os

# Import configuration
from config import STRATEGY_CONFIG, DATA_CONFIG, EXECUTION_CONFIG, OUTPUT_CONFIG, apply_preset

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class TradeResult:
    """Represents a single trade result"""
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: int
    strategy: str
    pnl: float
    return_pct: float
    hold_duration_minutes: int


@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    # Moving Average Crossover
    ma_short_period: int = field(default_factory=lambda: STRATEGY_CONFIG['ma_short_period'])
    ma_long_period: int = field(default_factory=lambda: STRATEGY_CONFIG['ma_long_period'])
    
    # Bollinger Bands
    bb_period: int = field(default_factory=lambda: STRATEGY_CONFIG['bb_period'])
    bb_std_dev: float = field(default_factory=lambda: STRATEGY_CONFIG['bb_std_dev'])
    
    # Breakout Strategy
    breakout_lookback: int = field(default_factory=lambda: STRATEGY_CONFIG['breakout_lookback'])
    profit_target_pct: float = field(default_factory=lambda: STRATEGY_CONFIG['profit_target_pct'])
    stop_loss_pct: float = field(default_factory=lambda: STRATEGY_CONFIG['stop_loss_pct'])
    
    # RSI Divergence Strategy
    rsi_period: int = field(default_factory=lambda: STRATEGY_CONFIG['rsi_period'])
    rsi_overbought: float = field(default_factory=lambda: STRATEGY_CONFIG['rsi_overbought'])
    rsi_oversold: float = field(default_factory=lambda: STRATEGY_CONFIG['rsi_oversold'])
    divergence_lookback: int = field(default_factory=lambda: STRATEGY_CONFIG['divergence_lookback'])
    
    # Opening Range Breakout Strategy
    orb_range_minutes: int = field(default_factory=lambda: STRATEGY_CONFIG['orb_range_minutes'])
    orb_market_open: str = field(default_factory=lambda: STRATEGY_CONFIG['orb_market_open'])
    
    # Enhanced Risk Management
    use_trend_filter: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_trend_filter'])
    trend_ma_period: int = field(default_factory=lambda: STRATEGY_CONFIG['trend_ma_period'])
    use_atr_targets: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_atr_targets'])
    atr_period: int = field(default_factory=lambda: STRATEGY_CONFIG['atr_period'])
    atr_stop_multiplier: float = field(default_factory=lambda: STRATEGY_CONFIG['atr_stop_multiplier'])
    atr_target_multiplier: float = field(default_factory=lambda: STRATEGY_CONFIG['atr_target_multiplier'])
    use_volume_filter: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_volume_filter'])
    volume_ma_period: int = field(default_factory=lambda: STRATEGY_CONFIG['volume_ma_period'])
    volume_threshold: float = field(default_factory=lambda: STRATEGY_CONFIG['volume_threshold'])
    
    # Portfolio-Level Risk Management
    use_dynamic_position_sizing: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_dynamic_position_sizing'])
    risk_per_trade_pct: float = field(default_factory=lambda: STRATEGY_CONFIG['risk_per_trade_pct'])
    max_concurrent_positions: int = field(default_factory=lambda: STRATEGY_CONFIG['max_concurrent_positions'])
    correlation_threshold: float = field(default_factory=lambda: STRATEGY_CONFIG['correlation_threshold'])
    
    # Market Regime Filtering
    use_market_regime_filter: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_market_regime_filter'])
    volatility_lookback: int = field(default_factory=lambda: STRATEGY_CONFIG['volatility_lookback'])
    low_vol_threshold: float = field(default_factory=lambda: STRATEGY_CONFIG['low_vol_threshold'])
    high_vol_threshold: float = field(default_factory=lambda: STRATEGY_CONFIG['high_vol_threshold'])
    
    # Enhanced Strategy Logic
    bb_opposite_band_exit: bool = field(default_factory=lambda: STRATEGY_CONFIG['bb_opposite_band_exit'])
    orb_end_of_day_exit: bool = field(default_factory=lambda: STRATEGY_CONFIG['orb_end_of_day_exit'])
    orb_fadeout_bars: int = field(default_factory=lambda: STRATEGY_CONFIG['orb_fadeout_bars'])
    
    # Advanced Strategy Features
    use_trailing_stops: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_trailing_stops'])
    trailing_stop_atr_multiplier: float = field(default_factory=lambda: STRATEGY_CONFIG['trailing_stop_atr_multiplier'])
    use_adx_filter: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_adx_filter'])
    adx_threshold: float = field(default_factory=lambda: STRATEGY_CONFIG['adx_threshold'])
    use_profit_zones: bool = field(default_factory=lambda: STRATEGY_CONFIG['use_profit_zones'])
    profit_zone_partial_exit: float = field(default_factory=lambda: STRATEGY_CONFIG['profit_zone_partial_exit'])
    
    # Strategy-Specific Market Regime Preferences
    enable_regime_specific_filtering: bool = field(default_factory=lambda: STRATEGY_CONFIG['enable_regime_specific_filtering'])
    bb_preferred_regime: str = field(default_factory=lambda: STRATEGY_CONFIG['bb_preferred_regime'])
    ma_preferred_regime: str = field(default_factory=lambda: STRATEGY_CONFIG['ma_preferred_regime'])
    breakout_preferred_regime: str = field(default_factory=lambda: STRATEGY_CONFIG['breakout_preferred_regime'])
    rsi_preferred_regime: str = field(default_factory=lambda: STRATEGY_CONFIG['rsi_preferred_regime'])
    orb_preferred_regime: str = field(default_factory=lambda: STRATEGY_CONFIG['orb_preferred_regime'])
    
    # Correlation Filtering
    enable_correlation_filter: bool = field(default_factory=lambda: STRATEGY_CONFIG['enable_correlation_filter'])
    correlation_lookback_period: int = field(default_factory=lambda: STRATEGY_CONFIG['correlation_lookback_period'])
    max_position_correlation: float = field(default_factory=lambda: STRATEGY_CONFIG['max_position_correlation'])
    
    # Walk-Forward Optimization
    enable_walk_forward: bool = field(default_factory=lambda: STRATEGY_CONFIG['enable_walk_forward'])
    training_period_months: int = field(default_factory=lambda: STRATEGY_CONFIG['training_period_months'])
    testing_period_months: int = field(default_factory=lambda: STRATEGY_CONFIG['testing_period_months'])
    
    # Trading Parameters
    initial_capital: float = field(default_factory=lambda: STRATEGY_CONFIG['initial_capital'])
    position_size_pct: float = field(default_factory=lambda: STRATEGY_CONFIG['position_size_pct'])
    transaction_cost_pct: float = field(default_factory=lambda: STRATEGY_CONFIG['transaction_cost_pct'])
    slippage_pct: float = field(default_factory=lambda: STRATEGY_CONFIG['slippage_pct'])


class DataLoader:
    """Handles loading and preprocessing of the stock market data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.symbols = []
        
    def load_data(self, max_rows: Optional[int] = None, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and parse the CSV data with complex header structure
        
        Args:
            max_rows: Maximum number of rows to load (None for no limit)
            start_date: Start date in 'YYYY-MM-DD' format (None for no limit)
            end_date: End date in 'YYYY-MM-DD' format (None for no limit)
        """
        print("Loading data from CSV...")
        
        if start_date or end_date:
            date_filter_msg = f"Date range: {start_date or 'beginning'} to {end_date or 'end'}"
            print(f"Applying date filter: {date_filter_msg}")
        
        # Read the first few rows to understand structure
        header_rows = pd.read_csv(self.file_path, nrows=3, header=None)
        
        # Extract symbols and data types from first two rows
        symbols_row = header_rows.iloc[0].fillna('').tolist()[1:]  # Skip first column (timestamp)
        datatypes_row = header_rows.iloc[1].fillna('').tolist()[1:]  # Skip first column
        
        # Create proper column names by combining symbol and data type
        columns = ['timestamp']
        current_symbol = None
        
        for i, (symbol, datatype) in enumerate(zip(symbols_row, datatypes_row)):
            if symbol and symbol.strip():  # New symbol found
                current_symbol = symbol.strip()
            
            if current_symbol and datatype and datatype.strip():
                col_name = f"{current_symbol}_{datatype.strip()}"
                columns.append(col_name)
            else:
                # Handle cases where datatype might be missing
                columns.append(f"col_{i+1}")
        
        print(f"Reading data with {len(columns)} columns...")
        
        # Load data in chunks to handle large file
        chunk_size = 5000  # Reduced chunk size for safety
        chunks = []
        
        try:
            reader = pd.read_csv(self.file_path, skiprows=3, names=columns, 
                               chunksize=chunk_size, nrows=max_rows, 
                               dtype='str', na_values=['', 'NaN', 'nan'])
            
            for i, chunk in enumerate(reader):
                # Convert timestamp
                try:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
                except:
                    print(f"Warning: Could not parse timestamps in chunk {i}")
                    continue
                
                # Remove rows with invalid timestamps
                chunk = chunk.dropna(subset=['timestamp'])
                
                # Apply date filtering if specified
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    chunk = chunk[chunk['timestamp'] >= start_dt]
                
                if end_date:
                    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include the entire end date
                    chunk = chunk[chunk['timestamp'] < end_dt]
                
                # Skip empty chunks after filtering
                if len(chunk) == 0:
                    continue
                
                # Convert numeric columns to float
                for col in chunk.columns:
                    if col != 'timestamp':
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                
                if len(chunk) > 0:
                    chunks.append(chunk)
                
                if i % 5 == 0:
                    print(f"Processed {(i+1) * chunk_size:,} rows...")
                    
        except Exception as e:
            print(f"Error reading data: {e}")
            return pd.DataFrame()
        
        if not chunks:
            print("No valid data chunks found!")
            return pd.DataFrame()
        
        # Combine all chunks
        self.data = pd.concat(chunks, ignore_index=True)
        
        # Extract unique symbols from column names
        self.symbols = []
        for col in self.data.columns:
            if '_close' in col:
                symbol = col.replace('_close', '')
                self.symbols.append(symbol)
        
        print(f"Loaded {len(self.data):,} rows with {len(self.symbols)} symbols")
        if len(self.data) > 0:
            print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
        return self.data
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Extract data for a specific symbol"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Look for columns with this symbol
        symbol_cols = ['timestamp']
        for col in self.data.columns:
            if col.startswith(f'{symbol}_'):
                symbol_cols.append(col)
        
        if len(symbol_cols) < 5:  # Need at least timestamp + OHLC
            print(f"Warning: Insufficient columns for {symbol}. Found: {symbol_cols}")
            return pd.DataFrame()
            
        symbol_data = self.data[symbol_cols].copy()
        
        # Rename columns to standard names
        new_cols = ['timestamp']
        for col in symbol_cols[1:]:  # Skip timestamp
            if '_open' in col:
                new_cols.append('open')
            elif '_high' in col:
                new_cols.append('high')
            elif '_low' in col:
                new_cols.append('low')
            elif '_close' in col:
                new_cols.append('close')
            elif '_volume' in col:
                new_cols.append('volume')
            else:
                new_cols.append(col.split('_')[-1])  # Use the last part after underscore
        
        symbol_data.columns = new_cols
        
        # Ensure we have the required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        available_cols = [col for col in required_cols if col in symbol_data.columns]
        
        if len(available_cols) < 4:
            print(f"Warning: Missing required OHLC columns for {symbol}")
            return pd.DataFrame()
        
        # Remove rows with missing price data
        symbol_data = symbol_data.dropna(subset=available_cols)
        
        # Set timestamp as index
        if 'timestamp' in symbol_data.columns:
            symbol_data.set_index('timestamp', inplace=True)
        
        return symbol_data


class TechnicalIndicators:
    """Calculate technical indicators for trading strategies"""
    
    @staticmethod
    def moving_average(prices: pd.Series, period: int) -> pd.Series:
        """Calculate simple moving average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def rolling_high_low(high_prices: pd.Series, low_prices: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate rolling maximum high and minimum low"""
        rolling_high = high_prices.rolling(window=period).max()
        rolling_low = low_prices.rolling(window=period).min()
        return rolling_high, rolling_low
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 20) -> pd.Series:
        """Detect bullish/bearish divergence between price and indicator"""
        divergence = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            price_window = price.iloc[i-lookback:i+1]
            indicator_window = indicator.iloc[i-lookback:i+1]
            
            # Bullish divergence: price makes lower low, indicator makes higher low
            if (price_window.iloc[-1] < price_window.iloc[0] and 
                indicator_window.iloc[-1] > indicator_window.iloc[0]):
                divergence.iloc[i] = 1
                
            # Bearish divergence: price makes higher high, indicator makes lower high
            elif (price_window.iloc[-1] > price_window.iloc[0] and 
                  indicator_window.iloc[-1] < indicator_window.iloc[0]):
                divergence.iloc[i] = -1
                
        return divergence
    
    @staticmethod
    def calculate_market_volatility(data: pd.Series, period: int = 20) -> pd.Series:
        """Calculate rolling volatility (standard deviation of returns)"""
        returns = data.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        return volatility
    
    @staticmethod
    def detect_market_regime(volatility: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
        """Classify market regime based on volatility percentiles"""
        vol_percentiles = volatility.rolling(window=252).rank(pct=True) * 100  # Use 1-year rolling percentiles
        
        regime = pd.Series('medium', index=volatility.index)
        regime[vol_percentiles <= low_threshold] = 'low_vol'
        regime[vol_percentiles >= high_threshold] = 'high_vol'
        
        return regime
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) for trend strength"""
        if len(high) < period * 2:
            return pd.Series([np.nan] * len(high), index=high.index)
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high.index)
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=high.index)
        
        # Calculate True Range
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        # Calculate smoothed +DM and -DM
        plus_dm_smooth = plus_dm.ewm(alpha=1/period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx


class MovingAverageCrossoverStrategy:
    """Moving Average Crossover Strategy Implementation"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = "Moving Average Crossover"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MA crossover with ADX trend filter"""
        signals = data.copy()
        
        # Calculate moving averages
        signals['ma_short'] = TechnicalIndicators.moving_average(
            data['close'], self.config.ma_short_period)
        signals['ma_long'] = TechnicalIndicators.moving_average(
            data['close'], self.config.ma_long_period)
        
        # Calculate ADX for trend strength filter
        if self.config.use_adx_filter:
            signals['adx'] = TechnicalIndicators.calculate_adx(
                data['high'], data['low'], data['close'])
        
        # Generate signals
        signals['signal'] = 0
        signals['position'] = 0
        
        # Create boolean masks for crossovers
        short_above_long = signals['ma_short'] > signals['ma_long']
        short_above_long_prev = signals['ma_short'].shift(1) > signals['ma_long'].shift(1)
        
        # Buy signal: short MA crosses above long MA (golden cross)
        buy_signal = short_above_long & ~short_above_long_prev
        
        # Sell signal: short MA crosses below long MA (death cross)  
        sell_signal = ~short_above_long & short_above_long_prev
        
        # Apply ADX filter if enabled
        if self.config.use_adx_filter:
            # Only allow signals when ADX indicates strong trend
            strong_trend = signals['adx'] >= self.config.adx_threshold
            buy_signal = buy_signal & strong_trend
            sell_signal = sell_signal & strong_trend
        
        # Apply signals
        signals.loc[buy_signal, 'signal'] = 1
        signals.loc[sell_signal, 'signal'] = -1
        
        # Calculate positions (1 when above, 0 when below)
        signals['position'] = short_above_long.astype(int)
        
        return signals


class BollingerBandsStrategy:
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = "Bollinger Bands"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on Bollinger Bands"""
        signals = data.copy()
        
        # Calculate Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            data['close'], self.config.bb_period, self.config.bb_std_dev)
        
        signals['bb_upper'] = upper
        signals['bb_middle'] = middle
        signals['bb_lower'] = lower
        
        # Generate signals
        signals['signal'] = 0
        signals['position'] = 0
        
        # Calculate positions (only hold position from buy to sell)
        position = 0
        position_size = 1.0  # Track remaining position size for partial exits
        positions = []
        buy_signals = []
        sell_signals = []
        
        for i, (idx, row) in enumerate(signals.iterrows()):
            signal = 0
            
            # Skip if we don't have valid Bollinger Bands data
            if pd.isna(row['bb_lower']) or pd.isna(row['bb_middle']) or pd.isna(row['bb_upper']):
                buy_signals.append(0)
                sell_signals.append(0)
                positions.append(position)
                continue
            
            # Buy signal: price touches or goes below lower band (oversold)
            if row['close'] <= row['bb_lower'] and position == 0:
                signal = 1
                position = 1
                position_size = 1.0  # Reset to full position
            
            # Enhanced exit logic with profit zones
            elif position == 1:
                if self.config.use_profit_zones and position_size > 0.5:
                    # Profit Zone 1: Partial exit at middle band
                    if row['close'] >= row['bb_middle']:
                        # Exit partial position at profit zone
                        exit_percentage = self.config.profit_zone_partial_exit
                        position_size -= exit_percentage
                        signal = -0.5  # Indicate partial exit
                        
                        # If we've exited enough, close remaining position
                        if position_size <= 0.5:
                            position_size = 0.5
                
                # Final exit conditions
                if self.config.bb_opposite_band_exit:
                    # Exit remaining position at upper band for maximum profit
                    if row['close'] >= row['bb_upper']:
                        signal = -1
                        position = 0
                        position_size = 0.0
                else:
                    # Original logic: exit at middle band (if not using profit zones)
                    if not self.config.use_profit_zones and row['close'] >= row['bb_middle']:
                        signal = -1
                        position = 0
                        position_size = 0.0
            
            buy_signals.append(1 if signal == 1 else 0)
            sell_signals.append(1 if signal == -1 or signal == -0.5 else 0)
            positions.append(position * position_size)  # Scale position by remaining size
        
        signals['signal'] = [1 if buy else (-1 if sell else 0) 
                           for buy, sell in zip(buy_signals, sell_signals)]
        signals['position'] = positions
        
        signals['signal'] = [1 if buy else (-1 if sell else 0) 
                           for buy, sell in zip(buy_signals, sell_signals)]
        signals['position'] = positions
        
        return signals


class BreakoutStrategy:
    """Price Action Breakout Strategy"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = "Breakout Strategy"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on breakout logic with trailing stops"""
        signals = data.copy()
        
        # Calculate rolling resistance and support levels
        resistance, support = TechnicalIndicators.rolling_high_low(
            data['high'], data['low'], self.config.breakout_lookback)
        
        signals['resistance'] = resistance
        signals['support'] = support
        
        # Calculate ATR for trailing stops
        signals['atr'] = TechnicalIndicators.calculate_atr(
            data['high'], data['low'], data['close'])
        
        # Generate signals
        signals['signal'] = 0
        signals['position'] = 0
        
        # Buy signal: price breaks above resistance
        signals['signal'] = np.where(
            (signals['close'] > signals['resistance'].shift(1)) & 
            (signals['close'].shift(1) <= signals['resistance'].shift(1)), 1, 0)
        
        # Calculate positions with enhanced exit logic including trailing stops
        position = 0
        entry_price = 0
        highest_price = 0
        positions = []
        
        for idx, row in signals.iterrows():
            if row['signal'] == 1 and position == 0:  # Buy signal
                position = 1
                entry_price = row['close']
                highest_price = row['close']
            elif position == 1:  # Currently in position
                # Update highest price
                highest_price = max(highest_price, row['close'])
                
                # Check profit target
                profit_pct = (row['close'] - entry_price) / entry_price * 100
                
                # Enhanced exit conditions with trailing stop
                should_exit = False
                
                # 1. Traditional profit target and stop loss
                if (profit_pct >= self.config.profit_target_pct or 
                    profit_pct <= -self.config.stop_loss_pct):
                    should_exit = True
                
                # 2. Support level break
                elif row['close'] < row['support']:
                    should_exit = True
                
                # 3. Trailing stop based on ATR
                elif (self.config.use_trailing_stops and 
                      not pd.isna(row['atr']) and row['atr'] > 0):
                    trailing_stop_price = highest_price - (row['atr'] * self.config.trailing_stop_atr_multiplier)
                    if row['close'] <= trailing_stop_price:
                        should_exit = True
                
                if should_exit:
                    position = 0
                    entry_price = 0
                    highest_price = 0
            
            positions.append(position)
        
        signals['position'] = positions
        
        return signals


class RSIDivergenceStrategy:
    """RSI Divergence Strategy - Advanced Mean Reversion"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = "RSI Divergence"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on RSI divergence"""
        signals = data.copy()
        
        # Calculate RSI
        signals['rsi'] = TechnicalIndicators.calculate_rsi(
            signals['close'], self.config.rsi_period)
        
        # Detect divergence
        signals['divergence'] = TechnicalIndicators.detect_divergence(
            signals['close'], signals['rsi'], self.config.divergence_lookback)
        
        # Generate signals
        signals['signal'] = 0
        signals['position'] = 0
        
        # Signal conditions with RSI levels
        buy_signals = ((signals['divergence'] == 1) & 
                      (signals['rsi'] < self.config.rsi_oversold))
        sell_signals = ((signals['divergence'] == -1) & 
                       (signals['rsi'] > self.config.rsi_overbought))
        
        # Calculate positions
        position = 0
        positions = []
        
        for idx, row in signals.iterrows():
            if buy_signals.loc[idx] and position == 0:
                position = 1
            elif sell_signals.loc[idx] and position == 1:
                position = 0
            elif position == 1 and row['rsi'] > self.config.rsi_overbought:
                position = 0  # Exit on overbought
            
            positions.append(position)
        
        signals['signal'] = [1 if buy else (-1 if sell else 0) 
                           for buy, sell in zip(buy_signals, sell_signals)]
        signals['position'] = positions
        
        return signals


class OpeningRangeBreakoutStrategy:
    """Opening Range Breakout Strategy - Time-based Day Trading"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = "Opening Range Breakout"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on opening range breakouts"""
        signals = data.copy()
        signals['signal'] = 0
        signals['position'] = 0
        signals['orb_high'] = np.nan
        signals['orb_low'] = np.nan
        
        # Convert timestamp to datetime if it's not already
        signals.index = pd.to_datetime(signals.index)
        
        # Group by date to handle multiple days
        for date, day_data in signals.groupby(signals.index.date):
            if len(day_data) == 0:
                continue
                
            # Define opening range (first N minutes of trading)
            market_open = pd.Timestamp.combine(date, pd.Timestamp(self.config.orb_market_open).time())
            range_end = market_open + pd.Timedelta(minutes=self.config.orb_range_minutes)
            
            # Get opening range data
            opening_range = day_data[(day_data.index >= market_open) & 
                                   (day_data.index <= range_end)]
            
            if len(opening_range) == 0:
                continue
                
            orb_high = opening_range['high'].max()
            orb_low = opening_range['low'].min()
            
            # Apply ORB levels to the entire day
            day_mask = signals.index.date == date
            signals.loc[day_mask, 'orb_high'] = orb_high
            signals.loc[day_mask, 'orb_low'] = orb_low
            
            # Generate signals for post-range period
            post_range = day_data[day_data.index > range_end]
            current_position = 0
            entry_bar_count = 0
            
            for idx, row in post_range.iterrows():
                current_time = idx.time()
                
                # End-of-day exit (force close before market close)
                if self.config.orb_end_of_day_exit and current_time >= pd.Timestamp('15:50').time():
                    if current_position != 0:
                        signals.loc[idx, 'signal'] = -current_position  # Exit signal
                        current_position = 0
                        entry_bar_count = 0
                    continue
                
                # Entry signals
                if current_position == 0:
                    # Buy signal: breakout above ORB high
                    if row['close'] > orb_high:
                        signals.loc[idx, 'signal'] = 1
                        current_position = 1
                        entry_bar_count = 0
                    # Sell signal: breakdown below ORB low  
                    elif row['close'] < orb_low:
                        signals.loc[idx, 'signal'] = -1
                        current_position = -1
                        entry_bar_count = 0
                
                # Fadeout exit: if breakout reverses back into range quickly
                elif current_position != 0:
                    entry_bar_count += 1
                    
                    if self.config.orb_fadeout_bars > 0 and entry_bar_count <= self.config.orb_fadeout_bars:
                        # Check for fadeout (reversal back into range)
                        if current_position == 1 and row['close'] < orb_high:
                            signals.loc[idx, 'signal'] = -1  # Exit long
                            current_position = 0
                            entry_bar_count = 0
                        elif current_position == -1 and row['close'] > orb_low:
                            signals.loc[idx, 'signal'] = 1   # Exit short
                            current_position = 0
                            entry_bar_count = 0
                
                # Update position tracking
                signals.loc[idx, 'position'] = current_position
        
        return signals


class BacktestEngine:
    """Main backtesting engine that runs strategies and calculates performance"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy(config),
            'bollinger_bands': BollingerBandsStrategy(config),
            'breakout': BreakoutStrategy(config),
            'rsi_divergence': RSIDivergenceStrategy(config),
            'opening_range_breakout': OpeningRangeBreakoutStrategy(config)
        }
        
        # Portfolio-level tracking
        self.open_positions = {}  # Track open positions across all strategies
        self.position_count = 0   # Current number of open positions
    
    def calculate_dynamic_position_size(self, entry_price: float, stop_loss_price: float) -> int:
        """Calculate position size based on risk per trade"""
        if not self.config.use_dynamic_position_sizing:
            # Use original fixed percentage method
            return int(self.config.initial_capital * self.config.position_size_pct / entry_price)
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Calculate total risk amount (percentage of portfolio)
        total_risk_amount = self.config.initial_capital * (self.config.risk_per_trade_pct / 100)
        
        # Calculate position size
        if risk_per_share > 0:
            position_size = int(total_risk_amount / risk_per_share)
            # Ensure position doesn't exceed maximum percentage of portfolio
            max_position_value = self.config.initial_capital * 0.2  # Max 20% per position
            max_shares = int(max_position_value / entry_price)
            return min(position_size, max_shares)
        else:
            return int(self.config.initial_capital * self.config.position_size_pct / entry_price)
    
    def check_portfolio_constraints(self, symbol: str, strategy_name: str, current_data: pd.DataFrame, all_data: dict) -> bool:
        """Check if new position violates portfolio constraints including correlation"""
        # Check maximum concurrent positions
        if self.position_count >= self.config.max_concurrent_positions:
            return False
        
        # Check for exact duplicate position
        position_key = f"{symbol}_{strategy_name}"
        if position_key in self.open_positions:
            return False
        
        # Check correlation if enabled and we have other positions
        if (self.config.enable_correlation_filter and 
            len(self.open_positions) > 0 and 
            symbol in all_data):
            
            # Calculate correlation with existing positions
            current_returns = current_data['close'].pct_change().dropna()
            
            for existing_position in self.open_positions.keys():
                existing_symbol = existing_position.split('_')[0]
                if existing_symbol in all_data and existing_symbol != symbol:
                    existing_returns = all_data[existing_symbol]['close'].pct_change().dropna()
                    
                    # Align the series for correlation calculation
                    aligned_current, aligned_existing = current_returns.align(existing_returns, join='inner')
                    
                    if len(aligned_current) >= self.config.correlation_lookback_period:
                        # Use recent data for correlation
                        recent_current = aligned_current.tail(self.config.correlation_lookback_period)
                        recent_existing = aligned_existing.tail(self.config.correlation_lookback_period)
                        
                        if len(recent_current) > 10:  # Ensure sufficient data
                            correlation = recent_current.corr(recent_existing)
                            
                            # Block if correlation is too high
                            if abs(correlation) > self.config.max_position_correlation:
                                return False
            
        return True
    
    def apply_trailing_stop(self, entry_price: float, current_price: float, highest_price: float, data_row: pd.Series) -> tuple:
        """Apply trailing stop loss logic
        Returns: (should_exit, new_highest_price)
        """
        if not self.config.use_trailing_stops:
            return False, highest_price
        
        # Update highest price if current price is higher
        new_highest_price = max(highest_price, current_price)
        
        # Calculate ATR-based trailing stop
        if 'atr' in data_row and not pd.isna(data_row['atr']):
            atr_value = data_row['atr']
        else:
            # Fallback to percentage-based stop if ATR not available
            atr_value = entry_price * 0.02  # 2% fallback
        
        # Calculate trailing stop price
        trailing_stop_price = new_highest_price - (atr_value * self.config.trailing_stop_atr_multiplier)
        
        # Exit if current price falls below trailing stop
        should_exit = current_price <= trailing_stop_price
        
        return should_exit, new_highest_price
    
    def update_portfolio_tracking(self, symbol: str, strategy_name: str, action: str):
        """Update portfolio-level position tracking"""
        position_key = f"{symbol}_{strategy_name}"
        
        if action == 'open':
            self.open_positions[position_key] = True
            self.position_count += 1
        elif action == 'close':
            if position_key in self.open_positions:
                del self.open_positions[position_key]
                self.position_count -= 1
    
    def apply_filters_and_enhancements(self, data: pd.DataFrame, signals: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        """Apply trend filter, volume filter, market regime, and dynamic risk management"""
        enhanced_signals = signals.copy()
        
        # 1. Enhanced Market Regime Filter with Strategy-Specific Logic
        if self.config.use_market_regime_filter:
            market_volatility = TechnicalIndicators.calculate_market_volatility(
                data['close'], self.config.volatility_lookback)
            market_regime = TechnicalIndicators.detect_market_regime(
                market_volatility, self.config.low_vol_threshold, self.config.high_vol_threshold)
            
            enhanced_signals['market_regime'] = market_regime
            enhanced_signals['market_volatility'] = market_volatility
            
            # Strategy-specific regime filtering
            if self.config.enable_regime_specific_filtering:
                regime_mask = pd.Series(True, index=enhanced_signals.index)
                
                # Get strategy preference from config
                strategy_regime_map = {
                    'bollinger_bands': self.config.bb_preferred_regime,
                    'ma_crossover': self.config.ma_preferred_regime,
                    'breakout': self.config.breakout_preferred_regime,
                    'rsi_divergence': self.config.rsi_preferred_regime,
                    'opening_range_breakout': self.config.orb_preferred_regime
                }
                
                preferred_regime = strategy_regime_map.get(strategy_name, 'any')
                
                if preferred_regime != 'any':
                    if preferred_regime == 'low_vol':
                        regime_mask = market_regime == 'low_vol'
                    elif preferred_regime == 'high_vol':
                        regime_mask = market_regime == 'high_vol'
                    elif preferred_regime == 'medium':
                        regime_mask = market_regime == 'medium'
                
                # Apply regime filter
                enhanced_signals['signal'] = np.where(regime_mask, enhanced_signals['signal'], 0)
            else:
                # Original logic for backward compatibility
                regime_mask = pd.Series(True, index=enhanced_signals.index)
                
                if strategy_name in ['ma_crossover', 'breakout']:
                    # Trend-following strategies work better in low/medium volatility
                    regime_mask = market_regime != 'high_vol'
                elif strategy_name in ['bollinger_bands', 'rsi_divergence']:
                    # Mean reversion strategies can work in all regimes but excel in high volatility
                    regime_mask = pd.Series(True, index=enhanced_signals.index)
                
                # Apply regime filter
                enhanced_signals['signal'] = np.where(regime_mask, enhanced_signals['signal'], 0)
        
        # 2. Trend Filter
        if self.config.use_trend_filter:
            trend_ma = TechnicalIndicators.moving_average(
                data['close'], self.config.trend_ma_period)
            enhanced_signals['trend_ma'] = trend_ma
            
            # Only allow buy signals when price is above trend MA
            # Only allow sell signals when price is below trend MA
            buy_mask = data['close'] > trend_ma
            sell_mask = data['close'] < trend_ma
            
            # Filter signals based on trend
            enhanced_signals['signal'] = np.where(
                (enhanced_signals['signal'] == 1) & buy_mask, 1,
                np.where((enhanced_signals['signal'] == -1) & sell_mask, -1, 0)
            )
        
        # 3. Volume Filter
        if self.config.use_volume_filter and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(window=self.config.volume_ma_period).mean()
            volume_condition = data['volume'] > (volume_ma * self.config.volume_threshold)
            enhanced_signals['volume_ma'] = volume_ma
            enhanced_signals['volume_confirmed'] = volume_condition
            
            # Only keep signals with volume confirmation
            enhanced_signals['signal'] = np.where(
                volume_condition, enhanced_signals['signal'], 0
            )
        
        # 4. ATR-based Dynamic Risk Management
        if self.config.use_atr_targets:
            atr = TechnicalIndicators.calculate_atr(
                data['high'], data['low'], data['close'], self.config.atr_period)
            enhanced_signals['atr'] = atr
            
            # Calculate dynamic stop loss and profit targets
            enhanced_signals['dynamic_stop_loss'] = data['close'] - (atr * self.config.atr_stop_multiplier)
            enhanced_signals['dynamic_profit_target'] = data['close'] + (atr * self.config.atr_target_multiplier)
        
        return enhanced_signals
        
    def run_backtest(self, symbol: str, data: pd.DataFrame, all_data: dict = None) -> Dict:
        """Run backtest for all strategies on a single symbol"""
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            print(f"Running {strategy.name} strategy on {symbol}...")
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Apply filters and enhancements
            enhanced_signals = self.apply_filters_and_enhancements(data, signals, strategy_name)
            
            # Apply portfolio-level constraints (now with correlation checking)
            portfolio_filtered_signals = self._apply_portfolio_constraints(
                enhanced_signals, symbol, strategy_name, data, all_data)
            
            # Calculate trades and performance
            trades = self._calculate_trades(symbol, portfolio_filtered_signals, strategy_name)
            performance = self._calculate_performance(trades)
            
            results[strategy_name] = {
                'signals': signals,
                'trades': trades,
                'performance': performance,
                'strategy_name': strategy.name
            }
        
        return results
    
    def _apply_portfolio_constraints(self, signals: pd.DataFrame, symbol: str, strategy_name: str, current_data: pd.DataFrame, all_data: dict = None) -> pd.DataFrame:
        """Apply portfolio-level position limits and constraints"""
        filtered_signals = signals.copy()
        
        # Track positions for this specific symbol/strategy combination
        current_position = 0
        
        for idx, row in filtered_signals.iterrows():
            original_signal = row['signal']
            
            # Entry signal
            if original_signal != 0 and current_position == 0:
                # Check portfolio constraints before allowing entry (including correlation)
                if self.check_portfolio_constraints(symbol, strategy_name, current_data, all_data or {}):
                    # Allow the signal and update tracking
                    current_position = original_signal
                    self.update_portfolio_tracking(symbol, strategy_name, 'open')
                else:
                    # Block the signal due to portfolio constraints
                    filtered_signals.loc[idx, 'signal'] = 0
            
            # Exit signal
            elif original_signal != 0 and current_position != 0:
                # Allow exit and update tracking
                current_position = 0
                self.update_portfolio_tracking(symbol, strategy_name, 'close')
            
            # Update position column
            filtered_signals.loc[idx, 'position'] = current_position
        
        return filtered_signals
    
    def _calculate_trades(self, symbol: str, signals: pd.DataFrame, strategy_name: str) -> List[TradeResult]:
        """Calculate individual trades from position signals"""
        trades = []
        position = 0
        entry_time = None
        entry_price = 0
        
        for timestamp, row in signals.iterrows():
            current_position = row['position']
            
            # Entry: position changes from 0 to 1
            if current_position == 1 and position == 0:
                position = 1
                entry_time = timestamp
                entry_price = row['close']
            
            # Exit: position changes from 1 to 0
            elif current_position == 0 and position == 1:
                position = 0
                exit_price = row['close']
                
                # Calculate dynamic position size based on risk
                stop_loss_price = entry_price
                if 'dynamic_stop_loss' in row and pd.notna(row['dynamic_stop_loss']):
                    stop_loss_price = row['dynamic_stop_loss']
                elif 'atr' in row and pd.notna(row['atr']):
                    stop_loss_price = entry_price - (row['atr'] * self.config.atr_stop_multiplier)
                else:
                    # Fallback to percentage-based stop
                    stop_loss_price = entry_price * (1 - self.config.stop_loss_pct / 100)
                
                quantity = self.calculate_dynamic_position_size(entry_price, stop_loss_price)
                gross_pnl = (exit_price - entry_price) * quantity
                
                # Apply transaction costs and slippage
                transaction_cost = (entry_price + exit_price) * quantity * self.config.transaction_cost_pct
                slippage_cost = (entry_price + exit_price) * quantity * self.config.slippage_pct
                net_pnl = gross_pnl - transaction_cost - slippage_cost
                
                return_pct = (exit_price - entry_price) / entry_price * 100
                hold_duration = (timestamp - entry_time).total_seconds() / 60  # minutes
                
                trade = TradeResult(
                    symbol=symbol,
                    entry_time=str(entry_time),
                    exit_time=str(timestamp),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    strategy=strategy_name,
                    pnl=net_pnl,
                    return_pct=return_pct,
                    hold_duration_minutes=int(hold_duration)
                )
                
                trades.append(trade)
        
        return trades
    
    def _calculate_performance(self, trades: List[TradeResult]) -> Dict:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'avg_return_pct': 0,
                'max_win': 0,
                'max_loss': 0,
                'avg_hold_time_minutes': 0
            }
        
        returns = [trade.return_pct for trade in trades]
        pnls = [trade.pnl for trade in trades]
        
        # Calculate true cumulative return based on actual P&L vs initial capital
        # NOTE: We don't sum individual trade returns (mathematically incorrect)
        # Instead, we calculate total return as: (Total P&L / Initial Capital) * 100
        initial_capital = self.config.initial_capital
        total_pnl = sum(pnls)
        true_total_return_pct = (total_pnl / initial_capital) * 100
        
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'total_pnl': total_pnl,
            'total_return_pct': true_total_return_pct,
            'avg_return_pct': np.mean(returns),
            'max_win': max(returns) if returns else 0,
            'max_loss': min(returns) if returns else 0,
            'avg_hold_time_minutes': np.mean([trade.hold_duration_minutes for trade in trades])
        }


class ResultsAnalyzer:
    """Analyze and visualize backtest results"""
    
    def __init__(self):
        pass
    
    def print_performance_summary(self, results: Dict, symbol: str):
        """Print a formatted performance summary"""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE SUMMARY FOR {symbol}")
        print(f"{'='*60}")
        
        for strategy_name, strategy_results in results.items():
            perf = strategy_results['performance']
            strategy_display_name = strategy_results['strategy_name']
            
            print(f"\n{strategy_display_name}:")
            print(f"  Total Trades: {perf['total_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.1f}%")
            print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
            print(f"  Total Return: {perf['total_return_pct']:.2f}%")
            print(f"  Avg Return per Trade: {perf['avg_return_pct']:.2f}%")
            print(f"  Best Trade: {perf['max_win']:.2f}%")
            print(f"  Worst Trade: {perf['max_loss']:.2f}%")
            print(f"  Avg Hold Time: {perf['avg_hold_time_minutes']:.0f} minutes")
    
    def plot_strategy_comparison(self, results: Dict, symbol: str):
        """Create comparison plots for all strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Strategy Comparison for {symbol}', fontsize=16)
        
        # Prepare data for plotting
        strategies = []
        total_returns = []
        win_rates = []
        total_trades = []
        avg_returns = []
        
        for strategy_name, strategy_results in results.items():
            perf = strategy_results['performance']
            strategies.append(strategy_results['strategy_name'])
            total_returns.append(perf['total_return_pct'])
            win_rates.append(perf['win_rate'])
            total_trades.append(perf['total_trades'])
            avg_returns.append(perf['avg_return_pct'])
        
        # Plot 1: Total Returns
        axes[0, 0].bar(strategies, total_returns, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Total Return (%)')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Win Rates
        axes[0, 1].bar(strategies, win_rates, color=['gold', 'orange', 'red'])
        axes[0, 1].set_title('Win Rate (%)')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Number of Trades
        axes[1, 0].bar(strategies, total_trades, color=['purple', 'blue', 'green'])
        axes[1, 0].set_title('Total Number of Trades')
        axes[1, 0].set_ylabel('Number of Trades')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Average Return per Trade
        axes[1, 1].bar(strategies, avg_returns, color=['pink', 'yellow', 'cyan'])
        axes[1, 1].set_title('Average Return per Trade (%)')
        axes[1, 1].set_ylabel('Avg Return (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'/home/malmorga/ml/{symbol}_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trading Strategy Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trading_simulator.py                           # Use default settings
  python trading_simulator.py --preset quick_test       # Use quick test preset
  python trading_simulator.py --symbols AAPL,MSFT,GOOGL # Test specific symbols
  python trading_simulator.py --start-date 2017-10-01 --end-date 2017-12-31  # October-December 2017
  python trading_simulator.py --start-date 2018-01-01   # From January 2018 onwards
  python trading_simulator.py --end-date 2017-11-30     # Up to November 2017
  python trading_simulator.py --start-date 2017-09-15 --symbols AAPL,MSFT  # Date range + specific symbols
  
Available presets: quick_test, major_stocks, full_analysis, tech_stocks, sample_run
Dataset date range: 2017-09-11 to 2018-02-16
        """
    )
    
    parser.add_argument('--preset', type=str, help='Use a predefined configuration preset')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to test (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--start-date', type=str, help='Start date for analysis (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, help='End date for analysis (YYYY-MM-DD format)')
    parser.add_argument('--max-rows', type=int, help='Maximum number of data rows to load (legacy option)')
    parser.add_argument('--count', type=int, help='Number of symbols to test (if not using --symbols)')
    parser.add_argument('--exclude', type=str, help='Comma-separated list of symbols to exclude')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-charts', action='store_true', help='Disable chart generation')
    parser.add_argument('--output-dir', type=str, help='Output directory for charts and reports')
    
    return parser.parse_args()

def configure_from_args(args):
    """Configure the simulator based on command line arguments"""
    
    # Apply preset first if specified
    if args.preset:
        try:
            apply_preset(args.preset)
            print(f"✅ Applied preset: {args.preset}")
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    # Override with command line arguments
    if args.symbols:
        DATA_CONFIG['specific_symbols'] = [s.strip() for s in args.symbols.split(',')]
        DATA_CONFIG['test_symbols_count'] = None
        print(f"✅ Testing specific symbols: {DATA_CONFIG['specific_symbols']}")
    
    if args.start_date:
        DATA_CONFIG['date_range']['start_date'] = args.start_date
        print(f"✅ Start date: {args.start_date}")
    
    if args.end_date:
        DATA_CONFIG['date_range']['end_date'] = args.end_date
        print(f"✅ End date: {args.end_date}")
    
    if args.max_rows:
        DATA_CONFIG['max_rows'] = args.max_rows
        print(f"✅ Limited to {args.max_rows:,} rows (legacy option)")
    
    if args.count:
        DATA_CONFIG['test_symbols_count'] = args.count
        if not args.symbols:  # Only apply if no specific symbols provided
            DATA_CONFIG['specific_symbols'] = None
        print(f"✅ Testing {args.count} symbols")
    
    if args.exclude:
        DATA_CONFIG['exclude_symbols'] = [s.strip() for s in args.exclude.split(',')]
        print(f"✅ Excluding symbols: {DATA_CONFIG['exclude_symbols']}")
    
    if args.verbose:
        EXECUTION_CONFIG['verbose_output'] = True
        print("✅ Verbose output enabled")
    
    if args.no_charts:
        OUTPUT_CONFIG['save_charts'] = False
        print("✅ Chart generation disabled")
    
    if args.output_dir:
        OUTPUT_CONFIG['output_directory'] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"✅ Output directory: {args.output_dir}")

def select_test_symbols(data_loader):
    """Select symbols to test based on configuration"""
    available_symbols = data_loader.symbols
    
    if EXECUTION_CONFIG['verbose_output']:
        print(f"Available symbols: {len(available_symbols)}")
        print(f"First 10: {available_symbols[:10]}")
    
    # Apply exclusions
    if DATA_CONFIG['exclude_symbols']:
        available_symbols = [s for s in available_symbols if s not in DATA_CONFIG['exclude_symbols']]
        if EXECUTION_CONFIG['verbose_output']:
            print(f"After exclusions: {len(available_symbols)} symbols")
    
    # Select symbols based on configuration
    if DATA_CONFIG['specific_symbols']:
        # Use specific symbols (filter to those available)
        test_symbols = [s for s in DATA_CONFIG['specific_symbols'] if s in available_symbols]
        missing = [s for s in DATA_CONFIG['specific_symbols'] if s not in available_symbols]
        if missing and EXECUTION_CONFIG['verbose_output']:
            print(f"⚠️  Symbols not found in data: {missing}")
    else:
        # Use count-based selection
        count = DATA_CONFIG['test_symbols_count']
        if count is None:
            test_symbols = available_symbols  # Use all
        else:
            test_symbols = available_symbols[:count]
    
    return test_symbols

def main():
    """Main function to run the trading strategy simulator"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("Trading Strategy Simulator")
    print("=" * 40)
    
    # Configure based on arguments
    configure_from_args(args)
    
    # Display current configuration
    if EXECUTION_CONFIG['verbose_output']:
        print(f"\n📊 Configuration:")
        print(f"   Max rows: {DATA_CONFIG['max_rows']}")
        print(f"   Specific symbols: {DATA_CONFIG['specific_symbols']}")
        print(f"   Symbol count: {DATA_CONFIG['test_symbols_count']}")
        print(f"   Exclude symbols: {DATA_CONFIG['exclude_symbols']}")
        print(f"   Save charts: {OUTPUT_CONFIG['save_charts']}")
    
    # Configuration
    config = StrategyConfig()
    
    # Display strategy parameters if verbose
    if EXECUTION_CONFIG['verbose_output']:
        print(f"\n🔧 Strategy Parameters:")
        print(f"   MA Periods: {config.ma_short_period}/{config.ma_long_period}")
        print(f"   BB Period/StdDev: {config.bb_period}/{config.bb_std_dev}")
        print(f"   Breakout Lookback: {config.breakout_lookback}")
        print(f"   Profit Target: {config.profit_target_pct}%, Stop Loss: {config.stop_loss_pct}%")
    
    # Load data
    data_loader = DataLoader('/home/malmorga/ml/dataset.csv')
    
    if EXECUTION_CONFIG['verbose_output']:
        print(f"\n📁 Loading data...")
    
    # Load data with date filtering
    data_loader.load_data(
        max_rows=DATA_CONFIG['max_rows'],
        start_date=DATA_CONFIG['date_range']['start_date'],
        end_date=DATA_CONFIG['date_range']['end_date']
    )
    
    if data_loader.data is None or len(data_loader.data) == 0:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Select test symbols
    test_symbols = select_test_symbols(data_loader)
    
    if not test_symbols:
        print("❌ No symbols selected for testing. Exiting.")
        return
    
    print(f"\n🎯 Testing strategies on {len(test_symbols)} symbols: {test_symbols[:10]}")
    if len(test_symbols) > 10:
        print(f"    ... and {len(test_symbols) - 10} more")
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config)
    analyzer = ResultsAnalyzer()
    
    all_results = {}
    successful_runs = 0
    
    for i, symbol in enumerate(test_symbols, 1):
        if EXECUTION_CONFIG['verbose_output']:
            print(f"\n{'='*50}")
            print(f"Processing {symbol} ({i}/{len(test_symbols)})")
            print(f"{'='*50}")
        else:
            print(f"Processing {symbol}... ", end='', flush=True)
        
        # Get symbol data
        symbol_data = data_loader.get_symbol_data(symbol)
        
        if len(symbol_data) < DATA_CONFIG['min_data_points']:
            if EXECUTION_CONFIG['verbose_output']:
                print(f"⚠️  Insufficient data for {symbol} ({len(symbol_data)} points), skipping...")
            else:
                print("❌ (insufficient data)")
            continue
        
        # Run backtest
        try:
            results = backtest_engine.run_backtest(symbol, symbol_data)
            all_results[symbol] = results
            successful_runs += 1
            
            if EXECUTION_CONFIG['verbose_output']:
                # Print results
                analyzer.print_performance_summary(results, symbol)
            else:
                print("✅")
            
            # Create plots if enabled
            if OUTPUT_CONFIG['save_charts']:
                analyzer.plot_strategy_comparison(results, symbol)
                
        except Exception as e:
            if EXECUTION_CONFIG['verbose_output']:
                print(f"❌ Error processing {symbol}: {e}")
            else:
                print("❌ (error)")
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_runs}/{len(test_symbols)} symbols")
    
    if OUTPUT_CONFIG['save_charts'] and successful_runs > 0:
        output_dir = OUTPUT_CONFIG['output_directory']
        print(f"📊 Charts saved to: {output_dir}")
    
    if successful_runs == 0:
        print("⚠️  No symbols were successfully processed.")
    
    return all_results


if __name__ == "__main__":
    main()
