#!/usr/bin/env python3
"""
Quick Data Validation Script
============================

Run this script first to validate that the data loading works correctly
before running the full trading strategy simulator.
"""

import sys
import os
sys.path.append('/home/malmorga/ml')

from trading_simulator import DataLoader, StrategyConfig, TechnicalIndicators
import pandas as pd
import numpy as np

def test_data_loading():
    """Test the data loading functionality"""
    print("Testing Data Loading...")
    print("=" * 40)
    
    # Initialize data loader
    data_loader = DataLoader('/home/malmorga/ml/dataset.csv')
    
    # Load a small sample of data
    print("Loading first 1000 rows...")
    data = data_loader.load_data(max_rows=1000)
    
    if data is None or len(data) == 0:
        print("❌ Data loading failed!")
        return False
    
    print(f"✅ Successfully loaded {len(data):,} rows")
    print(f"✅ Found {len(data_loader.symbols)} symbols")
    print(f"✅ Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Test symbol extraction
    print(f"\nFirst 10 symbols: {data_loader.symbols[:10]}")
    
    # Test individual symbol data extraction
    if data_loader.symbols:
        test_symbol = data_loader.symbols[0]
        print(f"\nTesting data extraction for symbol: {test_symbol}")
        
        symbol_data = data_loader.get_symbol_data(test_symbol)
        
        if len(symbol_data) > 0:
            print(f"✅ Successfully extracted {len(symbol_data)} rows for {test_symbol}")
            print(f"✅ Columns: {symbol_data.columns.tolist()}")
            print(f"\nSample data for {test_symbol}:")
            print(symbol_data.head())
            
            # Check for missing data
            missing_data = symbol_data.isnull().sum()
            print(f"\nMissing data count:")
            print(missing_data)
            
            return True
        else:
            print(f"❌ No data extracted for {test_symbol}")
            return False
    
    return False

def test_technical_indicators():
    """Test technical indicator calculations"""
    print("\n" + "=" * 40)
    print("Testing Technical Indicators...")
    print("=" * 40)
    
    # Create sample data
    dates = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(100) * 0.05,
        'high': prices + abs(np.random.randn(100) * 0.1),
        'low': prices - abs(np.random.randn(100) * 0.1),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }).set_index('timestamp')
    
    from trading_simulator import TechnicalIndicators
    
    # Test moving average
    ma_20 = TechnicalIndicators.moving_average(sample_data['close'], 20)
    print(f"✅ Moving Average (20): {ma_20.dropna().iloc[-1]:.2f}")
    
    # Test Bollinger Bands
    upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_data['close'], 20, 2.0)
    print(f"✅ Bollinger Bands - Upper: {upper.dropna().iloc[-1]:.2f}, "
          f"Middle: {middle.dropna().iloc[-1]:.2f}, Lower: {lower.dropna().iloc[-1]:.2f}")
    
    # Test rolling high/low
    rolling_high, rolling_low = TechnicalIndicators.rolling_high_low(
        sample_data['high'], sample_data['low'], 20)
    print(f"✅ Rolling High/Low - High: {rolling_high.dropna().iloc[-1]:.2f}, "
          f"Low: {rolling_low.dropna().iloc[-1]:.2f}")
    
    return True

def main():
    """Main test function"""
    print("Trading Strategy Simulator - Data Validation")
    print("=" * 50)
    
    # Test data loading
    data_success = test_data_loading()
    
    if not data_success:
        print("\n❌ Data loading tests failed. Please check your dataset.csv file.")
        return
    
    # Test technical indicators
    indicator_success = test_technical_indicators()
    
    if data_success and indicator_success:
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("✅ Ready to run the full trading simulator")
        print("=" * 50)
        print("\nTo run the full simulation, execute:")
        print("python trading_simulator.py")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")

if __name__ == "__main__":
    main()
