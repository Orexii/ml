#!/usr/bin/env python3
"""
Strategy Analysis and Optimization
==================================

Analyze why certain strategies aren't generating trades and optimize parameters.
"""

import sys
sys.path.append('/home/malmorga/ml')

from trading_simulator import DataLoader, StrategyConfig, MovingAverageCrossoverStrategy, BollingerBandsStrategy, TechnicalIndicators
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_ma_strategy(data, symbol):
    """Analyze Moving Average strategy for a specific symbol"""
    print(f"\n{'='*50}")
    print(f"Moving Average Analysis for {symbol}")
    print(f"{'='*50}")
    
    config = StrategyConfig()
    
    # Calculate MAs
    ma_short = TechnicalIndicators.moving_average(data['close'], config.ma_short_period)
    ma_long = TechnicalIndicators.moving_average(data['close'], config.ma_long_period)
    
    # Check for valid data
    valid_ma_short = ~ma_short.isna()
    valid_ma_long = ~ma_long.isna()
    valid_data = valid_ma_short & valid_ma_long
    
    if valid_data.sum() == 0:
        print(f"❌ No valid MA data for {symbol}")
        return
    
    print(f"Data points with valid MAs: {valid_data.sum()}")
    if len(ma_short.dropna()) > 0:
        print(f"MA Short ({config.ma_short_period}): {ma_short.dropna().iloc[-1]:.2f}")
    if len(ma_long.dropna()) > 0:
        print(f"MA Long ({config.ma_long_period}): {ma_long.dropna().iloc[-1]:.2f}")
    
    # Check for crossovers
    crossovers_up = 0
    crossovers_down = 0
    
    for i in range(1, len(data)):
        if (not pd.isna(ma_short.iloc[i]) and not pd.isna(ma_long.iloc[i]) and
            not pd.isna(ma_short.iloc[i-1]) and not pd.isna(ma_long.iloc[i-1])):
            if (ma_short.iloc[i] > ma_long.iloc[i] and 
                ma_short.iloc[i-1] <= ma_long.iloc[i-1]):
                crossovers_up += 1
            elif (ma_short.iloc[i] < ma_long.iloc[i] and 
                  ma_short.iloc[i-1] >= ma_long.iloc[i-1]):
                crossovers_down += 1
    
    print(f"Golden crosses (bullish): {crossovers_up}")
    print(f"Death crosses (bearish): {crossovers_down}")
    
    # Plot the MAs
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label='Close Price', alpha=0.7)
    plt.plot(data.index, ma_short, label=f'MA {config.ma_short_period}', linewidth=2)
    plt.plot(data.index, ma_long, label=f'MA {config.ma_long_period}', linewidth=2)
    plt.title(f'Moving Averages for {symbol}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'/home/malmorga/ml/{symbol}_moving_averages.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_bb_strategy(data, symbol):
    """Analyze Bollinger Bands strategy for a specific symbol"""
    print(f"\n{'='*50}")
    print(f"Bollinger Bands Analysis for {symbol}")
    print(f"{'='*50}")
    
    config = StrategyConfig()
    
    # Calculate Bollinger Bands
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        data['close'], config.bb_period, config.bb_std_dev)
    
    # Check for touches
    lower_touches = (data['close'] <= lower).sum()
    upper_touches = (data['close'] >= upper).sum()
    
    print(f"Lower band touches: {lower_touches}")
    print(f"Upper band touches: {upper_touches}")
    print(f"Current price: {data['close'].iloc[-1]:.2f}")
    print(f"Current bands - Upper: {upper.iloc[-1]:.2f}, Middle: {middle.iloc[-1]:.2f}, Lower: {lower.iloc[-1]:.2f}")
    
    # Calculate band width
    band_width = ((upper - lower) / middle * 100).dropna()
    print(f"Average band width: {band_width.mean():.2f}%")
    print(f"Current band width: {band_width.iloc[-1]:.2f}%")
    
    # Plot Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label='Close Price', linewidth=2)
    plt.plot(data.index, upper, label='Upper Band', linestyle='--', alpha=0.7)
    plt.plot(data.index, middle, label='Middle Band (SMA)', alpha=0.7)
    plt.plot(data.index, lower, label='Lower Band', linestyle='--', alpha=0.7)
    plt.fill_between(data.index, upper, lower, alpha=0.1)
    plt.title(f'Bollinger Bands for {symbol}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'/home/malmorga/ml/{symbol}_bollinger_bands.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_alternative_parameters():
    """Test different parameter combinations"""
    print(f"\n{'='*60}")
    print("TESTING ALTERNATIVE PARAMETERS")
    print(f"{'='*60}")
    
    # Load data
    data_loader = DataLoader('/home/malmorga/ml/dataset.csv')
    data_loader.load_data(max_rows=10000)  # Smaller dataset for testing
    
    # Test different MA periods
    ma_combinations = [
        (5, 15),   # Very short term
        (10, 20),  # Short term
        (20, 50),  # Original
        (15, 30),  # Medium term
    ]
    
    symbol = 'AAPL'  # Test on Apple
    symbol_data = data_loader.get_symbol_data(symbol)
    
    if len(symbol_data) == 0:
        print(f"No data for {symbol}")
        return
    
    print(f"Testing MA combinations on {symbol} ({len(symbol_data)} data points)")
    
    for short, long in ma_combinations:
        ma_short = TechnicalIndicators.moving_average(symbol_data['close'], short)
        ma_long = TechnicalIndicators.moving_average(symbol_data['close'], long)
        
        # Count crossovers
        crossovers = 0
        for i in range(1, len(symbol_data)):
            if (not pd.isna(ma_short.iloc[i]) and not pd.isna(ma_long.iloc[i]) and
                not pd.isna(ma_short.iloc[i-1]) and not pd.isna(ma_long.iloc[i-1])):
                if (ma_short.iloc[i] > ma_long.iloc[i] and 
                    ma_short.iloc[i-1] <= ma_long.iloc[i-1]) or \
                   (ma_short.iloc[i] < ma_long.iloc[i] and 
                    ma_short.iloc[i-1] >= ma_long.iloc[i-1]):
                    crossovers += 1
        
        print(f"MA({short},{long}): {crossovers} crossovers")

def main():
    """Main analysis function"""
    print("Trading Strategy Analysis")
    print("=" * 40)
    
    # Load data
    data_loader = DataLoader('/home/malmorga/ml/dataset.csv')
    data_loader.load_data(max_rows=5000)  # Smaller sample for analysis
    
    # Analyze first available symbol
    if data_loader.symbols:
        symbol = data_loader.symbols[0]  # Use first symbol
        symbol_data = data_loader.get_symbol_data(symbol)
        
        if len(symbol_data) > 100:  # Need sufficient data
            analyze_ma_strategy(symbol_data, symbol)
            analyze_bb_strategy(symbol_data, symbol)
        else:
            print(f"Insufficient data for {symbol}")
    
    # Test alternative parameters
    test_alternative_parameters()

if __name__ == "__main__":
    main()
