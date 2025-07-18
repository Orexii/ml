#!/usr/bin/env python3
"""
SPY Trading Strategy Analysis
Analyzes trading strategies on SPY ETF data and compares against buy-and-hold returns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our existing trading components
from trading_simulator import (
    StrategyConfig, TechnicalIndicators, MovingAverageCrossoverStrategy,
    BollingerBandsStrategy, BreakoutStrategy, RSIDivergenceStrategy,
    OpeningRangeBreakoutStrategy
)
from config import STRATEGY_CONFIG


class SPYDataLoader:
    """Handles loading and preprocessing of SPY ETF data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        
    def load_data(self, start_date: str = None, end_date: str = None, max_rows: int = None) -> pd.DataFrame:
        """Load SPY CSV data with proper date parsing"""
        print(f"Loading SPY data from: {self.file_path}")
        
        try:
            # Load the CSV
            df = pd.read_csv(self.file_path)
            
            # Parse the date format: "20090522  07:30:00" -> "2009-05-22 07:30:00"
            df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d  %H:%M:%S')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Sort by datetime to ensure proper order
            df.sort_index(inplace=True)
            
            # Apply date filtering if specified
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
                print(f"Filtered data from {start_date}")
                
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
                print(f"Filtered data to {end_date}")
            
            # Apply row limit if specified
            if max_rows and len(df) > max_rows:
                df = df.tail(max_rows)  # Take most recent data
                print(f"Limited to most recent {max_rows} rows")
            
            # Ensure we have the required columns and clean data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in SPY data")
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with missing price data
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Store the cleaned data
            self.data = df[required_columns].copy()
            
            print(f"✅ Loaded {len(self.data):,} rows of SPY data")
            print(f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"💰 Price range: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading SPY data: {e}")
            return None


class SPYBenchmarkAnalyzer:
    """Analyzes buy-and-hold SPY performance for benchmarking"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def calculate_buy_hold_returns(self) -> Dict:
        """Calculate buy-and-hold returns for SPY"""
        start_price = self.data['close'].iloc[0]
        end_price = self.data['close'].iloc[-1]
        
        total_return = (end_price - start_price) / start_price
        num_years = (self.data.index[-1] - self.data.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1/num_years) - 1
        
        # Calculate daily returns for volatility
        daily_data = self.data.resample('D').last().dropna()
        daily_returns = daily_data['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'start_price': start_price,
            'end_price': end_price,
            'start_date': self.data.index[0],
            'end_date': self.data.index[-1],
            'num_years': num_years
        }
    
    def calculate_annual_returns(self) -> pd.DataFrame:
        """Calculate annual returns for year-by-year comparison"""
        daily_data = self.data.resample('D').last().dropna()
        annual_returns = []
        
        for year in range(daily_data.index.year.min(), daily_data.index.year.max() + 1):
            year_data = daily_data[daily_data.index.year == year]
            if len(year_data) > 0:
                start_price = year_data['close'].iloc[0]
                end_price = year_data['close'].iloc[-1]
                annual_return = (end_price - start_price) / start_price
                annual_returns.append({
                    'year': year,
                    'return': annual_return,
                    'start_price': start_price,
                    'end_price': end_price
                })
        
        return pd.DataFrame(annual_returns)


class SPYStrategyBacktester:
    """Backtests trading strategies on SPY data"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy(config),
            'bollinger_bands': BollingerBandsStrategy(config),
            'breakout': BreakoutStrategy(config),
            'rsi_divergence': RSIDivergenceStrategy(config),
            'opening_range_breakout': OpeningRangeBreakoutStrategy(config)
        }
    
    def run_strategy_backtest(self, strategy_name: str, data: pd.DataFrame) -> Dict:
        """Run backtest for a single strategy"""
        strategy = self.strategies[strategy_name]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate trades and performance
        trades = self._calculate_trades(signals, strategy_name)
        performance = self._calculate_performance(trades, data)
        
        return {
            'strategy_name': strategy.name,
            'trades': trades,
            'performance': performance,
            'signals': signals
        }
    
    def _calculate_trades(self, signals: pd.DataFrame, strategy_name: str) -> list:
        """Calculate individual trades from signals"""
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for idx, row in signals.iterrows():
            try:
                signal = row.get('signal', 0)
                position_value = row.get('position', 0)
                
                # Handle potential Series/array values
                if hasattr(signal, '__len__') and len(signal) > 1:
                    signal = signal.iloc[0] if hasattr(signal, 'iloc') else signal[0]
                if hasattr(position_value, '__len__') and len(position_value) > 1:
                    position_value = position_value.iloc[0] if hasattr(position_value, 'iloc') else position_value[0]
                
                if signal == 1 and position == 0:  # Enter long position
                    position = 1
                    entry_price = row['close']
                    entry_time = idx
                elif (signal == -1 or position_value == 0) and position == 1:  # Exit position
                    exit_price = row['close']
                    exit_time = idx
                    
                    # Calculate trade metrics
                    pnl = exit_price - entry_price
                    return_pct = (exit_price - entry_price) / entry_price
                    hold_duration = (exit_time - entry_time).total_seconds() / 60  # minutes
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'hold_duration_minutes': hold_duration,
                        'strategy': strategy_name
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_time = None
                    
            except Exception as e:
                # Skip problematic rows
                continue
        
        return trades
    
    def _calculate_performance(self, trades: list, data: pd.DataFrame) -> Dict:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'annualized_return': 0,
                'avg_return_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_hold_time_hours': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Return metrics
        total_return = trades_df['return_pct'].sum()
        avg_return_per_trade = trades_df['return_pct'].mean()
        best_trade = trades_df['return_pct'].max()
        worst_trade = trades_df['return_pct'].min()
        
        # Time metrics
        avg_hold_time_hours = trades_df['hold_duration_minutes'].mean() / 60
        
        # Calculate annualized return
        num_years = (data.index[-1] - data.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1/num_years) - 1 if num_years > 0 else 0
        
        # Calculate Sharpe ratio and max drawdown
        if len(trades_df) > 1:
            returns_std = trades_df['return_pct'].std()
            sharpe_ratio = (avg_return_per_trade - 0.02/252) / returns_std if returns_std > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + trades_df['return_pct']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_return_per_trade': avg_return_per_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_hold_time_hours': avg_hold_time_hours,
            'total_pnl': trades_df['pnl'].sum(),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


def run_spy_analysis(file_path: str, start_date: str = None, end_date: str = None, max_rows: int = None):
    """Main function to run complete SPY analysis"""
    
    print("🎯 SPY Trading Strategy Analysis")
    print("=" * 50)
    
    # Load SPY data
    loader = SPYDataLoader(file_path)
    spy_data = loader.load_data(start_date, end_date, max_rows)
    
    if spy_data is None or len(spy_data) == 0:
        print("❌ Failed to load SPY data. Exiting.")
        return
    
    # Calculate buy-and-hold benchmark
    print("\n📊 Calculating Buy-and-Hold Benchmark...")
    benchmark = SPYBenchmarkAnalyzer(spy_data)
    buy_hold_performance = benchmark.calculate_buy_hold_returns()
    annual_returns = benchmark.calculate_annual_returns()
    
    print(f"📈 SPY Buy-and-Hold Performance:")
    print(f"   📅 Period: {buy_hold_performance['start_date'].strftime('%Y-%m-%d')} to {buy_hold_performance['end_date'].strftime('%Y-%m-%d')}")
    print(f"   💰 Total Return: {buy_hold_performance['total_return']*100:.2f}%")
    print(f"   📊 Annualized Return: {buy_hold_performance['annualized_return']*100:.2f}%")
    print(f"   📉 Max Drawdown: {buy_hold_performance['max_drawdown']*100:.2f}%")
    print(f"   🎯 Sharpe Ratio: {buy_hold_performance['sharpe_ratio']:.2f}")
    
    # Initialize strategy backtester with enhanced config
    config = StrategyConfig()
    
    # Apply our enhanced settings
    config.use_adx_filter = True
    config.use_trailing_stops = True
    config.use_profit_zones = True
    config.enable_regime_specific_filtering = True
    
    backtester = SPYStrategyBacktester(config)
    
    # Run all strategies
    print(f"\n🔬 Running Enhanced Trading Strategies...")
    strategy_results = {}
    
    for strategy_name in backtester.strategies.keys():
        print(f"\n📈 Testing {backtester.strategies[strategy_name].name}...")
        
        try:
            result = backtester.run_strategy_backtest(strategy_name, spy_data)
            strategy_results[strategy_name] = result
            
            perf = result['performance']
            print(f"   ✅ Trades: {perf['total_trades']}")
            print(f"   📊 Win Rate: {perf['win_rate']*100:.1f}%")
            print(f"   💰 Total Return: {perf['total_return']*100:.2f}%")
            print(f"   📈 Annualized: {perf['annualized_return']*100:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            strategy_results[strategy_name] = None
    
    # Create comparison summary
    print(f"\n🏆 STRATEGY vs BUY-AND-HOLD COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    
    # Add buy-and-hold as baseline
    comparison_data.append({
        'Strategy': 'SPY Buy-and-Hold',
        'Total Return (%)': f"{buy_hold_performance['total_return']*100:.2f}%",
        'Annualized (%)': f"{buy_hold_performance['annualized_return']*100:.2f}%",
        'Sharpe Ratio': f"{buy_hold_performance['sharpe_ratio']:.2f}",
        'Max Drawdown (%)': f"{buy_hold_performance['max_drawdown']*100:.2f}%",
        'Beats Benchmark': '📊 Benchmark'
    })
    
    # Add strategy results
    for strategy_name, result in strategy_results.items():
        if result and result['performance']['total_trades'] > 0:
            perf = result['performance']
            beats_benchmark = perf['annualized_return'] > buy_hold_performance['annualized_return']
            
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Total Return (%)': f"{perf['total_return']*100:.2f}%",
                'Annualized (%)': f"{perf['annualized_return']*100:.2f}%",
                'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                'Max Drawdown (%)': f"{perf['max_drawdown']*100:.2f}%",
                'Beats Benchmark': '🏆 YES' if beats_benchmark else '❌ NO'
            })
    
    # Display comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Annual comparison if we have multiple years
    if len(annual_returns) > 1:
        print(f"\n📅 ANNUAL RETURNS COMPARISON")
        print("=" * 40)
        print(annual_returns[['year', 'return']].to_string(index=False, 
                                                         formatters={'return': lambda x: f"{x*100:.2f}%"}))
    
    return {
        'spy_data': spy_data,
        'buy_hold_performance': buy_hold_performance,
        'strategy_results': strategy_results,
        'annual_returns': annual_returns
    }


if __name__ == "__main__":
    # Run the analysis
    spy_file = "/home/malmorga/ml/1_min_SPY_2008-2021.csv"
    
    print("🚀 Running Full SPY Analysis (2008-2021)...")
    print("This may take a few minutes to process the full dataset...\n")
    
    # Run analysis on different time periods
    periods = [
        {"name": "Full Dataset", "start": None, "end": None, "max_rows": None},
        {"name": "Financial Crisis", "start": "2008-01-01", "end": "2009-12-31", "max_rows": None},
        {"name": "Recovery Period", "start": "2010-01-01", "end": "2012-12-31", "max_rows": None},
        {"name": "Bull Market", "start": "2013-01-01", "end": "2019-12-31", "max_rows": None},
        {"name": "COVID Period", "start": "2020-01-01", "end": "2021-12-31", "max_rows": None},
    ]
    
    all_results = {}
    
    for period in periods:
        print(f"\n{'='*60}")
        print(f"🎯 ANALYZING: {period['name']}")
        print(f"{'='*60}")
        
        try:
            results = run_spy_analysis(
                spy_file, 
                start_date=period['start'], 
                end_date=period['end'], 
                max_rows=period['max_rows']
            )
            all_results[period['name']] = results
            
        except Exception as e:
            print(f"❌ Error analyzing {period['name']}: {e}")
    
    # Summary across all periods
    print(f"\n🎊 FINAL SUMMARY: STRATEGIES vs SPY BUY-AND-HOLD")
    print("="*80)
    
    for period_name, results in all_results.items():
        if results:
            buy_hold = results['buy_hold_performance']
            print(f"\n📊 {period_name}:")
            print(f"   SPY Buy-Hold: {buy_hold['annualized_return']*100:.2f}% annually")
            
            for strategy_name, strategy_result in results['strategy_results'].items():
                if strategy_result and strategy_result['performance']['total_trades'] > 0:
                    perf = strategy_result['performance']
                    beats = "🏆" if perf['annualized_return'] > buy_hold['annualized_return'] else "❌"
                    print(f"   {strategy_result['strategy_name']}: {perf['annualized_return']*100:.2f}% {beats}")
            
