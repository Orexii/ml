#!/usr/bin/env python3
"""
SPY Monthly Analysis
Comprehensive month-by-month analysis of trading strategies vs SPY buy-and-hold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our existing trading components
from spy_analysis import SPYDataLoader, SPYBenchmarkAnalyzer, SPYStrategyBacktester
from trading_simulator import StrategyConfig


def analyze_monthly_performance(file_path: str):
    """Analyze strategy performance month by month across entire dataset"""
    
    print("🗓️  SPY MONTHLY PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Loading full SPY dataset for monthly analysis...")
    
    # Load full dataset
    loader = SPYDataLoader(file_path)
    full_data = loader.load_data()
    
    if full_data is None or len(full_data) == 0:
        print("❌ Failed to load SPY data. Exiting.")
        return
    
    print(f"✅ Loaded {len(full_data):,} rows covering {full_data.index.min()} to {full_data.index.max()}")
    
    # Get list of all months in the dataset
    monthly_periods = []
    current_date = full_data.index.min().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = full_data.index.max()
    
    while current_date <= end_date:
        # Calculate month end
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1)
        
        month_end = next_month - timedelta(days=1)
        month_end = month_end.replace(hour=23, minute=59, second=59)
        
        # Check if we have data for this month
        month_data = full_data[(full_data.index >= current_date) & (full_data.index <= month_end)]
        
        if len(month_data) > 100:  # Ensure sufficient data points
            monthly_periods.append({
                'year': current_date.year,
                'month': current_date.month,
                'start_date': current_date,
                'end_date': min(month_end, end_date),
                'period_name': f"{current_date.year}-{current_date.month:02d}"
            })
        
        current_date = next_month
    
    print(f"📅 Found {len(monthly_periods)} months with sufficient data")
    
    # Initialize results storage
    monthly_results = []
    
    # Setup enhanced strategy configuration
    config = StrategyConfig()
    config.use_adx_filter = True
    config.use_trailing_stops = True
    config.use_profit_zones = True
    config.enable_regime_specific_filtering = True
    
    strategies_to_test = ['bollinger_bands', 'opening_range_breakout', 'ma_crossover']
    
    print(f"\n🔬 Analyzing {len(strategies_to_test)} strategies across {len(monthly_periods)} months...")
    print("This will take several minutes to complete...\n")
    
    # Process each month
    for i, period in enumerate(monthly_periods, 1):
        period_name = period['period_name']
        print(f"📊 Processing {period_name} ({i}/{len(monthly_periods)})... ", end='', flush=True)
        
        try:
            # Extract month data
            month_data = full_data[
                (full_data.index >= period['start_date']) & 
                (full_data.index <= period['end_date'])
            ]
            
            if len(month_data) < 50:  # Skip months with insufficient data
                print("❌ Insufficient data")
                continue
            
            # Calculate buy-and-hold benchmark for this month
            benchmark = SPYBenchmarkAnalyzer(month_data)
            buy_hold_perf = benchmark.calculate_buy_hold_returns()
            
            # Test strategies
            backtester = SPYStrategyBacktester(config)
            month_result = {
                'period': period_name,
                'year': period['year'],
                'month': period['month'],
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'data_points': len(month_data),
                'spy_return': buy_hold_perf['total_return'],
                'spy_start_price': buy_hold_perf['start_price'],
                'spy_end_price': buy_hold_perf['end_price']
            }
            
            # Test each strategy
            for strategy_name in strategies_to_test:
                try:
                    result = backtester.run_strategy_backtest(strategy_name, month_data)
                    perf = result['performance']
                    
                    month_result[f'{strategy_name}_return'] = perf['total_return']
                    month_result[f'{strategy_name}_trades'] = perf['total_trades']
                    month_result[f'{strategy_name}_win_rate'] = perf['win_rate']
                    month_result[f'{strategy_name}_max_drawdown'] = perf['max_drawdown']
                    month_result[f'{strategy_name}_beats_spy'] = perf['total_return'] > buy_hold_perf['total_return']
                    
                except Exception as e:
                    # Fill with zeros if strategy fails
                    month_result[f'{strategy_name}_return'] = 0
                    month_result[f'{strategy_name}_trades'] = 0
                    month_result[f'{strategy_name}_win_rate'] = 0
                    month_result[f'{strategy_name}_max_drawdown'] = 0
                    month_result[f'{strategy_name}_beats_spy'] = False
            
            monthly_results.append(month_result)
            print("✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(monthly_results)
    
    if len(results_df) == 0:
        print("❌ No monthly results generated")
        return
    
    print(f"\n📈 MONTHLY ANALYSIS COMPLETE")
    print(f"✅ Successfully analyzed {len(results_df)} months")
    
    # Analyze worst performing months
    analyze_worst_months(results_df)
    
    # Analyze best performing months
    analyze_best_months(results_df)
    
    # Overall statistics
    analyze_overall_statistics(results_df)
    
    # Create visualizations
    create_monthly_visualizations(results_df)
    
    return results_df


def analyze_worst_months(df):
    """Identify and analyze worst performing months"""
    print(f"\n🔴 WORST PERFORMING MONTHS ANALYSIS")
    print("=" * 50)
    
    strategies = ['bollinger_bands', 'opening_range_breakout', 'ma_crossover']
    
    for strategy in strategies:
        return_col = f'{strategy}_return'
        if return_col in df.columns:
            # Find worst 10 months for this strategy
            worst_months = df.nsmallest(10, return_col)
            
            print(f"\n📉 {strategy.replace('_', ' ').title()} - Top 10 Worst Months:")
            print("-" * 40)
            
            for _, row in worst_months.iterrows():
                spy_return = row['spy_return'] * 100
                strategy_return = row[return_col] * 100
                outperformance = strategy_return - spy_return
                
                print(f"   {row['period']}: {strategy_return:+6.2f}% "
                      f"(SPY: {spy_return:+6.2f}%, Diff: {outperformance:+6.2f}%)")
            
            # Calculate statistics for worst months
            worst_returns = worst_months[return_col] * 100
            print(f"\n   📊 Worst Month Statistics:")
            print(f"      Average Loss: {worst_returns.mean():.2f}%")
            print(f"      Median Loss: {worst_returns.median():.2f}%")
            print(f"      Worst Single Month: {worst_returns.min():.2f}%")
            print(f"      Standard Deviation: {worst_returns.std():.2f}%")


def analyze_best_months(df):
    """Identify and analyze best performing months"""
    print(f"\n🟢 BEST PERFORMING MONTHS ANALYSIS")
    print("=" * 50)
    
    strategies = ['bollinger_bands', 'opening_range_breakout', 'ma_crossover']
    
    for strategy in strategies:
        return_col = f'{strategy}_return'
        if return_col in df.columns:
            # Find best 10 months for this strategy
            best_months = df.nlargest(10, return_col)
            
            print(f"\n📈 {strategy.replace('_', ' ').title()} - Top 10 Best Months:")
            print("-" * 40)
            
            for _, row in best_months.iterrows():
                spy_return = row['spy_return'] * 100
                strategy_return = row[return_col] * 100
                outperformance = strategy_return - spy_return
                
                print(f"   {row['period']}: {strategy_return:+6.2f}% "
                      f"(SPY: {spy_return:+6.2f}%, Diff: {outperformance:+6.2f}%)")
            
            # Calculate statistics for best months
            best_returns = best_months[return_col] * 100
            print(f"\n   📊 Best Month Statistics:")
            print(f"      Average Gain: {best_returns.mean():.2f}%")
            print(f"      Median Gain: {best_returns.median():.2f}%")
            print(f"      Best Single Month: {best_returns.max():.2f}%")


def analyze_overall_statistics(df):
    """Analyze overall performance statistics"""
    print(f"\n📊 OVERALL PERFORMANCE STATISTICS")
    print("=" * 50)
    
    strategies = ['bollinger_bands', 'opening_range_breakout', 'ma_crossover']
    
    summary_stats = []
    
    for strategy in strategies:
        return_col = f'{strategy}_return'
        beats_col = f'{strategy}_beats_spy'
        
        if return_col in df.columns:
            returns = df[return_col] * 100
            beats_spy_count = df[beats_col].sum() if beats_col in df.columns else 0
            total_months = len(df)
            
            stats = {
                'Strategy': strategy.replace('_', ' ').title(),
                'Avg Monthly Return': f"{returns.mean():.2f}%",
                'Median Monthly Return': f"{returns.median():.2f}%",
                'Best Month': f"{returns.max():.2f}%",
                'Worst Month': f"{returns.min():.2f}%",
                'Std Deviation': f"{returns.std():.2f}%",
                'Positive Months': f"{(returns > 0).sum()}/{total_months} ({(returns > 0).mean()*100:.1f}%)",
                'Beats SPY': f"{beats_spy_count}/{total_months} ({beats_spy_count/total_months*100:.1f}%)"
            }
            summary_stats.append(stats)
    
    # Display summary table
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))
    
    # SPY statistics for comparison
    spy_returns = df['spy_return'] * 100
    print(f"\n📊 SPY Buy-and-Hold Statistics:")
    print(f"   Average Monthly Return: {spy_returns.mean():.2f}%")
    print(f"   Best Month: {spy_returns.max():.2f}%")
    print(f"   Worst Month: {spy_returns.min():.2f}%")
    print(f"   Positive Months: {(spy_returns > 0).sum()}/{len(df)} ({(spy_returns > 0).mean()*100:.1f}%)")


def create_monthly_visualizations(df):
    """Create visualizations of monthly performance"""
    print(f"\n📈 Creating performance visualizations...")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SPY Trading Strategies - Monthly Performance Analysis', fontsize=16, fontweight='bold')
        
        strategies = ['bollinger_bands', 'opening_range_breakout', 'ma_crossover']
        colors = ['blue', 'green', 'red']
        
        # 1. Monthly returns comparison
        ax1 = axes[0, 0]
        spy_returns = df['spy_return'] * 100
        ax1.plot(range(len(df)), spy_returns, label='SPY Buy-Hold', color='black', alpha=0.7, linewidth=2)
        
        for i, strategy in enumerate(strategies):
            return_col = f'{strategy}_return'
            if return_col in df.columns:
                strategy_returns = df[return_col] * 100
                ax1.plot(range(len(df)), strategy_returns, 
                        label=strategy.replace('_', ' ').title(), 
                        color=colors[i], alpha=0.8)
        
        ax1.set_title('Monthly Returns Over Time')
        ax1.set_xlabel('Month Index')
        ax1.set_ylabel('Monthly Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Return distribution histogram
        ax2 = axes[0, 1]
        ax2.hist(spy_returns, bins=30, alpha=0.7, label='SPY', color='black', density=True)
        
        for i, strategy in enumerate(strategies):
            return_col = f'{strategy}_return'
            if return_col in df.columns:
                strategy_returns = df[return_col] * 100
                ax2.hist(strategy_returns, bins=30, alpha=0.6, 
                        label=strategy.replace('_', ' ').title(), 
                        color=colors[i], density=True)
        
        ax2.set_title('Monthly Return Distributions')
        ax2.set_xlabel('Monthly Return (%)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative performance
        ax3 = axes[1, 0]
        spy_cumulative = (1 + df['spy_return']).cumprod()
        ax3.plot(range(len(df)), spy_cumulative, label='SPY Buy-Hold', color='black', linewidth=2)
        
        for i, strategy in enumerate(strategies):
            return_col = f'{strategy}_return'
            if return_col in df.columns:
                strategy_cumulative = (1 + df[return_col]).cumprod()
                ax3.plot(range(len(df)), strategy_cumulative, 
                        label=strategy.replace('_', ' ').title(), 
                        color=colors[i], linewidth=2)
        
        ax3.set_title('Cumulative Performance')
        ax3.set_xlabel('Month Index')
        ax3.set_ylabel('Cumulative Return Multiple')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Win rate analysis
        ax4 = axes[1, 1]
        win_rates = []
        strategy_names = []
        
        for strategy in strategies:
            beats_col = f'{strategy}_beats_spy'
            if beats_col in df.columns:
                win_rate = df[beats_col].mean() * 100
                win_rates.append(win_rate)
                strategy_names.append(strategy.replace('_', ' ').title())
        
        bars = ax4.bar(strategy_names, win_rates, color=colors[:len(win_rates)], alpha=0.7)
        ax4.set_title('Percentage of Months Beating SPY')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
        
        plt.tight_layout()
        plt.savefig('/home/malmorga/ml/spy_monthly_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved visualization to: spy_monthly_analysis.png")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


if __name__ == "__main__":
    spy_file = "/home/malmorga/ml/1_min_SPY_2008-2021.csv"
    
    print("🚀 Starting comprehensive monthly analysis...")
    print("⚠️  This analysis will process the entire SPY dataset month by month.")
    print("⏰ Expected runtime: 15-30 minutes depending on system performance.\n")
    
    results = analyze_monthly_performance(spy_file)
    
    if results is not None:
        # Save results to CSV for further analysis
        output_file = "/home/malmorga/ml/spy_monthly_results.csv"
        results.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        print(f"\n🎊 MONTHLY ANALYSIS COMPLETE!")
        print("="*50)
