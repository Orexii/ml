#!/usr/bin/env python3
"""
Full SPY Dataset Monthly Analysis
Comprehensive month-by-month analysis across entire SPY dataset (2008-2021)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from spy_analysis import SPYDataLoader, SPYBenchmarkAnalyzer, SPYStrategyBacktester
from trading_simulator import StrategyConfig


def run_full_dataset_monthly_analysis():
    """Run monthly analysis on entire SPY dataset"""
    
    print('🚀 FULL SPY DATASET MONTHLY ANALYSIS (2008-2021)')
    print('='*70)
    print('Loading entire SPY dataset...')
    print('⏰ This will take 15-30 minutes to complete...\n')
    
    # Load full dataset
    loader = SPYDataLoader('/home/malmorga/ml/1_min_SPY_2008-2021.csv')
    full_data = loader.load_data()
    
    if full_data is None or len(full_data) == 0:
        print("❌ Failed to load SPY data. Exiting.")
        return
    
    print(f"✅ Loaded {len(full_data):,} rows")
    print(f"📅 Full date range: {full_data.index.min()} to {full_data.index.max()}")
    
    # Generate all monthly periods
    monthly_periods = []
    current_date = full_data.index.min().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = full_data.index.max()
    
    print(f"\n📊 Generating monthly periods...")
    
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
    
    # Setup enhanced strategy configuration
    config = StrategyConfig()
    config.use_adx_filter = True
    config.use_trailing_stops = True
    config.use_profit_zones = True
    config.enable_regime_specific_filtering = True
    
    backtester = SPYStrategyBacktester(config)
    
    # Initialize results storage
    monthly_results = []
    
    print(f"\n🔬 Processing {len(monthly_periods)} months...")
    print("Progress: ", end='', flush=True)
    
    # Process each month
    for i, period in enumerate(monthly_periods, 1):
        period_name = period['period_name']
        
        # Progress indicator
        if i % 10 == 0:
            print(f"{i}", end='', flush=True)
        elif i % 5 == 0:
            print(".", end='', flush=True)
        
        try:
            # Extract month data
            month_data = full_data[
                (full_data.index >= period['start_date']) & 
                (full_data.index <= period['end_date'])
            ]
            
            if len(month_data) < 50:  # Skip months with insufficient data
                continue
            
            # Calculate buy-and-hold benchmark for this month
            benchmark = SPYBenchmarkAnalyzer(month_data)
            buy_hold_perf = benchmark.calculate_buy_hold_returns()
            
            # Test Bollinger Bands strategy (our best performer)
            bb_result = backtester.run_strategy_backtest('bollinger_bands', month_data)
            bb_perf = bb_result['performance']
            
            # Store results
            month_result = {
                'period': period_name,
                'year': period['year'],
                'month': period['month'],
                'data_points': len(month_data),
                'spy_return_pct': buy_hold_perf['total_return'] * 100,
                'spy_start_price': buy_hold_perf['start_price'],
                'spy_end_price': buy_hold_perf['end_price'],
                'bb_return_pct': bb_perf['total_return'] * 100,
                'bb_trades': bb_perf['total_trades'],
                'bb_win_rate': bb_perf['win_rate'] * 100,
                'bb_beats_spy': bb_perf['total_return'] > buy_hold_perf['total_return'],
                'outperformance': (bb_perf['total_return'] - buy_hold_perf['total_return']) * 100,
                'bb_sharpe': bb_perf['sharpe_ratio'],
                'bb_max_drawdown': bb_perf['max_drawdown'] * 100
            }
            
            monthly_results.append(month_result)
            
        except Exception as e:
            # Skip problematic months
            print(f"❌", end='', flush=True)
            continue
    
    print(f"\n\n✅ Processing complete!")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(monthly_results)
    
    if len(results_df) == 0:
        print("❌ No monthly results generated")
        return
    
    # Save comprehensive results
    output_file = '/home/malmorga/ml/full_spy_monthly_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"💾 Full results saved to: {output_file}")
    
    # Run comprehensive analysis
    analyze_full_results(results_df)
    
    return results_df


def analyze_full_results(df):
    """Comprehensive analysis of full dataset results"""
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS - FULL DATASET")
    print("="*60)
    
    total_months = len(df)
    wins = len(df[df['bb_beats_spy'] == True])
    positive_months = len(df[df['bb_return_pct'] > 0])
    
    print(f"\n📈 OVERALL PERFORMANCE ({total_months} months):")
    print(f"   ✅ Months with positive returns: {positive_months}/{total_months} ({positive_months/total_months*100:.1f}%)")
    print(f"   🏆 Months beating SPY: {wins}/{total_months} ({wins/total_months*100:.1f}%)")
    print(f"   📊 Average monthly return: {df['bb_return_pct'].mean():.2f}%")
    print(f"   📈 Average outperformance: {df['outperformance'].mean():.2f}%")
    
    # Annual breakdown
    print(f"\n📅 ANNUAL PERFORMANCE BREAKDOWN:")
    annual_stats = df.groupby('year').agg({
        'bb_return_pct': ['mean', 'sum', 'count'],
        'spy_return_pct': ['mean', 'sum'],
        'bb_beats_spy': 'sum',
        'outperformance': 'mean'
    }).round(2)
    
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        wins_year = len(year_data[year_data['bb_beats_spy'] == True])
        total_year = len(year_data)
        annual_bb_return = year_data['bb_return_pct'].sum()
        annual_spy_return = year_data['spy_return_pct'].sum()
        
        print(f"   {year}: BB {annual_bb_return:+6.1f}% vs SPY {annual_spy_return:+6.1f}% "
              f"(Win rate: {wins_year}/{total_year} = {wins_year/total_year*100:.0f}%)")
    
    # Best and worst periods
    print(f"\n🏆 BEST MONTHS:")
    best_months = df.nlargest(5, 'outperformance')
    for _, month in best_months.iterrows():
        print(f"   {month['period']}: +{month['outperformance']:.2f}% outperformance "
              f"(BB: {month['bb_return_pct']:+.2f}%, SPY: {month['spy_return_pct']:+.2f}%)")
    
    print(f"\n⚠️  WORST MONTHS:")
    worst_months = df.nsmallest(5, 'outperformance')
    for _, month in worst_months.iterrows():
        print(f"   {month['period']}: {month['outperformance']:.2f}% underperformance "
              f"(BB: {month['bb_return_pct']:+.2f}%, SPY: {month['spy_return_pct']:+.2f}%)")
    
    # Risk analysis
    print(f"\n🛡️ RISK ANALYSIS:")
    loss_months = df[df['bb_return_pct'] < 0]
    print(f"   Months with losses: {len(loss_months)}/{total_months} ({len(loss_months)/total_months*100:.1f}%)")
    if len(loss_months) > 0:
        print(f"   Average loss: {loss_months['bb_return_pct'].mean():.2f}%")
        print(f"   Worst loss: {loss_months['bb_return_pct'].min():.2f}%")
    
    print(f"   Monthly return volatility: {df['bb_return_pct'].std():.2f}%")
    print(f"   Average Sharpe ratio: {df['bb_sharpe'].mean():.2f}")
    
    # Market condition analysis
    print(f"\n📊 MARKET CONDITION PERFORMANCE:")
    
    # Bull vs Bear months (based on SPY performance)
    bull_months = df[df['spy_return_pct'] > 0]
    bear_months = df[df['spy_return_pct'] <= 0]
    
    if len(bull_months) > 0:
        bull_wins = len(bull_months[bull_months['bb_beats_spy'] == True])
        print(f"   Bull months (SPY +): {len(bull_months)} months, "
              f"BB beats SPY {bull_wins} times ({bull_wins/len(bull_months)*100:.1f}%)")
        print(f"      Avg BB return: {bull_months['bb_return_pct'].mean():.2f}%")
    
    if len(bear_months) > 0:
        bear_wins = len(bear_months[bear_months['bb_beats_spy'] == True])
        print(f"   Bear months (SPY -): {len(bear_months)} months, "
              f"BB beats SPY {bear_wins} times ({bear_wins/len(bear_months)*100:.1f}%)")
        print(f"      Avg BB return: {bear_months['bb_return_pct'].mean():.2f}%")
    
    # Consecutive analysis
    print(f"\n⚠️  CONSECUTIVE PERIODS ANALYSIS:")
    df_sorted = df.sort_values('period')
    
    # Consecutive losses
    consecutive_losses = []
    consecutive_underperformance = []
    current_loss_streak = 0
    current_underperf_streak = 0
    
    for _, row in df_sorted.iterrows():
        if row['bb_return_pct'] < 0:
            current_loss_streak += 1
        else:
            if current_loss_streak > 0:
                consecutive_losses.append(current_loss_streak)
            current_loss_streak = 0
            
        if not row['bb_beats_spy']:
            current_underperf_streak += 1
        else:
            if current_underperf_streak > 0:
                consecutive_underperformance.append(current_underperf_streak)
            current_underperf_streak = 0
    
    # Add final streaks if ended on negative
    if current_loss_streak > 0:
        consecutive_losses.append(current_loss_streak)
    if current_underperf_streak > 0:
        consecutive_underperformance.append(current_underperf_streak)
    
    if consecutive_losses:
        print(f"   Max consecutive loss months: {max(consecutive_losses)}")
        print(f"   Avg loss streak length: {np.mean(consecutive_losses):.1f} months")
    else:
        print(f"   No consecutive loss months!")
    
    if consecutive_underperformance:
        print(f"   Max consecutive underperformance: {max(consecutive_underperformance)} months")
        print(f"   Avg underperformance streak: {np.mean(consecutive_underperformance):.1f} months")
    
    # Summary statistics
    print(f"\n🎯 KEY PERFORMANCE METRICS:")
    print(f"   Total return (cumulative): {((1 + df['bb_return_pct']/100).prod() - 1)*100:.1f}%")
    print(f"   SPY total return (cumulative): {((1 + df['spy_return_pct']/100).prod() - 1)*100:.1f}%")
    
    # Annualized metrics
    years = (df['year'].max() - df['year'].min() + 1)
    annual_bb_return = ((1 + df['bb_return_pct']/100).prod()) ** (1/years) - 1
    annual_spy_return = ((1 + df['spy_return_pct']/100).prod()) ** (1/years) - 1
    
    print(f"   Annualized BB return: {annual_bb_return*100:.1f}%")
    print(f"   Annualized SPY return: {annual_spy_return*100:.1f}%")
    print(f"   Annual outperformance: {(annual_bb_return - annual_spy_return)*100:.1f}%")


if __name__ == "__main__":
    print("🚀 Starting Full SPY Dataset Monthly Analysis...")
    print("📊 This will analyze every month from 2008-2021")
    print("⏰ Estimated time: 15-30 minutes\n")
    
    results = run_full_dataset_monthly_analysis()
    
    if results is not None:
        print(f"\n🎊 FULL DATASET ANALYSIS COMPLETE!")
        print("="*50)
        print(f"📈 Analyzed {len(results)} months across 13+ years")
        print(f"💾 Results saved to: full_spy_monthly_results.csv")
