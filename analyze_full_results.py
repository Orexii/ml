#!/usr/bin/env python3
"""
Full Dataset Results Analysis
Comprehensive analysis of the complete SPY dataset results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def analyze_full_dataset_results():
    """Comprehensive analysis of full dataset results"""
    
    print('📊 FULL DATASET RESULTS ANALYSIS')
    print('='*50)
    
    # Load results
    try:
        df = pd.read_csv('/home/malmorga/ml/full_spy_monthly_results.csv')
        print(f"✅ Loaded {len(df)} months of results")
    except FileNotFoundError:
        print("❌ Results file not found. Run full_spy_analysis.py first.")
        return
    
    # Basic statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Period: {df['year'].min()}-{df['year'].max()} ({len(df)} months)")
    print(f"   Total data points processed: {df['data_points'].sum():,}")
    print(f"   Total trades executed: {df['bb_trades'].sum():,}")
    
    # Performance metrics
    positive_months = len(df[df['bb_return_pct'] > 0])
    winning_months = len(df[df['bb_beats_spy'] == True])
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"   Positive months: {positive_months}/{len(df)} ({positive_months/len(df)*100:.1f}%)")
    print(f"   Months beating SPY: {winning_months}/{len(df)} ({winning_months/len(df)*100:.1f}%)")
    print(f"   Average monthly return: {df['bb_return_pct'].mean():.2f}%")
    print(f"   Median monthly return: {df['bb_return_pct'].median():.2f}%")
    print(f"   Average outperformance: {df['outperformance'].mean():.2f}%")
    
    # Cumulative returns
    cumulative_bb = (1 + df['bb_return_pct']/100).cumprod().iloc[-1] - 1
    cumulative_spy = (1 + df['spy_return_pct']/100).cumprod().iloc[-1] - 1
    
    print(f"\n💰 CUMULATIVE RETURNS:")
    print(f"   Bollinger Bands strategy: {cumulative_bb*100:.1f}%")
    print(f"   SPY buy-and-hold: {cumulative_spy*100:.1f}%")
    print(f"   Total outperformance: {(cumulative_bb - cumulative_spy)*100:.1f}%")
    
    # Annualized metrics
    years = len(df) / 12
    annual_bb = (1 + cumulative_bb) ** (1/years) - 1
    annual_spy = (1 + cumulative_spy) ** (1/years) - 1
    
    print(f"   Annualized BB return: {annual_bb*100:.1f}%")
    print(f"   Annualized SPY return: {annual_spy*100:.1f}%")
    print(f"   Annual alpha: {(annual_bb - annual_spy)*100:.1f}%")
    
    # Risk analysis
    print(f"\n🛡️ RISK ANALYSIS:")
    loss_months = df[df['bb_return_pct'] < 0]
    print(f"   Months with losses: {len(loss_months)}/{len(df)} ({len(loss_months)/len(df)*100:.1f}%)")
    
    if len(loss_months) > 0:
        print(f"   Average loss: {loss_months['bb_return_pct'].mean():.2f}%")
        print(f"   Worst single month: {loss_months['bb_return_pct'].min():.2f}%")
        print(f"   95th percentile loss: {loss_months['bb_return_pct'].quantile(0.05):.2f}%")
    
    print(f"   Monthly volatility: {df['bb_return_pct'].std():.2f}%")
    print(f"   Downside deviation: {df[df['bb_return_pct'] < 0]['bb_return_pct'].std():.2f}%")
    
    # Sharpe ratio analysis
    print(f"   Average Sharpe ratio: {df['bb_sharpe'].mean():.3f}")
    print(f"   Median Sharpe ratio: {df['bb_sharpe'].median():.3f}")
    
    # Drawdown analysis
    print(f"   Average max drawdown: {df['bb_max_drawdown'].mean():.2f}%")
    print(f"   Worst max drawdown: {df['bb_max_drawdown'].min():.2f}%")
    
    # Market condition analysis
    print(f"\n📊 MARKET CONDITION ANALYSIS:")
    
    # Bull vs bear performance
    bull_months = df[df['spy_return_pct'] > 0]
    bear_months = df[df['spy_return_pct'] <= 0]
    
    print(f"   Bull market months (SPY positive): {len(bull_months)}")
    if len(bull_months) > 0:
        bull_wins = len(bull_months[bull_months['bb_beats_spy'] == True])
        print(f"      Win rate: {bull_wins}/{len(bull_months)} ({bull_wins/len(bull_months)*100:.1f}%)")
        print(f"      Avg return: {bull_months['bb_return_pct'].mean():.2f}%")
        print(f"      Avg outperformance: {bull_months['outperformance'].mean():.2f}%")
    
    print(f"   Bear market months (SPY negative): {len(bear_months)}")
    if len(bear_months) > 0:
        bear_wins = len(bear_months[bear_months['bb_beats_spy'] == True])
        print(f"      Win rate: {bear_wins}/{len(bear_months)} ({bear_wins/len(bear_months)*100:.1f}%)")
        print(f"      Avg return: {bear_months['bb_return_pct'].mean():.2f}%")
        print(f"      Avg outperformance: {bear_months['outperformance'].mean():.2f}%")
    
    # Crisis period analysis
    print(f"\n⚠️ CRISIS PERIOD ANALYSIS:")
    
    # 2008 Financial Crisis
    crisis_2008 = df[(df['year'] == 2008)]
    if len(crisis_2008) > 0:
        crisis_wins = len(crisis_2008[crisis_2008['bb_beats_spy'] == True])
        crisis_return = (1 + crisis_2008['bb_return_pct']/100).prod() - 1
        spy_return = (1 + crisis_2008['spy_return_pct']/100).prod() - 1
        print(f"   2008 Financial Crisis ({len(crisis_2008)} months):")
        print(f"      Win rate: {crisis_wins}/{len(crisis_2008)} ({crisis_wins/len(crisis_2008)*100:.1f}%)")
        print(f"      BB return: {crisis_return*100:.1f}%")
        print(f"      SPY return: {spy_return*100:.1f}%")
        print(f"      Outperformance: {(crisis_return - spy_return)*100:.1f}%")
    
    # 2020 COVID Crisis
    covid_2020 = df[(df['year'] == 2020)]
    if len(covid_2020) > 0:
        covid_wins = len(covid_2020[covid_2020['bb_beats_spy'] == True])
        covid_return = (1 + covid_2020['bb_return_pct']/100).prod() - 1
        spy_return = (1 + covid_2020['spy_return_pct']/100).prod() - 1
        print(f"   2020 COVID Crisis ({len(covid_2020)} months):")
        print(f"      Win rate: {covid_wins}/{len(covid_2020)} ({covid_wins/len(covid_2020)*100:.1f}%)")
        print(f"      BB return: {covid_return*100:.1f}%")
        print(f"      SPY return: {spy_return*100:.1f}%")
        print(f"      Outperformance: {(covid_return - spy_return)*100:.1f}%")
    
    # Consecutive analysis
    analyze_consecutive_periods(df)
    
    # Best and worst periods
    analyze_extremes(df)
    
    # Trading analysis
    analyze_trading_metrics(df)
    
    # Final assessment
    print(f"\n🏆 INSTITUTIONAL QUALITY ASSESSMENT:")
    
    # Calculate key institutional metrics
    calmar_ratio = annual_bb / (abs(df['bb_max_drawdown'].min()) / 100)
    sortino_ratio = df['bb_return_pct'].mean() / df[df['bb_return_pct'] < 0]['bb_return_pct'].std()
    
    print(f"   Calmar Ratio: {calmar_ratio:.2f} (>1.0 is excellent)")
    print(f"   Sortino Ratio: {sortino_ratio:.2f} (>1.0 is good)")
    print(f"   Max Drawdown: {df['bb_max_drawdown'].min():.1f}% (institutional limit: -10%)")
    print(f"   Win Rate: {winning_months/len(df)*100:.1f}% (target: >40%)")
    print(f"   Positive Months: {positive_months/len(df)*100:.1f}% (target: >60%)")
    
    # Overall assessment
    score = 0
    if calmar_ratio > 1.0: score += 1
    if sortino_ratio > 1.0: score += 1
    if df['bb_max_drawdown'].min() > -10: score += 1
    if winning_months/len(df) > 0.4: score += 1
    if positive_months/len(df) > 0.6: score += 1
    if annual_bb > 0.1: score += 1  # >10% annual return
    
    print(f"\n🎯 INSTITUTIONAL SCORE: {score}/6")
    if score >= 5:
        print("   ⭐ EXCELLENT - Institutional quality strategy")
    elif score >= 4:
        print("   ✅ GOOD - Professional grade strategy")
    elif score >= 3:
        print("   📈 ACCEPTABLE - Retail trading suitable")
    else:
        print("   ⚠️ NEEDS IMPROVEMENT")


def analyze_consecutive_periods(df):
    """Analyze consecutive winning/losing periods"""
    print(f"\n⚠️ CONSECUTIVE PERIODS ANALYSIS:")
    
    # Sort by period
    df_sorted = df.sort_values('period').reset_index(drop=True)
    
    # Consecutive losses
    loss_streaks = []
    underperform_streaks = []
    current_loss = 0
    current_underperform = 0
    
    for _, row in df_sorted.iterrows():
        if row['bb_return_pct'] < 0:
            current_loss += 1
        else:
            if current_loss > 0:
                loss_streaks.append(current_loss)
            current_loss = 0
            
        if not row['bb_beats_spy']:
            current_underperform += 1
        else:
            if current_underperform > 0:
                underperform_streaks.append(current_underperform)
            current_underperform = 0
    
    # Add final streaks
    if current_loss > 0:
        loss_streaks.append(current_loss)
    if current_underperform > 0:
        underperform_streaks.append(current_underperform)
    
    if loss_streaks:
        print(f"   Loss streaks: {len(loss_streaks)} total")
        print(f"   Max consecutive losses: {max(loss_streaks)} months")
        print(f"   Average loss streak: {np.mean(loss_streaks):.1f} months")
    else:
        print(f"   No consecutive loss periods!")
    
    if underperform_streaks:
        print(f"   Underperformance streaks: {len(underperform_streaks)} total")
        print(f"   Max consecutive underperformance: {max(underperform_streaks)} months")
        print(f"   Average underperformance streak: {np.mean(underperform_streaks):.1f} months")


def analyze_extremes(df):
    """Analyze best and worst performing periods"""
    print(f"\n🏆 BEST PERFORMING MONTHS:")
    best = df.nlargest(5, 'outperformance')
    for _, row in best.iterrows():
        print(f"   {row['period']}: +{row['outperformance']:.1f}% outperformance "
              f"(BB: {row['bb_return_pct']:+.1f}%, SPY: {row['spy_return_pct']:+.1f}%)")
    
    print(f"\n⚠️ WORST PERFORMING MONTHS:")
    worst = df.nsmallest(5, 'outperformance')
    for _, row in worst.iterrows():
        print(f"   {row['period']}: {row['outperformance']:.1f}% underperformance "
              f"(BB: {row['bb_return_pct']:+.1f}%, SPY: {row['spy_return_pct']:+.1f}%)")
    
    print(f"\n📈 BEST ABSOLUTE RETURNS:")
    best_abs = df.nlargest(5, 'bb_return_pct')
    for _, row in best_abs.iterrows():
        print(f"   {row['period']}: {row['bb_return_pct']:+.1f}% (vs SPY: {row['spy_return_pct']:+.1f}%)")
    
    print(f"\n📉 WORST ABSOLUTE RETURNS:")
    worst_abs = df.nsmallest(5, 'bb_return_pct')
    for _, row in worst_abs.iterrows():
        print(f"   {row['period']}: {row['bb_return_pct']:+.1f}% (vs SPY: {row['spy_return_pct']:+.1f}%)")


def analyze_trading_metrics(df):
    """Analyze trading-specific metrics"""
    print(f"\n📊 TRADING ANALYSIS:")
    
    print(f"   Total trades: {df['bb_trades'].sum():,}")
    print(f"   Average trades per month: {df['bb_trades'].mean():.0f}")
    print(f"   Min trades per month: {df['bb_trades'].min()}")
    print(f"   Max trades per month: {df['bb_trades'].max()}")
    
    print(f"   Average win rate: {df['bb_win_rate'].mean():.1f}%")
    print(f"   Min win rate: {df['bb_win_rate'].min():.1f}%")
    print(f"   Max win rate: {df['bb_win_rate'].max():.1f}%")
    
    # Correlation between trades and performance
    trade_corr = df['bb_trades'].corr(df['bb_return_pct'])
    winrate_corr = df['bb_win_rate'].corr(df['bb_return_pct'])
    
    print(f"   Correlation (trades vs returns): {trade_corr:.3f}")
    print(f"   Correlation (win rate vs returns): {winrate_corr:.3f}")


if __name__ == "__main__":
    analyze_full_dataset_results()
