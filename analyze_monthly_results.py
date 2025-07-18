#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load the results
df = pd.read_csv('/home/malmorga/ml/monthly_analysis_results.csv')

print('📊 COMPREHENSIVE MONTHLY ANALYSIS RESULTS')
print('='*60)

# Overall statistics
total_months = len(df)
wins = len(df[df['bb_beats_spy'] == True])
losses = len(df[df['bb_beats_spy'] == False])

print(f'\n📈 OVERALL PERFORMANCE:')
print(f'   Total months analyzed: {total_months}')
print(f'   Months beating SPY: {wins} ({wins/total_months*100:.1f}%)')
print(f'   Months losing to SPY: {losses} ({losses/total_months*100:.1f}%)')

# Best and worst months
best_month = df.loc[df['outperformance'].idxmax()]
worst_month = df.loc[df['outperformance'].idxmin()]

print(f'\n🏆 BEST MONTH:')
print(f'   {best_month["period"]}: +{best_month["outperformance"]:.2f}% outperformance')
print(f'   BB Return: {best_month["bb_return_pct"]:.2f}% vs SPY: {best_month["spy_return_pct"]:.2f}%')

print(f'\n⚠️  WORST MONTH:')
print(f'   {worst_month["period"]}: {worst_month["outperformance"]:.2f}% underperformance')
print(f'   BB Return: {worst_month["bb_return_pct"]:.2f}% vs SPY: {worst_month["spy_return_pct"]:.2f}%')

# Crisis period analysis
print(f'\n🔥 CRISIS PERIOD ANALYSIS:')

for period in df['period_group'].unique():
    period_data = df[df['period_group'] == period]
    wins_period = len(period_data[period_data['bb_beats_spy'] == True])
    total_period = len(period_data)
    avg_outperf = period_data['outperformance'].mean()
    
    print(f'\n   📊 {period}:')
    print(f'      Win rate: {wins_period}/{total_period} ({wins_period/total_period*100:.1f}%)')
    print(f'      Avg outperformance: {avg_outperf:.2f}%')
    print(f'      Best month: +{period_data["outperformance"].max():.2f}%')
    print(f'      Worst month: {period_data["outperformance"].min():.2f}%')

# Monthly returns distribution
print(f'\n📊 BOLLINGER BANDS RETURN DISTRIBUTION:')
print(f'   Mean monthly return: {df["bb_return_pct"].mean():.2f}%')
print(f'   Median monthly return: {df["bb_return_pct"].median():.2f}%')
print(f'   Best month: {df["bb_return_pct"].max():.2f}%')
print(f'   Worst month: {df["bb_return_pct"].min():.2f}%')
print(f'   Standard deviation: {df["bb_return_pct"].std():.2f}%')

# Months with losses
loss_months = df[df['bb_return_pct'] < 0]
print(f'\n❌ MONTHS WITH ACTUAL LOSSES:')
print(f'   Total months with losses: {len(loss_months)}')
if len(loss_months) > 0:
    for _, month in loss_months.iterrows():
        print(f'   {month["period"]}: {month["bb_return_pct"]:.2f}% (SPY: {month["spy_return_pct"]:.2f}%)')

# Consecutive losses analysis
print(f'\n⚠️  RISK ANALYSIS - CONSECUTIVE UNDERPERFORMANCE:')
df_sorted = df.sort_values('period')
consecutive_losses = []
current_streak = 0

for idx, row in df_sorted.iterrows():
    if not row['bb_beats_spy']:
        current_streak += 1
    else:
        if current_streak > 0:
            consecutive_losses.append(current_streak)
        current_streak = 0

if current_streak > 0:
    consecutive_losses.append(current_streak)

if consecutive_losses:
    max_consecutive = max(consecutive_losses)
    print(f'   Maximum consecutive underperforming months: {max_consecutive}')
    print(f'   Average losing streak length: {np.mean(consecutive_losses):.1f} months')
else:
    print('   No consecutive underperforming periods!')

# High-level summary
positive_months = len(df[df['bb_return_pct'] > 0])
print(f'\n🎯 KEY INSIGHTS:')
print(f'   ✅ Months with positive returns: {positive_months}/{total_months} ({positive_months/total_months*100:.1f}%)')
print(f'   📈 Average monthly outperformance: {df["outperformance"].mean():.2f}%')
print(f'   🏆 Win rate vs SPY: {wins/total_months*100:.1f}%')
