#!/usr/bin/env python3
"""
Final Results Visualization
===========================

Creates enhanced visualizations and summary statistics for the trading strategy results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def create_summary_dashboard():
    """Create a comprehensive dashboard of all results"""
    
    # Data from our simulation results
    results_data = {
        'Strategy': ['Moving Average', 'Bollinger Bands', 'Breakout'] * 3,
        'Stock': ['A', 'A', 'A', 'AAL', 'AAL', 'AAL', 'AAP', 'AAP', 'AAP'],
        'Total_Trades': [491, 721, 241, 504, 756, 355, 487, 693, 393],
        'Win_Rate': [35.0, 67.4, 39.8, 31.2, 68.1, 38.0, 32.4, 64.8, 38.2],
        'Total_Return': [3.84, 20.39, 7.61, 5.72, 12.45, -7.90, 0.84, 18.54, -0.15],
        'Best_Trade': [3.45, 2.00, 5.80, 4.14, 2.05, 2.06, 8.51, 1.55, 2.33],
        'Worst_Trade': [-1.37, -2.32, -1.36, -6.81, -6.05, -6.70, -3.33, -3.13, -3.48],
        'Avg_Hold_Time': [220, 84, 599, 203, 92, 280, 179, 79, 224]
    }
    
    df = pd.DataFrame(results_data)
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trading Strategy Performance Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Win Rate by Strategy
    win_rate_avg = df.groupby('Strategy')['Win_Rate'].mean()
    axes[0, 0].bar(win_rate_avg.index, win_rate_avg.values, 
                   color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[0, 0].set_title('Average Win Rate by Strategy', fontweight='bold')
    axes[0, 0].set_ylabel('Win Rate (%)')
    axes[0, 0].set_ylim(0, 80)
    for i, v in enumerate(win_rate_avg.values):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 2. Total Return by Strategy and Stock
    pivot_return = df.pivot(index='Stock', columns='Strategy', values='Total_Return')
    pivot_return.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
    axes[0, 1].set_title('Total Return by Strategy and Stock', fontweight='bold')
    axes[0, 1].set_ylabel('Total Return (%)')
    axes[0, 1].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Number of Trades
    pivot_trades = df.pivot(index='Stock', columns='Strategy', values='Total_Trades')
    pivot_trades.plot(kind='bar', ax=axes[0, 2], alpha=0.8)
    axes[0, 2].set_title('Number of Trades by Strategy', fontweight='bold')
    axes[0, 2].set_ylabel('Total Trades')
    axes[0, 2].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].tick_params(axis='x', rotation=0)
    
    # 4. Risk-Return Scatter
    for strategy in df['Strategy'].unique():
        strategy_data = df[df['Strategy'] == strategy]
        axes[1, 0].scatter(strategy_data['Worst_Trade'].abs(), strategy_data['Total_Return'], 
                          label=strategy, s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Maximum Loss (%)')
    axes[1, 0].set_ylabel('Total Return (%)')
    axes[1, 0].set_title('Risk vs Return Analysis', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Average Holding Time
    hold_time_avg = df.groupby('Strategy')['Avg_Hold_Time'].mean()
    axes[1, 1].bar(hold_time_avg.index, hold_time_avg.values, 
                   color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[1, 1].set_title('Average Holding Time by Strategy', fontweight='bold')
    axes[1, 1].set_ylabel('Average Hold Time (minutes)')
    for i, v in enumerate(hold_time_avg.values):
        axes[1, 1].text(i, v + 10, f'{v:.0f}m', ha='center', fontweight='bold')
    
    # 6. Strategy Efficiency (Return per Trade)
    df['Return_per_Trade'] = df['Total_Return'] / df['Total_Trades']
    efficiency_avg = df.groupby('Strategy')['Return_per_Trade'].mean()
    axes[1, 2].bar(efficiency_avg.index, efficiency_avg.values * 100,  # Convert to percentage
                   color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[1, 2].set_title('Strategy Efficiency (Return per Trade)', fontweight='bold')
    axes[1, 2].set_ylabel('Average Return per Trade (%)')
    for i, v in enumerate(efficiency_avg.values * 100):
        axes[1, 2].text(i, v + 0.001, f'{v:.3f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/malmorga/ml/strategy_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_strategy_rankings(df):
    """Create strategy rankings based on different metrics"""
    
    print("\n" + "="*60)
    print("STRATEGY RANKINGS")
    print("="*60)
    
    # Calculate average metrics by strategy
    strategy_metrics = df.groupby('Strategy').agg({
        'Win_Rate': 'mean',
        'Total_Return': 'mean',
        'Total_Trades': 'mean',
        'Best_Trade': 'mean',
        'Worst_Trade': 'mean',
        'Avg_Hold_Time': 'mean'
    }).round(2)
    
    strategy_metrics['Return_per_Trade'] = (strategy_metrics['Total_Return'] / strategy_metrics['Total_Trades'] * 100).round(4)
    
    print("\nAVERAGE PERFORMANCE METRICS:")
    print(strategy_metrics)
    
    # Rankings
    rankings = {}
    rankings['Win_Rate'] = strategy_metrics['Win_Rate'].rank(ascending=False)
    rankings['Total_Return'] = strategy_metrics['Total_Return'].rank(ascending=False)  
    rankings['Return_per_Trade'] = strategy_metrics['Return_per_Trade'].rank(ascending=False)
    rankings['Risk_Management'] = strategy_metrics['Worst_Trade'].rank(ascending=True)  # Higher is better (less negative)
    
    ranking_df = pd.DataFrame(rankings)
    ranking_df['Overall_Rank'] = ranking_df.mean(axis=1).rank()
    
    print(f"\nSTRATEGY RANKINGS (1=Best, 3=Worst):")
    print(ranking_df.round(1))
    
    # Best strategy overall
    best_strategy = ranking_df['Overall_Rank'].idxmin()
    print(f"\n🏆 BEST OVERALL STRATEGY: {best_strategy}")
    
    return strategy_metrics, ranking_df

def print_key_insights():
    """Print key insights and recommendations"""
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    insights = [
        "📈 BOLLINGER BANDS is the clear winner with:",
        "   • Highest win rates (64-68%)",
        "   • Best total returns (12-20%)",
        "   • Most consistent performance across stocks",
        "",
        "📊 MOVING AVERAGE CROSSOVER shows:",
        "   • Low win rates but reasonable returns when it works",
        "   • Best suited for strong trending markets",
        "   • Needs optimization to reduce whipsaws",
        "",
        "🚀 BREAKOUT STRATEGY demonstrates:",
        "   • Highest individual trade potential (up to 8.51%)",
        "   • Inconsistent performance across different stocks", 
        "   • Best for high-volatility environments",
        "",
        "💰 TRANSACTION COSTS are significant:",
        "   • 0.2% per round trip severely impacts profitability",
        "   • High-frequency strategies (Bollinger) hit hardest",
        "   • Consider broker selection and position sizing",
        "",
        "🎯 OPTIMIZATION OPPORTUNITIES:",
        "   • Test different parameter combinations",
        "   • Implement dynamic position sizing",
        "   • Add market regime filters",
        "   • Consider portfolio-based approaches"
    ]
    
    for insight in insights:
        print(insight)

def main():
    """Main function"""
    print("Creating Trading Strategy Analysis Dashboard...")
    
    # Create comprehensive dashboard
    df = create_summary_dashboard()
    
    # Create rankings and analysis
    metrics, rankings = create_strategy_rankings(df)
    
    # Print key insights
    print_key_insights()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print("Generated files:")
    print("• strategy_dashboard.png - Comprehensive performance dashboard")
    print("• results_summary.md - Detailed written analysis")
    print("• Individual stock comparison charts (A_*, AAL_*, AAP_*)")

if __name__ == "__main__":
    main()
