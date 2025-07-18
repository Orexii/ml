"""
Configuration file for Trading Strategy Simulator
=================================================

Modify these parameters to test different strategy configurations
"""

# Strategy Parameters
STRATEGY_CONFIG = {
    # Moving Average Crossover Strategy
    'ma_short_period': 20,      # Short-term moving average period (minutes)
    'ma_long_period': 50,       # Long-term moving average period (minutes)
    
    # Bollinger Bands Strategy
    'bb_period': 20,            # Bollinger Bands period (minutes)
    'bb_std_dev': 2.0,          # Number of standard deviations for bands
    
    # Breakout Strategy
    'breakout_lookback': 60,    # Lookback period for support/resistance (minutes)
    'profit_target_pct': 0.5,   # Profit target percentage
    'stop_loss_pct': 0.3,       # Stop loss percentage
    
    # RSI Divergence Strategy
    'rsi_period': 14,           # RSI calculation period
    'rsi_overbought': 70,       # RSI overbought level
    'rsi_oversold': 30,         # RSI oversold level
    'divergence_lookback': 20,  # Lookback period for divergence detection
    
    # Opening Range Breakout Strategy
    'orb_range_minutes': 30,    # Opening range duration (minutes from market open)
    'orb_market_open': '09:30', # Market open time (HH:MM format)
    
    # Enhanced Risk Management
    'use_trend_filter': True,   # Enable trend filter using 200-period MA
    'trend_ma_period': 200,     # Trend filter moving average period
    'use_atr_targets': True,    # Use ATR-based dynamic targets instead of fixed %
    'atr_period': 14,           # ATR calculation period
    'atr_stop_multiplier': 2.0, # Stop loss = entry_price ± (ATR * multiplier)
    'atr_target_multiplier': 4.0, # Profit target = entry_price ± (ATR * multiplier)
    'use_volume_filter': True,  # Enable volume confirmation
    'volume_ma_period': 20,     # Volume moving average period
    'volume_threshold': 1.5,    # Volume must be > threshold * avg_volume
    
    # Portfolio-Level Risk Management
    'use_dynamic_position_sizing': True,  # Enable volatility-based position sizing
    'risk_per_trade_pct': 1.0,          # Percentage of capital to risk per trade
    'max_concurrent_positions': 5,       # Maximum number of open positions
    'correlation_threshold': 0.7,        # Skip trades on correlated assets above this
    
    # Market Regime Filtering
    'use_market_regime_filter': True,    # Enable market regime adaptation
    'volatility_lookback': 20,           # Period for volatility calculation
    'low_vol_threshold': 15,             # Low volatility threshold (percentile)
    'high_vol_threshold': 85,            # High volatility threshold (percentile)
    
    # Enhanced Strategy Logic
    'bb_opposite_band_exit': True,       # Exit BB trades at opposite band instead of middle
    'orb_end_of_day_exit': True,         # Force ORB exit at end of trading day
    'orb_fadeout_bars': 5,               # Exit ORB if reversal within N bars
    
    # Advanced Strategy Features
    'use_trailing_stops': True,          # Enable trailing stop-losses
    'trailing_stop_atr_multiplier': 2.0, # ATR multiplier for trailing stops
    'use_adx_filter': True,              # Enable ADX trend strength filter
    'adx_threshold': 25.0,               # Minimum ADX for trend strategies
    'use_profit_zones': True,            # Enable profit-taking zones
    'profit_zone_partial_exit': 0.5,     # Percentage to exit at profit zone (0.5 = 50%)
    
    # Strategy-Specific Market Regime Preferences
    'enable_regime_specific_filtering': True,  # Enable strategy-specific regime filtering
    'bb_preferred_regime': 'low_vol',          # Bollinger Bands: best in low volatility
    'ma_preferred_regime': 'high_vol',         # MA Crossover: best in trending markets
    'breakout_preferred_regime': 'medium',     # Breakout: best in medium volatility
    'rsi_preferred_regime': 'any',             # RSI Divergence: works in any regime
    'orb_preferred_regime': 'medium',          # ORB: best in medium volatility
    
    # Correlation Filtering
    'enable_correlation_filter': True,    # Prevent highly correlated positions
    'correlation_lookback_period': 60,    # Days to calculate correlation
    'max_position_correlation': 0.8,      # Maximum correlation between positions
    
    # Walk-Forward Optimization
    'enable_walk_forward': False,         # Enable walk-forward analysis
    'training_period_months': 12,         # Training period for optimization
    'testing_period_months': 3,           # Testing period for validation
    
    # Trading Parameters
    'initial_capital': 100000.0,        # Starting capital ($)
    'position_size_pct': 0.1,           # Position size as % of capital (0.1 = 10%)
    'transaction_cost_pct': 0.001,      # Transaction cost (0.001 = 0.1%)
    'slippage_pct': 0.001,              # Slippage cost (0.001 = 0.1%)
}

# Data Loading Parameters
DATA_CONFIG = {
    'max_rows': 50000,          # Limit data rows for testing (None for full dataset)
    'test_symbols_count': 3,    # Number of symbols to test (None for all available)
    'specific_symbols': None,   # List of specific symbols to test (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    'exclude_symbols': [],      # List of symbols to exclude from testing
    'min_data_points': 100,     # Minimum data points required per symbol
    'date_range': {             # Optional date range filtering
        'start_date': None,     # 'YYYY-MM-DD' format or None for no limit
        'end_date': None,       # 'YYYY-MM-DD' format or None for no limit
    }
}

# Execution Parameters
EXECUTION_CONFIG = {
    'parallel_processing': False,     # Enable parallel strategy execution (future feature)
    'verbose_output': True,          # Print detailed progress information
    'save_individual_trades': True,  # Save detailed trade logs
    'real_time_plotting': False,     # Show plots in real-time (vs batch at end)
}

# Output Parameters
OUTPUT_CONFIG = {
    'save_charts': True,        # Save strategy comparison charts
    'chart_format': 'png',      # Chart format (png, pdf, svg)
    'chart_dpi': 300,           # Chart resolution
    'output_directory': './',   # Directory to save outputs
    'filename_prefix': '',      # Prefix for output files
}

# Advanced Strategy Variations (for future experimentation)
STRATEGY_VARIATIONS = {
    'ma_crossover': [
        {'ma_short_period': 5, 'ma_long_period': 15},   # Fast - Quick signals
        {'ma_short_period': 10, 'ma_long_period': 30},  # Medium-fast with wider gap
        {'ma_short_period': 10, 'ma_long_period': 50},  # Wide gap for trend confirmation
        {'ma_short_period': 20, 'ma_long_period': 50},  # Current default
        {'ma_short_period': 15, 'ma_long_period': 60},  # Conservative with strong trend filter
    ],
    
    'bollinger_bands': [
        {'bb_period': 15, 'bb_std_dev': 1.5},   # Tight bands, more signals
        {'bb_period': 20, 'bb_std_dev': 1.8},   # Slightly tighter than default
        {'bb_period': 20, 'bb_std_dev': 2.0},   # Current default
        {'bb_period': 25, 'bb_std_dev': 2.2},   # Wider period, slightly wider bands
        {'bb_period': 30, 'bb_std_dev': 2.5},   # Wide bands, fewer but stronger signals
    ],
    
    'breakout': [
        {'breakout_lookback': 30, 'profit_target_pct': 0.8, 'stop_loss_pct': 0.3},   # Fast, improved R:R ratio
        {'breakout_lookback': 45, 'profit_target_pct': 1.0, 'stop_loss_pct': 0.4},   # Medium-term, 2.5:1 R:R
        {'breakout_lookback': 60, 'profit_target_pct': 1.2, 'stop_loss_pct': 0.4},   # Current period, better R:R
        {'breakout_lookback': 90, 'profit_target_pct': 1.5, 'stop_loss_pct': 0.5},   # Longer-term, 3:1 R:R
        {'breakout_lookback': 120, 'profit_target_pct': 2.0, 'stop_loss_pct': 0.6},  # Major levels, 3.3:1 R:R
    ],
    
    'rsi_divergence': [
        {'rsi_period': 10, 'divergence_lookback': 15, 'rsi_overbought': 75, 'rsi_oversold': 25},  # Sensitive
        {'rsi_period': 14, 'divergence_lookback': 20, 'rsi_overbought': 70, 'rsi_oversold': 30},  # Standard
        {'rsi_period': 21, 'divergence_lookback': 25, 'rsi_overbought': 65, 'rsi_oversold': 35},  # Conservative
    ],
    
    'opening_range_breakout': [
        {'orb_range_minutes': 15},   # Quick 15-minute range
        {'orb_range_minutes': 30},   # Standard 30-minute range
        {'orb_range_minutes': 60},   # Extended 1-hour range
    ]
}

# Preset Configurations for Common Use Cases
PRESETS = {
    'quick_test': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'date_range': {
            'start_date': '2017-09-15',
            'end_date': '2017-10-15'
        },
        'max_rows': 5000,
        'verbose_output': True,
        'save_charts': True
    },
    
    'major_stocks': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.A', 'JNJ', 'V', 'WMT'],
        'date_range': {
            'start_date': '2017-10-01',
            'end_date': '2017-12-31'
        },
        'verbose_output': False,
        'save_charts': True
    },
    
    'tech_stocks': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'CRM', 'ADBE'],
        'date_range': {
            'start_date': '2018-01-01',
            'end_date': '2018-02-16'
        },
        'verbose_output': True,
        'save_charts': True
    },
    
    'sample_run': {
        'test_symbols_count': 5,
        'date_range': {
            'start_date': '2017-11-01',
            'end_date': '2017-11-30'
        },
        'verbose_output': False,
        'save_charts': False
    },
    
    'full_analysis': {
        'test_symbols_count': 50,
        'date_range': {
            'start_date': None,  # Use full dataset
            'end_date': None
        },
        'verbose_output': False,
        'save_charts': True
    },
    
    # Optimized strategy presets based on performance analysis
    'optimized_fast': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'date_range': {
            'start_date': '2017-10-01',
            'end_date': '2018-01-31'
        },
        'strategy_overrides': {
            'ma_short_period': 10,
            'ma_long_period': 50,      # Wide gap for better trend confirmation
            'bb_std_dev': 1.8,         # Tighter bands for more signals
            'profit_target_pct': 1.0,  # Better risk/reward ratio
            'stop_loss_pct': 0.4
        },
        'verbose_output': True,
        'save_charts': True
    },
    
    'optimized_conservative': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ'],
        'date_range': {
            'start_date': '2017-11-01',
            'end_date': '2018-02-16'
        },
        'strategy_overrides': {
            'ma_short_period': 15,
            'ma_long_period': 60,      # Very wide gap for strong trend filter
            'bb_period': 25,           # Longer period for stability
            'bb_std_dev': 2.2,         # Slightly wider bands
            'breakout_lookback': 90,   # Longer lookback for major levels
            'profit_target_pct': 1.5,  # 3:1 risk/reward ratio
            'stop_loss_pct': 0.5
        },
        'verbose_output': False,
        'save_charts': True
    },
    
    'risk_reward_optimized': {
        'test_symbols_count': 10,
        'date_range': {
            'start_date': '2017-09-15',
            'end_date': '2018-02-16'
        },
        'strategy_overrides': {
            'ma_short_period': 10,
            'ma_long_period': 30,      # Medium-fast with good separation
            'bb_std_dev': 1.5,         # Tight bands for more entry opportunities
            'breakout_lookback': 45,   # Medium-term breakouts
            'profit_target_pct': 1.2,  # 3:1 risk/reward
            'stop_loss_pct': 0.4
        },
        'verbose_output': False,
        'save_charts': True
    },
    
    # Enhanced strategies with trend filters and dynamic risk management
    'enhanced_adaptive': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'date_range': {
            'start_date': '2017-10-01',
            'end_date': '2018-01-31'
        },
        'strategy_overrides': {
            'use_trend_filter': True,
            'use_atr_targets': True,
            'use_volume_filter': True,
            'ma_short_period': 10,
            'ma_long_period': 30,
            'bb_std_dev': 1.8,
            'breakout_lookback': 45,
            'atr_stop_multiplier': 2.0,
            'atr_target_multiplier': 4.0,
            'volume_threshold': 1.5
        },
        'verbose_output': True,
        'save_charts': True
    },
    
    'new_strategies_test': {
        'specific_symbols': ['AAPL', 'MSFT'],
        'date_range': {
            'start_date': '2017-11-01',
            'end_date': '2017-12-31'
        },
        'strategy_overrides': {
            'use_trend_filter': True,
            'use_atr_targets': True,
            'rsi_period': 14,
            'divergence_lookback': 20,
            'orb_range_minutes': 30
        },
        'verbose_output': True,
        'save_charts': True
    },
    
    # Ultimate enhanced strategy with all improvements
    'ultimate_enhanced': {
        'specific_symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'date_range': {
            'start_date': '2017-10-01',
            'end_date': '2018-01-31'
        },
        'strategy_overrides': {
            # Core enhancements
            'use_trend_filter': True,
            'use_atr_targets': True,
            'use_volume_filter': True,
            'use_market_regime_filter': True,
            'use_dynamic_position_sizing': True,
            
            # NEW: Advanced strategy features
            'use_trailing_stops': True,
            'trailing_stop_atr_multiplier': 2.0,
            'use_adx_filter': True,
            'adx_threshold': 25.0,
            'use_profit_zones': True,
            'profit_zone_partial_exit': 0.5,
            
            # NEW: Strategy-specific regime filtering
            'enable_regime_specific_filtering': True,
            'bb_preferred_regime': 'low_vol',
            'ma_preferred_regime': 'high_vol',
            'breakout_preferred_regime': 'medium',
            'rsi_preferred_regime': 'any',
            'orb_preferred_regime': 'medium',
            
            # NEW: Correlation filtering
            'enable_correlation_filter': True,
            'correlation_lookback_period': 60,
            'max_position_correlation': 0.8,
            
            # Strategy-specific improvements
            'bb_opposite_band_exit': True,
            'orb_end_of_day_exit': True,
            'orb_fadeout_bars': 5,
            
            # Risk management
            'risk_per_trade_pct': 0.8,  # Reduced for better risk control
            'max_concurrent_positions': 3,  # Reduced for diversification
            'atr_stop_multiplier': 2.0,
            'atr_target_multiplier': 4.0,
            
            # Optimized parameters
            'ma_short_period': 10,
            'ma_long_period': 30,
            'bb_std_dev': 1.8,
            'rsi_period': 14,
            'orb_range_minutes': 30
        },
        'verbose_output': True,
        'save_charts': True
    }
}

# Function to apply preset configuration
def apply_preset(preset_name):
    """Apply a preset configuration"""
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")
    
    preset = PRESETS[preset_name]
    
    # Update DATA_CONFIG with preset values
    for key, value in preset.items():
        if key == 'strategy_overrides':
            # Apply strategy parameter overrides
            for strategy_key, strategy_value in value.items():
                if strategy_key in STRATEGY_CONFIG:
                    STRATEGY_CONFIG[strategy_key] = strategy_value
        elif key in DATA_CONFIG:
            DATA_CONFIG[key] = value
        elif key in EXECUTION_CONFIG:
            EXECUTION_CONFIG[key] = value
        elif key in OUTPUT_CONFIG:
            OUTPUT_CONFIG[key] = value
    
    return preset
