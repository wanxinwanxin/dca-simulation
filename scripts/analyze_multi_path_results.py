"""Analyze multi-path comparison results and generate summary insights."""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_multi_path_results():
    """Analyze the multi-path comparison results."""
    
    print("📊 Multi-Path Analysis Summary")
    print("=" * 60)
    
    # Load results
    results_dir = Path("results/order_debug")
    market_df = pd.read_csv(results_dir / "market_orders_results.csv")
    dutch_df = pd.read_csv(results_dir / "dutch_orders_results.csv")
    
    print(f"📈 Loaded results from {len(market_df)} price paths")
    print()
    
    # Key metrics comparison
    print("🔍 KEY METRICS COMPARISON")
    print("-" * 40)
    
    metrics = {
        "Fill Rate": ("fill_rate", "%"),
        "Avg Fill Price": ("avg_fill_price", ""),
        "Fill vs TWAP": ("fill_vs_twap", ""),
        "Fill vs Creation": ("fill_vs_creation", ""),
        "Execution Delay": ("avg_execution_delay", "blocks"),
        "Price Volatility": ("price_volatility", ""),
    }
    
    for metric_name, (column, unit) in metrics.items():
        market_val = market_df[column].mean()
        dutch_val = dutch_df[column].mean()
        
        market_std = market_df[column].std()
        dutch_std = dutch_df[column].std()
        
        diff = dutch_val - market_val
        diff_pct = (diff / market_val * 100) if market_val != 0 else 0
        
        unit_str = f" {unit}" if unit else ""
        
        print(f"{metric_name:<20} Market: {market_val:7.3f}±{market_std:5.3f}{unit_str}")
        print(f"{'':<20} Dutch:  {dutch_val:7.3f}±{dutch_std:5.3f}{unit_str}")
        print(f"{'':<20} Diff:   {diff:+7.3f} ({diff_pct:+5.1f}%)")
        print()
    
    # Fill rate analysis
    print("📋 FILL RATE ANALYSIS")
    print("-" * 40)
    
    market_fill_rates = market_df['fill_rate']
    dutch_fill_rates = dutch_df['fill_rate']
    
    print(f"Market Orders:")
    print(f"  Always fill:     {(market_fill_rates == 1.0).sum()}/{len(market_df)} paths ({(market_fill_rates == 1.0).mean():.1%})")
    print(f"  Partial fills:   {((market_fill_rates > 0) & (market_fill_rates < 1.0)).sum()}/{len(market_df)} paths")
    print(f"  No fills:        {(market_fill_rates == 0).sum()}/{len(market_df)} paths")
    
    print(f"Dutch Orders:")
    print(f"  Always fill:     {(dutch_fill_rates == 1.0).sum()}/{len(dutch_df)} paths ({(dutch_fill_rates == 1.0).mean():.1%})")
    print(f"  Partial fills:   {((dutch_fill_rates > 0) & (dutch_fill_rates < 1.0)).sum()}/{len(dutch_df)} paths")
    print(f"  No fills:        {(dutch_fill_rates == 0).sum()}/{len(dutch_df)} paths")
    print()
    
    # Performance analysis
    print("🎯 PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Only analyze paths where both strategies had fills
    both_filled = (market_df['fill_rate'] > 0) & (dutch_df['fill_rate'] > 0)
    n_both = both_filled.sum()
    
    if n_both > 0:
        market_valid = market_df[both_filled]
        dutch_valid = dutch_df[both_filled]
        
        # Price performance
        price_diffs = dutch_valid['avg_fill_price'].values - market_valid['avg_fill_price'].values
        dutch_better_price = (price_diffs > 0).sum()
        
        print(f"Paths with both strategies filled: {n_both}/{len(market_df)}")
        print(f"Dutch achieved better price:       {dutch_better_price}/{n_both} paths ({dutch_better_price/n_both:.1%})")
        print(f"Average price difference:          {price_diffs.mean():+.4f} ± {price_diffs.std():.4f}")
        
        # Execution timing
        delay_diffs = dutch_valid['avg_execution_delay'].values - market_valid['avg_execution_delay'].values
        print(f"Average delay difference:          {delay_diffs.mean():+.2f} ± {delay_diffs.std():.2f} blocks")
        
        # Market conditions analysis
        high_vol_threshold = market_df['price_volatility'].quantile(0.75)
        high_vol_paths = market_valid['price_volatility'] > high_vol_threshold
        
        if high_vol_paths.sum() > 0:
            high_vol_price_diffs = price_diffs[high_vol_paths]
            low_vol_price_diffs = price_diffs[~high_vol_paths]
            
            print(f"\nMarket Condition Analysis:")
            print(f"High volatility paths (top 25%):   {high_vol_paths.sum()}")
            print(f"  Dutch vs Market price diff:     {high_vol_price_diffs.mean():+.4f} ± {high_vol_price_diffs.std():.4f}")
            print(f"Low volatility paths (bottom 75%): {(~high_vol_paths).sum()}")
            print(f"  Dutch vs Market price diff:     {low_vol_price_diffs.mean():+.4f} ± {low_vol_price_diffs.std():.4f}")
    
    # Timing distribution analysis
    print("\n⏱️ TIMING ANALYSIS")
    print("-" * 40)
    
    dutch_delays = dutch_df[dutch_df['avg_execution_delay'].notna()]['avg_execution_delay']
    
    if len(dutch_delays) > 0:
        print(f"Dutch Order Execution Delays:")
        print(f"  Mean:    {dutch_delays.mean():.2f} blocks")
        print(f"  Median:  {dutch_delays.median():.2f} blocks")
        print(f"  Min:     {dutch_delays.min():.2f} blocks")
        print(f"  Max:     {dutch_delays.max():.2f} blocks")
        print(f"  25th %:  {dutch_delays.quantile(0.25):.2f} blocks")
        print(f"  75th %:  {dutch_delays.quantile(0.75):.2f} blocks")
        
        # Categorize delays
        immediate = (dutch_delays <= 1).sum()
        fast = ((dutch_delays > 1) & (dutch_delays <= 5)).sum()
        medium = ((dutch_delays > 5) & (dutch_delays <= 15)).sum()
        slow = (dutch_delays > 15).sum()
        
        total = len(dutch_delays)
        print(f"\nDelay Categories:")
        print(f"  Immediate (≤1 block):   {immediate}/{total} ({immediate/total:.1%})")
        print(f"  Fast (1-5 blocks):      {fast}/{total} ({fast/total:.1%})")
        print(f"  Medium (5-15 blocks):   {medium}/{total} ({medium/total:.1%})")
        print(f"  Slow (>15 blocks):      {slow}/{total} ({slow/total:.1%})")
    
    # Economic analysis
    print("\n💰 ECONOMIC ANALYSIS")
    print("-" * 40)
    
    if n_both > 0:
        # Calculate total value traded and price improvements
        market_total_value = (market_valid['vwap'] * market_valid['total_qty_filled']).sum()
        dutch_total_value = (dutch_valid['vwap'] * dutch_valid['total_qty_filled']).sum()
        
        total_qty_market = market_valid['total_qty_filled'].sum()
        total_qty_dutch = dutch_valid['total_qty_filled'].sum()
        
        print(f"Total quantity traded:")
        print(f"  Market orders: {total_qty_market:,.0f} units")
        print(f"  Dutch orders:  {total_qty_dutch:,.0f} units")
        
        print(f"Total value traded:")
        print(f"  Market orders: {market_total_value:,.2f}")
        print(f"  Dutch orders:  {dutch_total_value:,.2f}")
        
        if total_qty_dutch > 0:
            value_diff = dutch_total_value - market_total_value
            print(f"Value difference:      {value_diff:+,.2f} ({value_diff/market_total_value*100:+.2f}%)")
            
            # Calculate fill vs creation improvement (for sell orders, higher is better)
            market_creation_improvement = market_valid['fill_vs_creation'].mean()
            dutch_creation_improvement = dutch_valid['fill_vs_creation'].mean()
            
            print(f"\nPrice improvement vs creation price:")
            print(f"  Market orders: {market_creation_improvement:+.4f}")
            print(f"  Dutch orders:  {dutch_creation_improvement:+.4f}")
            print(f"  Difference:    {dutch_creation_improvement - market_creation_improvement:+.4f}")
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Full visualization available at: results/order_debug/multi_path_market_vs_dutch_comparison.png")


if __name__ == "__main__":
    analyze_multi_path_results() 