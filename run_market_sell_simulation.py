#!/usr/bin/env python3
"""Simulate 10 market sell orders of 10k each on a single price path."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.test_utils.analyzers import PricePathAnalyzer
from src.test_utils.strategies import MarketOrderStrategy
from src.test_utils.simulation import create_default_config, SimulationRunner
from src.test_utils.analysis import analyze_order_performance, print_analysis_summary
from src.test_utils.visualization import (
    setup_figure, plot_price_path, plot_fills, plot_cumulative_fills,
    plot_performance_summary, save_figure, finalize_plot
)
from src.core.orders import Side
import matplotlib.pyplot as plt
import numpy as np


def run_market_sell_simulation():
    """Run simulation with 10 market sell orders of 10k each."""
    
    print("🎯 Market Sell Orders Simulation")
    print("=" * 50)
    print("Configuration:")
    print("- Order type: Market SELL orders")
    print("- Order size: 10,000 units each")
    print("- Number of orders: 10")
    print("- Total quantity: 100,000 units")
    print()
    
    # Configure simulation
    config = create_default_config(
        horizon=250.0,  # Just over 4 minutes (enough for 10 orders at 25s intervals)
        volatility=0.01,  # 1% volatility (reduced to avoid overflow)
        drift=0.005,  # 0.5% upward drift (reduced)
        random_seed=42,
        initial_price=100.0,
        spread=0.05,  # 5 cent spread
        impact_gamma=0.001,  # Market impact coefficient
    )
    
    # Create strategy with analyzer
    analyzer = PricePathAnalyzer()
    strategy = MarketOrderStrategy(
        order_qty=10000.0,  # 10k per order
        n_orders=10,        # 10 orders total
        interval=25.0,      # 25 seconds between orders
        side=Side.SELL,     # SELL orders
        analyzer=analyzer
    )
    
    print("🚀 Running simulation...")
    
    # Run simulation
    runner = SimulationRunner(config=config)
    results = runner.run_simulation(
        strategy, 
        analyzer, 
        target_qty=200000.0  # Set higher than our 100k total to not limit simulation
    )
    
    print("✅ Simulation completed!")
    print()
    
    # Debug: print what's actually in results
    print("🔍 Results keys:", list(results.keys()))
    
    # Analyze and print results
    analysis = analyze_order_performance(
        fills=results["fills"],
        order_creations=results["order_creations"],
        price_data=results.get("price_path", []),  # Use .get() with default
        order_prefix="market"
    )
    
    print_analysis_summary(analysis, "📊 Market Sell Orders Analysis")
    
    # Additional detailed statistics
    print("\n📈 Detailed Performance Metrics")
    print("=" * 40)
    
    fills = results["fills"]
    order_creations = results["order_creations"]
    price_path = results.get("price_path", [])
    
    if fills:
        # Calculate VWAP
        total_value = sum(f.price * f.qty for f in fills)
        total_qty = sum(f.qty for f in fills)
        vwap = total_value / total_qty if total_qty > 0 else 0
        
        # Calculate market price at start and end
        start_price = price_path[0][1] if price_path else 100.0
        end_price = price_path[-1][1] if price_path else 100.0
        
        # Calculate fill price statistics
        fill_prices = [f.price for f in fills]
        
        print(f"VWAP (Volume Weighted Avg Price): ${vwap:.3f}")
        print(f"Market Price at Start: ${start_price:.3f}")
        print(f"Market Price at End: ${end_price:.3f}")
        print(f"Price Movement During Execution: ${end_price - start_price:+.3f} ({(end_price/start_price-1)*100:+.2f}%)")
        print()
        
        print(f"Fill Price Statistics:")
        print(f"- Average: ${np.mean(fill_prices):.3f}")
        print(f"- Median: ${np.median(fill_prices):.3f}")
        print(f"- Min: ${np.min(fill_prices):.3f}")
        print(f"- Max: ${np.max(fill_prices):.3f}")
        print(f"- Standard Deviation: ${np.std(fill_prices):.3f}")
        print()
        
        # Market impact analysis
        theoretical_impact_per_order = config.impact_gamma * 10000  # γ * order_size
        total_theoretical_impact = theoretical_impact_per_order * len(fills)
        
        print(f"Market Impact Analysis:")
        print(f"- Theoretical impact per order: ${theoretical_impact_per_order:.3f}")
        print(f"- Total theoretical impact: ${total_theoretical_impact:.3f}")
        print(f"- Actual VWAP vs start price: ${vwap - start_price:+.3f}")
        print()
        
        # Gas costs
        total_gas_paid = sum(f.gas_paid for f in fills)
        print(f"Transaction Costs:")
        print(f"- Total gas paid: ${total_gas_paid:.6f}")
        print(f"- Gas per fill: ${total_gas_paid/len(fills):.6f}")
        
        # Execution timing
        fill_times = [f.timestamp for f in fills]
        creation_times = [oc[0] for oc in order_creations]
        
        print(f"\nExecution Timing:")
        print(f"- First order created: {min(creation_times):.1f}s")
        print(f"- Last order created: {max(creation_times):.1f}s")
        print(f"- First fill: {min(fill_times):.1f}s")
        print(f"- Last fill: {max(fill_times):.1f}s")
        print(f"- Total execution window: {max(fill_times) - min(creation_times):.1f}s")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    create_market_sell_visualization(results, analysis)
    print("📈 Visualization saved to: results/market_sell_simulation.png")
    
    return results, analysis


def create_market_sell_visualization(results, analysis):
    """Create comprehensive visualization of the market sell simulation."""
    
    # Create figure with subplots
    fig, axes = setup_figure(2, 3, (18, 12))
    
    fills = results["fills"]
    order_creations = results["order_creations"] 
    price_path = results.get("price_path", [])
    
    # Plot 1: Price path with order executions
    if price_path and fills:
        times = [p[0] for p in price_path]
        prices = [p[1] for p in price_path]
        
        plot_price_path(axes[0, 0], times, prices, "Price Path with Market Sell Orders")
        plot_fills(axes[0, 0], fills, Side.SELL)
        
        # Add order creation markers
        creation_times = [oc[0] for oc in order_creations]
        creation_prices = []
        for ct in creation_times:
            # Find closest price point
            closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - ct))
            creation_prices.append(prices[closest_idx])
        
        axes[0, 0].scatter(creation_times, creation_prices, 
                          color='orange', marker='v', s=60, alpha=0.7, 
                          label='Order Created', zorder=5)
        axes[0, 0].legend()
    
    # Plot 2: Cumulative fills over time
    if fills:
        fill_times = [f.timestamp for f in fills]
        fill_qtys = [f.qty for f in fills]
        cumulative_qty = np.cumsum(fill_qtys)
        
        axes[0, 1].plot(fill_times, cumulative_qty, 'go-', linewidth=2, markersize=8, label='Cumulative Quantity')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Cumulative Quantity')
        axes[0, 1].set_title('Cumulative Quantity Sold', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot 3: Fill price evolution
    if fills:
        fill_times = [f.timestamp for f in fills]
        fill_prices = [f.price for f in fills]
        
        axes[0, 2].plot(fill_times, fill_prices, 'ro-', linewidth=2, markersize=8, label='Fill Prices')
        axes[0, 2].set_xlabel('Time (seconds)')
        axes[0, 2].set_ylabel('Fill Price ($)')
        axes[0, 2].set_title('Fill Price Evolution')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Add VWAP line
        if len(fills) > 0:
            total_value = sum(f.price * f.qty for f in fills)
            total_qty = sum(f.qty for f in fills)
            vwap = total_value / total_qty
            axes[0, 2].axhline(y=vwap, color='green', linestyle='--', linewidth=2, label=f'VWAP: ${vwap:.3f}')
            axes[0, 2].legend()
    
    # Plot 4: Price distribution
    if fills:
        fill_prices = [f.price for f in fills]
        axes[1, 0].hist(fill_prices, bins=max(3, len(fills)//2), alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Fill Price ($)')
        axes[1, 0].set_ylabel('Number of Fills')
        axes[1, 0].set_title('Fill Price Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add mean line
        mean_price = np.mean(fill_prices)
        axes[1, 0].axvline(x=mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.3f}')
        axes[1, 0].legend()
    
    # Plot 5: Execution delays
    if fills and order_creations:
        creation_times = {oc[1]: oc[0] for oc in order_creations}
        execution_delays = []
        
        for fill in fills:
            if fill.order_id in creation_times:
                delay = fill.timestamp - creation_times[fill.order_id]
                execution_delays.append(delay)
        
        if execution_delays:
            axes[1, 1].bar(range(len(execution_delays)), execution_delays, alpha=0.7, color='lightcoral')
            axes[1, 1].set_xlabel('Order Number')
            axes[1, 1].set_ylabel('Execution Delay (seconds)')
            axes[1, 1].set_title('Order Execution Delays')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add average line
            avg_delay = np.mean(execution_delays)
            axes[1, 1].axhline(y=avg_delay, color='blue', linestyle='--', linewidth=2, label=f'Avg: {avg_delay:.1f}s')
            axes[1, 1].legend()
    
    # Plot 6: Performance summary (text-based)
    axes[1, 2].axis('off')
    
    # Handle None values safely
    vwap_str = f"${analysis.vwap:.3f}" if analysis.vwap else "N/A"
    avg_price_str = f"${analysis.avg_fill_price:.3f}" if analysis.avg_fill_price else "N/A"
    min_price_str = f"${analysis.min_fill_price:.3f}" if analysis.min_fill_price else "N/A"
    max_price_str = f"${analysis.max_fill_price:.3f}" if analysis.max_fill_price else "N/A"
    twap_str = f"${analysis.fill_vs_twap:+.3f}" if analysis.fill_vs_twap else "N/A"
    delay_str = f"{analysis.avg_execution_delay:.1f}s" if analysis.avg_execution_delay else "N/A"
    
    summary_text = f"""Market Sell Performance Summary
    
Total Orders: {analysis.total_orders}
Total Fills: {analysis.total_fills}  
Fill Rate: {analysis.fill_rate:.1%}

Quantity:
- Total Filled: {analysis.total_qty_filled:,.0f}
- Avg per Fill: {analysis.avg_fill_qty:,.0f}

Prices:
- VWAP: {vwap_str}
- Average: {avg_price_str}
- Range: {min_price_str} - {max_price_str}

Performance:
- vs TWAP: {twap_str}
- Avg Delay: {delay_str}
"""
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save figure
    output_path = save_figure("market_sell_simulation.png")
    finalize_plot()
    
    return output_path


if __name__ == "__main__":
    try:
        results, analysis = run_market_sell_simulation()
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 