#!/usr/bin/env python3
"""Generate and visualize 100 GBM price paths with no drift to verify fan shape."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from market.gbm import GBM


def generate_gbm_paths(n_paths: int = 100, horizon: float = 3600, dt: float = 1.0, 
                      initial_price: float = 100.0, volatility: float = 0.02, 
                      drift: float = 0.0, base_seed: int = 42) -> tuple:
    """Generate multiple GBM price paths with no drift.
    
    Args:
        n_paths: Number of price paths to generate
        horizon: Time horizon in seconds
        dt: Time step in seconds
        initial_price: Starting price
        volatility: Volatility parameter (sigma)
        drift: Drift parameter (mu) - set to 0 for no drift
        base_seed: Base random seed
    
    Returns:
        Tuple of (time_points, price_paths) where price_paths is n_paths x time_steps
    """
    time_points = np.arange(0, horizon + dt, dt)
    n_steps = len(time_points)
    price_paths = np.zeros((n_paths, n_steps))
    
    print(f"🎲 Generating {n_paths} GBM paths...")
    print(f"   • Initial price: ${initial_price:.2f}")
    print(f"   • Volatility (σ): {volatility:.3f}")
    print(f"   • Drift (μ): {drift:.3f}")
    print(f"   • Time horizon: {horizon:.0f} seconds ({horizon/3600:.1f} hours)")
    print(f"   • Time step: {dt:.1f} seconds")
    
    for i in range(n_paths):
        # Create a fresh random state for each path
        random_state = np.random.RandomState(base_seed + i)
        
        # Create GBM instance
        gbm = GBM(
            mu=drift,
            sigma=volatility,
            dt=dt,
            s0=initial_price,
            random_state=random_state
        )
        
        # Extract price path
        for j, t in enumerate(time_points):
            price_paths[i, j] = gbm.mid_price(t)
    
    return time_points, price_paths


def create_fan_shape_visualization(time_points: np.ndarray, price_paths: np.ndarray, 
                                 output_dir: str = "results/price_analysis") -> Path:
    """Create comprehensive fan shape visualization."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GBM Fan Shape Analysis: 100 Paths with No Drift (μ=0)', fontsize=16, fontweight='bold')
    
    # Convert time to hours for better readability
    time_hours = time_points / 3600
    initial_price = price_paths[0, 0]
    
    # Plot 1: All paths
    ax1 = axes[0, 0]
    for i in range(len(price_paths)):
        ax1.plot(time_hours, price_paths[i], alpha=0.3, linewidth=0.5, color='blue')
    
    ax1.axhline(y=initial_price, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Initial Price: ${initial_price:.2f}')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('All 100 GBM Price Paths')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Price distribution at different time points
    ax2 = axes[0, 1]
    time_snapshots = [0.25, 0.5, 0.75, 1.0]  # Hours
    colors = ['red', 'orange', 'green', 'blue']
    
    for i, (t_hour, color) in enumerate(zip(time_snapshots, colors)):
        t_idx = int(t_hour * 3600 / (time_points[1] - time_points[0]))
        if t_idx < len(time_points):
            prices_at_t = price_paths[:, t_idx]
            ax2.hist(prices_at_t, bins=20, alpha=0.6, label=f't={t_hour:.1f}h', 
                    color=color, density=True)
    
    ax2.axvline(x=initial_price, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Initial Price')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Density')
    ax2.set_title('Price Distribution at Different Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fan shape envelope (percentiles)
    ax3 = axes[1, 0]
    percentiles = [5, 25, 50, 75, 95]
    price_percentiles = np.percentile(price_paths, percentiles, axis=0)
    
    # Plot percentile bands
    ax3.fill_between(time_hours, price_percentiles[0], price_percentiles[-1], 
                     alpha=0.2, color='blue', label='5th-95th percentile')
    ax3.fill_between(time_hours, price_percentiles[1], price_percentiles[-2], 
                     alpha=0.3, color='blue', label='25th-75th percentile')
    
    # Plot median
    ax3.plot(time_hours, price_percentiles[2], 'r-', linewidth=3, label='Median (50th percentile)')
    ax3.axhline(y=initial_price, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Initial Price')
    
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Price ($)')
    ax3.set_title('GBM Fan Shape Envelope (Percentiles)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Theoretical vs observed variance
    ax4 = axes[1, 1]
    
    # Calculate observed variance at each time point
    observed_variance = np.var(price_paths, axis=0)
    
    # Theoretical variance for GBM with no drift: Var[S(t)] = S₀² * (e^(σ²t) - 1)
    sigma = 0.02  # volatility used
    S0 = initial_price
    theoretical_variance = S0**2 * (np.exp(sigma**2 * time_points) - 1)
    
    ax4.plot(time_hours, observed_variance, 'b-', linewidth=2, label='Observed Variance', alpha=0.8)
    ax4.plot(time_hours, theoretical_variance, 'r--', linewidth=2, label='Theoretical Variance', alpha=0.8)
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Price Variance')
    ax4.set_title('Theoretical vs Observed Variance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale to better see the exponential growth
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_path / "gbm_100_paths_no_drift_fan_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_file


def print_statistical_summary(price_paths: np.ndarray, time_points: np.ndarray):
    """Print statistical summary of the GBM paths."""
    print(f"\n📊 Statistical Summary of {len(price_paths)} GBM Paths")
    print("=" * 60)
    
    initial_price = price_paths[0, 0]
    final_prices = price_paths[:, -1]
    
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"\nFinal Prices (after {time_points[-1]/3600:.1f} hours):")
    print(f"  • Mean:     ${np.mean(final_prices):.2f}")
    print(f"  • Median:   ${np.median(final_prices):.2f}")
    print(f"  • Std Dev:  ${np.std(final_prices):.2f}")
    print(f"  • Min:      ${np.min(final_prices):.2f}")
    print(f"  • Max:      ${np.max(final_prices):.2f}")
    print(f"  • Range:    ${np.max(final_prices) - np.min(final_prices):.2f}")
    
    # Check if median is close to initial price (expected for no drift)
    median_deviation = (np.median(final_prices) - initial_price) / initial_price * 100
    print(f"\nMedian deviation from initial price: {median_deviation:+.2f}%")
    
    # Calculate price spread (fan width) at different times
    print(f"\nFan Width Analysis (95th - 5th percentile):")
    time_checkpoints = [0.25, 0.5, 0.75, 1.0]  # Hours
    
    for t_hour in time_checkpoints:
        t_idx = int(t_hour * 3600 / (time_points[1] - time_points[0]))
        if t_idx < len(time_points):
            prices_at_t = price_paths[:, t_idx]
            p5, p95 = np.percentile(prices_at_t, [5, 95])
            spread = p95 - p5
            spread_pct = spread / initial_price * 100
            print(f"  • t={t_hour:.1f}h: ${spread:.2f} ({spread_pct:.1f}% of initial price)")


def main():
    """Main function to generate and analyze GBM fan shape."""
    print("🌀 GBM Fan Shape Analysis")
    print("=" * 50)
    
    # Parameters
    n_paths = 100
    horizon = 3600  # 1 hour
    dt = 10.0  # 10 second steps for smoother paths
    initial_price = 100.0
    volatility = 0.02  # 2% volatility
    drift = 0.0  # No drift - this is key for fan shape
    
    # Generate paths
    time_points, price_paths = generate_gbm_paths(
        n_paths=n_paths, 
        horizon=horizon, 
        dt=dt, 
        initial_price=initial_price, 
        volatility=volatility, 
        drift=drift
    )
    
    # Create visualization
    print(f"\n📈 Creating fan shape visualization...")
    output_file = create_fan_shape_visualization(time_points, price_paths)
    
    # Print statistical summary
    print_statistical_summary(price_paths, time_points)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Visualization saved to: {output_file}")
    print(f"\n🔍 Key Fan Shape Characteristics:")
    print(f"   • The paths start at a single point (${initial_price:.2f})")
    print(f"   • They spread out over time in a characteristic 'fan' shape")
    print(f"   • With no drift (μ=0), the median path stays near the initial price")
    print(f"   • The variance grows exponentially with time")
    print(f"   • This is the classic GBM behavior without drift!")


if __name__ == "__main__":
    main() 