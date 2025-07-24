"""Common visualization utilities for test simulations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.core.events import Fill
from src.core.orders import Side


def setup_figure(nrows: int = 2, ncols: int = 3, figsize: Tuple[int, int] = (18, 12)) -> Tuple[plt.Figure, np.ndarray]:
    """Create a standard figure layout for simulation plots."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # Always ensure axes is a numpy array with at least 1 dimension
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    elif axes.ndim == 0:
        axes = np.array([axes])
    return fig, axes


def save_figure(filename: str, output_dir: str = "results/order_debug", dpi: int = 300) -> Path:
    """Save figure with standard settings."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / filename
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    return output_file


def plot_price_path(ax: plt.Axes, price_data: List[Tuple[float, float]], 
                   title: str = "Price Path", **kwargs) -> None:
    """Plot a price path time series."""
    if not price_data:
        return
    
    times, prices = zip(*price_data)
    ax.plot(times, prices, label='Mid Price', **kwargs)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Price ($)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_fills(ax: plt.Axes, fills: List[Fill], price_data: Optional[List[Tuple[float, float]]] = None,
               side: Optional[Side] = None, **kwargs) -> None:
    """Plot fill events on a price chart."""
    if not fills:
        return
    
    # Filter fills by side if specified
    if side:
        fills = [f for f in fills if f.order_id.startswith(side.value.lower())]
    
    fill_times = [f.timestamp for f in fills]
    fill_prices = [f.price for f in fills]
    
    # Plot background price path if provided
    if price_data:
        plot_price_path(ax, price_data, title="")
    
    # Plot fills
    ax.scatter(fill_times, fill_prices, s=100, marker='o', 
              edgecolors='black', linewidth=2, zorder=10, 
              label='Fills', **kwargs)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Price ($)')
    ax.set_title('Order Fills', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_limit_price_evolution(ax: plt.Axes, price_updates: Dict[str, List[Tuple[float, float]]],
                               title: str = "Limit Price Evolution") -> None:
    """Plot the evolution of limit prices over time."""
    for order_id, updates in price_updates.items():
        if not updates:
            continue
        
        times, prices = zip(*updates)
        ax.plot(times, prices, marker='o', markersize=3, label=f'Order {order_id}', alpha=0.7)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Limit Price ($)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if price_updates:
        ax.legend()


def plot_order_size_evolution(ax: plt.Axes, step_data: List[Dict[str, Any]], 
                             title: str = "Order Size Evolution") -> None:
    """Plot how order size changes over time."""
    times = [s['time'] for s in step_data if 'time' in s]
    sizes = [s.get('order_size', 0) for s in step_data]
    
    ax.plot(times, sizes, 'b-', linewidth=2, label='Order Size')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Order Size')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_cumulative_fills(ax: plt.Axes, fills: List[Fill], 
                         title: str = "Cumulative Fill Quantity") -> None:
    """Plot cumulative fill quantity over time."""
    if not fills:
        ax.text(0.5, 0.5, 'No fills', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title, fontweight='bold')
        return
    
    fill_times = [f.timestamp for f in fills]
    fill_qtys = [f.qty for f in fills]
    cumulative_qty = np.cumsum(fill_qtys)
    
    # Create step function
    times = [0] + [t for t in fill_times for _ in range(2)] 
    qtys = [0] + [q for q in cumulative_qty for _ in range(2)][:-1]
    
    ax.plot(times, qtys, 'g-', linewidth=2, label='Cumulative Quantity')
    
    # Mark individual fills
    ax.scatter(fill_times, cumulative_qty, color='red', s=100, 
              marker='o', edgecolors='black', linewidth=2, zorder=10, label='Fill Events')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Cumulative Quantity')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_performance_summary(ax: plt.Axes, summary_data: Dict[str, Any], 
                           title: str = "Performance Summary") -> None:
    """Plot a text-based performance summary."""
    summary_text = []
    
    for key, value in summary_data.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                summary_text.append(f"{key}: {value:.4f}")
            else:
                summary_text.append(f"{key}: {value}")
        else:
            summary_text.append(f"{key}: {value}")
    
    ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontweight='bold')


def plot_distribution_comparison(ax: plt.Axes, data1: List[float], data2: List[float],
                               labels: List[str], title: str, xlabel: str = "Value") -> None:
    """Plot histogram comparison of two distributions."""
    ax.hist(data1, bins=20, alpha=0.7, label=labels[0], color='blue')
    ax.hist(data2, bins=20, alpha=0.7, label=labels[1], color='orange')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_scatter_comparison(ax: plt.Axes, x_data: List[float], y_data: List[float],
                          xlabel: str, ylabel: str, title: str) -> None:
    """Plot scatter comparison with diagonal reference line."""
    ax.scatter(x_data, y_data, alpha=0.6, s=30, color='green')
    
    # Add diagonal line for reference
    if x_data and y_data:
        min_val = min(min(x_data), min(y_data))
        max_val = max(max(x_data), max(y_data))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Equal Performance')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_metrics_comparison(ax: plt.Axes, metrics1: Dict[str, float], metrics2: Dict[str, float],
                          labels: List[str], title: str = "Metrics Comparison") -> None:
    """Plot bar chart comparison of metrics."""
    metric_names = list(metrics1.keys())
    values1 = list(metrics1.values())
    values2 = list(metrics2.values())
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    ax.bar(x - width/2, values1, width, label=labels[0], alpha=0.7)
    ax.bar(x + width/2, values2, width, label=labels[1], alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_multi_path_visualization(market_results: List[Dict], dutch_results: List[Dict],
                                  output_filename: str = "multi_path_comparison.png") -> Path:
    """Create a comprehensive multi-path comparison visualization."""
    fig, axes = setup_figure(2, 3, (18, 12))
    
    # Convert to DataFrames for easier analysis
    market_df = pd.DataFrame(market_results)
    dutch_df = pd.DataFrame(dutch_results)
    
    # Plot 1: Fill rates
    if 'fill_rate' in market_df.columns and 'fill_rate' in dutch_df.columns:
        plot_distribution_comparison(
            axes[0, 0], market_df['fill_rate'].tolist(), dutch_df['fill_rate'].tolist(),
            ['Market Orders', 'Dutch Orders'], 'Fill Rate Distribution', 'Fill Rate'
        )
    
    # Plot 2: Fill prices
    if 'avg_fill_price' in market_df.columns and 'avg_fill_price' in dutch_df.columns:
        market_prices = market_df['avg_fill_price'].dropna().tolist()
        dutch_prices = dutch_df['avg_fill_price'].dropna().tolist()
        plot_distribution_comparison(
            axes[0, 1], market_prices, dutch_prices,
            ['Market Orders', 'Dutch Orders'], 'Fill Price Distribution', 'Average Fill Price'
        )
    
    # Plot 3: Performance vs TWAP
    if 'fill_vs_twap' in market_df.columns and 'fill_vs_twap' in dutch_df.columns:
        market_vs_twap = market_df['fill_vs_twap'].dropna().tolist()
        dutch_vs_twap = dutch_df['fill_vs_twap'].dropna().tolist()
        plot_distribution_comparison(
            axes[0, 2], market_vs_twap, dutch_vs_twap,
            ['Market Orders', 'Dutch Orders'], 'Price Improvement vs TWAP', 'Fill Price vs TWAP'
        )
        axes[0, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add more plots based on available data...
    
    plt.tight_layout()
    return save_figure(output_filename)


def finalize_plot() -> None:
    """Clean up and close the current plot."""
    plt.close() 