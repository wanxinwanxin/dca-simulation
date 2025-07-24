"""Common analysis utilities for simulation results."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from src.core.events import Fill


@dataclass
class OrderAnalysisResult:
    """Standard result structure for order analysis."""
    
    # Basic metrics
    total_orders: int
    total_fills: int
    fill_rate: float
    
    # Price metrics (if any fills occurred)
    avg_fill_price: Optional[float] = None
    min_fill_price: Optional[float] = None
    max_fill_price: Optional[float] = None
    std_fill_price: Optional[float] = None
    
    # Quantity metrics
    total_qty_filled: float = 0.0
    avg_fill_qty: Optional[float] = None
    
    # Timing metrics
    avg_execution_delay: Optional[float] = None
    min_execution_delay: Optional[float] = None
    max_execution_delay: Optional[float] = None
    
    # Performance metrics (relative to benchmarks)
    fill_vs_twap: Optional[float] = None
    price_improvement_pct: Optional[float] = None
    
    # Additional context
    twap: Optional[float] = None
    vwap: Optional[float] = None


def calculate_twap(price_data: List[Tuple[float, float]]) -> float:
    """Calculate Time-Weighted Average Price from price path data."""
    if not price_data or len(price_data) < 2:
        return 0.0
    
    total_time_weighted = 0.0
    total_time = 0.0
    
    for i in range(len(price_data) - 1):
        t1, p1 = price_data[i]
        t2, _ = price_data[i + 1]
        dt = t2 - t1
        total_time_weighted += p1 * dt
        total_time += dt
    
    return total_time_weighted / total_time if total_time > 0 else 0.0


def calculate_vwap(fills: List[Fill]) -> float:
    """Calculate Volume-Weighted Average Price from fills."""
    if not fills:
        return 0.0
    
    total_value = sum(f.price * f.qty for f in fills)
    total_qty = sum(f.qty for f in fills)
    
    return total_value / total_qty if total_qty > 0 else 0.0


def analyze_order_performance(
    fills: List[Fill],
    order_creations: List[Tuple[float, str, float]],
    price_data: Optional[List[Tuple[float, float]]] = None,
    order_prefix: str = ""
) -> OrderAnalysisResult:
    """Analyze order performance with comprehensive metrics."""
    
    # Filter fills and creations by prefix if specified
    if order_prefix:
        fills = [f for f in fills if f.order_id.startswith(order_prefix)]
        order_creations = [oc for oc in order_creations if oc[1].startswith(order_prefix)]
    
    total_orders = len(order_creations)
    total_fills = len(fills)
    fill_rate = total_fills / total_orders if total_orders > 0 else 0.0
    
    # Initialize result with basic metrics
    result = OrderAnalysisResult(
        total_orders=total_orders,
        total_fills=total_fills,
        fill_rate=fill_rate,
    )
    
    if not fills:
        return result
    
    # Price metrics
    fill_prices = [f.price for f in fills]
    result.avg_fill_price = np.mean(fill_prices)
    result.min_fill_price = np.min(fill_prices)
    result.max_fill_price = np.max(fill_prices)
    result.std_fill_price = np.std(fill_prices)
    
    # Quantity metrics
    fill_qtys = [f.qty for f in fills]
    result.total_qty_filled = sum(fill_qtys)
    result.avg_fill_qty = np.mean(fill_qtys)
    
    # Timing metrics (execution delay from order creation to fill)
    execution_delays = []
    creation_times = {oc[1]: oc[0] for oc in order_creations}
    
    for fill in fills:
        if fill.order_id in creation_times:
            delay = fill.timestamp - creation_times[fill.order_id]
            execution_delays.append(delay)
    
    if execution_delays:
        result.avg_execution_delay = np.mean(execution_delays)
        result.min_execution_delay = np.min(execution_delays)
        result.max_execution_delay = np.max(execution_delays)
    
    # Calculate benchmarks if price data available
    if price_data:
        result.twap = calculate_twap(price_data)
        result.vwap = calculate_vwap(fills)
        
        if result.avg_fill_price and result.twap:
            result.fill_vs_twap = result.avg_fill_price - result.twap
            result.price_improvement_pct = (result.avg_fill_price - result.twap) / result.twap * 100
    
    return result


def compare_strategies(
    results1: List[OrderAnalysisResult],
    results2: List[OrderAnalysisResult],
    strategy_names: Tuple[str, str] = ("Strategy 1", "Strategy 2")
) -> Dict[str, Any]:
    """Compare two sets of strategy results statistically."""
    
    comparison = {
        "strategy_names": strategy_names,
        "n_paths": (len(results1), len(results2)),
    }
    
    # Define metrics to compare
    metrics = [
        "fill_rate", "avg_fill_price", "fill_vs_twap", 
        "avg_execution_delay", "price_improvement_pct"
    ]
    
    for metric in metrics:
        values1 = [getattr(r, metric) for r in results1 if getattr(r, metric) is not None]
        values2 = [getattr(r, metric) for r in results2 if getattr(r, metric) is not None]
        
        if values1 and values2:
            comparison[f"{metric}_mean"] = (np.mean(values1), np.mean(values2))
            comparison[f"{metric}_std"] = (np.std(values1), np.std(values2))
            comparison[f"{metric}_diff"] = np.mean(values2) - np.mean(values1)
            
            # Simple t-test approximation (if both have reasonable sample size)
            if len(values1) >= 10 and len(values2) >= 10:
                try:
                    from scipy import stats
                    t_stat, p_val = stats.ttest_ind(values1, values2)
                    comparison[f"{metric}_ttest"] = {"t_stat": t_stat, "p_value": p_val}
                except ImportError:
                    pass  # scipy not available
    
    return comparison


def calculate_multi_path_summary(results: List[Dict[str, Any]], order_prefix: str = "") -> Dict[str, Any]:
    """Calculate summary statistics from multi-path simulation results."""
    
    # Extract relevant data from results
    all_fill_rates = []
    all_avg_prices = []
    all_vs_twap = []
    all_delays = []
    
    for result in results:
        fills = result.get("fills", [])
        order_creations = result.get("order_creations", [])
        price_path = result.get("price_path", [])
        
        analysis = analyze_order_performance(fills, order_creations, price_path, order_prefix)
        
        all_fill_rates.append(analysis.fill_rate)
        if analysis.avg_fill_price is not None:
            all_avg_prices.append(analysis.avg_fill_price)
        if analysis.fill_vs_twap is not None:
            all_vs_twap.append(analysis.fill_vs_twap)
        if analysis.avg_execution_delay is not None:
            all_delays.append(analysis.avg_execution_delay)
    
    summary = {
        "n_paths": len(results),
        "fill_rate": {
            "mean": np.mean(all_fill_rates),
            "std": np.std(all_fill_rates),
            "min": np.min(all_fill_rates),
            "max": np.max(all_fill_rates),
        },
    }
    
    if all_avg_prices:
        summary["avg_fill_price"] = {
            "mean": np.mean(all_avg_prices),
            "std": np.std(all_avg_prices),
            "min": np.min(all_avg_prices),
            "max": np.max(all_avg_prices),
        }
    
    if all_vs_twap:
        summary["fill_vs_twap"] = {
            "mean": np.mean(all_vs_twap),
            "std": np.std(all_vs_twap),
            "min": np.min(all_vs_twap),
            "max": np.max(all_vs_twap),
        }
    
    if all_delays:
        summary["avg_execution_delay"] = {
            "mean": np.mean(all_delays),
            "std": np.std(all_delays),
            "min": np.min(all_delays),
            "max": np.max(all_delays),
        }
    
    return summary


def create_results_dataframe(results: List[Dict[str, Any]], order_prefix: str = "") -> pd.DataFrame:
    """Convert multi-path results to a pandas DataFrame for analysis."""
    
    rows = []
    for i, result in enumerate(results):
        fills = result.get("fills", [])
        order_creations = result.get("order_creations", [])
        price_path = result.get("price_path", [])
        
        analysis = analyze_order_performance(fills, order_creations, price_path, order_prefix)
        
        row = {
            "path_id": i,
            "random_seed": result.get("random_seed", i),
            "total_orders": analysis.total_orders,
            "total_fills": analysis.total_fills,
            "fill_rate": analysis.fill_rate,
            "total_qty_filled": analysis.total_qty_filled,
        }
        
        # Add optional metrics
        optional_metrics = [
            "avg_fill_price", "min_fill_price", "max_fill_price", "std_fill_price",
            "avg_fill_qty", "avg_execution_delay", "min_execution_delay", "max_execution_delay",
            "fill_vs_twap", "price_improvement_pct", "twap", "vwap"
        ]
        
        for metric in optional_metrics:
            value = getattr(analysis, metric)
            row[metric] = value if value is not None else np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_analysis_summary(analysis: OrderAnalysisResult, title: str = "Order Analysis") -> None:
    """Print a formatted summary of order analysis results."""
    
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Total Orders: {analysis.total_orders}")
    print(f"Total Fills: {analysis.total_fills}")
    print(f"Fill Rate: {analysis.fill_rate:.1%}")
    
    if analysis.total_fills > 0:
        print(f"Total Quantity Filled: {analysis.total_qty_filled:.1f}")
        print(f"Average Fill Price: ${analysis.avg_fill_price:.3f}")
        print(f"Price Range: ${analysis.min_fill_price:.3f} - ${analysis.max_fill_price:.3f}")
        
        if analysis.avg_execution_delay is not None:
            print(f"Average Execution Delay: {analysis.avg_execution_delay:.1f} blocks")
        
        if analysis.twap is not None:
            print(f"TWAP: ${analysis.twap:.3f}")
            
        if analysis.fill_vs_twap is not None:
            print(f"Fill vs TWAP: ${analysis.fill_vs_twap:+.3f}")
            
        if analysis.price_improvement_pct is not None:
            print(f"Price Improvement: {analysis.price_improvement_pct:+.2f}%")
    else:
        print("No fills occurred")


def calculate_order_size_impact(
    small_results: List[Dict[str, Any]], 
    large_results: List[Dict[str, Any]],
    order_prefix: str = ""
) -> Dict[str, Any]:
    """Analyze how order size affects performance metrics."""
    
    small_df = create_results_dataframe(small_results, order_prefix)
    large_df = create_results_dataframe(large_results, order_prefix)
    
    # Calculate means for comparison
    small_means = small_df.select_dtypes(include=[np.number]).mean()
    large_means = large_df.select_dtypes(include=[np.number]).mean()
    
    impact = {}
    for metric in small_means.index:
        if not pd.isna(small_means[metric]) and not pd.isna(large_means[metric]):
            impact[f"{metric}_small"] = small_means[metric]
            impact[f"{metric}_large"] = large_means[metric]
            impact[f"{metric}_difference"] = large_means[metric] - small_means[metric]
            
            # Calculate percentage change if base value is non-zero
            if abs(small_means[metric]) > 1e-10:
                impact[f"{metric}_pct_change"] = (large_means[metric] - small_means[metric]) / small_means[metric] * 100
    
    return impact 