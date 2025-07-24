"""Multi-path simulation framework utilities."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass

from .simulation import SimulationConfig, MultiPathRunner
from .analysis import (
    analyze_order_performance, create_results_dataframe, 
    calculate_multi_path_summary, print_analysis_summary,
    compare_strategies, OrderAnalysisResult
)
from .visualization import create_multi_path_visualization, finalize_plot


@dataclass
class MultiPathConfig:
    """Configuration for multi-path simulation experiments."""
    
    # Simulation parameters
    simulation_config: SimulationConfig
    n_paths: int = 100
    base_seed: int = 42
    
    # Output settings
    output_dir: str = "results/order_debug"
    save_csv: bool = True
    save_plots: bool = True
    
    # Analysis settings
    order_prefixes: List[str] = None  # If None, will analyze all orders
    comparison_pairs: List[tuple] = None  # Pairs of prefixes to compare


class MultiPathExperiment:
    """Framework for running multi-path experiments with standardized analysis."""
    
    def __init__(self, config: MultiPathConfig):
        self.config = config
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.analysis_results: Dict[str, List[OrderAnalysisResult]] = {}
        
    def run_strategy(self, 
                    strategy_factory: Callable[[], Any],
                    analyzer_factory: Callable[[], Any],
                    strategy_name: str,
                    **engine_kwargs) -> List[Dict[str, Any]]:
        """Run a strategy across multiple paths."""
        
        print(f"🚀 Running {strategy_name} across {self.config.n_paths} paths...")
        
        runner = MultiPathRunner(
            config=self.config.simulation_config,
            n_paths=self.config.n_paths,
            base_seed=self.config.base_seed
        )
        
        results = runner.run_multi_path(strategy_factory, analyzer_factory, **engine_kwargs)
        self.results[strategy_name] = results
        
        # Analyze results for each order prefix
        if self.config.order_prefixes:
            strategy_analysis = {}
            for prefix in self.config.order_prefixes:
                prefix_results = []
                for result in results:
                    analysis = analyze_order_performance(
                        fills=result.get("fills", []),
                        order_creations=result.get("order_creations", []),
                        price_data=result.get("price_path", []),
                        order_prefix=prefix
                    )
                    prefix_results.append(analysis)
                strategy_analysis[prefix] = prefix_results
            self.analysis_results[strategy_name] = strategy_analysis
        
        print(f"✅ {strategy_name} completed!")
        return results
    
    def save_results(self, strategy_name: str) -> None:
        """Save results to CSV files."""
        if not self.config.save_csv or strategy_name not in self.results:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.results[strategy_name]
        
        if self.config.order_prefixes:
            # Save separate CSV for each order type
            for prefix in self.config.order_prefixes:
                df = create_results_dataframe(results, order_prefix=prefix)
                filename = f"{strategy_name}_{prefix}_results.csv"
                df.to_csv(output_dir / filename, index=False)
                print(f"📊 Saved {prefix} results to: {output_dir / filename}")
        else:
            # Save all results
            df = create_results_dataframe(results)
            filename = f"{strategy_name}_results.csv"
            df.to_csv(output_dir / filename, index=False)
            print(f"📊 Saved results to: {output_dir / filename}")
    
    def print_summary(self, strategy_name: str) -> None:
        """Print summary statistics for a strategy."""
        if strategy_name not in self.results:
            print(f"No results found for strategy: {strategy_name}")
            return
        
        results = self.results[strategy_name]
        
        if self.config.order_prefixes:
            for prefix in self.config.order_prefixes:
                summary = calculate_multi_path_summary(results, order_prefix=prefix)
                print(f"\n📈 {strategy_name} - {prefix} Orders Summary:")
                print(f"Analyzed {summary['n_paths']} paths")
                
                for metric, stats in summary.items():
                    if metric != 'n_paths' and isinstance(stats, dict):
                        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                              f"(range: {stats['min']:.4f} - {stats['max']:.4f})")
        else:
            summary = calculate_multi_path_summary(results)
            print(f"\n📈 {strategy_name} Summary:")
            print(f"Analyzed {summary['n_paths']} paths")
            
            for metric, stats in summary.items():
                if metric != 'n_paths' and isinstance(stats, dict):
                    print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"(range: {stats['min']:.4f} - {stats['max']:.4f})")
    
    def compare_strategies(self, strategy1: str, strategy2: str, order_prefix: str = "") -> None:
        """Compare two strategies statistically."""
        if strategy1 not in self.analysis_results or strategy2 not in self.analysis_results:
            print(f"Cannot compare: missing analysis results for {strategy1} or {strategy2}")
            return
        
        if self.config.order_prefixes and order_prefix:
            results1 = self.analysis_results[strategy1][order_prefix]
            results2 = self.analysis_results[strategy2][order_prefix]
            title = f"{strategy1} vs {strategy2} ({order_prefix} orders)"
        else:
            # Use first available prefix or all results
            key = list(self.analysis_results[strategy1].keys())[0] if self.config.order_prefixes else ""
            results1 = self.analysis_results[strategy1][key] if key else []
            results2 = self.analysis_results[strategy2][key] if key else []
            title = f"{strategy1} vs {strategy2}"
        
        if not results1 or not results2:
            print(f"No results to compare for {title}")
            return
        
        comparison = compare_strategies(results1, results2, (strategy1, strategy2))
        
        print(f"\n🔍 {title}")
        print("=" * len(title))
        
        for metric, data in comparison.items():
            if metric.endswith("_mean"):
                base_metric = metric.replace("_mean", "")
                mean1, mean2 = data
                diff = comparison.get(f"{base_metric}_diff", 0)
                print(f"{base_metric}: {mean1:.4f} vs {mean2:.4f} (diff: {diff:+.4f})")
                
                # Add statistical significance if available
                ttest_key = f"{base_metric}_ttest"
                if ttest_key in comparison:
                    p_val = comparison[ttest_key]["p_value"]
                    significance = " *" if p_val < 0.05 else ""
                    print(f"  p-value: {p_val:.4f}{significance}")
    
    def create_visualizations(self) -> List[Path]:
        """Create comparison visualizations for all strategy pairs."""
        if not self.config.save_plots:
            return []
        
        output_files = []
        
        # Create pairwise comparisons if specified
        if self.config.comparison_pairs:
            for strategy1, strategy2 in self.config.comparison_pairs:
                if strategy1 in self.results and strategy2 in self.results:
                    filename = f"{strategy1}_vs_{strategy2}_comparison.png"
                    
                    try:
                        output_file = create_multi_path_visualization(
                            market_results=self.results[strategy1],
                            dutch_results=self.results[strategy2],
                            output_filename=filename
                        )
                        output_files.append(output_file)
                        print(f"📈 Created comparison visualization: {output_file}")
                    except Exception as e:
                        print(f"Failed to create visualization for {strategy1} vs {strategy2}: {e}")
                    finally:
                        finalize_plot()
        
        return output_files
    
    def run_complete_experiment(self, experiments: List[Dict[str, Any]]) -> None:
        """Run a complete multi-strategy experiment with analysis and visualization."""
        
        print(f"🎯 Starting Multi-Path Experiment")
        print(f"Configuration: {self.config.n_paths} paths, seed {self.config.base_seed}")
        print(f"Order types: {self.config.order_prefixes or 'all'}")
        print()
        
        # Run all strategies
        for experiment in experiments:
            strategy_name = experiment["name"]
            strategy_factory = experiment["strategy_factory"]
            analyzer_factory = experiment["analyzer_factory"]
            engine_kwargs = experiment.get("engine_kwargs", {})
            
            self.run_strategy(strategy_factory, analyzer_factory, strategy_name, **engine_kwargs)
            self.save_results(strategy_name)
            self.print_summary(strategy_name)
        
        # Run comparisons
        if self.config.comparison_pairs:
            print("\n🔍 Strategy Comparisons:")
            for strategy1, strategy2 in self.config.comparison_pairs:
                if self.config.order_prefixes:
                    for prefix in self.config.order_prefixes:
                        self.compare_strategies(strategy1, strategy2, prefix)
                else:
                    self.compare_strategies(strategy1, strategy2)
        
        # Create visualizations
        output_files = self.create_visualizations()
        
        print(f"\n✅ Experiment completed!")
        print(f"📁 Results saved to: {self.config.output_dir}")
        if output_files:
            print(f"🎨 Visualizations: {len(output_files)} files created")


def create_standard_experiment_config(**overrides) -> MultiPathConfig:
    """Create a standard multi-path experiment configuration."""
    defaults = {
        "simulation_config": SimulationConfig(),
        "n_paths": 100,
        "base_seed": 42,
        "output_dir": "results/order_debug",
        "save_csv": True,
        "save_plots": True,
        "order_prefixes": ["market", "dutch"],
        "comparison_pairs": [("market_strategy", "dutch_strategy")],
    }
    defaults.update(overrides)
    
    return MultiPathConfig(**defaults) 