"""CLI entry point for running DCA simulations."""

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from src.config.factory import create_component, create_filler_decision
from src.engine.matching import MatchingEngine


def run_single_simulation(
    config: dict[str, Any], 
    strategy_config: dict[str, Any],
    run_id: int
) -> dict[str, Any]:
    """Run a single simulation path."""
    # Create random seed for this run
    base_seed = config["random_seed"]
    run_seed = base_seed + run_id
    
    # Create market components
    price_process = create_component(
        config["price_process"]["name"],
        config["price_process"]["params"],
        random_seed=run_seed,
    )
    
    liquidity_model = create_component(
        config["liquidity"]["name"],
        config["liquidity"]["params"],
    )
    
    impact_model = create_component(
        config["impact"]["name"],
        config["impact"]["params"],
    )
    
    gas_model = create_component(
        config["gas"]["name"],
        config["gas"]["params"],
    )
    
    filler_decision = create_filler_decision()
    
    # Create probes
    probes = []
    for probe_name in config["probes"]:
        probe = create_component(probe_name, {})
        probes.append(probe)
    
    # Create strategy
    strategy = create_component(
        strategy_config["algo"],
        strategy_config["params"],
        random_seed=run_seed,
    )
    
    # Create engine
    engine = MatchingEngine(
        price_process=price_process,
        liquidity_model=liquidity_model,
        impact_model=impact_model,
        gas_model=gas_model,
        filler_decision=filler_decision,
        probes=probes,
        time_step=1.0,
    )
    
    # Run simulation
    results = engine.run(
        algo=strategy,
        target_qty=config["target_qty"],
        horizon=config["horizon"],
    )
    
    # Add metadata
    results["run_id"] = run_id
    results["strategy"] = strategy_config["algo"]
    results["seed"] = run_seed
    
    return results


def run_experiment(config_path: str) -> None:
    """Run full experiment from configuration file."""
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    run_name = config["run_name"]
    paths = config["paths"]
    
    # Create results directory
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running experiment: {run_name}")
    print(f"Configuration: {config_path}")
    print(f"Simulation paths: {paths}")
    
    # Run simulations for each strategy
    for strategy_idx, strategy_config in enumerate(config["strategies"]):
        strategy_name = strategy_config["algo"]
        print(f"\nRunning strategy: {strategy_name}")
        
        strategy_results = []
        
        if paths == 1:
            # Single-threaded
            result = run_single_simulation(config, strategy_config, 0)
            strategy_results.append(result)
        else:
            # Multi-threaded
            with multiprocessing.Pool() as pool:
                tasks = [
                    (config, strategy_config, run_id) 
                    for run_id in range(paths)
                ]
                strategy_results = pool.starmap(run_single_simulation, tasks)
        
        # Save individual run results
        for result in strategy_results:
            run_file = results_dir / f"{strategy_name}_run_{result['run_id']}.json"
            with open(run_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        
        # Aggregate results
        if strategy_results:
            # Extract probe results for aggregation
            probe_keys = [
                key for key in strategy_results[0].keys() 
                if key.startswith("probe_")
            ]
            
            aggregated = {
                "strategy": strategy_name,
                "total_runs": len(strategy_results),
            }
            
            for probe_key in probe_keys:
                probe_data = [result[probe_key] for result in strategy_results]
                
                # Aggregate numeric metrics
                numeric_keys = [
                    k for k, v in probe_data[0].items() 
                    if isinstance(v, (int, float)) and v is not None
                ]
                
                for metric_key in numeric_keys:
                    values = [
                        data[metric_key] for data in probe_data 
                        if data[metric_key] is not None
                    ]
                    if values:
                        aggregated[f"{probe_key}_{metric_key}_mean"] = sum(values) / len(values)
                        aggregated[f"{probe_key}_{metric_key}_std"] = (
                            sum((x - aggregated[f"{probe_key}_{metric_key}_mean"])**2 for x in values) / len(values)
                        ) ** 0.5 if len(values) > 1 else 0.0
            
            # Save aggregated results
            summary_file = results_dir / f"{strategy_name}_summary.json"
            with open(summary_file, "w") as f:
                json.dump(aggregated, f, indent=2, default=str)
            
            print(f"Results saved to {results_dir}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run DCA execution algorithm simulation")
    parser.add_argument("config", help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        sys.exit(1)
    
    try:
        run_experiment(args.config)
        print("\nSimulation completed successfully!")
    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 