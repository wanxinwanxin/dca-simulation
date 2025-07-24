"""Bridge utilities to interface with existing simulation components."""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.market.gbm import GBM


@dataclass
class PricePathResult:
    """Container for generated price path results."""
    
    # Path configuration
    volatility: float
    drift: float  
    initial_price: float
    dt: float
    horizon: float
    seed: int
    
    # Generated data
    times: List[float]
    prices: List[float]
    
    # Metadata
    path_id: str
    timestamp: str
    
    # Statistics
    mean_price: float
    std_price: float
    min_price: float
    max_price: float
    final_price: float
    total_return: float
    volatility_realized: float


class GBMPathGenerator:
    """Simplified interface for generating GBM price paths for the UI."""
    
    def __init__(self):
        self.generated_paths: Dict[str, PricePathResult] = {}
    
    def generate_paths(self, 
                      n_paths: int = 10,
                      volatility: float = 0.02,
                      drift: float = 0.0,  # Phase 1: Fixed to zero drift
                      initial_price: float = 100.0,  # Phase 1: Fixed S₀
                      dt: float = 1.0,  # Phase 1: Fixed time step
                      horizon: float = 200.0,  # Phase 1: Fixed horizon
                      base_seed: int = 42) -> List[PricePathResult]:
        """Generate multiple GBM price paths with given parameters."""
        
        paths = []
        times = [i * dt for i in range(int(horizon / dt) + 1)]
        
        for path_idx in range(n_paths):
            # Create unique seed for each path
            seed = base_seed + path_idx
            
            # Generate GBM path using existing implementation
            random_state = np.random.RandomState(seed)
            gbm = GBM(
                mu=drift,
                sigma=volatility,
                dt=dt,
                s0=initial_price,
                random_state=random_state
            )
            
            # Extract prices for our time horizon
            prices = []
            for t in times:
                prices.append(gbm.mid_price(t))
            
            # Calculate statistics
            prices_array = np.array(prices)
            mean_price = float(np.mean(prices_array))
            std_price = float(np.std(prices_array))
            min_price = float(np.min(prices_array))
            max_price = float(np.max(prices_array))
            final_price = float(prices[-1])
            total_return = (final_price / initial_price - 1.0) * 100  # Percentage
            
            # Realized volatility (annualized from daily returns)
            if len(prices) > 1:
                log_returns = np.diff(np.log(prices_array))
                realized_vol = float(np.std(log_returns) * np.sqrt(252))  # Annualized
            else:
                realized_vol = 0.0
            
            # Create result object
            path_result = PricePathResult(
                volatility=volatility,
                drift=drift,
                initial_price=initial_price,
                dt=dt,
                horizon=horizon,
                seed=seed,
                times=times,
                prices=prices,
                path_id=f"path_{path_idx:03d}_{seed}",
                timestamp=str(datetime.now()),
                mean_price=mean_price,
                std_price=std_price,
                min_price=min_price,
                max_price=max_price,
                final_price=final_price,
                total_return=total_return,
                volatility_realized=realized_vol
            )
            
            paths.append(path_result)
            
        return paths
    
    def save_paths_to_session(self, paths: List[PricePathResult], session_state) -> None:
        """Save generated paths to Streamlit session state."""
        if 'saved_paths' not in session_state:
            session_state.saved_paths = {}
        
        for path in paths:
            session_state.saved_paths[path.path_id] = path
    
    def export_paths_to_json(self, paths: List[PricePathResult], filename: str) -> Path:
        """Export paths to JSON file for later use."""
        # Convert paths to serializable format
        paths_data = []
        for path in paths:
            path_data = {
                'config': {
                    'volatility': path.volatility,
                    'drift': path.drift,
                    'initial_price': path.initial_price,
                    'dt': path.dt,
                    'horizon': path.horizon,
                    'seed': path.seed
                },
                'data': {
                    'times': path.times,
                    'prices': path.prices,
                    'path_id': path.path_id,
                    'timestamp': path.timestamp
                },
                'statistics': {
                    'mean_price': path.mean_price,
                    'std_price': path.std_price,
                    'min_price': path.min_price,
                    'max_price': path.max_price,
                    'final_price': path.final_price,
                    'total_return': path.total_return,
                    'volatility_realized': path.volatility_realized
                }
            }
            paths_data.append(path_data)
        
        # Ensure output directory exists
        output_dir = Path("results/streamlit_exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(paths_data, f, indent=2)
        
        return output_file
    
    def load_paths_from_json(self, filepath: Path) -> List[PricePathResult]:
        """Load paths from JSON file."""
        with open(filepath, 'r') as f:
            paths_data = json.load(f)
        
        paths = []
        for path_data in paths_data:
            config = path_data['config']
            data = path_data['data']
            stats = path_data['statistics']
            
            path = PricePathResult(
                volatility=config['volatility'],
                drift=config['drift'],
                initial_price=config['initial_price'],
                dt=config['dt'],
                horizon=config['horizon'],
                seed=config['seed'],
                times=data['times'],
                prices=data['prices'],
                path_id=data['path_id'],
                timestamp=data['timestamp'],
                mean_price=stats['mean_price'],
                std_price=stats['std_price'],
                min_price=stats['min_price'],
                max_price=stats['max_price'],
                final_price=stats['final_price'],
                total_return=stats['total_return'],
                volatility_realized=stats['volatility_realized']
            )
            paths.append(path)
        
        return paths


def calculate_ensemble_statistics(paths: List[PricePathResult]) -> Dict[str, Any]:
    """Calculate statistics across multiple price paths."""
    if not paths:
        return {}
    
    # Extract final prices and returns
    final_prices = [path.final_price for path in paths]
    total_returns = [path.total_return for path in paths]
    realized_vols = [path.volatility_realized for path in paths]
    
    # Calculate ensemble statistics
    ensemble_stats = {
        'n_paths': len(paths),
        'config': {
            'volatility': paths[0].volatility,
            'drift': paths[0].drift,
            'initial_price': paths[0].initial_price,
            'dt': paths[0].dt,
            'horizon': paths[0].horizon
        },
        'final_prices': {
            'mean': float(np.mean(final_prices)),
            'std': float(np.std(final_prices)),
            'min': float(np.min(final_prices)),
            'max': float(np.max(final_prices)),
            'percentiles': {
                '5': float(np.percentile(final_prices, 5)),
                '25': float(np.percentile(final_prices, 25)),
                '50': float(np.percentile(final_prices, 50)),
                '75': float(np.percentile(final_prices, 75)),
                '95': float(np.percentile(final_prices, 95))
            }
        },
        'total_returns': {
            'mean': float(np.mean(total_returns)),
            'std': float(np.std(total_returns)),
            'min': float(np.min(total_returns)),
            'max': float(np.max(total_returns))
        },
        'realized_volatility': {
            'mean': float(np.mean(realized_vols)),
            'std': float(np.std(realized_vols)),
            'target': paths[0].volatility
        }
    }
    
    return ensemble_stats 