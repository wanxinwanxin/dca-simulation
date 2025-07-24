"""Common simulation setup and execution utilities."""

import numpy as np
import simpy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar, Generic, List
from pathlib import Path

from src.market.gbm import GBM
from src.market.const_spread import ConstSpread
from src.market.realistic_impact import RealisticImpact
from src.market.percentage_impact import PercentageImpact
from src.market.dutch_impact import DutchImpact
from src.cost.gas_model import Evm1559
from src.cost.filler import FillerDecision
from src.engine.matching import MatchingEngine
from src.test_utils.engines import TrackingMatchingEngine


@dataclass
class SimulationConfig:
    """Configuration for common simulation parameters."""
    
    # Simulation timing
    horizon: float = 200.0
    dt: float = 1.0
    
    # Price process
    initial_price: float = 100.0
    drift: float = 0.01
    volatility: float = 0.02
    random_seed: int = 42
    
    # Market structure
    spread: float = 0.05
    impact_gamma: float = 0.001
    
    # Gas model
    base_fee: float = 2e-9
    tip: float = 1e-9


EngineType = TypeVar('EngineType', bound=MatchingEngine)


@dataclass
class SimulationRunner(Generic[EngineType]):
    """Generic simulation runner that handles common setup and execution."""
    
    config: SimulationConfig
    engine_class: Type[EngineType] = TrackingMatchingEngine
    
    def create_price_process(self, random_seed: Optional[int] = None) -> GBM:
        """Create a GBM price process with configuration parameters."""
        seed = random_seed if random_seed is not None else self.config.random_seed
        random_state = np.random.RandomState(seed)
        
        return GBM(
            mu=self.config.drift,
            sigma=self.config.volatility,
            dt=self.config.dt,
            s0=self.config.initial_price,
            random_state=random_state,
        )
    
    def create_market_models(self):
        """Create standard market models."""
        liquidity_model = ConstSpread(spread=self.config.spread)
        impact_model = PercentageImpact(
            spread=self.config.spread,
            gamma=self.config.impact_gamma
        )
        return liquidity_model, impact_model
    
    def create_cost_models(self):
        """Create standard cost models."""
        gas_model = Evm1559(
            base_fee=self.config.base_fee,
            tip=self.config.tip,
        )
        filler_decision = FillerDecision()
        return gas_model, filler_decision
    
    def create_engine(self, analyzer: Optional[Any] = None, **kwargs) -> EngineType:
        """Create the simulation engine with standard components."""
        price_process = self.create_price_process()
        liquidity_model, impact_model = self.create_market_models()
        gas_model, filler_decision = self.create_cost_models()
        
        # Set up probes list
        probes = []
        if analyzer:
            probes.append(analyzer)
        
        # Create engine - handle tracking engines differently
        from src.test_utils.engines import TrackingMatchingEngine
        if issubclass(self.engine_class, TrackingMatchingEngine):
            # For tracking engines, pass analyzer separately
            engine = self.engine_class(
                price_process=price_process,
                liquidity_model=liquidity_model,
                impact_model=impact_model,
                gas_model=gas_model,
                filler_decision=filler_decision,
                probes=probes,
                analyzer=analyzer,
                **kwargs
            )
        else:
            # For standard engines, use probes
            engine = self.engine_class(
                price_process=price_process,
                liquidity_model=liquidity_model,
                impact_model=impact_model,
                gas_model=gas_model,
                filler_decision=filler_decision,
                probes=probes,
                **kwargs
            )
        
        return engine
    
    def run_simulation(self, strategy: Any, analyzer: Optional[Any] = None, 
                      target_qty: float = 1000.0, **engine_kwargs) -> Dict[str, Any]:
        """Run a complete simulation with the given strategy."""
        engine = self.create_engine(analyzer=analyzer, **engine_kwargs)
        
        # Run simulation using MatchingEngine API
        results = engine.run(
            algo=strategy,
            target_qty=target_qty,
            horizon=self.config.horizon
        )
        
        # Return analyzer results if available
        if analyzer and hasattr(analyzer, 'final'):
            analyzer_results = analyzer.final()
            analyzer_results.update(results)
            return analyzer_results
        else:
            return results


@dataclass  
class MultiPathRunner:
    """Runner for multi-path simulations."""
    
    config: SimulationConfig
    n_paths: int = 100
    base_seed: int = 42
    
    def run_multi_path(self, strategy_factory, analyzer_factory, **engine_kwargs) -> List[Dict[str, Any]]:
        """Run multiple simulations with different random seeds."""
        results = []
        
        for i in range(self.n_paths):
            # Create fresh strategy and analyzer for each path
            strategy = strategy_factory()
            analyzer = analyzer_factory()
            
            # Use different seed for each path
            config = SimulationConfig(
                horizon=self.config.horizon,
                dt=self.config.dt,
                initial_price=self.config.initial_price,
                drift=self.config.drift,
                volatility=self.config.volatility,
                random_seed=self.base_seed + i,
                spread=self.config.spread,
                impact_gamma=self.config.impact_gamma,
                base_fee=self.config.base_fee,
                tip=self.config.tip,
            )
            
            runner = SimulationRunner(config=config)
            result = runner.run_simulation(strategy, analyzer, **engine_kwargs)
            result["path_id"] = i
            result["random_seed"] = self.base_seed + i
            results.append(result)
        
        return results


class ControlledPriceProcess:
    """Price process with predefined price schedule for testing."""
    
    def __init__(self, price_schedule: Dict[float, float]):
        """Initialize with time -> price mapping."""
        self.price_schedule = price_schedule
        self.times = sorted(price_schedule.keys())
    
    def mid_price(self, t: float) -> float:
        """Get price with linear interpolation between key points."""
        if t <= self.times[0]:
            return self.price_schedule[self.times[0]]
        elif t >= self.times[-1]:
            return self.price_schedule[self.times[-1]]
        
        # Linear interpolation
        for i in range(len(self.times) - 1):
            if self.times[i] <= t <= self.times[i + 1]:
                t1, t2 = self.times[i], self.times[i + 1]
                p1, p2 = self.price_schedule[t1], self.price_schedule[t2]
                
                if t2 == t1:  # Avoid division by zero
                    return p1
                
                weight = (t - t1) / (t2 - t1)
                price = p1 + weight * (p2 - p1)
                return price
        
        return self.price_schedule[self.times[-1]]


def create_default_config(**overrides) -> SimulationConfig:
    """Create a default simulation configuration with optional overrides."""
    defaults = {
        "horizon": 200.0,
        "dt": 1.0,
        "initial_price": 100.0,
        "drift": 0.01,
        "volatility": 0.02,
        "random_seed": 42,
        "spread": 0.05,
        "impact_gamma": 0.001,
        "base_fee": 2e-9,
        "tip": 1e-9,
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def ensure_output_dir(subdir: str = "order_debug") -> Path:
    """Ensure output directory exists and return path."""
    output_dir = Path("results") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir 