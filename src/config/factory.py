"""Component factory registry for YAML configuration."""

from typing import Any, Type

import numpy as np

from src.core.orders import Side
from src.cost.filler import FillerDecision
from src.cost.gas_model import Evm1559
from src.market.const_spread import ConstSpread
from src.market.gbm import GBM
from src.market.linear_impact import LinearImpact
from src.market.realistic_impact import RealisticImpact
from src.market.percentage_impact import PercentageImpact
from src.market.dutch_impact import DutchImpact
from src.metrics.fill_prices import FillPrices
from src.metrics.fill_timings import FillTimings
from src.strategy.true_dutch_limit import TrueDutchLimit  
from src.strategy.twap_market import TwapMarket


# Component registry mapping names to classes
REGISTRY: dict[str, Type[Any]] = {
    # Price processes
    "GBM": GBM,
    
    # Liquidity models
    "ConstSpread": ConstSpread,
    
    # Impact models
    "LinearImpact": LinearImpact,
    "RealisticImpact": RealisticImpact,
    "PercentageImpact": PercentageImpact,
    "DutchImpact": DutchImpact,
    
    # Gas models
    "Evm1559": Evm1559,
    
    # Strategies
    "TwapMarket": TwapMarket,
    "DutchLimit": TrueDutchLimit,  # Using correct implementation
    
    # Probes
    "FillTimings": FillTimings,
    "FillPrices": FillPrices,
}


def create_component(name: str, params: dict[str, Any], random_seed: int | None = None) -> Any:
    """Create component instance from name and parameters."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown component: {name}")
    
    component_class = REGISTRY[name]
    
    # Handle special parameters
    if "side" in params:
        params["side"] = Side(params["side"])
    
    # Add random state for components that need it
    if name == "GBM" and random_seed is not None:
        params["random_state"] = np.random.RandomState(random_seed)
    
    return component_class(**params)


def create_filler_decision() -> FillerDecision:
    """Create default filler decision."""
    return FillerDecision() 