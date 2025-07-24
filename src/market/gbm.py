"""Geometric Brownian Motion price process."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian Motion price process.
    
    dS = μ * S * dt + σ * S * dW
    """
    
    mu: float  # drift coefficient
    sigma: float  # volatility
    dt: float  # time step
    s0: float  # initial price
    random_state: np.random.RandomState
    
    def __post_init__(self) -> None:
        """Generate price path on initialization."""
        # Pre-generate a reasonable number of price points
        # In practice, this would be done more dynamically
        max_steps = int(86400 / self.dt)  # 1 day worth of steps
        
        randn = self.random_state.randn(max_steps)
        
        # Store pre-computed values for efficiency
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt)
        
        log_returns = drift + diffusion * randn
        
        # Compute cumulative log prices
        log_prices = np.cumsum(log_returns)
        
        # Convert to absolute prices
        prices = self.s0 * np.exp(log_prices)
        
        # Store using object.__setattr__ since dataclass is frozen
        object.__setattr__(self, '_prices', prices)
        object.__setattr__(self, '_max_time', max_steps * self.dt)
    
    def mid_price(self, t: float) -> float:
        """Get mid price at time t."""
        if t <= 0:
            return self.s0
            
        if t >= self._max_time:
            # Extend price path if needed
            return self._prices[-1]
            
        # Find closest time step
        step = int(t / self.dt)
        
        if step >= len(self._prices):
            return self._prices[-1]
            
        return float(self._prices[step]) 