"""Market layer protocols for price, liquidity, and impact models."""

from typing import Protocol

from src.core.orders import Order, Side


class PriceProcess(Protocol):
    """Protocol for price evolution models."""
    
    def mid_price(self, t: float) -> float:
        """Get mid price at time t."""
        ...


class LiquidityModel(Protocol):
    """Protocol for liquidity availability models."""
    
    def crossed(self, side: Side, limit_px: float, mid_px: float) -> bool:
        """Check if order would be filled given current market conditions."""
        ...


class ImpactModel(Protocol):
    """Protocol for market impact models."""
    
    def exec_price(self, order: Order, mid_px: float) -> float:
        """Calculate execution price including market impact."""
        ... 