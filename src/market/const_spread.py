"""Constant spread liquidity model."""

from dataclasses import dataclass

from src.core.orders import Side


@dataclass(frozen=True)
class ConstSpread:
    """Constant half-spread liquidity model.
    
    Fill if abs(mid - limit) >= spread.
    """
    
    spread: float  # half-spread
    
    def crossed(self, side: Side, limit_px: float, mid_px: float) -> bool:
        """Check if order would be filled given current market conditions."""
        if side == Side.BUY:
            # Buy order fills if limit price >= mid + spread
            return limit_px >= mid_px + self.spread
        else:
            # Sell order fills if limit price <= mid - spread  
            return limit_px <= mid_px - self.spread 