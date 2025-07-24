"""Filler decision logic."""

from dataclasses import dataclass

from src.core.orders import Side


@dataclass(frozen=True)
class FillerDecision:
    """Determine if a fill is profitable after gas costs."""
    
    def should_fill(
        self, 
        side: Side, 
        limit_px: float, 
        mid_px: float,
        qty: float, 
        gas_fee: float
    ) -> bool:
        """Return True if profit ≥ gas fee."""
        if side == Side.SELL:
            # Seller profits when mid > limit (buy low, sell high)
            pnl = (mid_px - limit_px) * qty
        else:
            # Buyer profits when limit > mid (sell high, buy low)  
            pnl = (limit_px - mid_px) * qty
            
        return pnl >= gas_fee 