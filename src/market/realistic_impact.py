"""Realistic market impact model with proper bid-ask spread handling."""

from dataclasses import dataclass

from src.core.orders import Order, Side


@dataclass(frozen=True)
class RealisticImpact:
    """Realistic market impact model.
    
    For market orders: pays bid-ask spread + linear impact
    For limit orders: only linear impact (since they already cross spread)
    
    Market orders execution:
    - Buy: (mid + spread) + γ·qty  (pay ask + impact)
    - Sell: (mid - spread) - γ·qty (hit bid - impact)
    
    Limit orders execution: 
    - Buy: mid + γ·qty  (crossed spread already)
    - Sell: mid - γ·qty (crossed spread already)
    """
    
    spread: float  # half-spread (same as liquidity model)
    gamma: float   # linear impact coefficient
    
    def exec_price(self, order: Order, mid_px: float) -> float:
        """Calculate execution price including spread and market impact."""
        impact = self.gamma * order.qty
        
        if order.limit_px is None:
            # Market order: pays bid-ask spread + impact
            if order.side == Side.BUY:
                # Buy market order: pay ask + impact
                return mid_px + self.spread + impact
            else:
                # Sell market order: hit bid - impact  
                return mid_px - self.spread - impact
        else:
            # Limit order: already crossed spread, just apply impact
            if order.side == Side.BUY:
                # Buy limit: price improvement from limit + impact
                return mid_px + impact
            else:
                # Sell limit: price improvement from limit - impact
                return mid_px - impact 