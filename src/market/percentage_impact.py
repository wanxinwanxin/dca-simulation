"""Percentage-based market impact model with proper bid-ask spread handling."""

from dataclasses import dataclass

from src.core.orders import Order, Side


@dataclass(frozen=True)
class PercentageImpact:
    """Percentage-based market impact model.
    
    Impact scales with price level: impact = γ * qty * mid_price
    This makes economic sense since impact should be proportional to price level.
    
    For market orders: pays bid-ask spread + percentage impact
    For limit orders: only percentage impact (since they already cross spread)
    
    Market orders execution:
    - Buy: (mid + spread) + γ·qty·mid  (pay ask + impact)
    - Sell: (mid - spread) - γ·qty·mid (hit bid - impact)
    
    Limit orders execution: 
    - Buy: mid + γ·qty·mid  (crossed spread already)
    - Sell: mid - γ·qty·mid (crossed spread already)
    """
    
    spread: float  # half-spread (same as liquidity model)
    gamma: float   # percentage impact coefficient (e.g., 0.0001 = 0.01% per unit)
    
    def exec_price(self, order: Order, mid_px: float) -> float:
        """Calculate execution price including spread and percentage-based market impact."""
        # Impact is now proportional to price level
        impact = self.gamma * order.qty * mid_px
        
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
    
    def get_impact_breakdown(self, order: Order, mid_px: float) -> dict:
        """Get detailed breakdown of impact calculation for analysis."""
        impact_dollar = self.gamma * order.qty * mid_px
        impact_percent = self.gamma * order.qty * 100  # Convert to percentage
        
        return {
            "order_id": order.id,
            "order_qty": order.qty,
            "mid_price": mid_px,
            "gamma_percent": self.gamma * 100,  # Convert to percentage
            "impact_dollar": impact_dollar,
            "impact_percent": impact_percent,
            "spread": self.spread,
        } 