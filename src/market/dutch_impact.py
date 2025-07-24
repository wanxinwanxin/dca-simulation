"""Dutch-aware market impact model with configurable fill price split."""

from dataclasses import dataclass

from src.core.orders import Order, Side


@dataclass(frozen=True)
class DutchImpact:
    """Dutch-aware market impact model with configurable fill price split.
    
    This model allows controlling where Dutch orders fill between their limit price
    and the equivalent market order price through the dutch_price_split parameter.
    
    dutch_price_split interpretation:
    - 0.0: Dutch orders fill at limit price (no filler competition)
    - 1.0: Dutch orders fill at market order equivalent price (maximum filler competition)
    - 0.5: Dutch orders fill halfway between limit and market prices
    
    For market orders: pays bid-ask spread + linear impact
    For regular limit orders: uses limit price + impact adjustment
    For Dutch orders (detected by order ID): interpolates between limit and market prices
    """
    
    spread: float  # half-spread (same as liquidity model)
    gamma: float   # linear impact coefficient
    dutch_price_split: float = 0.0  # Default: fill at limit price
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        if not (0.0 <= self.dutch_price_split <= 1.0):
            raise ValueError(f"dutch_price_split must be between 0.0 and 1.0, got {self.dutch_price_split}")
    
    def _is_dutch_order(self, order: Order) -> bool:
        """Detect if this is a Dutch order based on ID pattern."""
        return (order.id.startswith("dutch_") or 
                order.id.startswith("true_dutch_") or
                "dutch" in order.id.lower())
    
    def exec_price(self, order: Order, mid_px: float) -> float:
        """Calculate execution price with Dutch order fill price control."""
        impact = self.gamma * order.qty
        
        if order.limit_px is None:
            # Market order: pays bid-ask spread + impact
            if order.side == Side.BUY:
                return mid_px + self.spread + impact
            else:
                return mid_px - self.spread - impact
        elif self._is_dutch_order(order):
            # Dutch order: interpolate between limit price and market order price
            market_order_price = self._market_order_price(order.side, mid_px, impact)
            limit_price = order.limit_px
            
            # Interpolate: split=0 -> limit_price, split=1 -> market_order_price
            fill_price = limit_price + self.dutch_price_split * (market_order_price - limit_price)
            return fill_price
        else:
            # Regular limit order: use limit price with impact adjustment
            if order.side == Side.BUY:
                return order.limit_px + impact
            else:
                return order.limit_px - impact
    
    def _market_order_price(self, side: Side, mid_px: float, impact: float) -> float:
        """Calculate what a market order would pay/receive."""
        if side == Side.BUY:
            return mid_px + self.spread + impact
        else:
            return mid_px - self.spread - impact
    
    def get_fill_price_breakdown(self, order: Order, mid_px: float) -> dict:
        """Get detailed breakdown of fill price calculation for analysis."""
        impact = self.gamma * order.qty
        
        result = {
            "order_id": order.id,
            "order_side": order.side.value,
            "is_dutch": self._is_dutch_order(order),
            "mid_price": mid_px,
            "limit_price": order.limit_px,
            "impact": impact,
            "spread": self.spread,
            "dutch_split": self.dutch_price_split,
        }
        
        if order.limit_px is None:
            # Market order
            result["fill_price"] = self._market_order_price(order.side, mid_px, impact)
            result["price_type"] = "market_order"
        elif self._is_dutch_order(order):
            # Dutch order
            market_price = self._market_order_price(order.side, mid_px, impact)
            result["market_equivalent_price"] = market_price
            result["fill_price"] = order.limit_px + self.dutch_price_split * (market_price - order.limit_px)
            result["price_type"] = "dutch_order"
            result["price_improvement_vs_limit"] = result["fill_price"] - order.limit_px
            result["price_improvement_vs_market"] = result["fill_price"] - market_price
        else:
            # Regular limit order
            if order.side == Side.BUY:
                result["fill_price"] = order.limit_px + impact
            else:
                result["fill_price"] = order.limit_px - impact
            result["price_type"] = "limit_order"
        
        return result 