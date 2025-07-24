"""Dutch-aware matching engine that properly handles dynamic Dutch order limits."""

from typing import Any, Dict
from dataclasses import dataclass

from src.engine.matching import MatchingEngine
from src.core.events import Fill
from src.strategy.true_dutch_limit import TrueDutchLimit


@dataclass
class DutchAwareMatchingEngine(MatchingEngine):
    """Matching engine that properly handles Dutch order dynamic limit prices.
    
    Key enhancement: For Dutch orders, checks the theoretical current limit price
    at the time of fill evaluation, not the static limit price from when the order
    was last placed/updated.
    """
    
    def __init__(self, *args, dutch_strategy: TrueDutchLimit = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dutch_strategy = dutch_strategy
    
    def _is_dutch_order(self, order_id: str) -> bool:
        """Check if this is a Dutch order based on ID pattern."""
        return (order_id.startswith("dutch_") or 
                order_id.startswith("true_dutch_") or
                "dutch" in order_id.lower())
    
    def _get_current_dutch_limit(self, order_id: str, current_time: float) -> float | None:
        """Get the theoretical current limit price for a Dutch order."""
        if not self.dutch_strategy or not self._is_dutch_order(order_id):
            return None
        
        return self.dutch_strategy.get_current_limit_price(current_time)
    
    def _check_fills(self, current_time: float, mid_price: float) -> None:
        """Enhanced fill checking that uses dynamic Dutch order limits."""
        filled_orders = []
        
        for order_id, order in self.open_orders.items():
            # Check if order is expired
            if current_time > order.valid_to:
                filled_orders.append(order_id)
                continue
            
            # For Dutch orders, use theoretical current limit price
            limit_price_to_check = order.limit_px
            
            if self._is_dutch_order(order_id) and self.dutch_strategy:
                theoretical_limit = self._get_current_dutch_limit(order_id, current_time)
                if theoretical_limit is not None:
                    limit_price_to_check = theoretical_limit
                    print(f"  Dutch order {order_id}: static_limit={order.limit_px:.3f}, theoretical_limit={theoretical_limit:.3f}")
                else:
                    # Dutch order expired based on theoretical calculation
                    print(f"  Dutch order {order_id}: theoretical limit expired")
                    filled_orders.append(order_id)
                    continue
            
            # Check if order crosses the spread (for limit orders)
            if limit_price_to_check is not None:
                crossed = self.liquidity_model.crossed(order.side, limit_price_to_check, mid_price)
                print(f"  Order {order_id}: limit={limit_price_to_check:.3f}, crossed={crossed}")
                
                if not crossed:
                    continue
            
            # Calculate execution price (still use original order for impact model)
            exec_price = self.impact_model.exec_price(order, mid_price)
            
            # Calculate gas cost
            gas_cost = self.gas_model.gas_fee(self.gas_per_fill)
            
            # Check if filler would execute (use theoretical limit for decision)
            if limit_price_to_check is not None:
                should_fill = self.filler_decision.should_fill(
                    order.side, limit_price_to_check, mid_price, order.qty, gas_cost
                )
                if not should_fill:
                    continue
            
            print(f"  FILLING {order_id} at theoretical_limit={limit_price_to_check:.3f}")
            
            # Execute the fill
            fill = Fill(
                order_id=order.id,
                timestamp=current_time,
                qty=order.qty,
                price=exec_price,
                gas_paid=gas_cost,
            )
            
            # Update state
            self.total_filled_qty += order.qty
            filled_orders.append(order_id)
            
            # Notify probes
            for probe in self.probes:
                probe.on_fill(fill)
        
        # Remove filled orders
        for order_id in filled_orders:
            if order_id in self.open_orders:
                del self.open_orders[order_id] 