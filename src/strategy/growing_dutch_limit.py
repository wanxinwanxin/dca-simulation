"""Growing Self-Adjusting Dutch Limit Order Strategy."""

from dataclasses import dataclass
from typing import List

from src.core.orders import Side
from src.core.events import Fill
from src.strategy.protocols import (
    BrokerState,
    InstructionType,
    MarketSnapshot,
    OrderInstruction,
)


@dataclass(frozen=True)
class GrowingSelfAdjustingDutchLimit:
    """Growing Self-Adjusting Dutch Limit Order Strategy.
    
    Features:
    1. Size grows over time (configurable growth rate)
    2. Limit price decays over time (like regular Dutch orders)
    3. When filled, limit price adjusts up by a percentage FROM CURRENT PRICE
    4. Continues to decay from the adjusted current price (not a separate base)
    5. Expires by time limit or when total filled exceeds threshold
    """
    
    # Size parameters
    initial_size: float
    size_growth_rate: float  # units per second
    max_size: float
    
    # Price parameters
    side: Side
    starting_limit_price: float  # p_0
    decay_rate: float  # price change per unit time
    adjustment_percentage: float  # percentage to adjust up after fills
    
    # Expiry parameters
    order_duration: float  # time limit (seconds)
    max_total_filled: float  # quantity limit
    
    def __post_init__(self) -> None:
        """Initialize internal state."""
        object.__setattr__(self, '_current_order_id', None)
        object.__setattr__(self, '_order_start_time', None)
        object.__setattr__(self, '_orders_completed', 0)
        object.__setattr__(self, '_total_filled_qty', 0.0)
        object.__setattr__(self, '_last_adjustment_time', None)  # When last price adjustment happened
        object.__setattr__(self, '_last_adjustment_price', self.starting_limit_price)  # Price at last adjustment
        object.__setattr__(self, '_price_adjustments', 0)
        object.__setattr__(self, '_last_fill_check_time', None)
        object.__setattr__(self, '_last_limit_price', None)

    def _calculate_current_size(self, clock: float) -> float:
        """Calculate current fillable order size."""
        if self._order_start_time is None:
            return self.initial_size
        
        time_elapsed = clock - self._order_start_time
        
        # Calculate grown size
        grown_size = self.initial_size + self.size_growth_rate * time_elapsed
        
        # Cap at max_size
        grown_size = min(grown_size, self.max_size)
        
        # Subtract what's already been filled
        current_size = grown_size - self._total_filled_qty
        
        return max(0.0, current_size)
    
    def _calculate_current_limit_price(self, clock: float) -> float:
        """Calculate current limit price with decay from last adjustment."""
        if self._order_start_time is None:
            return self.starting_limit_price
        
        # Time since last price adjustment (or order start if no adjustments)
        reference_time = self._last_adjustment_time or self._order_start_time
        time_since_adjustment = clock - reference_time
        
        # Start from the price at last adjustment and apply decay
        if self.side == Side.SELL:
            # For sell orders: decay down from last adjustment price
            current_price = self._last_adjustment_price - self.decay_rate * time_since_adjustment
        else:
            # For buy orders: decay up from last adjustment price
            current_price = self._last_adjustment_price + self.decay_rate * time_since_adjustment
        
        return current_price
    
    def _check_for_new_fills(self, broker_state: BrokerState) -> List[Fill]:
        """Check if there have been new fills since last check."""
        new_fills = []
        
        if self._current_order_id and self._current_order_id not in broker_state.open_orders:
            # Order was filled or cancelled - assume filled for now
            # In a real implementation, we'd track fills more precisely
            if self._last_fill_check_time is not None:
                # Create a mock fill for the missing quantity
                # This is simplified - in practice we'd get fill events
                current_size = self._calculate_current_size(self._last_fill_check_time)
                if current_size > 0:
                    fill = Fill(
                        order_id=self._current_order_id,
                        timestamp=self._last_fill_check_time,
                        qty=current_size,
                        price=self._calculate_current_limit_price(self._last_fill_check_time),
                        gas_paid=0.0
                    )
                    new_fills.append(fill)
        
        return new_fills
    
    def _adjust_price_after_fill(self, fill: Fill) -> None:
        """Adjust limit price up after a fill."""
        if self.side == Side.SELL:
            # For sell: adjust price up
            adjustment = self._calculate_current_limit_price(fill.timestamp) * self.adjustment_percentage
            new_limit = self._calculate_current_limit_price(fill.timestamp) + adjustment
        else:
            # For buy: adjust price down
            adjustment = self._calculate_current_limit_price(fill.timestamp) * self.adjustment_percentage
            new_limit = self._calculate_current_limit_price(fill.timestamp) - adjustment
        
        object.__setattr__(self, '_last_adjustment_price', new_limit)
        object.__setattr__(self, '_last_adjustment_time', fill.timestamp)
        object.__setattr__(self, '_price_adjustments', self._price_adjustments + 1)
        object.__setattr__(self, '_total_filled_qty', self._total_filled_qty + fill.qty)
        
        print(f"📈 Fill detected! Adjusted limit: ${new_limit:.3f} "
              f"(+{self.adjustment_percentage*100:.1f}%), total filled: {self._total_filled_qty:.1f}")
    
    def step(
        self,
        clock: float,
        broker_state: BrokerState,
        market_state: MarketSnapshot,
    ) -> List[OrderInstruction]:
        """Generate order instructions for current time step."""
        instructions = []
        
        # Check if we're done due to quantity limit
        if self._total_filled_qty >= self.max_total_filled:
            return []
        
        # Check if we're done due to time limit
        if self._order_start_time and clock - self._order_start_time >= self.order_duration:
            if self._current_order_id and self._current_order_id in broker_state.open_orders:
                instructions.append(OrderInstruction(
                    instruction_type=InstructionType.CANCEL,
                    order_id=self._current_order_id,
                ))
            return instructions
        
        # If no current order, place initial one
        if self._current_order_id is None:
            order_id = f"growing_dutch_{self._orders_completed}"
            current_size = self._calculate_current_size(clock)
            
            if current_size <= 0:
                return []
            
            instruction = OrderInstruction(
                instruction_type=InstructionType.PLACE,
                order_id=order_id,
                side=self.side,
                qty=current_size,
                limit_px=self.starting_limit_price,
                valid_to=clock + self.order_duration,
            )
            
            instructions.append(instruction)
            
            # Update internal state
            object.__setattr__(self, '_current_order_id', order_id)
            object.__setattr__(self, '_order_start_time', clock)
            object.__setattr__(self, '_last_limit_price', self.starting_limit_price)
            object.__setattr__(self, '_last_fill_check_time', clock)
            
            print(f"🌱 Created growing Dutch order {order_id} at t={clock:.1f}: "
                  f"size={current_size:.1f}, limit=${self.starting_limit_price:.3f}")
            
        else:
            # Update last fill check time
            object.__setattr__(self, '_last_fill_check_time', clock)
            
            # Check if current order exists
            if self._current_order_id in broker_state.open_orders:
                # Calculate new size and limit price
                current_size = self._calculate_current_size(clock)
                new_limit_price = self._calculate_current_limit_price(clock)
                
                # Check if we need to update size or price
                current_order = broker_state.open_orders[self._current_order_id]
                size_changed = abs(current_order.qty - current_size) >= 0.1
                price_changed = abs(current_order.limit_px - new_limit_price) >= 0.001
                
                if size_changed or price_changed:
                    if current_size <= 0:
                        # Cancel order if size is zero or negative
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.CANCEL,
                            order_id=self._current_order_id,
                        ))
                        object.__setattr__(self, '_current_order_id', None)
                    else:
                        # Update order with new size and/or price
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.CANCEL,
                            order_id=self._current_order_id,
                        ))
                        
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.PLACE,
                            order_id=self._current_order_id,  # Same ID
                            side=self.side,
                            qty=current_size,
                            limit_px=new_limit_price,
                            valid_to=self._order_start_time + self.order_duration,
                        ))
                        
                        object.__setattr__(self, '_last_limit_price', new_limit_price)
                        
                        # Debug output (can be removed in production)
                        if size_changed and price_changed:
                            print(f"🔄 Updated order {self._current_order_id}: size={current_size:.1f}, limit=${new_limit_price:.3f}")
                        elif size_changed:
                            print(f"📏 Size update: {current_size:.1f}")
                        elif price_changed:
                            print(f"💰 Price update: ${new_limit_price:.3f}")
            
            else:
                # Order was filled or cancelled - handle fill adjustments
                if self._last_limit_price is not None:
                    # Estimate fill based on missing order
                    old_size = self._calculate_current_size(clock - 1.0)  # Approximate
                    if old_size > 0:
                        # Assume the order was filled
                        mock_fill = Fill(
                            order_id=self._current_order_id,
                            timestamp=clock,
                            qty=old_size,
                            price=self._last_limit_price,
                            gas_paid=0.0
                        )
                        self._adjust_price_after_fill(mock_fill)
                
                # Reset for next order if not done
                object.__setattr__(self, '_current_order_id', None)
                object.__setattr__(self, '_orders_completed', self._orders_completed + 1)
        
        return instructions
    
    def get_current_state(self, clock: float) -> dict:
        """Get current state for debugging/monitoring."""
        return {
            "time": clock,
            "current_size": self._calculate_current_size(clock),
            "current_limit_price": self._calculate_current_limit_price(clock),
            "total_filled": self._total_filled_qty,
            "price_adjustments": self._price_adjustments,
            "time_remaining": max(0, self.order_duration - (clock - (self._order_start_time or clock))),
        } 