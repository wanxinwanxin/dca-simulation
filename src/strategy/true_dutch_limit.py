"""True Dutch limit order strategy matching the user's specification."""

from dataclasses import dataclass

from src.core.orders import Side
from src.strategy.protocols import (
    BrokerState,
    InstructionType,
    MarketSnapshot,
    OrderInstruction,
)


@dataclass(frozen=True)
class TrueDutchLimit:
    """True Dutch auction limit order strategy.
    
    Each order has:
    - Starting limit price p_0
    - Decay rate d (price change per unit time)  
    - Expiry T (duration from creation)
    - At time t: limit_price = p_0 - d * t (for sell orders)
    - Single order ID maintained throughout lifecycle via cancel+replace
    """
    
    total_qty: float
    slice_qty: float
    side: Side
    starting_limit_price: float  # p_0
    decay_rate: float  # d (positive value, applied as -d for sells, +d for buys)
    order_duration: float  # T (seconds until expiry)
    
    def __post_init__(self) -> None:
        """Initialize internal state."""
        object.__setattr__(self, '_current_order_id', None)
        object.__setattr__(self, '_order_start_time', None)
        object.__setattr__(self, '_orders_completed', 0)
        object.__setattr__(self, '_filled_qty', 0.0)
        object.__setattr__(self, '_last_limit_price', None)
    
    def step(
        self,
        clock: float,
        broker_state: BrokerState,
        market_state: MarketSnapshot,
    ) -> list[OrderInstruction]:
        """Generate order instructions for current time step."""
        instructions = []
        
        # Check if we're done
        if self._filled_qty >= self.total_qty:
            return []
        
        # If no current order, place a new one
        if self._current_order_id is None:
            order_id = f"true_dutch_{self._orders_completed}"
            
            instruction = OrderInstruction(
                instruction_type=InstructionType.PLACE,
                order_id=order_id,
                side=self.side,
                qty=min(self.slice_qty, self.total_qty - self._filled_qty),
                limit_px=self.starting_limit_price,
                valid_to=clock + self.order_duration,
            )
            
            instructions.append(instruction)
            
            # Update internal state
            object.__setattr__(self, '_current_order_id', order_id)
            object.__setattr__(self, '_order_start_time', clock)
            object.__setattr__(self, '_last_limit_price', self.starting_limit_price)
            
        else:
            # Check if current order exists
            if self._current_order_id in broker_state.open_orders:
                # Calculate time elapsed since order creation
                time_elapsed = clock - self._order_start_time
                
                # Check if order has expired
                if time_elapsed >= self.order_duration:
                    # Cancel expired order
                    instructions.append(OrderInstruction(
                        instruction_type=InstructionType.CANCEL,
                        order_id=self._current_order_id,
                    ))
                    # Reset for next order
                    object.__setattr__(self, '_current_order_id', None)
                    object.__setattr__(self, '_orders_completed', self._orders_completed + 1)
                else:
                    # Calculate new limit price based on decay
                    if self.side == Side.SELL:
                        # For sell orders: p_0 - d * t
                        new_limit_price = self.starting_limit_price - self.decay_rate * time_elapsed
                    else:
                        # For buy orders: p_0 + d * t  
                        new_limit_price = self.starting_limit_price + self.decay_rate * time_elapsed
                    
                    # Only update if price has changed significantly (avoid too frequent updates)
                    price_change = abs(new_limit_price - self._last_limit_price)
                    if price_change >= 0.001:  # 0.1 cent threshold
                        
                        # Cancel current order
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.CANCEL,
                            order_id=self._current_order_id,
                        ))
                        
                        # Replace with same ID but new limit price
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.PLACE,
                            order_id=self._current_order_id,  # SAME ID - this is key!
                            side=self.side,
                            qty=min(self.slice_qty, self.total_qty - self._filled_qty),
                            limit_px=new_limit_price,
                            valid_to=self._order_start_time + self.order_duration,  # Fixed expiry
                        ))
                        
                        object.__setattr__(self, '_last_limit_price', new_limit_price)
            else:
                # Order was filled or cancelled, start next slice
                object.__setattr__(self, '_current_order_id', None)
                object.__setattr__(self, '_filled_qty', self._filled_qty + self.slice_qty)
                object.__setattr__(self, '_orders_completed', self._orders_completed + 1)
        
        return instructions
    
    def get_current_limit_price(self, clock: float) -> float | None:
        """Get the theoretical limit price at current time (for debugging)."""
        if self._order_start_time is None:
            return None
            
        time_elapsed = clock - self._order_start_time
        
        if time_elapsed >= self.order_duration:
            return None  # Expired
            
        if self.side == Side.SELL:
            return self.starting_limit_price - self.decay_rate * time_elapsed
        else:
            return self.starting_limit_price + self.decay_rate * time_elapsed 