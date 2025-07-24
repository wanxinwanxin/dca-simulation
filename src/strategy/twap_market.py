"""TWAP market order strategy."""

from dataclasses import dataclass

from src.core.orders import Side
from src.strategy.protocols import (
    BrokerState,
    InstructionType,
    MarketSnapshot,
    OrderInstruction,
)


@dataclass(frozen=True)
class TwapMarket:
    """TWAP strategy using market orders.
    
    Posts equal-sized market orders at fixed intervals.
    """
    
    total_qty: float
    n_slices: int
    side: Side
    interval: float  # seconds between orders
    
    def __post_init__(self) -> None:
        """Calculate slice size."""
        object.__setattr__(self, '_slice_qty', self.total_qty / self.n_slices)
        object.__setattr__(self, '_orders_placed', 0)
        object.__setattr__(self, '_last_order_time', -1.0)
    
    def step(
        self,
        clock: float,
        broker_state: BrokerState,
        market_state: MarketSnapshot,
    ) -> list[OrderInstruction]:
        """Generate order instructions for current time step."""
        # Check if it's time for next order
        if self._orders_placed >= self.n_slices:
            return []
        
        # Check if enough time has passed since last order (skip for first order)
        if self._last_order_time >= 0 and clock < self._last_order_time + self.interval:
            return []
        
        # Place next market order
        order_id = f"twap_market_{self._orders_placed}"
        
        instruction = OrderInstruction(
            instruction_type=InstructionType.PLACE,
            order_id=order_id,
            side=self.side,
            qty=self._slice_qty,
            limit_px=None,  # Market order
            valid_to=clock + 3600,  # 1 hour validity
        )
        
        # Update internal state
        object.__setattr__(self, '_orders_placed', self._orders_placed + 1)
        object.__setattr__(self, '_last_order_time', clock)
        
        return [instruction] 