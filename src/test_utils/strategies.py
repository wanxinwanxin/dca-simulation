"""Common strategy classes for test simulations."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from src.core.orders import Side
from src.strategy.protocols import OrderInstruction, InstructionType


class IntervalOrderStrategy(ABC):
    """Base class for strategies that create orders at fixed intervals."""
    
    def __init__(self, order_qty: float, n_orders: int, interval: float, 
                 side: Side = Side.SELL, analyzer: Optional[Any] = None):
        self.order_qty = order_qty
        self.n_orders = n_orders
        self.interval = interval
        self.side = side
        self.current_index = 0
        self.analyzer = analyzer
    
    def step(self, clock: float, broker_state: Any, market_state: Any) -> List[OrderInstruction]:
        """Generate orders at predetermined times."""
        instructions = []
        
        if self.current_index < self.n_orders and clock >= self.current_index * self.interval:
            instruction = self._create_order_instruction(
                clock=clock,
                market_state=market_state,
                order_index=self.current_index
            )
            
            if instruction:
                instructions.append(instruction)
                
                # Record creation if analyzer supports it
                if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
                    market_price = market_state.mid_price
                    self.analyzer.record_order_creation(clock, instruction.order_id, market_price)
                
                self.current_index += 1
        
        return instructions
    
    @abstractmethod
    def _create_order_instruction(self, clock: float, market_state: Any, order_index: int) -> Optional[OrderInstruction]:
        """Create the specific order instruction. Must be implemented by subclasses."""
        pass


class MarketOrderStrategy(IntervalOrderStrategy):
    """Strategy that creates market orders at fixed intervals."""
    
    def _create_order_instruction(self, clock: float, market_state: Any, order_index: int) -> OrderInstruction:
        """Create a market order instruction."""
        return OrderInstruction(
            instruction_type=InstructionType.PLACE,
            order_id=f"market_{order_index}",
            side=self.side,
            qty=self.order_qty,
            limit_px=None,  # Market order
            valid_to=clock + 3600,  # Long expiry
        )


class LimitOrderStrategy(IntervalOrderStrategy):
    """Strategy that creates limit orders at fixed intervals."""
    
    def __init__(self, order_qty: float, n_orders: int, interval: float, 
                 limit_offset: float, side: Side = Side.SELL, analyzer: Optional[Any] = None):
        super().__init__(order_qty, n_orders, interval, side, analyzer)
        self.limit_offset = limit_offset  # Offset from mid price (positive for sell orders above mid)
    
    def _create_order_instruction(self, clock: float, market_state: Any, order_index: int) -> OrderInstruction:
        """Create a limit order instruction."""
        mid_price = market_state.mid_price
        
        if self.side == Side.SELL:
            limit_price = mid_price + self.limit_offset
        else:
            limit_price = mid_price - self.limit_offset
        
        return OrderInstruction(
            instruction_type=InstructionType.PLACE,
            order_id=f"limit_{order_index}",
            side=self.side,
            qty=self.order_qty,
            limit_px=limit_price,
            valid_to=clock + 3600,
        )


class DutchOrderStrategy(IntervalOrderStrategy):
    """Strategy that creates Dutch orders at fixed intervals."""
    
    def __init__(self, order_qty: float, n_orders: int, interval: float,
                 starting_offset: float, decay_rate: float, expiry_blocks: float,
                 side: Side = Side.SELL, analyzer: Optional[Any] = None, split: float = 0.0):
        super().__init__(order_qty, n_orders, interval, side, analyzer)
        self.starting_offset = starting_offset  # Starting price offset from mid (positive for sell orders)
        self.decay_rate = decay_rate      # Decay rate per second
        self.expiry_blocks = expiry_blocks   # Blocks until expiry
        self.split = split     # Split parameter for fill price
    
    def _create_order_instruction(self, clock: float, market_state: Any, order_index: int) -> OrderInstruction:
        """Create a Dutch order instruction."""
        mid_price = market_state.mid_price
        
        if self.side == Side.SELL:
            starting_limit = mid_price + self.starting_offset
        else:
            starting_limit = mid_price - self.starting_offset
        
        return OrderInstruction(
            instruction_type=InstructionType.PLACE,
            order_id=f"dutch_{order_index}",
            side=self.side,
            qty=self.order_qty,
            limit_px=starting_limit,
            valid_to=clock + self.expiry_blocks,
        )
    
    def step(self, clock: float, broker_state: Any, market_state: Any) -> List[OrderInstruction]:
        """Override to record Dutch-specific creation data."""
        instructions = []
        
        if self.current_index < self.n_orders and clock >= self.current_index * self.interval:
            instruction = self._create_order_instruction(clock, market_state, self.current_index)
            
            if instruction:
                instructions.append(instruction)
                
                # Record Dutch-specific creation data
                if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
                    market_price = market_state.mid_price
                    if len(self.analyzer.record_order_creation.__code__.co_varnames) > 4:
                        # DutchAnalyzer with start_limit parameter
                        self.analyzer.record_order_creation(
                            clock, instruction.order_id, market_price, instruction.limit_px
                        )
                    else:
                        # Standard analyzer
                        self.analyzer.record_order_creation(clock, instruction.order_id, market_price)
                
                self.current_index += 1
        
        return instructions


class ConfigurableStrategy:
    """Flexible strategy for custom order configurations."""
    
    def __init__(self, orders: List[Dict[str, Any]], analyzer: Optional[Any] = None):
        self.orders = orders  # List of order specifications
        self.current_index = 0
        self.analyzer = analyzer
    
    def step(self, clock: float, broker_state: Any, market_state: Any) -> List[OrderInstruction]:
        """Generate orders based on configuration."""
        instructions = []
        
        # Check if it's time for the next order
        if self.current_index < len(self.orders):
            order_spec = self.orders[self.current_index]
            
            if clock >= order_spec.get("time", 0):
                mid_price = market_state.mid_price
                
                # Calculate limit price if specified
                limit_px = order_spec.get("limit_px")
                if limit_px is None and "limit_offset" in order_spec:
                    offset = order_spec["limit_offset"]
                    side = order_spec.get("side", Side.SELL)
                    if side == Side.SELL:
                        limit_px = mid_price + offset
                    else:
                        limit_px = mid_price - offset
                
                instruction = OrderInstruction(
                    instruction_type=InstructionType.PLACE,
                    order_id=order_spec.get("order_id", f"order_{self.current_index}"),
                    side=order_spec.get("side", Side.SELL),
                    qty=order_spec["qty"],
                    limit_px=limit_px,
                    valid_to=order_spec.get("valid_to", clock + 3600),
                )
                
                instructions.append(instruction)
                
                # Record creation
                if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
                    self.analyzer.record_order_creation(clock, instruction.order_id, mid_price)
                
                self.current_index += 1
        
        return instructions 