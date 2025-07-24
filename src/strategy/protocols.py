"""Strategy layer protocols and instruction types."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from src.core.orders import Side


class InstructionType(str, Enum):
    """Order instruction types."""
    PLACE = "PLACE"
    CANCEL = "CANCEL"


@dataclass(frozen=True, slots=True)
class OrderInstruction:
    """Instruction for placing or canceling orders."""
    
    instruction_type: InstructionType
    order_id: str
    side: Side | None = None
    qty: float | None = None
    limit_px: float | None = None
    valid_to: float | None = None


@dataclass(frozen=True, slots=True)
class BrokerState:
    """Current state of broker (orders, fills, etc.)."""
    
    open_orders: dict[str, Any]  # order_id -> Order
    filled_qty: float
    remaining_qty: float


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Current market state."""
    
    mid_price: float
    timestamp: float


class ExecutionAlgo(Protocol):
    """Protocol for execution algorithms."""
    
    def step(
        self,
        clock: float,
        broker_state: BrokerState,
        market_state: MarketSnapshot,
    ) -> list[OrderInstruction]:
        """Generate order instructions for current time step."""
        ... 