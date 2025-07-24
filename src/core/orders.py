"""Core order domain objects."""

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Order:
    """Immutable order representation."""
    id: str
    side: Side
    qty: float
    limit_px: float | None  # None ⇒ market order
    placed_at: float  # simulation time (seconds)
    valid_to: float  # GTT time 