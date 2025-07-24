"""Core event domain objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fill:
    """Immutable fill event representation."""
    order_id: str
    timestamp: float
    qty: float
    price: float
    gas_paid: float 