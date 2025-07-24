"""Metrics collection protocols."""

from typing import Any, Protocol

from src.core.events import Fill


class Probe(Protocol):
    """Protocol for collecting metrics during simulation."""
    
    def on_fill(self, fill: Fill) -> None:
        """Called when an order is filled."""
        ...
    
    def on_step(self, t: float) -> None:
        """Called on each simulation time step."""
        ...
    
    def final(self) -> dict[str, Any]:
        """Return final metrics summary."""
        ... 