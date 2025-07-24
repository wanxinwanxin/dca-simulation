"""Common analyzer classes for test simulations."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from src.core.events import Fill


@dataclass
class BaseAnalyzer(ABC):
    """Base analyzer class with common tracking functionality."""
    
    fills: List[Fill] = field(default_factory=list)
    
    def on_fill(self, fill: Fill) -> None:
        """Record fill events."""
        self.fills.append(fill)
    
    def on_step(self, t: float) -> None:
        """Called on each time step. Override if needed."""
        pass
    
    @abstractmethod
    def final(self) -> Dict[str, Any]:
        """Return analysis results. Must be implemented by subclasses."""
        pass


@dataclass
class OrderTrackingAnalyzer(BaseAnalyzer):
    """Analyzer that tracks order creation timing and market prices."""
    
    order_creations: List[Tuple[float, str, float]] = field(default_factory=list)  # time, order_id, market_price
    
    def record_order_creation(self, time: float, order_id: str, market_price: float) -> None:
        """Record when an order was created and the market price at that time."""
        self.order_creations.append((time, order_id, market_price))
    
    def final(self) -> Dict[str, Any]:
        """Return tracking data."""
        return {
            "fills": self.fills,
            "order_creations": self.order_creations,
        }


@dataclass
class PricePathAnalyzer(BaseAnalyzer):
    """Analyzer that tracks order creation and full price path."""
    
    order_creations: List[Tuple[float, str, float]] = field(default_factory=list)
    price_path: List[Tuple[float, float]] = field(default_factory=list)  # time, price
    
    def record_order_creation(self, time: float, order_id: str, market_price: float) -> None:
        """Record order creation."""
        self.order_creations.append((time, order_id, market_price))
    
    def record_price_point(self, time: float, price: float) -> None:
        """Record price path data."""
        self.price_path.append((time, price))
    
    def final(self) -> Dict[str, Any]:
        """Return analysis results."""
        return {
            "fills": self.fills,
            "order_creations": self.order_creations,
            "price_path": self.price_path,
        }


@dataclass
class DutchAnalyzer(BaseAnalyzer):
    """Analyzer specialized for Dutch order tracking."""
    
    order_creations: List[Tuple[float, str, float, float]] = field(default_factory=list)  # time, order_id, market_price, start_limit
    price_path: List[Tuple[float, float]] = field(default_factory=list)
    
    def record_order_creation(self, time: float, order_id: str, market_price: float, start_limit: float) -> None:
        """Record when a Dutch order was created."""
        self.order_creations.append((time, order_id, market_price, start_limit))
    
    def record_price_point(self, time: float, price: float) -> None:
        """Record price path data."""
        self.price_path.append((time, price))
    
    def final(self) -> Dict[str, Any]:
        """Return analysis results."""
        return {
            "fills": self.fills,
            "order_creations": self.order_creations,
            "price_path": self.price_path,
        }


@dataclass
class DetailedStepAnalyzer(BaseAnalyzer):
    """Analyzer that records detailed step-by-step data for verification."""
    
    step_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_step(self, step_data: Dict[str, Any]) -> None:
        """Record data for each time step."""
        self.step_data.append(step_data.copy())
    
    def final(self) -> Dict[str, Any]:
        """Return all recorded data."""
        return {
            "fills": self.fills,
            "step_data": self.step_data,
        }


@dataclass
class MultiPathAnalyzer(BaseAnalyzer):
    """Analyzer for multi-path simulations."""
    
    order_creations: List[Tuple[float, str, float]] = field(default_factory=list)
    
    def record_order_creation(self, time: float, order_id: str, market_price: float) -> None:
        """Record order creation."""
        self.order_creations.append((time, order_id, market_price))
    
    def final(self) -> Dict[str, Any]:
        """Return analysis results."""
        return {
            "fills": self.fills,
            "order_creations": self.order_creations,
        } 