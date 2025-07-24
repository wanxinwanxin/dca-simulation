"""Common tracking engine classes for test simulations."""

from typing import Any, Optional

from src.engine.matching import MatchingEngine
from src.engine.dutch_aware_matching import DutchAwareMatchingEngine
from src.core.orders import Order, Side


class TrackingMatchingEngine(MatchingEngine):
    """Base matching engine that adds order tracking capabilities."""
    
    def __init__(self, *args, analyzer: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
    
    def _place_order(self, instruction: Any, current_time: float) -> None:
        """Track order placements."""
        super()._place_order(instruction, current_time)
        
        # Record order creation if analyzer supports it
        if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
            market_price = self.price_process.mid_price(current_time)
            self.analyzer.record_order_creation(current_time, instruction.order_id, market_price)


class MarketOrderTrackingEngine(MatchingEngine):
    """Engine that specifically tracks market order creation and execution."""
    
    def __init__(self, *args, analyzer: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
    
    def _place_order(self, instruction: Any, current_time: float) -> None:
        """Track market order placements."""
        if instruction.side is None or instruction.qty is None:
            return
            
        order = Order(
            id=instruction.order_id,
            side=instruction.side,
            qty=instruction.qty,
            limit_px=instruction.limit_px,
            placed_at=current_time,
            valid_to=instruction.valid_to or current_time + 3600,
        )
        
        self.open_orders[order.id] = order
        
        # Track order creation with market price (market orders only)
        if self.analyzer and order.limit_px is None:  # Market orders only
            market_price = self.price_process.mid_price(current_time)
            self.analyzer.record_order_creation(current_time, order.id, market_price)


class DutchOrderTrackingEngine(MatchingEngine):
    """Engine that specifically tracks Dutch order creation and execution."""
    
    def __init__(self, *args, analyzer: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
    
    def _place_order(self, instruction: Any, current_time: float) -> None:
        """Track Dutch order placements."""
        if instruction.side is None or instruction.qty is None:
            return
            
        order = Order(
            id=instruction.order_id,
            side=instruction.side,
            qty=instruction.qty,
            limit_px=instruction.limit_px,
            placed_at=current_time,
            valid_to=instruction.valid_to or current_time + 3600,
        )
        
        self.open_orders[order.id] = order
        
        # Track Dutch order creation
        if self.analyzer and order.id.startswith("dutch_") and order.limit_px is not None:
            market_price = self.price_process.mid_price(current_time)
            
            # Check if analyzer supports Dutch-specific recording
            if hasattr(self.analyzer, 'record_order_creation'):
                if len(self.analyzer.record_order_creation.__code__.co_varnames) > 4:
                    # DutchAnalyzer with start_limit parameter
                    self.analyzer.record_order_creation(current_time, order.id, market_price, order.limit_px)
                else:
                    # Standard analyzer
                    self.analyzer.record_order_creation(current_time, order.id, market_price)


class DutchAwareTrackingEngine(DutchAwareMatchingEngine):
    """Dutch-aware engine with tracking capabilities."""
    
    def __init__(self, *args, analyzer: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
    
    def _place_order(self, instruction: Any, current_time: float) -> None:
        """Track order placements in Dutch-aware engine."""
        super()._place_order(instruction, current_time)
        
        # Record order creation if analyzer supports it
        if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
            market_price = self.price_process.mid_price(current_time)
            self.analyzer.record_order_creation(current_time, instruction.order_id, market_price)


class PricePathTrackingEngine(MatchingEngine):
    """Engine that tracks both order creation and price path."""
    
    def __init__(self, *args, analyzer: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
    
    def _place_order(self, instruction: Any, current_time: float) -> None:
        """Track order placements and record price."""
        super()._place_order(instruction, current_time)
        
        if self.analyzer:
            market_price = self.price_process.mid_price(current_time)
            
            # Record order creation
            if hasattr(self.analyzer, 'record_order_creation'):
                self.analyzer.record_order_creation(current_time, instruction.order_id, market_price)
            
            # Record price point
            if hasattr(self.analyzer, 'record_price_point'):
                self.analyzer.record_price_point(current_time, market_price)
    
    def step(self, until: float) -> None:
        """Override step to record price points."""
        if self.analyzer and hasattr(self.analyzer, 'record_price_point'):
            current_time = self.env.now
            market_price = self.price_process.mid_price(current_time)
            self.analyzer.record_price_point(current_time, market_price)
        
        super().step(until) 