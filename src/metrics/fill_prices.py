"""Fill price metrics probe."""

from dataclasses import dataclass, field
from typing import Any

from src.core.events import Fill


@dataclass
class FillPrices:
    """Collects fill price data."""
    
    fill_prices: list[float] = field(default_factory=list)
    fill_quantities: list[float] = field(default_factory=list)
    
    def on_fill(self, fill: Fill) -> None:
        """Record fill price and quantity."""
        self.fill_prices.append(fill.price)
        self.fill_quantities.append(fill.qty)
    
    def on_step(self, t: float) -> None:
        """No-op for price probe."""
        pass
    
    def final(self) -> dict[str, Any]:
        """Return price statistics."""
        if not self.fill_prices:
            return {
                "avg_fill_price": None,
                "min_fill_price": None,
                "max_fill_price": None,
                "vwap": None,
                "total_qty": 0.0,
            }
        
        total_qty = sum(self.fill_quantities)
        
        # Volume-weighted average price
        vwap = None
        if total_qty > 0:
            vwap = sum(
                price * qty for price, qty in zip(self.fill_prices, self.fill_quantities)
            ) / total_qty
        
        return {
            "avg_fill_price": sum(self.fill_prices) / len(self.fill_prices),
            "min_fill_price": min(self.fill_prices),
            "max_fill_price": max(self.fill_prices),
            "vwap": vwap,
            "total_qty": total_qty,
        } 