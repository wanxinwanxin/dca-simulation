"""Fill timing metrics probe."""

from dataclasses import dataclass, field
from typing import Any

from src.core.events import Fill


@dataclass
class FillTimings:
    """Collects fill timing data."""
    
    fill_times: list[float] = field(default_factory=list)
    
    def on_fill(self, fill: Fill) -> None:
        """Record fill timestamp."""
        self.fill_times.append(fill.timestamp)
    
    def on_step(self, t: float) -> None:
        """No-op for timing probe."""
        pass
    
    def final(self) -> dict[str, Any]:
        """Return timing statistics."""
        if not self.fill_times:
            return {
                "first_fill_time": None,
                "last_fill_time": None,
                "avg_fill_interval": None,
                "total_fills": 0,
            }
        
        first_time = min(self.fill_times)
        last_time = max(self.fill_times)
        total_fills = len(self.fill_times)
        
        avg_interval = None
        if total_fills > 1:
            avg_interval = (last_time - first_time) / (total_fills - 1)
        
        return {
            "first_fill_time": first_time,
            "last_fill_time": last_time,
            "avg_fill_interval": avg_interval,
            "total_fills": total_fills,
        } 