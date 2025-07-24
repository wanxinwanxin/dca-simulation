"""Linear market impact model."""
# %%
from dataclasses import dataclass

from src.core.orders import Order, Side
# %%


@dataclass(frozen=True)
class LinearImpact:
    """Linear market impact model.
    
    Market impact: mid ± γ·qty
    """
    
    gamma: float  # impact coefficient
    
    def exec_price(self, order: Order, mid_px: float) -> float:
        """Calculate execution price including market impact."""
        impact = self.gamma * order.qty
        
        if order.side == Side.BUY:
            # Buy orders push price up
            return mid_px + impact
        else:
            # Sell orders push price down
            return mid_px - impact 