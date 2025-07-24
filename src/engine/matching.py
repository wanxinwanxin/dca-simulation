"""SimPy-based matching engine."""

from dataclasses import dataclass, field
from typing import Any

import simpy

from src.core.events import Fill
from src.core.orders import Order
from src.cost.filler import FillerDecision
from src.cost.gas_model import GasModel
from src.market.protocols import ImpactModel, LiquidityModel, PriceProcess
from src.metrics.protocols import Probe
from src.strategy.protocols import (
    BrokerState,
    ExecutionAlgo,
    InstructionType,
    MarketSnapshot,
)


@dataclass
class MatchingEngine:
    """SimPy-based order matching and execution engine."""
    
    price_process: PriceProcess
    liquidity_model: LiquidityModel
    impact_model: ImpactModel
    gas_model: GasModel
    filler_decision: FillerDecision
    probes: list[Probe]
    time_step: float = 1.0  # seconds
    gas_per_fill: int = 21000  # standard transfer gas
    
    # Internal state
    env: simpy.Environment = field(default_factory=simpy.Environment, init=False)
    open_orders: dict[str, Order] = field(default_factory=dict, init=False)
    total_filled_qty: float = field(default=0.0, init=False)
    
    def run(self, algo: ExecutionAlgo, target_qty: float, horizon: float) -> dict[str, Any]:
        """Run simulation with given algorithm until completion or horizon."""
        
        def simulation_process() -> Any:
            """Main simulation process."""
            while self.env.now < horizon and self.total_filled_qty < target_qty:
                current_time = float(self.env.now)
                
                # Get current market state
                mid_price = self.price_process.mid_price(current_time)
                market_state = MarketSnapshot(
                    mid_price=mid_price,
                    timestamp=current_time,
                )
                
                # Get broker state
                broker_state = BrokerState(
                    open_orders=self.open_orders.copy(),
                    filled_qty=self.total_filled_qty,
                    remaining_qty=target_qty - self.total_filled_qty,
                )
                
                # Notify probes of time step
                for probe in self.probes:
                    probe.on_step(current_time)
                
                # Get instructions from algorithm
                instructions = algo.step(current_time, broker_state, market_state)
                
                # Process instructions
                for instruction in instructions:
                    if instruction.instruction_type == InstructionType.PLACE:
                        self._place_order(instruction, current_time)
                    elif instruction.instruction_type == InstructionType.CANCEL:
                        self._cancel_order(instruction.order_id)
                
                # Check for fills
                self._check_fills(current_time, mid_price)
                
                # Advance time
                yield self.env.timeout(self.time_step)
        
        # Start simulation
        self.env.process(simulation_process())
        self.env.run()
        
        # Collect final metrics
        results = {}
        for i, probe in enumerate(self.probes):
            probe_results = probe.final()
            results[f"probe_{i}_{probe.__class__.__name__}"] = probe_results
        
        return results
    
    def _place_order(
        self, 
        instruction: Any, 
        current_time: float
    ) -> None:
        """Place a new order."""
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
    
    def _cancel_order(self, order_id: str) -> None:
        """Cancel an existing order."""
        if order_id in self.open_orders:
            del self.open_orders[order_id]
    
    def _check_fills(self, current_time: float, mid_price: float) -> None:
        """Check if any orders should fill and execute them."""
        filled_orders = []
        
        for order_id, order in self.open_orders.items():
            # Check if order is expired
            if current_time > order.valid_to:
                filled_orders.append(order_id)
                continue
            
            # Check if order crosses the spread (for limit orders)
            if order.limit_px is not None:
                if not self.liquidity_model.crossed(order.side, order.limit_px, mid_price):
                    continue
            
            # Calculate execution price
            exec_price = self.impact_model.exec_price(order, mid_price)
            
            # Calculate gas cost
            gas_cost = self.gas_model.gas_fee(self.gas_per_fill)
            
            # Check if filler would execute
            if order.limit_px is not None:
                should_fill = self.filler_decision.should_fill(
                    order.side, order.limit_px, mid_price, order.qty, gas_cost
                )
                if not should_fill:
                    continue
            
            # Execute the fill
            fill = Fill(
                order_id=order.id,
                timestamp=current_time,
                qty=order.qty,
                price=exec_price,
                gas_paid=gas_cost,
            )
            
            # Update state
            self.total_filled_qty += order.qty
            filled_orders.append(order_id)
            
            # Notify probes
            for probe in self.probes:
                probe.on_fill(fill)
        
        # Remove filled orders
        for order_id in filled_orders:
            if order_id in self.open_orders:
                del self.open_orders[order_id] 