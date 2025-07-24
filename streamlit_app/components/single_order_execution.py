"""Single Order Execution Detail Component.

This component provides:
- Selection of a saved price path
- Configuration of a single market order (timing, quantity)
- Execution simulation with detailed tracking
- Visualization of execution on price path
- Performance metrics including implementation shortfall
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# Import our simulation framework
from src.test_utils.strategies import MarketOrderStrategy, ConfigurableStrategy
from src.test_utils.engines import MarketOrderTrackingEngine
from src.test_utils.simulation import SimulationRunner, SimulationConfig
from src.test_utils.analyzers import BaseAnalyzer
from src.core.orders import Side
from src.core.events import Fill
from src.market.gbm import GBM
from src.market.dutch_impact import DutchImpact
from src.engine.dutch_aware_matching import DutchAwareMatchingEngine
from src.strategy.protocols import (
    BrokerState,
    InstructionType, 
    MarketSnapshot,
    OrderInstruction,
)


@dataclass
class OrderExecutionResult:
    """Results from single order execution."""
    order_creation_time: float
    order_creation_mid: float
    execution_time: float
    execution_price: float
    order_quantity: float
    implementation_shortfall: float  # execution_price - creation_mid (for sell orders)
    implementation_shortfall_bps: float  # implementation shortfall in basis points
    time_to_execution: float  # execution_time - creation_time
    success: bool


class SingleOrderAnalyzer(BaseAnalyzer):
    """Analyzer that captures single order execution details."""
    
    def __init__(self):
        super().__init__()
        self.order_creation_events = []
        self.price_history = []
        
    def record_order_creation(self, timestamp: float, order_id: str, market_price: float):
        """Record order creation event."""
        self.order_creation_events.append({
            "timestamp": timestamp,
            "order_id": order_id,
            "market_price": market_price
        })
    
    def on_step(self, t: float, market_price: float = None):
        """Record market state at each step."""
        if market_price is not None:
            self.price_history.append({
                "timestamp": t,
                "price": market_price
            })
    
    def final(self) -> Dict[str, Any]:
        """Return comprehensive execution analysis."""
        return {
            "fills": self.fills,
            "order_creation_events": self.order_creation_events,
            "price_history": self.price_history,
            "fill_count": len(self.fills),
            "total_filled": sum(f.qty for f in self.fills),
        }


def create_single_order_execution_result(analyzer_result: Dict[str, Any], 
                                       order_qty: float) -> Optional[OrderExecutionResult]:
    """Create OrderExecutionResult from analyzer data."""
    
    # Check if we have the necessary data
    if not analyzer_result.get("order_creation_events") or not analyzer_result.get("fills"):
        return None
    
    creation_event = analyzer_result["order_creation_events"][0]
    fills = analyzer_result["fills"]
    
    if not fills:
        return None
    
    # For simplicity, assume single fill (market orders should fill immediately)
    fill = fills[0]
    
    # Calculate implementation shortfall
    creation_mid = creation_event["market_price"]
    execution_price = fill.price
    
    # For sell orders: shortfall = execution_price - creation_mid (negative is bad)
    # For buy orders: shortfall = creation_mid - execution_price (negative is bad)
    implementation_shortfall = execution_price - creation_mid
    implementation_shortfall_bps = (implementation_shortfall / creation_mid) * 10000
    
    time_to_execution = fill.timestamp - creation_event["timestamp"]
    
    return OrderExecutionResult(
        order_creation_time=creation_event["timestamp"],
        order_creation_mid=creation_mid,
        execution_time=fill.timestamp,
        execution_price=execution_price,
        order_quantity=fill.qty,
        implementation_shortfall=implementation_shortfall,
        implementation_shortfall_bps=implementation_shortfall_bps,
        time_to_execution=time_to_execution,
        success=True
    )


def create_dutch_order_execution_result(analyzer_result: Dict[str, Any], 
                                       order_qty: float,
                                       strategy: 'SingleDutchOrderStrategy',
                                       starting_limit_offset_pct: float,
                                       decay_rate_pct: float,
                                       order_duration: float) -> Optional['DutchOrderExecutionResult']:
    """Create DutchOrderExecutionResult from analyzer data."""
    
    # Check if we have the necessary data
    if not analyzer_result.get("order_creation_events") or not analyzer_result.get("fills"):
        return None
    
    creation_event = analyzer_result["order_creation_events"][0]
    fills = analyzer_result["fills"]
    
    if not fills:
        return None
    
    # For simplicity, assume single fill
    fill = fills[0]
    
    # Calculate metrics
    creation_mid = creation_event["market_price"]
    execution_price = fill.price
    time_to_execution = fill.timestamp - creation_event["timestamp"]
    
    # Calculate implementation shortfall
    if strategy.side == Side.SELL:
        implementation_shortfall = execution_price - creation_mid
    else:
        implementation_shortfall = creation_mid - execution_price
    
    implementation_shortfall_bps = (implementation_shortfall / creation_mid) * 10000
    
    # Calculate theoretical limit at fill time
    theoretical_limit = strategy.get_current_limit_price(fill.timestamp)
    
    # Calculate price improvement vs starting limit
    price_improvement_vs_limit = execution_price - strategy.starting_limit_price
    if strategy.side == Side.BUY:
        price_improvement_vs_limit = strategy.starting_limit_price - execution_price
    
    return DutchOrderExecutionResult(
        order_creation_time=creation_event["timestamp"],
        order_creation_mid=creation_mid,
        execution_time=fill.timestamp,
        execution_price=execution_price,
        order_quantity=fill.qty,
        implementation_shortfall=implementation_shortfall,
        implementation_shortfall_bps=implementation_shortfall_bps,
        time_to_execution=time_to_execution,
        success=True,
        starting_limit_price=strategy.starting_limit_price,
        decay_rate=decay_rate_pct,
        order_duration=order_duration,
        theoretical_limit_at_fill=theoretical_limit,
        price_improvement_vs_limit=price_improvement_vs_limit,
        dutch_order_type="dutch"
    )


def run_single_order_simulation(price_path_data, 
                              order_timing: float, 
                              order_quantity: float,
                              side: Side,
                              order_type: str = "market",
                              starting_limit_offset: float = 0.50,
                              decay_rate: float = 0.01,
                              order_duration: float = 60.0) -> Optional[OrderExecutionResult]:
    """Run simulation with a single market or Dutch order."""
    
    # Create analyzer
    analyzer = SingleOrderAnalyzer()
    
    # Create strategy based on order type
    if order_type == "dutch":
        # Dutch strategy will calculate starting limit based on mid price at creation time
        strategy = SingleDutchOrderStrategy(
            order_qty=order_quantity,
            timing=order_timing,
            side=side,
            starting_limit_offset_pct=starting_limit_offset,
            decay_rate_pct=decay_rate,
            order_duration=order_duration,
            analyzer=analyzer
        )
    else:
        # Market order (default)
        # Create order specification for ConfigurableStrategy
        order_spec = {
            "time": order_timing,  # Absolute time to place the order
            "qty": order_quantity,
            "side": side,
            "order_id": "single_market_order",
            "limit_px": None,  # Market order
        }
        strategy = ConfigurableStrategy(
            orders=[order_spec],
            analyzer=analyzer
        )
    
    # Create simulation config
    config = SimulationConfig(
        horizon=price_path_data.horizon,
        dt=price_path_data.dt,
        initial_price=price_path_data.initial_price,
        drift=price_path_data.drift,
        volatility=price_path_data.volatility,
        random_seed=price_path_data.seed,
        spread=0.01,  # 1 cent spread
        impact_gamma=0.001,  # 0.1% impact per unit
    )
    
    # Custom simulation runner with proper engine setup
    class SingleOrderSimulationRunner(SimulationRunner):
        def create_engine(self, analyzer=None, order_type="market", dutch_strategy=None, **kwargs):
            """Create engine with analyzer support."""
            price_process = self.create_price_process()
            liquidity_model, impact_model = self.create_market_models()
            gas_model, filler_decision = self.create_cost_models()
            
            # For Dutch orders, use Dutch-aware impact model and engine
            if order_type == "dutch":
                impact_model = DutchImpact(
                    spread=liquidity_model.spread,
                    gamma=0.001,  # 0.1% impact per unit
                    dutch_price_split=0.0  # Fill at limit price
                )
            
            # Set up probes list
            probes = []
            if analyzer:
                probes.append(analyzer)
            
            # Create appropriate engine based on order type
            if order_type == "dutch":
                engine = DutchAwareMatchingEngine(
                    price_process=price_process,
                    liquidity_model=liquidity_model,
                    impact_model=impact_model,
                    gas_model=gas_model,
                    filler_decision=filler_decision,
                    probes=probes,
                    dutch_strategy=dutch_strategy,
                    **kwargs
                )
            else:
                engine = MarketOrderTrackingEngine(
                    price_process=price_process,
                    liquidity_model=liquidity_model,
                    impact_model=impact_model,
                    gas_model=gas_model,
                    filler_decision=filler_decision,
                    probes=probes,
                    analyzer=analyzer,
                    **kwargs
                )
            
            return engine
    
    # Create simulation runner
    if order_type == "dutch":
        runner = SingleOrderSimulationRunner(
            config=config,
            engine_class=DutchAwareMatchingEngine
        )
        
        # Run simulation with Dutch-aware setup
        results = runner.run_simulation(
            strategy=strategy,
            analyzer=analyzer,
            order_type=order_type,
            dutch_strategy=strategy
        )
    else:
        runner = SingleOrderSimulationRunner(
            config=config,
            engine_class=MarketOrderTrackingEngine
        )
        
        # Run simulation
        results = runner.run_simulation(
            strategy=strategy,
            analyzer=analyzer,
            order_type=order_type
        )
    
    # Extract execution details
    analyzer_result = analyzer.final()
    
    if order_type == "dutch":
        return create_dutch_order_execution_result(
            analyzer_result, order_quantity, strategy, 
            starting_limit_offset, decay_rate, order_duration
        )
    else:
        return create_single_order_execution_result(analyzer_result, order_quantity)


def render_execution_visualization(execution_result: OrderExecutionResult,
                                 price_path_data) -> None:
    """Render visualization of order execution on price path."""
    
    # Use the existing price path data instead of regenerating
    time_points = price_path_data.times
    prices = price_path_data.prices
    
    # Create the plot
    fig = go.Figure()
    
    # Add price path
    fig.add_trace(go.Scatter(
        x=time_points,
        y=prices,
        mode='lines',
        name='Price Path',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x:.1f}s<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # For Dutch orders, add the declining limit price visualization
    if isinstance(execution_result, DutchOrderExecutionResult):
        # Calculate declining limit price over time during order lifetime
        dutch_result = execution_result  # Type cast for clarity
        
        # Find time indices for the order's active period
        order_start_time = dutch_result.order_creation_time
        order_end_time = dutch_result.execution_time
        
        # Create time points for the limit price line (every second during order lifetime)
        limit_times = []
        limit_prices = []
        
        # Get the strategy from session state to calculate limit prices
        # We'll recalculate this based on the execution result parameters
        decay_rate_pct = dutch_result.decay_rate
        starting_limit = dutch_result.starting_limit_price
        
        # Calculate limit price at each time step during order's lifetime
        current_time = order_start_time
        dt = 1.0  # Calculate every second for smooth line
        
        while current_time <= order_end_time + dt:  # Go slightly past execution for completeness
            time_elapsed = current_time - order_start_time
            
            # Calculate theoretical limit price (same logic as in strategy)
            decay_amount = starting_limit * (decay_rate_pct / 100) * time_elapsed
            
            # Get order side from implementation shortfall sign pattern
            # For sell orders: shortfall = execution_price - creation_mid
            # For buy orders: shortfall = creation_mid - execution_price
            is_sell_order = dutch_result.implementation_shortfall == (dutch_result.execution_price - dutch_result.order_creation_mid)
            
            if is_sell_order:
                # Sell order: decay DOWN from starting price
                current_limit = starting_limit - decay_amount
            else:
                # Buy order: decay UP from starting price (toward market)
                current_limit = starting_limit + decay_amount
            
            limit_times.append(current_time)
            limit_prices.append(current_limit)
            
            current_time += dt
        
        # Add declining limit price line
        fig.add_trace(go.Scatter(
            x=limit_times,
            y=limit_prices,
            mode='lines',
            name='Dutch Limit Price (Declining)',
            line=dict(color='red', width=3, dash='dash'),
            hovertemplate='Time: %{x:.1f}s<br>Limit Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add a marker at the theoretical limit at fill time
        fig.add_trace(go.Scatter(
            x=[dutch_result.execution_time],
            y=[dutch_result.theoretical_limit_at_fill],
            mode='markers',
            name='Theoretical Limit at Fill',
            marker=dict(
                color='red',
                size=12,
                symbol='x',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='Theoretical Limit at Fill<br>Time: %{x:.1f}s<br>Limit: $%{y:.2f}<extra></extra>'
        ))
        
        # Highlight the intersection area where fill occurred
        # Find the market price at execution time for comparison
        exec_time_index = None
        for i, t in enumerate(time_points):
            if abs(t - dutch_result.execution_time) < price_path_data.dt / 2:
                exec_time_index = i
                break
        
        if exec_time_index is not None:
            market_price_at_fill = prices[exec_time_index]
            
            # Add annotation explaining the intersection
            fig.add_annotation(
                x=dutch_result.execution_time,
                y=(market_price_at_fill + dutch_result.theoretical_limit_at_fill) / 2,
                text=f"Fill Intersection<br>Market: ${market_price_at_fill:.2f}<br>Limit: ${dutch_result.theoretical_limit_at_fill:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                bgcolor="lightgreen",
                bordercolor="green",
                font=dict(size=10)
            )
    
    # Add order creation marker
    fig.add_trace(go.Scatter(
        x=[execution_result.order_creation_time],
        y=[execution_result.order_creation_mid],
        mode='markers',
        name='Order Created',
        marker=dict(
            color='orange',
            size=12,
            symbol='diamond',
            line=dict(color='black', width=2)
        ),
        hovertemplate='Order Created<br>Time: %{x:.1f}s<br>Mid Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add execution marker
    fig.add_trace(go.Scatter(
        x=[execution_result.execution_time],
        y=[execution_result.execution_price],
        mode='markers',
        name='Order Executed',
        marker=dict(
            color='purple',
            size=15,
            symbol='star',
            line=dict(color='black', width=2)
        ),
        hovertemplate='Order Executed<br>Time: %{x:.1f}s<br>Execution Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add connecting line between creation and execution
    fig.add_trace(go.Scatter(
        x=[execution_result.order_creation_time, execution_result.execution_time],
        y=[execution_result.order_creation_mid, execution_result.execution_price],
        mode='lines',
        name='Execution Timeline',
        line=dict(
            color='gray',
            width=2,
            dash='dash'
        ),
        hoverinfo='skip'
    ))
    
    # Update layout
    title = 'Single Order Execution Detail'
    if isinstance(execution_result, DutchOrderExecutionResult):
        title += ' - Dutch Auction with Declining Limit Price'
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Price ($)',
        hovermode='closest',
        showlegend=True,
        height=500
    )
    
    # Add annotation for implementation shortfall
    fig.add_annotation(
        x=execution_result.execution_time,
        y=execution_result.execution_price,
        text=f"Implementation Shortfall: {execution_result.implementation_shortfall_bps:.1f} bps",
        showarrow=True,
        arrowhead=2,
        arrowcolor='blue',
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_execution_metrics(execution_result: OrderExecutionResult) -> None:
    """Render detailed execution metrics."""
    
    # Check if this is a Dutch order result
    is_dutch = isinstance(execution_result, DutchOrderExecutionResult)
    
    # Basic timing metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Order Creation Time",
            f"{execution_result.order_creation_time:.1f}s",
            help="When the order was placed"
        )
    
    with col2:
        st.metric(
            "Execution Time", 
            f"{execution_result.execution_time:.1f}s",
            help="When the order was filled"
        )
    
    with col3:
        st.metric(
            "Time to Execution",
            f"{execution_result.time_to_execution:.1f}s",
            help="Duration between order placement and execution"
        )
    
    with col4:
        order_type_display = "Dutch Auction" if is_dutch else "Market Order"
        st.metric(
            "Order Type",
            order_type_display,
            help="Type of order executed"
        )
    
    st.divider()
    
    # Price and shortfall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mid Price at Creation",
            f"${execution_result.order_creation_mid:.2f}",
            help="Market mid price when order was created"
        )
    
    with col2:
        st.metric(
            "Execution Price",
            f"${execution_result.execution_price:.2f}",
            help="Actual price at which order was filled"
        )
    
    with col3:
        st.metric(
            "Implementation Shortfall",
            f"${execution_result.implementation_shortfall:.4f}",
            delta=f"{execution_result.implementation_shortfall_bps:.1f} bps",
            help="Difference between execution price and mid price at creation"
        )
    
    with col4:
        st.metric(
            "Order Quantity",
            f"{execution_result.order_quantity:.2f}",
            help="Quantity of the executed order"
        )
    
    # Dutch order specific metrics
    if is_dutch:
        st.divider()
        st.write("**🔄 Dutch Auction Specific Metrics**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Starting Limit Price",
                f"${execution_result.starting_limit_price:.2f}",
                help="Initial limit price when Dutch order was placed"
            )
        
        with col2:
            st.metric(
                "Theoretical Limit at Fill",
                f"${execution_result.theoretical_limit_at_fill:.2f}",
                help="What the limit price had decayed to at execution time"
            )
        
        with col3:
            st.metric(
                "Decay Rate",
                f"{execution_result.decay_rate:.3f}%/sec",
                help="How fast the limit price was decaying"
            )


@dataclass
class DutchOrderExecutionResult(OrderExecutionResult):
    """Extended results for Dutch auction orders."""
    starting_limit_price: float
    decay_rate: float  # % per second
    order_duration: float
    theoretical_limit_at_fill: float
    price_improvement_vs_limit: float
    dutch_order_type: str


class SingleDutchOrderStrategy:
    """Strategy that creates a single Dutch order at specified timing with continuous limit price updates."""
    
    def __init__(self, order_qty: float, timing: float, side: Side,
                 starting_limit_offset_pct: float, decay_rate_pct: float, 
                 order_duration: float, analyzer=None):
        self.order_qty = order_qty
        self.timing = timing
        self.side = side
        self.starting_limit_offset_pct = starting_limit_offset_pct  # % offset from creation mid price
        self.decay_rate_pct = decay_rate_pct  # % of starting limit per second
        self.order_duration = order_duration
        self.analyzer = analyzer
        
        # Order state tracking
        self.order_placed = False
        self.order_id = "single_dutch_order"
        self.starting_limit_price = None  # Will be set when order is placed
        self.creation_mid_price = None  # Will be set when order is placed
        self.order_start_time = None  # When the order was first placed
        self.last_limit_price = None  # Track last limit price to avoid unnecessary updates
    
    def step(self, clock: float, broker_state: BrokerState, 
             market_state: MarketSnapshot) -> list[OrderInstruction]:
        """Generate Dutch order instructions with continuous limit price updates."""
        instructions = []
        
        # Place initial order at specified time
        if not self.order_placed and clock >= self.timing:
            # Calculate starting limit price based on current mid price (at creation time)
            self.creation_mid_price = market_state.mid_price
            self.order_start_time = clock
            
            if self.side == Side.SELL:
                # For sell: start above mid price by offset percentage
                self.starting_limit_price = self.creation_mid_price * (1 + self.starting_limit_offset_pct / 100)
            else:
                # For buy: start below mid price by offset percentage  
                self.starting_limit_price = self.creation_mid_price * (1 - self.starting_limit_offset_pct / 100)
            
            instruction = OrderInstruction(
                instruction_type=InstructionType.PLACE,
                order_id=self.order_id,
                side=self.side,
                qty=self.order_qty,
                limit_px=self.starting_limit_price,
                valid_to=clock + self.order_duration,
            )
            
            instructions.append(instruction)
            self.order_placed = True
            self.last_limit_price = self.starting_limit_price
            
            # Record order creation
            if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
                self.analyzer.record_order_creation(
                    clock, self.order_id, market_state.mid_price
                )
                
        elif self.order_placed and self.order_start_time is not None:
            # Check if order still exists (not filled or cancelled)
            if self.order_id in broker_state.open_orders:
                # Check if order has expired
                time_elapsed = clock - self.order_start_time
                if time_elapsed >= self.order_duration:
                    # Cancel expired order
                    instructions.append(OrderInstruction(
                        instruction_type=InstructionType.CANCEL,
                        order_id=self.order_id,
                    ))
                else:
                    # Calculate new declining limit price
                    new_limit_price = self.get_current_limit_price(clock)
                    
                    # Only update if price has changed significantly (avoid too frequent updates)
                    if (self.last_limit_price is None or 
                        abs(new_limit_price - self.last_limit_price) >= 0.001):  # 0.1 cent threshold
                        
                        # Cancel current order
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.CANCEL,
                            order_id=self.order_id,
                        ))
                        
                        # Replace with same ID but new declining limit price
                        instructions.append(OrderInstruction(
                            instruction_type=InstructionType.PLACE,
                            order_id=self.order_id,  # SAME ID - this is key for Dutch auction!
                            side=self.side,
                            qty=self.order_qty,
                            limit_px=new_limit_price,
                            valid_to=self.order_start_time + self.order_duration,  # Fixed expiry from original start time
                        ))
                        
                        self.last_limit_price = new_limit_price
        
        return instructions
    
    def get_current_limit_price(self, current_time: float) -> float:
        """Calculate theoretical current limit price for Dutch order."""
        if not self.order_placed or self.starting_limit_price is None or self.order_start_time is None:
            return 0.0
        
        time_elapsed = current_time - self.order_start_time
        
        # Ensure time_elapsed is non-negative
        time_elapsed = max(0.0, time_elapsed)
        
        # Calculate decay as percentage of starting limit price per second
        decay_amount = self.starting_limit_price * (self.decay_rate_pct / 100) * time_elapsed
        
        # For sell orders: decay DOWN from starting price
        # For buy orders: decay UP from starting price (toward market)
        if self.side == Side.SELL:
            return self.starting_limit_price - decay_amount
        else:
            return self.starting_limit_price + decay_amount


def verify_dutch_fill_intersection(execution_result: 'DutchOrderExecutionResult', 
                                 price_path_data) -> dict:
    """Verify that Dutch order fill occurred at the first intersection of declining limit with market price.
    
    Returns:
        dict: Analysis results including verification status and details
    """
    if not isinstance(execution_result, DutchOrderExecutionResult):
        return {"error": "Not a Dutch order result"}
    
    # Get execution details
    order_start_time = execution_result.order_creation_time
    fill_time = execution_result.execution_time
    starting_limit = execution_result.starting_limit_price
    decay_rate_pct = execution_result.decay_rate
    theoretical_limit_at_fill = execution_result.theoretical_limit_at_fill
    
    # Determine order side from implementation shortfall calculation
    is_sell_order = execution_result.implementation_shortfall == (execution_result.execution_price - execution_result.order_creation_mid)
    
    # Get market prices from the path data
    time_points = price_path_data.times
    prices = price_path_data.prices
    
    # Find intersection points during order lifetime
    intersections = []
    spread = 0.01  # Using same spread as in simulation (1 cent)
    
    for i, t in enumerate(time_points):
        if t < order_start_time or t > fill_time + 1.0:  # Check slightly past fill time
            continue
            
        # Calculate theoretical limit price at this time
        time_elapsed = t - order_start_time
        decay_amount = starting_limit * (decay_rate_pct / 100) * time_elapsed
        
        if is_sell_order:
            current_limit = starting_limit - decay_amount
            # For sell order: crosses when limit <= mid - spread
            crossing_threshold = prices[i] - spread
            crossed = current_limit <= crossing_threshold
        else:
            current_limit = starting_limit + decay_amount
            # For buy order: crosses when limit >= mid + spread
            crossing_threshold = prices[i] + spread
            crossed = current_limit >= crossing_threshold
        
        if crossed:
            intersections.append({
                "time": t,
                "market_price": prices[i],
                "limit_price": current_limit,
                "crossing_threshold": crossing_threshold,
                "margin": abs(current_limit - crossing_threshold)
            })
    
    # Analysis results
    analysis = {
        "order_side": "SELL" if is_sell_order else "BUY",
        "intersections_found": len(intersections),
        "first_intersection": intersections[0] if intersections else None,
        "fill_time": fill_time,
        "theoretical_limit_at_fill": theoretical_limit_at_fill,
        "spread_used": spread
    }
    
    # Verification checks
    if not intersections:
        analysis["verification"] = "FAILED"
        analysis["issue"] = "No intersections found - order should not have filled"
    elif len(intersections) == 1:
        first_intersection = intersections[0]
        time_diff = abs(first_intersection["time"] - fill_time)
        
        if time_diff <= price_path_data.dt:  # Within one time step
            analysis["verification"] = "PASSED"
            analysis["message"] = "Fill occurred at the first intersection as expected"
        else:
            analysis["verification"] = "WARNING"
            analysis["issue"] = f"Fill time differs from first intersection by {time_diff:.1f}s"
    else:
        first_intersection = intersections[0]
        time_diff = abs(first_intersection["time"] - fill_time)
        
        if time_diff <= price_path_data.dt:
            analysis["verification"] = "PASSED"
            analysis["message"] = f"Fill occurred at first intersection (found {len(intersections)} total intersections)"
        else:
            analysis["verification"] = "WARNING"
            analysis["issue"] = f"Multiple intersections found, fill may not be at first one"
    
    return analysis


def render_dutch_verification_analysis(execution_result: 'DutchOrderExecutionResult', 
                                     price_path_data) -> None:
    """Render detailed verification analysis for Dutch order fills."""
    
    if not isinstance(execution_result, DutchOrderExecutionResult):
        return
    
    st.subheader("🔍 Dutch Order Fill Verification")
    st.markdown("Analyzing whether the order filled at the **first intersection** of declining limit price with market price.")
    
    # Perform verification analysis
    analysis = verify_dutch_fill_intersection(execution_result, price_path_data)
    
    if "error" in analysis:
        st.error(f"Verification error: {analysis['error']}")
        return
    
    # Display verification status
    verification_status = analysis.get("verification", "UNKNOWN")
    
    if verification_status == "PASSED":
        st.success(f"✅ **VERIFICATION PASSED**: {analysis.get('message', 'Fill logic appears correct')}")
    elif verification_status == "WARNING":
        st.warning(f"⚠️ **VERIFICATION WARNING**: {analysis.get('issue', 'Some concerns found')}")
    elif verification_status == "FAILED":
        st.error(f"❌ **VERIFICATION FAILED**: {analysis.get('issue', 'Fill logic appears incorrect')}")
    
    # Display detailed analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Order Side",
            analysis["order_side"],
            help="Whether this was a buy or sell order"
        )
    
    with col2:
        st.metric(
            "Intersections Found",
            analysis["intersections_found"],
            help="Number of times the declining limit price crossed the market spread"
        )
    
    with col3:
        st.metric(
            "Spread Used",
            f"${analysis['spread_used']:.3f}",
            help="Half-spread used for intersection calculation"
        )
    
    # Show first intersection details if available
    if analysis["first_intersection"]:
        st.subheader("📊 First Intersection Details")
        first_int = analysis["first_intersection"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Intersection Time",
                f"{first_int['time']:.1f}s",
                help="When the first intersection occurred"
            )
        
        with col2:
            st.metric(
                "Market Price",
                f"${first_int['market_price']:.3f}",
                help="Market mid price at intersection"
            )
        
        with col3:
            st.metric(
                "Limit Price",
                f"${first_int['limit_price']:.3f}",
                help="Theoretical declining limit price at intersection"
            )
        
        with col4:
            st.metric(
                "Crossing Margin",
                f"${first_int['margin']:.4f}",
                help="How far the limit price crossed the threshold"
            )
        
        # Explanation of the crossing logic
        order_side = analysis["order_side"]
        threshold = first_int['crossing_threshold']
        
        if order_side == "SELL":
            st.info(f"""
            **Sell Order Crossing Logic:**
            - Declining limit price: ${first_int['limit_price']:.3f}
            - Must cross: Market price - spread = ${first_int['market_price']:.3f} - ${analysis['spread_used']:.3f} = ${threshold:.3f}
            - ✅ Crossed because: ${first_int['limit_price']:.3f} ≤ ${threshold:.3f}
            """)
        else:
            st.info(f"""
            **Buy Order Crossing Logic:**
            - Declining limit price: ${first_int['limit_price']:.3f}
            - Must cross: Market price + spread = ${first_int['market_price']:.3f} + ${analysis['spread_used']:.3f} = ${threshold:.3f}
            - ✅ Crossed because: ${first_int['limit_price']:.3f} ≥ ${threshold:.3f}
            """)
    else:
        st.warning("No intersection details available")


def render_single_order_execution():
    """Render the single order execution detail interface."""
    
    st.subheader("🎯 Single Order Execution Detail")
    st.markdown("""
    Simulate the execution of **one single market order** on a selected price path.  
    Analyze timing, implementation shortfall, and visualize the execution.
    """)
    
    # Check if we have saved paths
    if 'path_collections' not in st.session_state or not st.session_state.path_collections:
        st.warning("📁 No saved price paths found.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("💡 **Quick Start**: Go to the **GBM Explorer** tab to generate price paths.")
        with col2:
            st.info("🔄 **Auto-save is enabled** by default - paths will be automatically available here!")
        
        # Quick link to GBM Explorer
        st.markdown("👆 Switch to the **📈 GBM Explorer** tab in the sidebar to get started.")
        return
    
    # Path selection interface
    st.subheader("📈 Select Price Path")
    
    # Get all available paths organized by collection type
    auto_saved_paths = {}
    manually_saved_paths = {}
    
    for collection_name, collection in st.session_state.path_collections.items():
        is_auto_saved = collection['metadata'].get('auto_saved', False)
        
        for i, path in enumerate(collection['paths']):
            path_key = f"{collection_name} - Path {i+1}"
            path_display = {
                'key': path_key,
                'path': path,
                'collection': collection_name,
                'metadata': collection['metadata']
            }
            
            if is_auto_saved:
                auto_saved_paths[path_key] = path_display
            else:
                manually_saved_paths[path_key] = path_display
    
    if not auto_saved_paths and not manually_saved_paths:
        st.warning("No paths found in collections.")
        return
    
    # Display path statistics
    total_collections = len(st.session_state.path_collections)
    total_paths = sum(len(collection['paths']) for collection in st.session_state.path_collections.values())
    auto_collections = sum(1 for collection in st.session_state.path_collections.values() 
                          if collection['metadata'].get('auto_saved', False))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Collections", total_collections)
    with col2:
        st.metric("Total Paths", total_paths)
    with col3:
        st.metric("Auto-saved Collections", auto_collections)
    
    # Path selection with preference for auto-saved paths
    st.write("**Available Paths:**")
    
    # Combine paths with auto-saved first
    all_paths_ordered = {}
    
    if auto_saved_paths:
        st.write("🔄 **Auto-saved from GBM Explorer** (most recent first):")
        # Sort auto-saved paths by creation time (most recent first)
        sorted_auto = sorted(auto_saved_paths.items(), 
                           key=lambda x: x[1]['metadata']['created'], 
                           reverse=True)
        for path_key, path_display in sorted_auto:
            all_paths_ordered[f"🔄 {path_key}"] = path_display['path']
    
    if manually_saved_paths:
        st.write("💾 **Manually saved:**")
        for path_key, path_display in manually_saved_paths.items():
            all_paths_ordered[f"💾 {path_key}"] = path_display['path']
    
    if not all_paths_ordered:
        st.error("No valid paths found.")
        return
    
    selected_path_key = st.selectbox(
        "Choose a price path:",
        options=list(all_paths_ordered.keys()),
        help="Select a saved price path to simulate order execution on. Auto-saved paths from GBM Explorer are shown first."
    )
    
    selected_path = all_paths_ordered[selected_path_key]
    
    # Show path preview with enhanced information
    with st.expander("📊 Price Path Preview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Volatility**: {selected_path.volatility:.3f}")
            st.write(f"**Initial Price**: ${selected_path.initial_price:.2f}")
            st.write(f"**Final Price**: ${selected_path.final_price:.2f}")
        with col2:
            st.write(f"**Duration**: {selected_path.horizon:.0f} seconds")
            st.write(f"**Time Step**: {selected_path.dt:.1f} seconds")
            st.write(f"**Total Return**: {selected_path.total_return:.2f}%")
        
        # Show if this was auto-saved
        if selected_path_key.startswith("🔄"):
            st.info("ℹ️ This path was auto-saved from the GBM Explorer")
    
    st.divider()
    
    # Order configuration
    st.subheader("📋 Order Configuration")
    
    # Order type selection
    col1, col2 = st.columns(2)
    with col1:
        order_type = st.selectbox(
            "Order Type",
            options=["market", "dutch"],
            format_func=lambda x: "Market Order" if x == "market" else "Dutch Auction Order",
            help="Choose between immediate market order or Dutch auction order with decaying limit price"
        )
    
    with col2:
        order_side = st.selectbox(
            "Order Side",
            options=["SELL", "BUY"],
            index=0,
            help="Direction of the order"
        )
    
    # Basic order parameters
    col1, col2 = st.columns(2)
    
    with col1:
        order_timing = st.slider(
            "Order Creation Time (seconds)",
            min_value=1.0,
            max_value=float(selected_path.horizon) - 10.0,
            value=50.0,
            step=1.0,
            help="When to place the order"
        )
    
    with col2:
        order_quantity = st.number_input(
            "Order Quantity",
            min_value=0.1,
            value=10.0,
            step=0.1,
            help="Size of the order"
        )
    
    # Dutch order specific parameters
    if order_type == "dutch":
        st.write("**🔄 Dutch Auction Parameters**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            starting_limit_offset = st.number_input(
                f"Starting Limit Offset (%)",
                min_value=0.1,
                max_value=100.0,
                value=50.0,
                step=0.1,
                help=f"Percentage {'above' if order_side == 'SELL' else 'below'} mid price at creation time to start the limit. Higher = more aggressive"
            )
        
        with col2:
            decay_rate = st.number_input(
                "Decay Rate (%)",
                min_value=0.001,
                max_value=10.0,
                value=1.0,
                step=0.001,
                format="%.3f",
                help="Percentage of starting limit price that decays per second toward mid price"
            )
        
        with col3:
            order_duration = st.number_input(
                "Order Duration (seconds)",
                min_value=10.0,
                max_value=float(selected_path.horizon) - order_timing - 1.0,
                value=min(140.0, float(selected_path.horizon) - order_timing - 10.0),
                step=5.0,
                help="Maximum time the order can remain active"
            )
        
        # Show Dutch order preview - estimate using price at order creation time
        # Get approximate mid price at order creation time from path
        creation_time_index = int(order_timing / selected_path.dt)
        if creation_time_index < len(selected_path.prices):
            creation_mid_price = selected_path.prices[creation_time_index]
        else:
            creation_mid_price = selected_path.prices[-1]  # Use final price if beyond path
        
        if order_side == "SELL":
            starting_limit = creation_mid_price * (1 + starting_limit_offset / 100)
            final_limit = starting_limit * (1 - (decay_rate / 100) * order_duration)
        else:
            starting_limit = creation_mid_price * (1 - starting_limit_offset / 100)  
            final_limit = starting_limit * (1 + (decay_rate / 100) * order_duration)
        
        st.info(f"""
        **Dutch Order Preview:**
        - **Starting Limit**: ${starting_limit:.2f} ({'+' if starting_limit > creation_mid_price else ''}{(starting_limit - creation_mid_price)/creation_mid_price*100:+.1f}% vs mid at creation)
        - **Final Limit**: ${final_limit:.2f} (after {order_duration:.0f}s decay)
        - **Decay Direction**: {'Downward' if order_side == 'SELL' else 'Upward'} at {decay_rate:.3f}%/second
        - **Mid Price at Creation**: ${creation_mid_price:.2f} (at t={order_timing:.0f}s)
        """)
    else:
        # Set default values for market orders
        starting_limit_offset = 0.0
        decay_rate = 0.0
        order_duration = 60.0
    
    # Execute simulation
    if st.button("🚀 Execute Order Simulation", type="primary"):
        with st.spinner("Running single order execution simulation..."):
            try:
                side = Side.SELL if order_side == "SELL" else Side.BUY
                execution_result = run_single_order_simulation(
                    selected_path,
                    order_timing,
                    order_quantity,
                    side,
                    order_type=order_type,
                    starting_limit_offset=starting_limit_offset,
                    decay_rate=decay_rate,
                    order_duration=order_duration
                )
                
                if execution_result and execution_result.success:
                    st.success("✅ Order execution simulation completed!")
                    
                    # Store result in session state for persistence
                    st.session_state['last_execution_result'] = execution_result
                    st.session_state['last_execution_path'] = selected_path
                    
                else:
                    st.error("❌ Order execution failed or no fills occurred.")
                    return
                    
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                return
    
    # Display results if available
    if 'last_execution_result' in st.session_state and 'last_execution_path' in st.session_state:
        execution_result = st.session_state['last_execution_result']
        path_data = st.session_state['last_execution_path']
        
        st.divider()
        st.subheader("📊 Execution Results")
        
        # Render metrics
        render_execution_metrics(execution_result)
        
        st.divider()
        
        # Render visualization
        st.subheader("📈 Execution Visualization")
        render_execution_visualization(execution_result, path_data)
        
        # Additional analysis
        st.divider()
        st.subheader("🔍 Analysis")
        
        # Display different analysis based on order type
        if isinstance(execution_result, DutchOrderExecutionResult):
            st.info(f"""
            **Dutch Auction Analysis**:
            - **Creation Mid Price**: ${execution_result.order_creation_mid:.2f}
            - **Starting Limit**: ${execution_result.starting_limit_price:.2f}
            - **Theoretical Limit at Fill**: ${execution_result.theoretical_limit_at_fill:.2f}
            - **Actual Execution Price**: ${execution_result.execution_price:.2f}
            - **Implementation Shortfall**: {execution_result.implementation_shortfall_bps:.1f} basis points
            - **Time to Fill**: {execution_result.time_to_execution:.1f} seconds
            - **Decay Duration**: {execution_result.time_to_execution:.1f} / {execution_result.order_duration:.0f} seconds
            
            **Dutch Order Performance**: The order decayed from ${execution_result.starting_limit_price:.2f} to ${execution_result.theoretical_limit_at_fill:.2f} 
            over {execution_result.time_to_execution:.1f} seconds before filling at ${execution_result.execution_price:.2f}.
            Price improvement vs starting limit: ${execution_result.price_improvement_vs_limit:+.4f}
            """)
            
            # Add verification analysis for Dutch orders
            st.divider()
            render_dutch_verification_analysis(execution_result, path_data)
        else:
            st.info(f"""
            **Market Order Analysis**:
            - **Creation Mid Price**: ${execution_result.order_creation_mid:.2f}
            - **Execution Price**: ${execution_result.execution_price:.2f}
            - **Price Difference**: ${execution_result.implementation_shortfall:.4f}
            - **Shortfall**: {execution_result.implementation_shortfall_bps:.1f} basis points
            - **Time to Fill**: {execution_result.time_to_execution:.1f} seconds
            
            **Implementation Shortfall**: Measures the difference between execution price and mid price at order creation.
            Analysis depends on order side and market conditions at execution time.
            """) 