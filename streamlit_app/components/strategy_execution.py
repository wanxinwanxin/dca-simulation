"""Strategy Execution Component for Phase 2A Implementation.

This component provides:
- Strategy configuration interface (TWAP, Dutch)
- Strategy execution on saved price paths  
- Basic execution visualization with order timeline and fill markers
- Performance comparison between strategies
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# Import our simulation framework
from src.strategy.twap_market import TwapMarket
from src.strategy.true_dutch_limit import TrueDutchLimit
from src.core.orders import Side
from src.test_utils.simulation import SimulationRunner, SimulationConfig
from src.test_utils.analyzers import BaseAnalyzer
from src.core.events import Fill


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    strategy_type: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        return cls(
            name=data["name"],
            strategy_type=data["strategy_type"],
            parameters=data["parameters"]
        )


@dataclass
class ExecutionResult:
    """Results from strategy execution."""
    strategy_config: StrategyConfig
    price_path_id: str
    fills: List[Fill]
    order_events: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    execution_timeline: List[Dict[str, Any]]


class StrategyAnalyzer(BaseAnalyzer):
    """Analyzer that captures detailed execution events."""
    
    def __init__(self):
        super().__init__()
        self.order_events = []
        self.price_history = []
        
    def record_order_creation(self, timestamp: float, order_id: str, market_price: float):
        """Record order creation event."""
        self.order_events.append({
            "timestamp": timestamp,
            "event_type": "order_created",
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
        if not self.fills:
            return {
                "fills": [],
                "order_events": self.order_events,
                "price_history": self.price_history,
                "fill_rate": 0.0,
                "avg_fill_price": None,
                "total_filled": 0.0,
                "execution_span": 0.0
            }
        
        total_filled = sum(fill.qty for fill in self.fills)
        avg_fill_price = sum(fill.price * fill.qty for fill in self.fills) / total_filled if total_filled > 0 else None
        
        fill_times = [fill.timestamp for fill in self.fills]
        execution_span = max(fill_times) - min(fill_times) if len(fill_times) > 1 else 0.0
        
        return {
            "fills": [{"timestamp": f.timestamp, "qty": f.qty, "price": f.price, "order_id": f.order_id, "gas_paid": f.gas_paid} for f in self.fills],
            "order_events": self.order_events,
            "price_history": self.price_history,
            "fill_rate": len(self.fills) / max(len(self.order_events), 1),
            "avg_fill_price": avg_fill_price,
            "total_filled": total_filled,
            "execution_span": execution_span
        }


def render_strategy_configuration():
    """Render the strategy configuration interface."""
    st.subheader("🎯 Strategy Configuration")
    
    # Strategy type selection
    strategy_type = st.selectbox(
        "Strategy Type",
        ["TWAP Market", "Dutch Limit"],
        help="Choose the execution strategy type"
    )
    
    # Strategy name
    strategy_name = st.text_input(
        "Strategy Name",
        value=f"{strategy_type.replace(' ', '_').lower()}_{len(st.session_state.get('configured_strategies', {}))}"
    )
    
    col1, col2 = st.columns(2)
    
    if strategy_type == "TWAP Market":
        with col1:
            total_qty = st.number_input(
                "Total Quantity",
                min_value=1.0,
                max_value=10000.0,
                value=1000.0,
                step=100.0,
                help="Total quantity to execute"
            )
            
            n_slices = st.number_input(
                "Number of Slices",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of equal-sized orders"
            )
        
        with col2:
            interval = st.number_input(
                "Interval (seconds)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Time between orders"
            )
            
            side = st.selectbox(
                "Side",
                ["SELL", "BUY"],
                help="Order side"
            )
        
        # Calculate and display derived metrics
        st.info(f"""
        **Strategy Summary:**
        - Slice size: {total_qty / n_slices:.1f} units
        - Total duration: {(n_slices - 1) * interval:.0f} seconds
        - Execution rate: {total_qty / ((n_slices - 1) * interval + 1):.2f} units/second
        """)
        
        config = StrategyConfig(
            name=strategy_name,
            strategy_type="TWAP",
            parameters={
                "total_qty": total_qty,
                "n_slices": n_slices,
                "interval": interval,
                "side": Side.SELL if side == "SELL" else Side.BUY
            }
        )
    
    elif strategy_type == "Dutch Limit":
        with col1:
            total_qty = st.number_input(
                "Total Quantity",
                min_value=1.0,
                max_value=10000.0,
                value=1000.0,
                step=100.0,
                help="Total quantity to execute"
            )
            
            slice_qty = st.number_input(
                "Slice Quantity", 
                min_value=1.0,
                max_value=total_qty,
                value=min(100.0, total_qty),
                step=10.0,
                help="Quantity per order"
            )
        
        with col2:
            starting_limit_price = st.number_input(
                "Starting Limit Price ($)",
                min_value=1.0,
                max_value=200.0,
                value=102.0,
                step=0.5,
                help="Initial limit price (above market for sells)"
            )
            
            decay_rate = st.number_input(
                "Decay Rate ($/second)",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Price decay per second"
            )
            
            order_duration = st.number_input(
                "Order Duration (seconds)",
                min_value=1.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                help="Individual order lifetime"
            )
            
            side = st.selectbox(
                "Side", 
                ["SELL", "BUY"],
                help="Order side"
            )
        
        # Calculate and display derived metrics
        n_orders = int(np.ceil(total_qty / slice_qty))
        final_price = starting_limit_price - decay_rate * order_duration
        
        st.info(f"""
        **Strategy Summary:**
        - Number of orders: {n_orders}
        - Final limit price: ${final_price:.2f}
        - Price range: ${final_price:.2f} - ${starting_limit_price:.2f}
        - Max execution time: {n_orders * order_duration:.0f} seconds
        """)
        
        config = StrategyConfig(
            name=strategy_name,
            strategy_type="DUTCH",
            parameters={
                "total_qty": total_qty,
                "slice_qty": slice_qty,
                "side": Side.SELL if side == "SELL" else Side.BUY,
                "starting_limit_price": starting_limit_price,
                "decay_rate": decay_rate,
                "order_duration": order_duration
            }
        )
    
    # Save strategy configuration
    if st.button("💾 Save Strategy Configuration", type="primary"):
        if 'configured_strategies' not in st.session_state:
            st.session_state.configured_strategies = {}
        
        st.session_state.configured_strategies[strategy_name] = config
        st.success(f"Strategy '{strategy_name}' saved!")
        st.rerun()
    
    return config


def render_path_selection():
    """Render price path selection interface."""
    st.subheader("📈 Select Price Path")
    
    # Check if we have saved paths
    if 'path_collections' not in st.session_state or not st.session_state.path_collections:
        st.warning("No saved price paths available. Please generate and save paths in the GBM Explorer first.")
        return None
    
    # Collection selection
    collection_names = list(st.session_state.path_collections.keys())
    selected_collection = st.selectbox(
        "Path Collection",
        collection_names,
        help="Choose a collection of saved price paths"
    )
    
    collection = st.session_state.path_collections[selected_collection]
    
    # Path selection within collection
    path_options = [f"Path {i} (σ={path.volatility:.3f})" for i, path in enumerate(collection['paths'])]
    selected_path_idx = st.selectbox(
        "Specific Path",
        range(len(path_options)),
        format_func=lambda x: path_options[x],
        help="Choose a specific price path for strategy execution"
    )
    
    selected_path = collection['paths'][selected_path_idx]
    
    # Display path preview
    with st.expander("📊 Path Preview"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(selected_path.prices))),
            y=selected_path.prices,
            mode='lines',
            name=f'Path {selected_path_idx}',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Selected Price Path",
            xaxis_title="Time Step",
            yaxis_title="Price ($)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Path statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Price", f"${selected_path.prices[0]:.2f}")
        with col2:
            st.metric("Final Price", f"${selected_path.final_price:.2f}")
        with col3:
            st.metric("Volatility", f"{selected_path.volatility:.3f}")
        with col4:
            st.metric("Total Return", f"{selected_path.total_return:.1f}%")
    
    return {
        "collection_name": selected_collection,
        "path_index": selected_path_idx,
        "path": selected_path
    }


def execute_strategy_on_path(strategy_config: StrategyConfig, path_data: Dict) -> ExecutionResult:
    """Execute a strategy on a specific price path."""
    
    # Create strategy instance
    if strategy_config.strategy_type == "TWAP":
        strategy = TwapMarket(**strategy_config.parameters)
    elif strategy_config.strategy_type == "DUTCH":
        strategy = TrueDutchLimit(**strategy_config.parameters)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_config.strategy_type}")
    
    # Create analyzer for tracking execution
    analyzer = StrategyAnalyzer()
    
    # Set up simulation configuration
    path = path_data["path"]
    sim_config = SimulationConfig(
        horizon=len(path.prices),
        dt=1.0,
        initial_price=path.prices[0],
        volatility=path.volatility,
        random_seed=42  # Use fixed seed for reproducible results
    )
    
    # Create controlled price process from saved path
    class ControlledGBM:
        def __init__(self, prices):
            self.prices = prices
            self.current_step = 0
        
        def mid_price(self, t: float) -> float:
            step = int(t)
            if step >= len(self.prices):
                return self.prices[-1]
            return self.prices[step]
    
    # Run simulation with controlled price process
    runner = SimulationRunner(config=sim_config)
    
    # Override the price process with our controlled one
    controlled_process = ControlledGBM(path.prices)
    
    # Create engine manually to use controlled price process
    liquidity_model, impact_model = runner.create_market_models()
    gas_model, filler_decision = runner.create_cost_models()
    
    from src.engine.matching import MatchingEngine
    
    engine = MatchingEngine(
        price_process=controlled_process,
        liquidity_model=liquidity_model,
        impact_model=impact_model,
        gas_model=gas_model,
        filler_decision=filler_decision,
        probes=[analyzer],
        time_step=1.0
    )
    
    # Run the simulation
    results = engine.run(
        algo=strategy,
        target_qty=strategy_config.parameters.get("total_qty", 1000.0),
        horizon=len(path.prices)
    )
    
    # Get analyzer results
    analysis = analyzer.final()
    
    # Calculate performance metrics
    if analysis["fills"]:
        avg_fill_price = analysis["avg_fill_price"]
        
        # TWAP benchmark (simple average of prices during execution period)
        fill_times = [fill["timestamp"] for fill in analysis["fills"]]
        if fill_times:
            start_time = min(fill_times)
            end_time = max(fill_times)
            twap_benchmark = np.mean([
                path.prices[int(t)] for t in range(int(start_time), min(int(end_time) + 1, len(path.prices)))
            ])
            
            performance_metrics = {
                "avg_fill_price": avg_fill_price,
                "twap_benchmark": twap_benchmark,
                "price_improvement": ((twap_benchmark - avg_fill_price) / twap_benchmark * 100) if strategy_config.parameters.get("side") == Side.SELL else ((avg_fill_price - twap_benchmark) / twap_benchmark * 100),
                "total_filled": analysis["total_filled"],
                "fill_rate": analysis["fill_rate"],
                "execution_span": analysis["execution_span"]
            }
        else:
            performance_metrics = {"error": "No fills recorded"}
    else:
        performance_metrics = {"error": "No fills recorded"}
    
    return ExecutionResult(
        strategy_config=strategy_config,
        price_path_id=f"{path_data['collection_name']}_{path_data['path_index']}",
        fills=analyzer.fills,
        order_events=analysis["order_events"],
        performance_metrics=performance_metrics,
        execution_timeline=analysis.get("price_history", [])
    )


def visualize_execution_result(execution_result: ExecutionResult, path_data: Dict):
    """Create visualization of strategy execution on price path."""
    
    path = path_data["path"]
    
    # Create subplot with price and execution events
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Price Path with Execution Events", "Order Timeline"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Main price path
    fig.add_trace(
        go.Scatter(
            x=list(range(len(path.prices))),
            y=path.prices,
            mode='lines',
            name='Price Path',
            line=dict(color='blue', width=2),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add fill markers if we have fills
    if execution_result.fills:
        fill_times = [fill.timestamp for fill in execution_result.fills]
        fill_prices = [fill.price for fill in execution_result.fills]
        fill_qtys = [fill.qty for fill in execution_result.fills]
        
        fig.add_trace(
            go.Scatter(
                x=fill_times,
                y=fill_prices,
                mode='markers',
                name='Fills',
                marker=dict(
                    size=[max(8, min(20, qty/50)) for qty in fill_qtys],  # Size based on quantity
                    color='red',
                    symbol='circle',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"Fill: {qty:.0f} @ ${price:.2f}" for qty, price in zip(fill_qtys, fill_prices)],
                hoverinfo='text'
            ),
            row=1, col=1
        )
    
    # Add order creation markers
    if execution_result.order_events:
        order_times = [event["timestamp"] for event in execution_result.order_events if event["event_type"] == "order_created"]
        order_prices = [event["market_price"] for event in execution_result.order_events if event["event_type"] == "order_created"]
        
        if order_times:
            fig.add_trace(
                go.Scatter(
                    x=order_times,
                    y=order_prices,
                    mode='markers',
                    name='Order Created',
                    marker=dict(
                        size=8,
                        color='orange',
                        symbol='diamond',
                        line=dict(width=1, color='darkorange')
                    ),
                    text=[f"Order created @ ${price:.2f}" for price in order_prices],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
    
    # Order timeline (simplified)
    if execution_result.order_events:
        event_times = [event["timestamp"] for event in execution_result.order_events]
        event_types = [event["event_type"] for event in execution_result.order_events]
        
        y_pos = [1 if event == "order_created" else 0.5 for event in event_types]
        colors = ['orange' if event == "order_created" else 'red' for event in event_types]
        
        fig.add_trace(
            go.Scatter(
                x=event_times,
                y=y_pos,
                mode='markers',
                name='Events Timeline',
                marker=dict(size=8, color=colors),
                text=[f"{event.replace('_', ' ').title()} @ t={time:.1f}" for event, time in zip(event_types, event_times)],
                hoverinfo='text',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"Strategy Execution: {execution_result.strategy_config.name}",
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Event Type", row=2, col=1, ticktext=["Fill", "Order"], tickvals=[0.5, 1])
    
    st.plotly_chart(fig, use_container_width=True)


def render_execution_results(execution_result: ExecutionResult):
    """Render execution results and performance metrics."""
    
    st.subheader("📊 Execution Results")
    
    metrics = execution_result.performance_metrics
    
    if "error" in metrics:
        st.error(f"Execution failed: {metrics['error']}")
        return
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Fill Price",
            f"${metrics.get('avg_fill_price', 0):.2f}" if metrics.get('avg_fill_price') else "N/A"
        )
    
    with col2:
        st.metric(
            "TWAP Benchmark", 
            f"${metrics.get('twap_benchmark', 0):.2f}" if metrics.get('twap_benchmark') else "N/A"
        )
    
    with col3:
        improvement = metrics.get('price_improvement', 0)
        st.metric(
            "Price Improvement",
            f"{improvement:+.2f}%" if improvement else "N/A",
            delta=f"{improvement:.2f}%" if improvement else None
        )
    
    with col4:
        st.metric(
            "Fill Rate",
            f"{metrics.get('fill_rate', 0):.1%}"
        )
    
    # Detailed execution information
    with st.expander("📋 Detailed Execution Info"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strategy Configuration:**")
            st.json(execution_result.strategy_config.to_dict())
        
        with col2:
            st.write("**Performance Metrics:**")
            st.json(metrics)


def render_strategy_execution():
    """Main strategy execution interface."""
    
    st.title("🎯 Strategy Execution")
    st.markdown("Execute trading strategies on saved price paths and analyze performance.")
    
    # Strategy configuration section
    with st.container():
        strategy_config = render_strategy_configuration()
    
    st.divider()
    
    # Path selection section
    path_data = render_path_selection()
    
    if path_data is None:
        return
    
    st.divider()
    
    # Execute strategy
    if st.button("🚀 Execute Strategy", type="primary", disabled=not path_data):
        with st.spinner("Running strategy execution..."):
            try:
                execution_result = execute_strategy_on_path(strategy_config, path_data)
                
                # Store result in session state
                if 'execution_results' not in st.session_state:
                    st.session_state.execution_results = []
                
                st.session_state.execution_results.append(execution_result)
                
                # Display results
                st.success("Strategy execution completed!")
                
                # Visualization
                visualize_execution_result(execution_result, path_data)
                
                # Results table
                render_execution_results(execution_result)
                
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")
                st.write("**Debug info:**")
                st.write(f"Strategy type: {strategy_config.strategy_type}")
                st.write(f"Path length: {len(path_data['path'].prices)}")
                
    # Show saved configured strategies
    if 'configured_strategies' in st.session_state and st.session_state.configured_strategies:
        st.divider()
        st.subheader("💾 Saved Strategy Configurations")
        
        for name, config in st.session_state.configured_strategies.items():
            with st.expander(f"📋 {name} ({config.strategy_type})"):
                st.json(config.to_dict()) 