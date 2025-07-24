"""Impact Model Explorer for DCA Simulation.

This component provides visualization and analysis of a selected market impact model
to help users understand its behavior and effects on order execution.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.market.linear_impact import LinearImpact
from src.market.percentage_impact import PercentageImpact
from src.market.realistic_impact import RealisticImpact
from src.core.orders import Order, Side


def create_test_order(side: Side, qty: float, order_id: str = "test") -> Order:
    """Create a test order for impact analysis."""
    return Order(
        id=order_id,
        side=side,
        qty=qty,
        limit_px=None,  # Market order
        placed_at=0.0,
        valid_to=float('inf')  # Order doesn't expire
    )


def analyze_single_model(model, model_name: str, mid_price: float, quantities: np.ndarray) -> pd.DataFrame:
    """Analyze a single impact model behavior across different order sizes."""
    results = []
    
    for qty in quantities:
        # Test both buy and sell orders
        for side in [Side.BUY, Side.SELL]:
            order = create_test_order(side, qty)
            exec_price = model.exec_price(order, mid_price)
            
            impact_abs = abs(exec_price - mid_price)
            impact_pct = (impact_abs / mid_price) * 100
            
            # Check for problematic scenarios
            is_negative = exec_price <= 0
            is_extreme = impact_pct > 50  # More than 50% impact
            
            results.append({
                'side': side.value,
                'quantity': qty,
                'mid_price': mid_price,
                'exec_price': exec_price,
                'impact_abs': impact_abs,
                'impact_pct': impact_pct,
                'is_negative': is_negative,
                'is_extreme': is_extreme
            })
    
    return pd.DataFrame(results)


def render_model_behavior_chart(analysis_data: pd.DataFrame, model_name: str) -> go.Figure:
    """Create a chart showing execution prices for the selected model."""
    fig = go.Figure()
    
    # Separate buy and sell
    buy_data = analysis_data[analysis_data['side'] == 'BUY']
    sell_data = analysis_data[analysis_data['side'] == 'SELL']
    
    # Add mid price reference line
    mid_price = analysis_data['mid_price'].iloc[0]
    fig.add_hline(
        y=mid_price, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Mid Price",
        annotation_position="left"
    )
    
    # Add buy line
    fig.add_trace(go.Scatter(
        x=buy_data['quantity'],
        y=buy_data['exec_price'],
        mode='lines+markers',
        name='BUY Orders',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(symbol='triangle-up', size=8),
        hovertemplate='<b>BUY Order</b><br>' +
                     'Quantity: %{x}<br>' +
                     'Execution Price: $%{y:.2f}<br>' +
                     'Impact: $%{customdata:.2f}<br>' +
                     '<extra></extra>',
        customdata=buy_data['impact_abs']
    ))
    
    # Add sell line
    fig.add_trace(go.Scatter(
        x=sell_data['quantity'],
        y=sell_data['exec_price'],
        mode='lines+markers', 
        name='SELL Orders',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(symbol='triangle-down', size=8),
        hovertemplate='<b>SELL Order</b><br>' +
                     'Quantity: %{x}<br>' +
                     'Execution Price: $%{y:.2f}<br>' +
                     'Impact: $%{customdata:.2f}<br>' +
                     '<extra></extra>',
        customdata=sell_data['impact_abs']
    ))
    
    fig.update_layout(
        title=f"{model_name} Impact Model - Execution Prices",
        xaxis_title="Order Quantity",
        yaxis_title="Execution Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    # Add annotations for negative prices if any
    negative_data = analysis_data[analysis_data['is_negative']]
    if len(negative_data) > 0:
        fig.add_annotation(
            x=0.98, y=0.02,
            xref="paper", yref="paper",
            text="⚠️ Negative prices detected!",
            showarrow=False,
            bgcolor="rgba(255,255,0,0.8)",
            bordercolor="red",
            borderwidth=1
        )
    
    return fig


def render_impact_analysis_chart(analysis_data: pd.DataFrame, model_name: str) -> go.Figure:
    """Create a chart showing impact percentage for the selected model."""
    fig = go.Figure()
    
    # Separate buy and sell
    buy_data = analysis_data[analysis_data['side'] == 'BUY'] 
    sell_data = analysis_data[analysis_data['side'] == 'SELL']
    
    # Add buy impact
    fig.add_trace(go.Scatter(
        x=buy_data['quantity'],
        y=buy_data['impact_pct'],
        mode='lines+markers',
        name='BUY Impact',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(symbol='triangle-up', size=8),
        hovertemplate='<b>BUY Impact</b><br>' +
                     'Quantity: %{x}<br>' +
                     'Impact: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Add sell impact  
    fig.add_trace(go.Scatter(
        x=sell_data['quantity'],
        y=sell_data['impact_pct'],
        mode='lines+markers',
        name='SELL Impact', 
        line=dict(color='#4ECDC4', width=3),
        marker=dict(symbol='triangle-down', size=8),
        hovertemplate='<b>SELL Impact</b><br>' +
                     'Quantity: %{x}<br>' +
                     'Impact: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Add warning zones
    fig.add_hline(y=10, line_dash="dot", line_color="orange", 
                  annotation_text="10% Impact", annotation_position="right")
    fig.add_hline(y=25, line_dash="dot", line_color="red",
                  annotation_text="25% Impact (High)", annotation_position="right")
    
    fig.update_layout(
        title=f"{model_name} Impact Model - Impact Percentage",
        xaxis_title="Order Quantity",
        yaxis_title="Impact (%)",
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig


def get_model_info(model_name: str) -> Dict[str, str]:
    """Get information about a specific model."""
    model_info = {
        "Linear": {
            "description": "Impact increases linearly with order size: impact = γ × quantity",
            "pros": "• Simple to understand and implement\n• Predictable behavior",
            "cons": "• Can produce negative prices for large orders\n• Impact not dependent on price level\n• Unrealistic for very large orders",
            "recommendation": "⚠️ Use with caution - monitor for negative prices",
            "color": "orange"
        },
        "Percentage": {
            "description": "Impact as percentage of price: impact = γ × quantity × price",
            "pros": "• Impact scales with price level\n• More realistic than linear\n• Less likely to produce negative prices",
            "cons": "• Still linear in quantity\n• May not capture complex market dynamics",
            "recommendation": "✅ Recommended for most use cases",
            "color": "green"
        },
        "Realistic": {
            "description": "Combines bid-ask spread with linear impact: spread + γ × quantity",
            "pros": "• Includes realistic bid-ask spread\n• Accounts for immediate liquidity costs",
            "cons": "• Still has linear impact component\n• Can produce negative prices for large orders",
            "recommendation": "⚠️ Better than linear, but watch for edge cases",
            "color": "yellow"
        }
    }
    return model_info.get(model_name, {})


def render_model_insights(analysis_data: pd.DataFrame, model_name: str) -> None:
    """Render insights and analysis for the selected model."""
    st.subheader(f"📊 {model_name} Model Analysis")
    
    # Model information
    model_info = get_model_info(model_name)
    
    if model_info:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Description:** {model_info['description']}")
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.markdown("**✅ Pros:**")
                st.markdown(model_info['pros'])
            
            with subcol2:
                st.markdown("**❌ Cons:**")
                st.markdown(model_info['cons'])
        
        with col2:
            color = model_info['color']
            if color == "green":
                st.success(model_info['recommendation'])
            elif color == "yellow":
                st.warning(model_info['recommendation'])
            else:
                st.error(model_info['recommendation'])
    
    # Statistics
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_impact = analysis_data['impact_pct'].max()
        st.metric("Max Impact", f"{max_impact:.1f}%")
    
    with col2:
        negative_count = analysis_data['is_negative'].sum()
        st.metric("Negative Prices", negative_count, 
                 delta="❌ Issues" if negative_count > 0 else "✅ OK")
    
    with col3:
        extreme_count = analysis_data['is_extreme'].sum()
        st.metric("Extreme Impact (>50%)", extreme_count,
                 delta="⚠️ High Risk" if extreme_count > 0 else "✅ OK")
    
    with col4:
        avg_impact = analysis_data['impact_pct'].mean()
        st.metric("Average Impact", f"{avg_impact:.1f}%")


def render_impact_model_explorer():
    """Main render function for the Impact Model Explorer."""
    st.header("📊 Impact Model Explorer")
    st.markdown("""
    Explore how a specific market impact model affects order execution prices.
    Select a model and parameters to understand its behavior across different order sizes.
    """)
    
    # Model selection
    st.subheader("🔧 Model Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Impact Model",
            ["Linear", "Percentage", "Realistic"],
            index=1,  # Default to Percentage (recommended)
            help="Choose which impact model to analyze"
        )
        
        mid_price = st.number_input(
            "Mid Price ($)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            help="Reference mid-market price for impact calculations"
        )
        
        max_quantity = st.number_input(
            "Max Order Size", 
            min_value=1.0,
            max_value=10000.0,
            value=1000.0,
            step=10.0,
            help="Maximum order size to analyze"
        )
    
    with col2:
        # Model-specific parameters
        if model_type == "Linear":
            st.markdown("**Linear Model Parameters:**")
            gamma = st.number_input(
                "γ (Impact per unit)",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.001,
                format="%.3f",
                help="Linear impact coefficient: impact = γ × quantity"
            )
            spread = 0.0  # Not used for linear
            
        elif model_type == "Percentage":
            st.markdown("**Percentage Model Parameters:**")
            gamma = st.number_input(
                "γ (Impact coefficient)",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Percentage impact coefficient: impact = γ × quantity × price"
            )
            spread = st.number_input(
                "Half-spread ($)",
                min_value=0.01,
                max_value=10.0,
                value=0.5,
                step=0.01,
                help="Half bid-ask spread"
            )
            
        else:  # Realistic
            st.markdown("**Realistic Model Parameters:**")
            spread = st.number_input(
                "Half-spread ($)",
                min_value=0.01,
                max_value=10.0,
                value=0.5,
                step=0.01,
                help="Half bid-ask spread for realistic model"
            )
            gamma = st.number_input(
                "γ (Linear impact)",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.001,
                format="%.3f",
                help="Linear impact coefficient added to spread"
            )
    
    # Generate analysis
    if st.button("🔄 Analyze Model", type="primary"):
        with st.spinner(f"Analyzing {model_type} impact model..."):
            # Create the selected model
            if model_type == "Linear":
                model = LinearImpact(gamma=gamma)
            elif model_type == "Percentage":
                model = PercentageImpact(spread=spread, gamma=gamma)
            else:  # Realistic
                model = RealisticImpact(spread=spread, gamma=gamma)
            
            # Generate test quantities
            quantities = np.linspace(1, max_quantity, 50)
            
            # Analyze the model
            analysis_data = analyze_single_model(model, model_type, mid_price, quantities)
            
            # Store in session state
            st.session_state.impact_analysis = {
                'data': analysis_data,
                'model_name': model_type,
                'mid_price': mid_price,
                'gamma': gamma,
                'spread': spread if model_type != "Linear" else 0.0
            }
    
    # Display results if available
    if 'impact_analysis' in st.session_state:
        analysis = st.session_state.impact_analysis
        data = analysis['data']
        model_name = analysis['model_name']
        
        st.divider()
        
        # Model insights
        render_model_insights(data, model_name)
        
        st.divider()
        
        # Execution price chart
        st.subheader("📈 Execution Price Behavior")
        fig1 = render_model_behavior_chart(data, model_name)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Impact analysis chart
        st.subheader("📊 Impact Analysis")
        fig2 = render_impact_analysis_chart(data, model_name)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Detailed Data"):
            st.dataframe(
                data.style.format({
                    'exec_price': '${:.2f}',
                    'impact_abs': '${:.2f}',
                    'impact_pct': '{:.2f}%'
                }),
                use_container_width=True
            )
    
    else:
        st.info("👆 Select a model and parameters, then click 'Analyze Model' to see the results.")
        
        # Show model comparison info
        st.divider()
        st.subheader("📚 Model Comparison Guide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔴 Linear Impact**
            - Simple linear relationship
            - ⚠️ Can produce negative prices
            - Best for: Testing extreme scenarios
            """)
        
        with col2:
            st.markdown("""
            **🟢 Percentage Impact**
            - Impact scales with price
            - ✅ More realistic behavior
            - Best for: Most trading simulations
            """)
        
        with col3:
            st.markdown("""
            **🟡 Realistic Impact**
            - Includes bid-ask spread
            - ⚠️ Still has linear components
            - Best for: Spread-sensitive analysis
            """)


if __name__ == "__main__":
    render_impact_model_explorer() 