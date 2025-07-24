"""Main Streamlit application for DCA Simulation GBM Price Path Explorer.

This is the Phase 1 MVP implementation providing:
- Interactive GBM parameter controls (volatility 0.01-0.20, paths 1-100)
- Real-time price path visualization with confidence bands
- Path persistence via session state and JSON export
- Basic statistics display and detailed analysis

Usage:
    streamlit run streamlit_app/main.py
"""

import streamlit as st
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import from src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from streamlit_app.components.gbm_explorer import render_gbm_explorer
from streamlit_app.components.path_manager import render_path_manager
from streamlit_app.components.single_order_execution import render_single_order_execution
from streamlit_app.components.impact_model_explorer import render_impact_model_explorer


def configure_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="DCA Simulation - GBM Explorer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
            # DCA Simulation - Phase 1 MVP
            
            Interactive **Geometric Brownian Motion** price path exploration tool.
            
            **Features:**
            - Volatility control (0.01-0.20)
            - Path count selection (1-100)
            - Real-time visualization
            - Statistical analysis
            - Path persistence & export
            
            **Phase 1 Specifications:**
            - Fixed drift (μ=0), initial price ($100), time step (1.0), horizon (200)
            - Session state management
            - JSON export/import capabilities
            
            Built as part of the DCA Simulation project roadmap.
            """
        }
    )


def render_home_tab():
    """Render the home/cover tab that explains what each tab does."""
    st.header("🏠 Welcome to DCA Simulation Explorer")
    
    st.markdown("""
    Welcome to the **DCA Simulation Explorer** - an interactive tool for understanding trading strategies 
    and market dynamics. This application provides several specialized tools to help you explore and 
    analyze different aspects of algorithmic trading.
    """)
    
    st.divider()
    
    # Tab explanations
    st.subheader("📚 Available Tools")
    
    # Create columns for organized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 GBM Explorer
        **Purpose:** Generate and analyze price paths using Geometric Brownian Motion
        
        **What it does:**
        - Create multiple price paths with different volatility settings
        - Visualize statistical properties and confidence bands
        - Save and manage price path collections
        - Export data for further analysis
        
        **Best for:** Understanding how volatility affects price movements and creating test data
        
        ---
        
        ### 🎯 Single Order Execution
        **Purpose:** Simulate individual order execution on saved price paths
        
        **What it does:**
        - Execute market orders at specific times
        - Calculate implementation shortfall and performance metrics
        - Visualize execution timing and price impact
        - Analyze order performance across different scenarios
        
        **Best for:** Understanding how timing affects individual order execution
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Path Manager
        **Purpose:** Organize and compare your saved price path collections
        
        **What it does:**
        - View and organize all your saved price paths
        - Compare different path collections side-by-side
        - Export and import path data
        - Manage your simulation library
        
        **Best for:** Organizing your simulation data and comparing different scenarios
        
        ---
        
        ### 📊 Impact Model Explorer
        **Purpose:** Understand how market impact models affect order execution
        
        **What it does:**
        - Visualize the effect of different impact models on execution price
        - Test model behavior across different order sizes
        - Identify potential issues with model parameters
        - Get recommendations for model selection
        
        **Best for:** Understanding market impact and choosing appropriate models
        """)
    
    st.divider()
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    **New to the application?** Follow this recommended workflow:
    
    1. **📈 Start with GBM Explorer** - Generate some price paths with different volatility settings
    2. **📁 Check Path Manager** - See your saved paths and organize your collections  
    3. **🎯 Try Single Order Execution** - Test how orders perform on your saved paths
    4. **📊 Explore Impact Models** - Understand how market impact affects execution prices
    """)
    
    # Current session info
    st.divider()
    st.subheader("💾 Your Current Session")
    
    # Show cache status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'path_cache' in st.session_state:
            cache_count = len(st.session_state.path_cache)
            st.metric("Cached Configurations", cache_count)
        else:
            st.metric("Cached Configurations", 0)
    
    with col2:
        if 'path_collections' in st.session_state:
            collections_count = len(st.session_state.path_collections)
            st.metric("Saved Collections", collections_count)
        else:
            st.metric("Saved Collections", 0)
    
    with col3:
        if 'path_collections' in st.session_state:
            total_paths = sum(len(collection['paths']) for collection in st.session_state.path_collections.values())
            st.metric("Total Saved Paths", total_paths)
        else:
            st.metric("Total Saved Paths", 0)
    
    # Show current data if available
    if 'last_generated_paths' in st.session_state and st.session_state.last_generated_paths:
        st.success("✅ You have recently generated price paths - ready to explore!")
        
        paths = st.session_state.last_generated_paths
        if paths:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Current Paths:** {len(paths)}")
            with col2:
                st.info(f"**Volatility:** {paths[0].volatility:.3f}")
            with col3:
                final_prices = [p.final_price for p in paths]
                st.info(f"**Mean Final Price:** ${np.mean(final_prices):.2f}")
    else:
        st.info("💡 Start by generating some price paths in the **GBM Explorer** tab!")
    
    st.divider()
    
    # Tips and tricks
    st.subheader("💡 Tips & Tricks")
    
    with st.expander("📝 Usage Tips"):
        st.markdown("""
        - **Auto-save is enabled by default** in GBM Explorer - your paths are automatically saved
        - **Use the sidebar** for quick navigation and session information
        - **Export your data** as JSON to use in external analysis tools
        - **Compare different volatility settings** to understand market behavior
        - **Test extreme scenarios** with very low or high volatility settings
        - **Save interesting configurations** in Path Manager for future reference
        """)
    
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        **Fixed Parameters:**
        - Drift (μ): 0.0 (no trend)
        - Initial Price: $100
        - Time Step: 1.0
        - Horizon: 200 steps
        
        **Supported Models:**
        - Linear Impact
        - Percentage Impact  
        - Realistic Impact
        
        **Export Formats:**
        - JSON (recommended)
        - CSV (for spreadsheet analysis)
        """)


def render_sidebar():
    """Render the sidebar with navigation and information."""
    with st.sidebar:
        st.title("🎯 Navigation")
        
        # Page selection
        page = st.radio(
            "Select Tool:",
            ["🏠 Home", "📈 GBM Explorer", "📁 Path Manager", "🎯 Single Order Execution", "📊 Impact Model Explorer"],
            index=0,
            help="Choose the tool you want to use"
        )
        
        st.divider()
        
        # Features information
        st.subheader("ℹ️ Quick Reference")
        st.info("""
        **🏠 Home:** Overview and quick start guide
        
        **📈 GBM Explorer:** Generate price paths with volatility control
        
        **📁 Path Manager:** Organize and compare saved paths
        
        **🎯 Single Order:** Execute and analyze individual orders
        
        **📊 Impact Models:** Understand market impact effects
        """)
        
        st.divider()
        
        # Session state information
        st.subheader("💾 Session Info")
        
        # Show cache status
        if 'path_cache' in st.session_state:
            cache_count = len(st.session_state.path_cache)
            st.write(f"Cached configurations: {cache_count}")
        else:
            st.write("Cache: Empty")
        
        # Show saved collections
        if 'path_collections' in st.session_state:
            collections_count = len(st.session_state.path_collections)
            total_paths = sum(len(collection['paths']) for collection in st.session_state.path_collections.values())
            st.write(f"Saved collections: {collections_count}")
            st.write(f"Total saved paths: {total_paths}")
        else:
            st.write("Saved paths: None")
        
        # Quick actions
        st.divider()
        st.subheader("🚀 Quick Actions")
        
        if st.button("🔄 Reset All Data", help="Clear all cached data and saved paths"):
            # Clear all session state data
            keys_to_clear = ['path_cache', 'path_collections', 'saved_paths', 'last_generated_paths']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("All data cleared!")
            st.rerun()
        
        # System information
        st.divider()
        st.subheader("🔧 System Info")
        st.caption("""
        **Version:** Phase 1.5+  
        **Mode:** Development  
        **Backend:** Existing GBM implementation  
        **Frontend:** Streamlit + Plotly
        """)
        
        return page


def render_header():
    """Render the main application header."""
    st.title("🎯 DCA Simulation - GBM Price Path Explorer")
    st.markdown("""
    **Phase 1 MVP:** Interactive exploration of **Geometric Brownian Motion** price processes.  
    Generate multiple price paths, analyze statistical properties, and export data for further analysis.
    """)
    
    # Quick stats if we have current data
    if 'last_generated_paths' in st.session_state:
        paths = st.session_state.last_generated_paths
        if paths:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Paths", len(paths))
            with col2:
                st.metric("Volatility", f"{paths[0].volatility:.3f}")
            with col3:
                final_prices = [p.final_price for p in paths]
                st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
            with col4:
                returns = [p.total_return for p in paths]
                st.metric("Mean Return", f"{np.mean(returns):.1f}%")
    
    st.divider()


def main():
    """Main application entry point."""
    # Configure page
    configure_page()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Render main content
    render_header()
    
    # Route to the selected page
    if selected_page == "🏠 Home":
        render_home_tab()
    elif selected_page == "📈 GBM Explorer":
        render_gbm_explorer()
    elif selected_page == "📁 Path Manager":
        # Pass current paths if available
        current_paths = st.session_state.get('last_generated_paths', [])
        render_path_manager(current_paths)
    elif selected_page == "🎯 Single Order Execution":
        render_single_order_execution()
    elif selected_page == "📊 Impact Model Explorer":
        render_impact_model_explorer()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
        DCA Simulation Project - Phase 1.5+ MVP<br>
        🚀 <strong>New:</strong> Impact Model Explorer for understanding market impact behavior<br>
        📧 Built with Streamlit + Plotly for interactive visualization
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 