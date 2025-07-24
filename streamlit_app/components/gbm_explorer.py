"""GBM Price Path Explorer component with parameter controls and visualization."""

import streamlit as st
import numpy as np
from typing import List, Optional, Tuple
import time
from datetime import datetime

from ..utils.simulation_bridge import GBMPathGenerator, PricePathResult, calculate_ensemble_statistics
from ..utils.plotting import (
    create_price_paths_chart, create_statistics_summary_chart, 
    display_path_statistics_table, create_downloadable_chart_data
)


class GBMExplorer:
    """Interactive GBM price path exploration component."""
    
    def __init__(self):
        self.generator = GBMPathGenerator()
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state for auto-save functionality."""
        if 'path_collections' not in st.session_state:
            st.session_state.path_collections = {}
        
        if 'saved_paths' not in st.session_state:
            st.session_state.saved_paths = {}
    
    def _auto_save_paths(self, paths: List[PricePathResult]) -> str:
        """Automatically save generated paths to collections."""
        if not paths:
            return ""
        
        # Generate descriptive collection name
        timestamp = datetime.now().strftime("%H%M%S")
        vol_str = f"{paths[0].volatility:.3f}".replace(".", "")
        collection_name = f"auto_vol{vol_str}_n{len(paths)}_{timestamp}"
        
        # Save to session state collections (same logic as PathManager)
        collection_data = {
            'paths': paths,
            'metadata': {
                'name': collection_name,
                'created': str(datetime.now()),
                'n_paths': len(paths),
                'volatility': paths[0].volatility,
                'base_seed': paths[0].seed - (paths[0].seed % 1000),  # Approximate base seed
                'auto_saved': True  # Mark as auto-saved
            }
        }
        
        st.session_state.path_collections[collection_name] = collection_data
        
        # Also add individual paths to saved_paths for compatibility
        for path in paths:
            st.session_state.saved_paths[path.path_id] = path
        
        return collection_name
        
    def render_parameter_controls(self) -> Tuple[float, int, int, bool, bool, bool]:
        """Render the parameter control interface."""
        
        st.subheader("🔧 GBM Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volatility slider (Phase 1 requirement: 0.01-0.20)
            volatility = st.slider(
                "Volatility (σ)",
                min_value=0.01,
                max_value=0.20,
                value=0.02,
                step=0.005,
                format="%.3f",
                help="Annual volatility parameter. Higher values create more price variation."
            )
            
            # Path count selector (Phase 1 requirement: 1-100)
            n_paths = st.slider(
                "Number of Paths",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                help="Number of price paths to generate. More paths show better statistical properties."
            )
        
        with col2:
            # Random seed for reproducibility
            base_seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=10000,
                value=42,
                step=1,
                help="Seed for random number generation. Same seed produces identical paths."
            )
            
            # Visualization options
            st.write("**Visualization Options**")
            show_confidence_bands = st.checkbox(
                "Show Confidence Bands",
                value=True,
                help="Display statistical confidence bands across all paths"
            )
            
            show_individual_paths = st.checkbox(
                "Show Individual Paths", 
                value=True,
                help="Display each individual price path"
            )
            
            # Auto-save option
            auto_save_enabled = st.checkbox(
                "🔄 Auto-save paths",
                value=True,
                help="Automatically save generated paths for use in Single Order Execution"
            )
        
        # Display fixed parameters (Phase 1 requirements)
        st.info("""
        **Fixed Parameters (Phase 1):**
        - Drift (μ): 0.0 (no trend)
        - Initial Price (S₀): $100.00
        - Time Step (dt): 1.0
        - Horizon: 200 time steps
        """)
        
        return volatility, n_paths, base_seed, show_confidence_bands, show_individual_paths, auto_save_enabled
    
    def generate_paths_with_caching(self, volatility: float, n_paths: int, base_seed: int, auto_save: bool = True) -> List[PricePathResult]:
        """Generate paths with intelligent caching to avoid unnecessary regeneration."""
        
        # Create cache key
        cache_key = f"vol_{volatility:.3f}_n_{n_paths}_seed_{base_seed}"
        
        # Initialize cache if needed
        if 'path_cache' not in st.session_state:
            st.session_state.path_cache = {}
        
        # Check if paths already exist in cache
        if cache_key in st.session_state.path_cache:
            cached_paths = st.session_state.path_cache[cache_key]
            
            # Auto-save cached paths if enabled and not already saved
            if auto_save:
                # Check if these paths are already in collections
                existing_collection = None
                for coll_name, coll_data in st.session_state.path_collections.items():
                    if (len(coll_data['paths']) == len(cached_paths) and 
                        coll_data['metadata']['volatility'] == volatility and
                        coll_data['metadata']['base_seed'] == base_seed):
                        existing_collection = coll_name
                        break
                
                if not existing_collection:
                    collection_name = self._auto_save_paths(cached_paths)
                    st.info(f"🔄 Auto-saved cached paths to collection '{collection_name}'")
            
            return cached_paths
        
        # Generate new paths with progress bar
        with st.spinner(f"Generating {n_paths} price paths..."):
            progress_bar = st.progress(0)
            start_time = time.time()
            
            # Generate paths
            paths = self.generator.generate_paths(
                n_paths=n_paths,
                volatility=volatility,
                base_seed=base_seed
            )
            
            progress_bar.progress(1.0)
            generation_time = time.time() - start_time
            
            # Cache the results
            st.session_state.path_cache[cache_key] = paths
            
            # Auto-save the newly generated paths
            if auto_save:
                collection_name = self._auto_save_paths(paths)
                st.success(f"Generated {n_paths} paths in {generation_time:.2f} seconds")
                st.info(f"🔄 Auto-saved to collection '{collection_name}' - ready for Single Order Execution!")
            else:
                st.success(f"Generated {n_paths} paths in {generation_time:.2f} seconds")
            
            # Clean up old cache entries to prevent memory issues
            if len(st.session_state.path_cache) > 10:
                # Remove oldest entries
                keys = list(st.session_state.path_cache.keys())
                for key in keys[:-10]:
                    del st.session_state.path_cache[key]
        
        return paths
    
    def render_main_visualization(self, paths: List[PricePathResult], 
                                show_confidence_bands: bool, 
                                show_individual_paths: bool) -> None:
        """Render the main price paths visualization."""
        
        st.subheader("📈 Price Path Visualization")
        
        if not paths:
            st.warning("No price paths to display. Please generate paths first.")
            return
        
        # Create and display the main chart
        fig = create_price_paths_chart(
            paths=paths,
            title=f"GBM Price Paths (σ={paths[0].volatility:.3f}, n={len(paths)})",
            show_individual_paths=show_individual_paths,
            show_confidence_bands=show_confidence_bands
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key statistics
        ensemble_stats = calculate_ensemble_statistics(paths)
        if ensemble_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Final Price",
                    f"${ensemble_stats['final_prices']['mean']:.2f}",
                    f"±${ensemble_stats['final_prices']['std']:.2f}"
                )
            
            with col2:
                st.metric(
                    "Mean Return",
                    f"{ensemble_stats['total_returns']['mean']:.2f}%",
                    f"±{ensemble_stats['total_returns']['std']:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Realized Volatility",
                    f"{ensemble_stats['realized_volatility']['mean']:.3f}",
                    f"Target: {ensemble_stats['realized_volatility']['target']:.3f}"
                )
            
            with col4:
                price_range = ensemble_stats['final_prices']['max'] - ensemble_stats['final_prices']['min']
                st.metric(
                    "Price Range",
                    f"${price_range:.2f}",
                    f"${ensemble_stats['final_prices']['min']:.2f} - ${ensemble_stats['final_prices']['max']:.2f}"
                )
    
    def render_detailed_statistics(self, paths: List[PricePathResult]) -> None:
        """Render detailed statistics section."""
        
        if not paths:
            return
        
        st.subheader("📊 Detailed Statistics")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Summary Charts", "Path Details", "Data Export"])
        
        with tab1:
            # Summary statistics chart
            stats_fig = create_statistics_summary_chart(paths)
            st.plotly_chart(stats_fig, use_container_width=True)
        
        with tab2:
            # Detailed path statistics table
            display_path_statistics_table(paths)
            
            # Path selector for highlighting
            if len(paths) > 1:
                st.subheader("Path Highlighting")
                selected_path_idx = st.selectbox(
                    "Select path to highlight in main chart:",
                    options=list(range(len(paths))),
                    format_func=lambda x: f"Path {x+1} (Return: {paths[x].total_return:.1f}%)"
                )
                
                # Store selected path for highlighting
                st.session_state.highlighted_path = selected_path_idx
        
        with tab3:
            # Data export options
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                if st.button("📄 Prepare CSV Download"):
                    df = create_downloadable_chart_data(paths)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"gbm_paths_{len(paths)}paths_vol{paths[0].volatility:.3f}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # JSON export
                if st.button("💾 Export to JSON"):
                    filename = f"gbm_paths_{len(paths)}paths_vol{paths[0].volatility:.3f}.json"
                    try:
                        export_path = self.generator.export_paths_to_json(paths, filename)
                        st.success(f"Exported to: {export_path}")
                        
                        # Add to saved paths
                        self.generator.save_paths_to_session(paths, st.session_state)
                        st.info(f"Added {len(paths)} paths to session state for future use.")
                        
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
    
    def render_generation_controls(self) -> Tuple[bool, bool]:
        """Render path generation controls."""
        
        st.subheader("🚀 Path Generation")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("Click **Generate Paths** to create new price paths with current parameters.")
            if st.session_state.get('path_collections'):
                num_collections = len(st.session_state.path_collections)
                total_paths = sum(len(coll['paths']) for coll in st.session_state.path_collections.values())
                st.caption(f"📁 {num_collections} saved collections with {total_paths} total paths")
        
        with col2:
            generate_button = st.button(
                "🎲 Generate Paths",
                type="primary",
                help="Generate new price paths with current parameters"
            )
        
        with col3:
            auto_generate = st.checkbox(
                "Auto-generate",
                value=False,
                help="Automatically regenerate paths when parameters change"
            )
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", help="Clear cached price paths to free memory"):
            if 'path_cache' in st.session_state:
                st.session_state.path_cache = {}
                st.success("Cache cleared!")
        
        return generate_button, auto_generate
    
    def render(self) -> None:
        """Render the complete GBM Explorer interface."""
        
        st.title("🎯 GBM Price Path Explorer")
        st.markdown("""
        Explore **Geometric Brownian Motion** price processes interactively. 
        Adjust volatility and path count to see how price paths evolve under different market conditions.
        
        🆕 **Auto-save enabled**: Generated paths are automatically saved for use in Single Order Execution!
        """)
        
        # Parameter controls
        volatility, n_paths, base_seed, show_confidence_bands, show_individual_paths, auto_save_enabled = self.render_parameter_controls()
        
        # Generation controls
        generate_button, auto_generate = self.render_generation_controls()
        
        # Determine if we should generate paths
        should_generate = generate_button or auto_generate
        
        # Generate paths if requested
        current_paths = []
        if should_generate:
            current_paths = self.generate_paths_with_caching(volatility, n_paths, base_seed, auto_save_enabled)
        elif 'last_generated_paths' in st.session_state:
            # Use previously generated paths if available
            current_paths = st.session_state.last_generated_paths
        
        # Store current paths for future use
        if current_paths:
            st.session_state.last_generated_paths = current_paths
        
        # Main visualization
        if current_paths:
            # Apply highlighting if selected
            highlight_idx = getattr(st.session_state, 'highlighted_path', None)
            if highlight_idx is not None and highlight_idx < len(current_paths):
                # Recreate chart with highlighting
                fig = create_price_paths_chart(
                    paths=current_paths,
                    title=f"GBM Price Paths (σ={current_paths[0].volatility:.3f}, n={len(current_paths)})",
                    show_individual_paths=show_individual_paths,
                    show_confidence_bands=show_confidence_bands,
                    highlight_path_idx=highlight_idx
                )
                st.subheader("📈 Price Path Visualization")
                st.plotly_chart(fig, use_container_width=True)
            else:
                self.render_main_visualization(current_paths, show_confidence_bands, show_individual_paths)
            
            # Detailed statistics
            self.render_detailed_statistics(current_paths)
        else:
            # Show instruction message
            st.info("👆 Click **Generate Paths** above to create and visualize GBM price paths!")
            
            # Show example configuration
            st.markdown("""
            ### 💡 Getting Started
            
            1. **Adjust Parameters**: Use the sliders to set volatility (0.01-0.20) and number of paths (1-100)
            2. **Generate Paths**: Click the generate button to create price paths
            3. **Auto-Save**: ✅ Paths are automatically saved for Single Order Execution
            4. **Explore Results**: View individual paths, confidence bands, and detailed statistics
            5. **Execute Orders**: Switch to Single Order Execution tab to simulate trades
            
            **Tip**: Start with low volatility (0.02) and 10 paths to see smooth price evolution.
            """)


def render_gbm_explorer():
    """Convenience function to render the GBM Explorer."""
    explorer = GBMExplorer()
    explorer.render() 