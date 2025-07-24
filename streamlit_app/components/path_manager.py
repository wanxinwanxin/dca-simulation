"""Path management component for saving, loading, and organizing price paths."""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.simulation_bridge import PricePathResult, GBMPathGenerator
from ..utils.plotting import create_path_comparison_chart, create_price_paths_chart


class PathManager:
    """Manages saved price paths and provides organization features."""
    
    def __init__(self):
        self.generator = GBMPathGenerator()
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for path management."""
        if 'saved_paths' not in st.session_state:
            st.session_state.saved_paths = {}
        
        if 'path_collections' not in st.session_state:
            st.session_state.path_collections = {}
        
        if 'selected_paths' not in st.session_state:
            st.session_state.selected_paths = []
    
    def save_current_paths(self, paths: List[PricePathResult], collection_name: str = "") -> None:
        """Save current paths to session state with optional collection name."""
        if not paths:
            st.warning("No paths to save.")
            return
        
        # Generate collection name if not provided
        if not collection_name:
            timestamp = datetime.now().strftime("%H%M%S")
            collection_name = f"paths_{len(paths)}_{timestamp}"
        
        # Save to session state
        collection_data = {
            'paths': paths,
            'metadata': {
                'name': collection_name,
                'created': str(datetime.now()),
                'n_paths': len(paths),
                'volatility': paths[0].volatility,
                'base_seed': paths[0].seed - (paths[0].seed % 1000)  # Approximate base seed
            }
        }
        
        st.session_state.path_collections[collection_name] = collection_data
        
        # Also add individual paths to saved_paths for easy access
        for path in paths:
            st.session_state.saved_paths[path.path_id] = path
        
        st.success(f"Saved {len(paths)} paths to collection '{collection_name}'")
    
    def load_paths_from_file(self, uploaded_file) -> Optional[List[PricePathResult]]:
        """Load paths from uploaded JSON file."""
        try:
            content = uploaded_file.read()
            paths = self.generator.load_paths_from_json(Path(uploaded_file.name))
            return paths
        except Exception as e:
            st.error(f"Failed to load paths from file: {str(e)}")
            return None
    
    def render_save_interface(self, current_paths: List[PricePathResult]) -> None:
        """Render interface for saving current paths."""
        st.subheader("💾 Save Current Paths")
        
        if not current_paths:
            st.info("No current paths to save. Generate some paths first!")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            collection_name = st.text_input(
                "Collection Name",
                placeholder=f"paths_{len(current_paths)}_{datetime.now().strftime('%H%M%S')}",
                help="Name for this collection of paths"
            )
        
        with col2:
            if st.button("💾 Save Paths", type="primary"):
                self.save_current_paths(current_paths, collection_name)
    
    def render_saved_collections(self) -> None:
        """Render interface for viewing and managing saved path collections."""
        st.subheader("📚 Saved Path Collections")
        
        if not st.session_state.path_collections:
            st.info("No saved path collections yet. Save some paths to see them here!")
            return
        
        # Display collections in a nice format
        for collection_name, collection_data in st.session_state.path_collections.items():
            with st.expander(f"📁 {collection_name}", expanded=False):
                metadata = collection_data['metadata']
                paths = collection_data['paths']
                
                # Collection info
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Created:** {metadata['created']}")
                    st.write(f"**Paths:** {metadata['n_paths']}")
                
                with col2:
                    st.write(f"**Volatility:** {metadata['volatility']:.3f}")
                    st.write(f"**Base Seed:** {metadata.get('base_seed', 'Unknown')}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{collection_name}"):
                        del st.session_state.path_collections[collection_name]
                        st.rerun()
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📈 View", key=f"view_{collection_name}"):
                        self._display_collection(collection_name, paths)
                
                with col2:
                    if st.button(f"📊 Compare", key=f"compare_{collection_name}"):
                        st.session_state.comparison_collection = collection_name
                
                with col3:
                    # Export to JSON
                    if st.button(f"📄 Export", key=f"export_{collection_name}"):
                        try:
                            filename = f"{collection_name}.json"
                            export_path = self.generator.export_paths_to_json(paths, filename)
                            st.success(f"Exported to {export_path}")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
    
    def _display_collection(self, collection_name: str, paths: List[PricePathResult]) -> None:
        """Display a collection of paths in a dedicated section."""
        st.subheader(f"Collection: {collection_name}")
        
        # Create visualization
        fig = create_price_paths_chart(
            paths=paths,
            title=f"{collection_name} (n={len(paths)}, σ={paths[0].volatility:.3f})",
            show_individual_paths=True,
            show_confidence_bands=len(paths) > 1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        if paths:
            final_prices = [p.final_price for p in paths]
            returns = [p.total_return for p in paths]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
            with col2:
                st.metric("Price Std Dev", f"${np.std(final_prices):.2f}")
            with col3:
                st.metric("Mean Return", f"{np.mean(returns):.2f}%")
            with col4:
                st.metric("Return Std Dev", f"{np.std(returns):.2f}%")
    
    def render_path_comparison(self) -> None:
        """Render interface for comparing individual paths."""
        st.subheader("🔍 Path Comparison")
        
        all_paths = []
        path_labels = []
        
        # Collect all available paths
        for collection_name, collection_data in st.session_state.path_collections.items():
            for i, path in enumerate(collection_data['paths']):
                all_paths.append(path)
                path_labels.append(f"{collection_name} - Path {i+1}")
        
        if len(all_paths) < 2:
            st.info("Need at least 2 saved paths to enable comparison. Save more path collections!")
            return
        
        # Path selection
        col1, col2 = st.columns(2)
        
        with col1:
            path1_idx = st.selectbox(
                "Select First Path",
                options=range(len(all_paths)),
                format_func=lambda x: path_labels[x],
                key="path1_selector"
            )
        
        with col2:
            path2_idx = st.selectbox(
                "Select Second Path", 
                options=range(len(all_paths)),
                format_func=lambda x: path_labels[x],
                index=1 if len(all_paths) > 1 else 0,
                key="path2_selector"
            )
        
        if path1_idx != path2_idx:
            # Create comparison chart
            path1 = all_paths[path1_idx]
            path2 = all_paths[path2_idx]
            
            fig = create_path_comparison_chart(path1, path2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison statistics
            st.subheader("Comparison Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Path 1:**")
                st.write(f"- Final Price: ${path1.final_price:.2f}")
                st.write(f"- Total Return: {path1.total_return:.2f}%")
                st.write(f"- Volatility: {path1.volatility:.3f}")
                st.write(f"- Min Price: ${path1.min_price:.2f}")
                st.write(f"- Max Price: ${path1.max_price:.2f}")
            
            with col2:
                st.write("**Path 2:**")
                st.write(f"- Final Price: ${path2.final_price:.2f}")
                st.write(f"- Total Return: {path2.total_return:.2f}%")
                st.write(f"- Volatility: {path2.volatility:.3f}")
                st.write(f"- Min Price: ${path2.min_price:.2f}")
                st.write(f"- Max Price: ${path2.max_price:.2f}")
        else:
            st.warning("Please select two different paths for comparison.")
    
    def render_import_interface(self) -> None:
        """Render interface for importing paths from files."""
        st.subheader("📥 Import Paths")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON file with exported paths",
            type=['json'],
            help="Upload a JSON file previously exported from this application"
        )
        
        if uploaded_file is not None:
            try:
                # Load paths from file
                content = json.loads(uploaded_file.read())
                
                # Convert back to PricePathResult objects
                imported_paths = []
                for path_data in content:
                    config = path_data['config']
                    data = path_data['data']
                    stats = path_data['statistics']
                    
                    path = PricePathResult(
                        volatility=config['volatility'],
                        drift=config['drift'],
                        initial_price=config['initial_price'],
                        dt=config['dt'],
                        horizon=config['horizon'],
                        seed=config['seed'],
                        times=data['times'],
                        prices=data['prices'],
                        path_id=data['path_id'],
                        timestamp=data['timestamp'],
                        mean_price=stats['mean_price'],
                        std_price=stats['std_price'],
                        min_price=stats['min_price'],
                        max_price=stats['max_price'],
                        final_price=stats['final_price'],
                        total_return=stats['total_return'],
                        volatility_realized=stats['volatility_realized']
                    )
                    imported_paths.append(path)
                
                if imported_paths:
                    # Save as new collection
                    collection_name = f"imported_{uploaded_file.name.replace('.json', '')}"
                    self.save_current_paths(imported_paths, collection_name)
                    st.success(f"Successfully imported {len(imported_paths)} paths!")
                else:
                    st.warning("No valid paths found in the uploaded file.")
                    
            except Exception as e:
                st.error(f"Failed to import file: {str(e)}")
    
    def render(self, current_paths: Optional[List[PricePathResult]] = None) -> None:
        """Render the complete path management interface."""
        st.title("📁 Path Manager")
        st.markdown("Organize, save, and compare your generated price paths.")
        
        # Create tabs for different functions
        tab1, tab2, tab3, tab4 = st.tabs(["💾 Save Paths", "📚 Saved Collections", "🔍 Compare Paths", "📥 Import"])
        
        with tab1:
            if current_paths:
                self.render_save_interface(current_paths)
            else:
                st.info("No current paths available. Generate some paths in the GBM Explorer first!")
        
        with tab2:
            self.render_saved_collections()
        
        with tab3:
            self.render_path_comparison()
        
        with tab4:
            self.render_import_interface()


def render_path_manager(current_paths: Optional[List[PricePathResult]] = None):
    """Convenience function to render the Path Manager."""
    manager = PathManager()
    manager.render(current_paths) 