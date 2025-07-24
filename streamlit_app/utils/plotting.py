"""Streamlit-optimized plotting utilities for price path visualization."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple

from .simulation_bridge import PricePathResult


def create_price_paths_chart(paths: List[PricePathResult], 
                           title: str = "GBM Price Paths",
                           show_individual_paths: bool = True,
                           show_confidence_bands: bool = True,
                           highlight_path_idx: Optional[int] = None) -> go.Figure:
    """Create interactive price paths chart with Plotly."""
    
    fig = go.Figure()
    
    if not paths:
        fig.add_annotation(
            text="No price paths to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract times (should be the same for all paths)
    times = paths[0].times
    
    # Create matrix of all prices for confidence bands
    all_prices = np.array([path.prices for path in paths])
    
    # Add confidence bands if requested
    if show_confidence_bands and len(paths) > 1:
        # Calculate percentiles across paths
        mean_prices = np.mean(all_prices, axis=0)
        p5 = np.percentile(all_prices, 5, axis=0)
        p25 = np.percentile(all_prices, 25, axis=0)
        p75 = np.percentile(all_prices, 75, axis=0)
        p95 = np.percentile(all_prices, 95, axis=0)
        
        # Add confidence bands (outer first for proper layering)
        fig.add_trace(go.Scatter(
            x=times, y=p95,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
            name='95th percentile'
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=p5,
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
            name='90% Confidence Band',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=p75,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
            name='75th percentile'
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=p25,
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
            name='50% Confidence Band',
            hoverinfo='skip'
        ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=times, y=mean_prices,
            mode='lines',
            line=dict(color='black', width=2),
            name='Mean Path',
            hovertemplate='<b>Mean Path</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Add individual paths if requested
    if show_individual_paths:
        # Limit number of paths shown to avoid performance issues
        max_paths_to_show = min(len(paths), 50 if len(paths) > 20 else len(paths))
        
        for i, path in enumerate(paths[:max_paths_to_show]):
            # Determine line style
            if highlight_path_idx is not None and i == highlight_path_idx:
                line_color = 'red'
                line_width = 3
                opacity = 1.0
                name = f'Path {i+1} (Highlighted)'
            else:
                line_color = f'rgba(31, 119, 180, {0.7 if len(paths) <= 10 else 0.3})'
                line_width = 1.5 if len(paths) <= 10 else 1
                opacity = 0.7 if len(paths) <= 10 else 0.3
                name = f'Path {i+1}'
            
            fig.add_trace(go.Scatter(
                x=times,
                y=path.prices,
                mode='lines',
                line=dict(color=line_color, width=line_width),
                opacity=opacity,
                name=name,
                showlegend=(i < 5 or (highlight_path_idx is not None and i == highlight_path_idx)),
                hovertemplate=f'<b>{name}</b><br>Time: %{{x}}<br>Price: $%{{y:.2f}}<br>Return: {path.total_return:.1f}%<extra></extra>'
            ))
        
        if max_paths_to_show < len(paths):
            fig.add_annotation(
                text=f"Showing {max_paths_to_show} of {len(paths)} paths",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )
    
    return fig


def create_statistics_summary_chart(paths: List[PricePathResult]) -> go.Figure:
    """Create a summary statistics chart."""
    
    if not paths:
        fig = go.Figure()
        fig.add_annotation(
            text="No paths available for statistics",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract statistics
    final_prices = [path.final_price for path in paths]
    total_returns = [path.total_return for path in paths]
    realized_vols = [path.volatility_realized for path in paths]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Final Prices Distribution', 'Total Returns Distribution', 
                       'Realized Volatility Distribution', 'Summary Statistics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )
    
    # Final prices histogram
    fig.add_trace(
        go.Histogram(x=final_prices, nbinsx=20, name="Final Prices",
                    hovertemplate='Price: $%{x:.2f}<br>Count: %{y}<extra></extra>'),
        row=1, col=1
    )
    
    # Total returns histogram
    fig.add_trace(
        go.Histogram(x=total_returns, nbinsx=20, name="Total Returns",
                    hovertemplate='Return: %{x:.1f}%<br>Count: %{y}<extra></extra>'),
        row=1, col=2
    )
    
    # Realized volatility histogram
    fig.add_trace(
        go.Histogram(x=realized_vols, nbinsx=20, name="Realized Volatility",
                    hovertemplate='Volatility: %{x:.3f}<br>Count: %{y}<extra></extra>'),
        row=2, col=1
    )
    
    # Summary statistics table
    stats_data = {
        'Metric': ['Count', 'Mean Final Price', 'Std Final Price', 'Mean Return', 
                   'Std Return', 'Mean Realized Vol', 'Target Vol'],
        'Value': [
            f'{len(paths)}',
            f'${np.mean(final_prices):.2f}',
            f'${np.std(final_prices):.2f}',
            f'{np.mean(total_returns):.2f}%',
            f'{np.std(total_returns):.2f}%',
            f'{np.mean(realized_vols):.3f}',
            f'{paths[0].volatility:.3f}'
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                       fill_color='lightblue'),
            cells=dict(values=[stats_data['Metric'], stats_data['Value']],
                      fill_color='lightgray')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Price Path Statistics Summary",
        showlegend=False,
        height=600,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


def create_path_comparison_chart(path1: PricePathResult, path2: PricePathResult) -> go.Figure:
    """Create a comparison chart between two specific paths."""
    
    fig = go.Figure()
    
    # Add both paths
    fig.add_trace(go.Scatter(
        x=path1.times, y=path1.prices,
        mode='lines',
        name=f'Path 1 (Return: {path1.total_return:.1f}%)',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Path 1</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=path2.times, y=path2.prices,
        mode='lines',
        name=f'Path 2 (Return: {path2.total_return:.1f}%)',
        line=dict(color='red', width=2),
        hovertemplate='<b>Path 2</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Path Comparison",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def display_path_statistics_table(paths: List[PricePathResult]) -> None:
    """Display a detailed statistics table in Streamlit."""
    
    if not paths:
        st.write("No paths available for statistics display.")
        return
    
    # Create dataframe for display
    data = []
    for i, path in enumerate(paths):
        data.append({
            'Path': i + 1,
            'Seed': path.seed,
            'Final Price': f'${path.final_price:.2f}',
            'Total Return': f'{path.total_return:.2f}%',
            'Min Price': f'${path.min_price:.2f}',
            'Max Price': f'${path.max_price:.2f}',
            'Realized Vol': f'{path.volatility_realized:.3f}',
            'Price Std Dev': f'${path.std_price:.2f}'
        })
    
    df = pd.DataFrame(data)
    
    # Display with formatting
    st.subheader("Detailed Path Statistics")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def create_downloadable_chart_data(paths: List[PricePathResult]) -> pd.DataFrame:
    """Create a DataFrame suitable for CSV download."""
    
    if not paths:
        return pd.DataFrame()
    
    # Create long-format dataframe
    data = []
    for i, path in enumerate(paths):
        for time, price in zip(path.times, path.prices):
            data.append({
                'path_id': path.path_id,
                'path_number': i + 1,
                'time': time,
                'price': price,
                'volatility': path.volatility,
                'seed': path.seed,
                'final_price': path.final_price,
                'total_return': path.total_return
            })
    
    return pd.DataFrame(data) 