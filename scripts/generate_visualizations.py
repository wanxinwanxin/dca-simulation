#!/usr/bin/env python3
"""Standalone script to generate price path visualizations."""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_price_path_visualization import (
    test_generate_sample_price_paths,
    test_generate_execution_scenario_visualization,
)
from tests.test_price_path_no_drift import (
    test_generate_gbm_fan_shape,
    test_generate_single_scenario_detailed,
)
from tests.test_comprehensive_order_debug import (
    test_comprehensive_order_debugging,
)
from tests.test_true_dutch_order import (
    test_true_dutch_order_behavior,
)

def main():
    """Generate all price path visualizations."""
    print("🎯 Generating Price Path Visualizations")
    print("=" * 50)
    
    try:
        print("\n📈 Generating sample price paths with different parameters...")
        test_generate_sample_price_paths()
        
        print("\n⏰ Generating execution timing analysis...")
        test_generate_execution_scenario_visualization()
        
        print("\n🌀 Generating GBM fan shape analysis (no drift)...")
        test_generate_gbm_fan_shape()
        
        print("\n📊 Generating detailed fan shape with theoretical comparison...")
        test_generate_single_scenario_detailed()
        
        print("\n🔧 Generating order execution debugging visualization...")
        test_comprehensive_order_debugging()
        
        print("\n🎯 Generating true Dutch order behavior visualization...")
        test_true_dutch_order_behavior()
        
        print("\n✅ All visualizations generated successfully!")
        print("\n📁 Files saved to:")
        print("   📈 Price Analysis: results/price_analysis/")
        print("      - price_paths_visualization.png")
        print("      - execution_timing_analysis.png")
        print("      - gbm_fan_shape_no_drift.png")
        print("      - gbm_detailed_fan_analysis.png")
        print("   🔧 Order Debugging: results/order_debug/")
        print("      - comprehensive_order_debug.png")
        print("      - true_dutch_order_demo.png")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 