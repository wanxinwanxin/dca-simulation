#!/usr/bin/env python3
"""
Quick test script for Phase 1 GBM Explorer functionality.
This verifies that the core components work without the Streamlit UI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from streamlit_app.utils.simulation_bridge import GBMPathGenerator, calculate_ensemble_statistics


def test_gbm_generation():
    """Test GBM path generation."""
    print("🧪 Testing GBM Path Generation...")
    
    generator = GBMPathGenerator()
    
    # Test basic path generation
    paths = generator.generate_paths(
        n_paths=5,
        volatility=0.02,
        base_seed=42
    )
    
    assert len(paths) == 5, f"Expected 5 paths, got {len(paths)}"
    assert all(len(path.prices) == 201 for path in paths), "All paths should have 201 prices"
    assert all(path.volatility == 0.02 for path in paths), "All paths should have volatility 0.02"
    
    print(f"✅ Generated {len(paths)} paths successfully")
    
    # Test statistics calculation
    stats = calculate_ensemble_statistics(paths)
    print(f"✅ Calculated ensemble statistics: {stats['n_paths']} paths")
    
    # Test JSON export
    filename = "test_export.json"
    export_path = generator.export_paths_to_json(paths, filename)
    print(f"✅ Exported to: {export_path}")
    
    # Test JSON import
    imported_paths = generator.load_paths_from_json(export_path)
    assert len(imported_paths) == len(paths), "Imported paths should match exported paths"
    print(f"✅ Imported {len(imported_paths)} paths successfully")
    
    # Clean up
    export_path.unlink()
    print("✅ Cleanup completed")


def test_parameter_ranges():
    """Test parameter validation and ranges."""
    print("\n🧪 Testing Parameter Ranges...")
    
    generator = GBMPathGenerator()
    
    # Test volatility range (Phase 1: 0.01-0.20)
    for vol in [0.01, 0.1, 0.20]:
        paths = generator.generate_paths(n_paths=2, volatility=vol, base_seed=42)
        assert all(path.volatility == vol for path in paths), f"Volatility {vol} not preserved"
        print(f"✅ Volatility {vol} works correctly")
    
    # Test path count range (Phase 1: 1-100)
    for n_paths in [1, 50, 100]:
        paths = generator.generate_paths(n_paths=n_paths, volatility=0.02, base_seed=42)
        assert len(paths) == n_paths, f"Expected {n_paths}, got {len(paths)}"
        print(f"✅ Path count {n_paths} works correctly")


def test_fixed_parameters():
    """Test that fixed parameters (Phase 1) are correctly applied."""
    print("\n🧪 Testing Fixed Parameters...")
    
    generator = GBMPathGenerator()
    paths = generator.generate_paths(n_paths=3, volatility=0.05, base_seed=42)
    
    for path in paths:
        assert path.drift == 0.0, f"Expected drift=0.0, got {path.drift}"
        assert path.initial_price == 100.0, f"Expected S₀=100, got {path.initial_price}"
        assert path.dt == 1.0, f"Expected dt=1.0, got {path.dt}"
        assert path.horizon == 200.0, f"Expected horizon=200, got {path.horizon}"
    
    print("✅ All fixed parameters are correctly applied")


def test_statistics_computation():
    """Test statistical calculations."""
    print("\n🧪 Testing Statistics Computation...")
    
    generator = GBMPathGenerator()
    paths = generator.generate_paths(n_paths=20, volatility=0.03, base_seed=123)
    
    stats = calculate_ensemble_statistics(paths)
    
    # Check required fields
    required_fields = ['n_paths', 'config', 'final_prices', 'total_returns', 'realized_volatility']
    for field in required_fields:
        assert field in stats, f"Missing required field: {field}"
    
    # Check values make sense
    assert stats['n_paths'] == 20, f"Expected n_paths=20, got {stats['n_paths']}"
    assert stats['config']['volatility'] == 0.03, "Volatility not preserved in stats"
    
    print("✅ Statistics computation works correctly")


def main():
    """Run all tests."""
    print("🚀 Phase 1 GBM Explorer - Core Functionality Test")
    print("=" * 50)
    
    try:
        test_gbm_generation()
        test_parameter_ranges()
        test_fixed_parameters()
        test_statistics_computation()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Phase 1 Summary:")
        print("✅ GBM path generation working")
        print("✅ Parameter controls (volatility 0.01-0.20, paths 1-100)")
        print("✅ Fixed parameters (μ=0, S₀=100, dt=1.0, horizon=200)")
        print("✅ Statistics calculation")
        print("✅ JSON export/import")
        print("✅ Data persistence functionality")
        
        print("\n🚀 Ready for Streamlit UI testing!")
        print("Run: streamlit run streamlit_app/main.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 