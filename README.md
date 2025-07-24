# DCA Simulation Framework

A modular Python framework for simulating and comparing execution algorithms under stochastic market conditions. Built for analyzing TWAP, Dutch auctions, and adaptive limit order strategies with realistic price impact and gas cost modeling.

## Features

- **Interactive Streamlit UI**: Web-based interface for real-time simulation and analysis
- **Modular Architecture**: Dependency injection design allows easy swapping of components
- **Reproducible Simulations**: Deterministic results with configurable random seeds
- **Multiple Strategies**: TWAP market orders, Dutch limit orders, and extensible strategy framework
- **Realistic Market Models**: Geometric Brownian Motion prices, constant spreads, linear impact
- **Gas Cost Modeling**: EIP-1559 gas fee simulation with filler profitability decisions
- **Comprehensive Metrics**: Fill timing and price analysis with aggregated statistics
- **YAML Configuration**: Easy experiment setup and parameter tuning
- **Visualization Tools**: Generate price path plots and execution timing analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dca-simulation

# Install dependencies
pip install -e .
```

### Run a Simulation

```bash
# Run the example configuration
python -m src.simulation configs/twap_vs_dutch.yml

# Generate price path visualizations
python scripts/generate_visualizations.py
```

### Interactive Streamlit UI

**🚀 NEW**: Launch the interactive web interface:

```bash
# Install with Streamlit dependencies
pip install -e ".[streamlit]"

# Launch the web interface
streamlit run streamlit_app/main.py --server.port 8501
```

**Available Features:**
- **🏠 Home**: Overview and quick start guide
- **📈 GBM Price Path Explorer**: Interactive price path generation with volatility controls
- **🎯 Single Order Execution**: Analyze individual market and Dutch auction orders
- **📊 Impact Model Explorer**: Compare different market impact models
- **📁 Path Manager**: Save, organize, and compare price path collections

**Live Demo**: [DCA Simulation on Streamlit Cloud](https://dca-simulation.streamlit.app) *(if deployed)*

### Example Output

```json
{
  "strategy": "TwapMarket",
  "total_runs": 1,
  "probe_1_FillPrices_vwap_mean": 87.39877437158812,
  "probe_0_FillTimings_avg_fill_interval_mean": 30.0,
  "probe_1_FillPrices_total_qty_mean": 100.0
}
```

## Architecture

### Core Components

- **`src/core/`**: Pure dataclasses for orders, fills, and events
- **`src/market/`**: Price processes, liquidity models, and impact models
- **`src/cost/`**: Gas fee models and filler decision logic
- **`src/strategy/`**: Execution algorithms (TWAP, Dutch, etc.)
- **`src/engine/`**: SimPy-based simulation engine
- **`src/metrics/`**: Data collection probes and analytics
- **`src/config/`**: Component factory and configuration system

### Execution Strategies

#### TWAP Market Orders
- Posts equal-sized market orders at fixed intervals
- Guaranteed execution but with market impact
- Configuration: `total_qty`, `n_slices`, `interval`

#### Dutch Limit Orders
- Starts with conservative limit price, drifts toward market
- Better price discovery but uncertain execution
- Configuration: `slice_qty`, `drift_rate`, `initial_spread`

## Configuration

Create YAML files in the `configs/` directory:

```yaml
run_name: my_experiment
horizon: 300  # seconds
random_seed: 42
target_qty: 100.0

price_process:
  name: GBM
  params: 
    mu: 0.00
    sigma: 0.02
    dt: 1.0
    s0: 100.0

liquidity:
  name: ConstSpread
  params: 
    spread: 0.5

impact:
  name: LinearImpact
  params: 
    gamma: 0.0001

gas:
  name: Evm1559
  params: 
    base_fee: 0.000000005
    tip: 0.000000002

strategies:
  - algo: TwapMarket
    params: 
      total_qty: 100.0
      n_slices: 10
      side: BUY
      interval: 30.0

probes:
  - FillTimings
  - FillPrices

paths: 1
```

## Visualization

The framework includes powerful visualization tools to analyze price paths and execution patterns:

### Price Path Analysis
```bash
# Generate price path visualizations
python scripts/generate_visualizations.py

# Or run specific tests
python -m tests.test_price_path_visualization
```

**Generated Plots:**
- **Price Paths with Different Parameters**: Shows how volatility (σ) and drift (μ) affect price evolution
- **Statistical Distribution Evolution**: Box plots showing price distribution over time
- **Execution Timing Analysis**: Compares different TWAP execution schedules
- **GBM Fan Shape Analysis**: Multiple realizations showing characteristic "fan-out" pattern without drift distortion
- **Detailed Fan Analysis**: Theoretical vs actual standard deviation growth with percentile bands
- **Order Execution Debugging**: Comprehensive visualization showing order lifecycle on price paths

### Sample Visualizations

The visualization suite creates:
1. **Multiple GBM scenarios** with varying volatility and drift parameters
2. **Statistical analysis** with 100+ simulation paths per scenario
3. **Execution timing comparison** showing how different TWAP intervals affect performance
4. **Order debugging scenarios** with market, limit, and Dutch order types

Key insights from visualizations:
- Higher volatility leads to wider price spreads over time (clear fan-out effect)
- Execution timing significantly affects VWAP results
- GBM paths show proper theoretical fan shape with √t standard deviation growth
- Multiple realizations of same parameters demonstrate stochastic nature of price evolution
- Order execution visualization helps debug matching logic and timing issues

## Project Structure

```
├── src/
│   ├── core/              # Domain objects (orders, events)
│   ├── market/            # Price & liquidity models
│   ├── cost/              # Gas fees & filler logic
│   ├── strategy/          # Execution algorithms
│   ├── engine/            # Simulation engine
│   ├── metrics/           # Data collection
│   ├── config/            # Factory registry
│   └── simulation.py      # CLI entry point
├── configs/               # YAML experiment recipes
├── scripts/               # Utility scripts
├── tests/                 # Test suite (includes visualization)
└── results/               # Simulation outputs & plots
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_core_hashable.py -v

# Run with coverage (install pytest-cov first)
python -m pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
ruff src/ tests/
```

### Adding New Strategies

1. Create a new file in `src/strategy/`
2. Implement the `ExecutionAlgo` protocol
3. Register in `src/config/factory.py`
4. Add configuration parameters
5. Write tests

### Adding New Market Models

1. Create implementation in `src/market/`
2. Follow the appropriate protocol (`PriceProcess`, `LiquidityModel`, `ImpactModel`)
3. Register in factory
4. Add tests

## Example Results

After running a simulation, results are saved to `results/<run_name>/`:

- `{Strategy}_run_{id}.json`: Individual simulation results
- `{Strategy}_summary.json`: Aggregated statistics across runs

Key metrics include:
- **Fill timing**: First/last fill times, average intervals
- **Fill prices**: VWAP, min/max prices, total quantity
- **Performance**: Success rate, price improvement

## License

This project is released under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions or issues, please open a GitHub issue with:
- Configuration file used
- Full error message
- Expected vs. actual behavior 