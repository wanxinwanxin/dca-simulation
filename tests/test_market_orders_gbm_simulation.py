"""Test script to simulate 1 GBM price path with 10 market orders and visualize results."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import simpy
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.orders import Side
from src.test_utils.simulation import SimulationConfig, SimulationRunner
from src.market.percentage_impact import PercentageImpact
from src.test_utils.strategies import MarketOrderStrategy
from src.test_utils.analyzers import PricePathAnalyzer
from src.test_utils.engines import PricePathTrackingEngine
from src.engine.matching import MatchingEngine
from src.strategy.protocols import BrokerState, MarketSnapshot
from src.test_utils.visualization import setup_figure, save_figure


class CustomPricePathEngine(MatchingEngine):
    """Custom engine that properly records price path data."""
    
    def __init__(self, *args, analyzer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
    
    def run(self, algo, target_qty: float, horizon: float):
        """Run simulation with price path recording."""
        
        def simulation_process():
            """Main simulation process with price recording."""
            while self.env.now < horizon and self.total_filled_qty < target_qty:
                current_time = float(self.env.now)
                
                # Get current market state
                mid_price = self.price_process.mid_price(current_time)
                
                # Record price point if analyzer supports it
                if self.analyzer and hasattr(self.analyzer, 'record_price_point'):
                    self.analyzer.record_price_point(current_time, mid_price)
                
                market_state = MarketSnapshot(
                    mid_price=mid_price,
                    timestamp=current_time,
                )
                
                # Get broker state
                broker_state = BrokerState(
                    open_orders=self.open_orders.copy(),
                    filled_qty=self.total_filled_qty,
                    remaining_qty=target_qty - self.total_filled_qty,
                )
                
                # Notify probes of time step
                for probe in self.probes:
                    probe.on_step(current_time)
                
                # Get instructions from algorithm
                from src.strategy.protocols import InstructionType
                instructions = algo.step(current_time, broker_state, market_state)
                
                # Process instructions
                for instruction in instructions:
                    if instruction.instruction_type == InstructionType.PLACE:
                        self._place_order(instruction, current_time)
                    elif instruction.instruction_type == InstructionType.CANCEL:
                        self._cancel_order(instruction.order_id)
                
                # Check for fills
                self._check_fills(current_time, mid_price)
                
                # Advance time
                yield self.env.timeout(self.time_step)
        
        # Start simulation
        self.env.process(simulation_process())
        self.env.run()
        
        # Collect final metrics
        results = {}
        for i, probe in enumerate(self.probes):
            probe_results = probe.final()
            results[f"probe_{i}_{probe.__class__.__name__}"] = probe_results
        
        return results
    
    def _place_order(self, instruction, current_time: float) -> None:
        """Track order placements and record price."""
        super()._place_order(instruction, current_time)
        
        if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
            market_price = self.price_process.mid_price(current_time)
            self.analyzer.record_order_creation(current_time, instruction.order_id, market_price)


def test_market_orders_with_gbm():
    """Test 10 market orders on a single GBM price path."""
    
    print("🎯 Testing Market Orders with GBM Price Path")
    print("=" * 50)
    
    # Configuration
    config = SimulationConfig(
        horizon=300.0,  # 5 minutes
        dt=1.0,         # 1 second steps
        initial_price=100.0,
        drift=0.0,      # No drift for cleaner visualization
        volatility=0.02,  # 2% volatility
        random_seed=42,
        spread=0.05,    # 5 cent spread
        impact_gamma=0.0001,  # 0.01% impact per unit (percentage-based)
        base_fee=2e-9,
        tip=1e-9,
    )
    
    # Strategy: 10 market orders, 5000 units each, every 30 seconds
    n_orders = 10
    order_size = 5000
    interval = 30.0  # 30 seconds between orders
    
    print(f"📋 Simulation Parameters:")
    print(f"   • Duration: {config.horizon:.0f} seconds ({config.horizon/60:.1f} minutes)")
    print(f"   • Initial price: ${config.initial_price:.2f}")
    print(f"   • Volatility: {config.volatility:.2%}")
    print(f"   • Drift: {config.drift:.2%}")
    print(f"   • Market orders: {n_orders} orders")
    print(f"   • Order size: {order_size:,} units each")
    print(f"   • Order interval: {interval:.0f} seconds")
    print(f"   • Total quantity: {n_orders * order_size:,} units")
    print(f"   • Impact model: PercentageImpact (0.01% per unit)")
    print(f"   • Spread: ${config.spread:.2f} half-spread")
    
    # Create analyzer to track everything
    analyzer = PricePathAnalyzer()
    
    # Create strategy
    strategy = MarketOrderStrategy(
        order_qty=order_size,
        n_orders=n_orders,
        interval=interval,
        side=Side.SELL,  # Sell orders
        analyzer=analyzer
    )
    
    # Create a custom runner that handles our engine properly
    class CustomSimulationRunner(SimulationRunner):
        def create_engine(self, analyzer=None, **kwargs):
            """Create engine with analyzer support and percentage-based impact."""
            price_process = self.create_price_process()
            liquidity_model, _ = self.create_market_models()  # Ignore default impact model
            gas_model, filler_decision = self.create_cost_models()
            
            # Create percentage-based impact model
            # gamma = 0.0001 means 0.01% impact per unit (50% impact for 5000 units!)
            impact_model = PercentageImpact(
                spread=self.config.spread,
                gamma=0.0001  # 0.01% impact per unit
            )
            
            # Set up probes list
            probes = []
            if analyzer:
                probes.append(analyzer)
            
            # Create our custom engine with analyzer
            engine = self.engine_class(
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
    runner = CustomSimulationRunner(
        config=config,
        engine_class=CustomPricePathEngine
    )
    
    print(f"\n🚀 Running simulation...")
    
    # Run simulation
    results = runner.run_simulation(
        strategy=strategy,
        analyzer=analyzer,
        target_qty=n_orders * order_size  # Total target quantity
    )
    
    print(f"✅ Simulation completed!")
    
    # Extract results
    fills = analyzer.fills
    order_creations = analyzer.order_creations
    price_path = analyzer.price_path
    
    print(f"\n📊 Results Summary:")
    print(f"   • Total orders placed: {len(order_creations)}")
    print(f"   • Total fills: {len(fills)}")
    print(f"   • Fill rate: {len(fills)/len(order_creations):.1%}")
    print(f"   • Price path points: {len(price_path)}")
    
    if fills:
        fill_prices = [f.price for f in fills]
        total_qty_filled = sum(f.qty for f in fills)
        avg_fill_price = np.mean(fill_prices)
        
        print(f"   • Total quantity filled: {total_qty_filled:,.0f} units")
        print(f"   • Average fill price: ${avg_fill_price:.2f}")
        print(f"   • Fill price range: ${min(fill_prices):.2f} - ${max(fill_prices):.2f}")
    
    # Create detailed fill vs creation price analysis
    print(f"\n📋 Fill Price vs Creation Market Price Analysis:")
    print("=" * 80)
    
    if fills and order_creations:
        # Create lookup for creation data
        creation_data = {order_id: (creation_time, market_price) 
                        for creation_time, order_id, market_price in order_creations}
        
        # Prepare table data
        table_data = []
        for fill in fills:
            if fill.order_id in creation_data:
                creation_time, creation_market_price = creation_data[fill.order_id]
                price_diff = fill.price - creation_market_price
                price_diff_pct = (price_diff / creation_market_price) * 100
                
                table_data.append({
                    'order_id': fill.order_id,
                    'creation_time': creation_time,
                    'creation_market_price': creation_market_price,
                    'fill_time': fill.timestamp,
                    'fill_price': fill.price,
                    'price_diff': price_diff,
                    'price_diff_pct': price_diff_pct,
                    'execution_delay': fill.timestamp - creation_time
                })
        
        if table_data:
            # Print table header
            print(f"{'Order ID':<12} {'Create':<8} {'Market@Create':<13} {'Fill':<8} {'Fill Price':<11} {'Diff ($)':<10} {'Diff (%)':<8} {'Delay (s)':<9}")
            print("-" * 80)
            
            # Print each row
            for row in table_data:
                print(f"{row['order_id']:<12} "
                      f"{row['creation_time']:<8.0f} "
                      f"${row['creation_market_price']:<12.2f} "
                      f"{row['fill_time']:<8.0f} "
                      f"${row['fill_price']:<10.2f} "
                      f"{row['price_diff']:<+9.2f} "
                      f"{row['price_diff_pct']:<+7.1f} "
                      f"{row['execution_delay']:<9.1f}")
            
            # Summary statistics
            avg_diff = np.mean([row['price_diff'] for row in table_data])
            avg_diff_pct = np.mean([row['price_diff_pct'] for row in table_data])
            max_diff = max([abs(row['price_diff']) for row in table_data])
            
            print("-" * 80)
            print(f"Summary:")
            print(f"  • Average price difference: ${avg_diff:+.2f} ({avg_diff_pct:+.1f}%)")
            print(f"  • Maximum absolute difference: ${max_diff:.2f}")
            print(f"  • Note: Negative = filled below creation market price (good for sells)")
            
            # Show percentage-based impact analysis
            print(f"\n💡 Percentage-Based Impact Analysis:")
            print("-" * 80)
            print(f"Impact Model: PercentageImpact (γ = 0.01% per unit)")
            print(f"Components for each 5000-unit sell order:")
            
            # Calculate impact components for a sample order
            sample_price = table_data[0]['creation_market_price']
            spread_cost = config.spread
            impact_pct = 0.0001 * 5000  # gamma * qty
            impact_dollar = impact_pct / 100 * sample_price
            total_cost = spread_cost + impact_dollar
            
            print(f"  • Spread cost (fixed): ${spread_cost:.2f}")
            print(f"  • Impact cost (% of price): {impact_pct:.2f}% = ${impact_dollar:.2f} at ${sample_price:.2f}")
            print(f"  • Total transaction cost: ${total_cost:.2f}")
            print(f"  • This scales proportionally with price level!")
            
            # Show how impact varies with price
            print(f"\nImpact scaling with price level (5000 units):")
            test_prices = [10, 50, 100, 500, 1000]
            for price in test_prices:
                impact_at_price = impact_pct / 100 * price
                total_at_price = spread_cost + impact_at_price
                print(f"  • At ${price:>4}: Impact = ${impact_at_price:.2f}, Total = ${total_at_price:.2f}")
            
            print(f"\n✅ Impact now scales proportionally - no more arbitrary denomination effects!")
    else:
        print("   No fill data available for comparison.")
    
    # Create visualization
    print(f"\n📈 Creating visualization...")
    
    fig, axes = setup_figure(2, 2, (16, 10))
    fig.suptitle('Market Orders on GBM Price Path', fontsize=16, fontweight='bold')
    
    # Convert time to minutes for better readability
    if price_path:
        times_min = [t/60 for t, _ in price_path]
        prices = [p for _, p in price_path]
    else:
        times_min, prices = [], []
    
    # Plot 1: Price path with order creation and fill events
    ax1 = axes[0, 0]
    
    # Plot price path (if available) or create synthetic path from order data
    if times_min and prices:
        ax1.plot(times_min, prices, 'b-', linewidth=1, alpha=0.7, label='GBM Price Path')
    elif order_creations and fills:
        # Create a simple visualization using order and fill data
        all_times = sorted(set([t/60 for t, _, _ in order_creations] + [f.timestamp/60 for f in fills]))
        ax1.text(0.5, 0.5, 'Price path tracking unavailable\nShowing order events only', 
                transform=ax1.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot order creation points
    if order_creations:
        creation_times = [t/60 for t, _, _ in order_creations]
        creation_prices = [p for _, _, p in order_creations]
        ax1.scatter(creation_times, creation_prices, s=40, color='orange', 
                   marker='^', edgecolors='black', linewidth=1, 
                   label='Order Placed', zorder=5)
    
    # Plot fill events
    if fills:
        fill_times = [f.timestamp/60 for f in fills]
        fill_prices = [f.price for f in fills]
        ax1.scatter(fill_times, fill_prices, s=50, color='red', 
                   marker='o', edgecolors='black', linewidth=1, 
                   label='Order Filled', zorder=10)
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Path with Order Events')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Fill prices over time
    ax2 = axes[0, 1]
    
    if fills:
        fill_times = [f.timestamp/60 for f in fills]
        fill_prices = [f.price for f in fills]
        
        # Line plot connecting fills
        ax2.plot(fill_times, fill_prices, 'ro-', linewidth=2, markersize=4, 
                markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
        
        # Add fill order numbers as annotations
        for i, (t, p) in enumerate(zip(fill_times, fill_prices)):
            ax2.annotate(f'{i+1}', (t, p), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', alpha=0.8))
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Fill Price ($)')
    ax2.set_title('Market Order Fill Prices')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative quantity filled
    ax3 = axes[1, 0]
    
    if fills:
        fill_times = [f.timestamp/60 for f in fills]
        fill_qtys = [f.qty for f in fills]
        cumulative_qty = np.cumsum(fill_qtys)
        
        # Step function for cumulative quantity - create proper step plot
        step_times = [0]
        step_qtys = [0]
        
        for i, (t, q) in enumerate(zip(fill_times, cumulative_qty)):
            step_times.extend([t, t])
            step_qtys.extend([step_qtys[-1], q])
        
        ax3.plot(step_times, step_qtys, 'g-', linewidth=3, label='Cumulative Quantity')
        ax3.scatter(fill_times, cumulative_qty, color='darkgreen', s=40, 
                   marker='s', edgecolors='black', linewidth=1, zorder=10)
        
        # Target line
        ax3.axhline(y=n_orders * order_size, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Target: {n_orders * order_size} units')
    
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Cumulative Quantity')
    ax3.set_title('Cumulative Fill Progress')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Order execution timing
    ax4 = axes[1, 1]
    
    if fills and order_creations:
        # Calculate execution delays
        creation_times_dict = {order_id: t for t, order_id, _ in order_creations}
        execution_delays = []
        order_numbers = []
        
        for i, fill in enumerate(fills):
            if fill.order_id in creation_times_dict:
                delay = fill.timestamp - creation_times_dict[fill.order_id]
                execution_delays.append(delay)
                order_numbers.append(i + 1)
        
        if execution_delays:
            ax4.bar(order_numbers, execution_delays, color='skyblue', 
                   edgecolor='black', linewidth=1, alpha=0.7)
            
            # Add value labels on bars
            for i, (order_num, delay) in enumerate(zip(order_numbers, execution_delays)):
                ax4.text(order_num, delay + max(execution_delays) * 0.01, 
                        f'{delay:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Order Number')
    ax4.set_ylabel('Execution Delay (seconds)')
    ax4.set_title('Order Execution Timing')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = save_figure("market_orders_gbm_simulation.png", "results/order_debug")
    
    print(f"📁 Visualization saved to: {output_file}")
    
    return {
        'fills': fills,
        'order_creations': order_creations,
        'price_path': price_path,
        'results': results,
        'config': config
    }


def main():
    """Main function."""
    return test_market_orders_with_gbm()


if __name__ == "__main__":
    main() 