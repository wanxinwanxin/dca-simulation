"""Test script to simulate Dutch limit orders on GBM price path."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import simpy
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.orders import Side
from src.core.events import Fill
from src.test_utils.simulation import SimulationConfig, SimulationRunner
from src.market.percentage_impact import PercentageImpact
from src.market.dutch_impact import DutchImpact
from src.test_utils.analyzers import DutchAnalyzer
from src.engine.matching import MatchingEngine
from src.strategy.protocols import BrokerState, MarketSnapshot
from src.test_utils.visualization import setup_figure, save_figure


class CustomDutchEngine(MatchingEngine):
    """Custom engine that properly handles Dutch order decay and theoretical limit prices."""
    
    def __init__(self, *args, analyzer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = analyzer
        self.dutch_orders = {}  # Track Dutch order decay parameters
    
    def run(self, algo, target_qty: float, horizon: float):
        """Run simulation with Dutch order tracking."""
        
        def simulation_process():
            """Main simulation process with price and order recording."""
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
        """Track order placements and record Dutch order parameters."""
        super()._place_order(instruction, current_time)
        
        if self.analyzer and hasattr(self.analyzer, 'record_order_creation'):
            market_price = self.price_process.mid_price(current_time)
            # DutchAnalyzer expects (time, order_id, market_price, start_limit)
            if instruction.limit_px is not None:
                self.analyzer.record_order_creation(current_time, instruction.order_id, market_price, instruction.limit_px)
            else:
                self.analyzer.record_order_creation(current_time, instruction.order_id, market_price)
    
    def register_dutch_order(self, order_id: str, creation_time: float, creation_mid: float, 
                           starting_limit: float, decay_bps_per_second: float):
        """Register Dutch order decay parameters."""
        self.dutch_orders[order_id] = {
            'creation_time': creation_time,
            'creation_mid': creation_mid,
            'starting_limit': starting_limit,
            'decay_bps_per_second': decay_bps_per_second
        }
    
    def _get_current_dutch_limit(self, order_id: str, current_time: float) -> float:
        """Calculate theoretical current limit price for Dutch order."""
        if order_id not in self.dutch_orders:
            return None
        
        params = self.dutch_orders[order_id]
        time_elapsed = current_time - params['creation_time']
        
        # Calculate decay: 10bp per second relative to creation mid price
        absolute_decay = (params['decay_bps_per_second'] / 10000) * params['creation_mid'] * time_elapsed
        
        # For sell orders: start high, decay DOWN
        current_limit = params['starting_limit'] - absolute_decay
        
        return max(0, current_limit)  # Don't go below zero
    
    def _check_fills(self, current_time: float, mid_price: float) -> None:
        """Check fills using theoretical current limit prices for Dutch orders."""
        filled_orders = []
        
        # Only debug when there are actually orders to check
        if self.open_orders:
            print(f"\n🔍 t={current_time:.0f}s, mid=${mid_price:.2f}, orders={len(self.open_orders)}")
        
        for order_id, order in self.open_orders.items():
            # Check if order is expired
            if current_time > order.valid_to:
                filled_orders.append(order_id)
                continue
            
            # For Dutch orders, use theoretical current limit price
            limit_price_to_check = order.limit_px
            theoretical_limit = None
            
            if order_id.startswith("dutch_") and order_id in self.dutch_orders:
                theoretical_limit = self._get_current_dutch_limit(order_id, current_time)
                if theoretical_limit is not None and theoretical_limit > 0:
                    limit_price_to_check = theoretical_limit
                    
                    print(f"  Dutch {order_id}: static=${order.limit_px:.2f}, current=${theoretical_limit:.2f}")
                    print(f"    Mid=${mid_price:.2f}, spread=${self.liquidity_model.spread:.2f}")
                    print(f"    Threshold=${mid_price - self.liquidity_model.spread:.2f}")
                else:
                    # Dutch order decayed to zero or below
                    print(f"  Dutch {order_id}: decayed below zero, expiring")
                    filled_orders.append(order_id)
                    continue
            
            # Check if order crosses the spread (for limit orders)
            if limit_price_to_check is not None:
                crossed = self.liquidity_model.crossed(order.side, limit_price_to_check, mid_price)
                print(f"    Crossed: {crossed}")
                
                if not crossed:
                    continue
            
            # Calculate execution price using theoretical limit for Dutch orders
            if order_id.startswith("dutch_") and theoretical_limit is not None:
                # For Dutch orders with split=0, fill at theoretical current limit
                exec_price = theoretical_limit
            else:
                # Regular impact model
                exec_price = self.impact_model.exec_price(order, mid_price)
            
            # Calculate gas cost
            gas_cost = self.gas_model.gas_fee(self.gas_per_fill)
            
            # Check if filler would execute
            if limit_price_to_check is not None:
                should_fill = self.filler_decision.should_fill(
                    order.side, limit_price_to_check, mid_price, order.qty, gas_cost
                )
                if not should_fill:
                    continue
            
            print(f"    ✅ FILLING {order_id} at ${exec_price:.2f} (theoretical limit)")
            
            # Execute the fill
            fill = Fill(
                order_id=order.id,
                timestamp=current_time,
                qty=order.qty,
                price=exec_price,
                gas_paid=gas_cost,
            )
            
            # Update state
            self.total_filled_qty += order.qty
            filled_orders.append(order_id)
            
            # Notify probes
            for probe in self.probes:
                probe.on_fill(fill)
        
        # Remove filled orders
        for order_id in filled_orders:
            if order_id in self.open_orders:
                del self.open_orders[order_id]
            if order_id in self.dutch_orders:
                del self.dutch_orders[order_id]


class FixedDutchStrategy:
    """Fixed Dutch strategy that registers orders with engine for proper decay handling."""
    
    def __init__(self, n_orders: int, order_size: float, interval: float, 
                 starting_offset_bps: float, decay_bps_per_second: float, 
                 order_duration: float, side: Side = Side.SELL, analyzer=None, engine=None):
        self.n_orders = n_orders
        self.order_size = order_size
        self.interval = interval
        self.starting_offset_bps = starting_offset_bps
        self.decay_bps_per_second = decay_bps_per_second  
        self.order_duration = order_duration
        self.side = side
        self.analyzer = analyzer
        self.engine = engine  # Reference to engine for registering Dutch orders
        self.current_index = 0
    
    def step(self, clock: float, broker_state, market_state) -> list:
        """Generate Dutch order instructions."""
        instructions = []
        
        # Create new orders at intervals
        if self.current_index < self.n_orders and clock >= self.current_index * self.interval:
            order_id = f"dutch_{self.current_index}"
            mid_price = market_state.mid_price
            
            # Calculate starting limit price - FIXED: 50bps ABOVE mid for sells
            if self.side == Side.SELL:
                starting_limit = mid_price + (self.starting_offset_bps / 10000) * mid_price
            else:
                starting_limit = mid_price - (self.starting_offset_bps / 10000) * mid_price
            
            from src.strategy.protocols import OrderInstruction, InstructionType
            instruction = OrderInstruction(
                instruction_type=InstructionType.PLACE,
                order_id=order_id,
                side=self.side,
                qty=self.order_size,
                limit_px=starting_limit,
                valid_to=clock + self.order_duration,
            )
            
            instructions.append(instruction)
            
            # Register Dutch order with engine for decay tracking
            if self.engine and hasattr(self.engine, 'register_dutch_order'):
                self.engine.register_dutch_order(
                    order_id=order_id,
                    creation_time=clock,
                    creation_mid=mid_price,
                    starting_limit=starting_limit,
                    decay_bps_per_second=self.decay_bps_per_second
                )
            
            print(f"🎯 Created Dutch order {order_id}: mid=${mid_price:.2f}, start_limit=${starting_limit:.2f} (+{self.starting_offset_bps}bps)")
            
            self.current_index += 1
        
        return instructions


def test_dutch_orders_with_gbm():
    """Test Dutch limit orders on a single GBM price path."""
    
    print("🎯 Testing Dutch Limit Orders with GBM Price Path")
    print("=" * 60)
    
    # Configuration
    config = SimulationConfig(
        horizon=600.0,  # 10 minutes (longer for Dutch orders)
        dt=1.0,         # 1 second steps
        initial_price=100.0,
        drift=0.0,      # No drift
        volatility=0.02,  # 2% volatility
        random_seed=42,
        spread=0.05,    # 5 cent spread
        impact_gamma=0.0001,  # 0.01% impact per unit (percentage-based)
        base_fee=2e-9,
        tip=1e-9,
    )
    
    # Dutch strategy parameters
    n_orders = 10
    order_size = 5000
    interval = 30.0  # 30 seconds between orders
    starting_offset_bps = 50  # 50bps above mid
    decay_bps_per_second = 10  # 10bp per second decay
    order_duration = 120.0  # 2 minutes max per order
    
    print(f"📋 Dutch Order Parameters:")
    print(f"   • Duration: {config.horizon:.0f} seconds ({config.horizon/60:.1f} minutes)")
    print(f"   • Initial price: ${config.initial_price:.2f}")
    print(f"   • Volatility: {config.volatility:.2%}")
    print(f"   • Dutch orders: {n_orders} orders")
    print(f"   • Order size: {order_size:,} units each")
    print(f"   • Order interval: {interval:.0f} seconds")
    print(f"   • Starting offset: {starting_offset_bps}bps above mid")
    print(f"   • Decay rate: {decay_bps_per_second}bp per second")
    print(f"   • Order duration: {order_duration:.0f} seconds")
    print(f"   • Split: 0 (fill at limit price)")
    
    # Create analyzer
    analyzer = DutchAnalyzer()
    
    # Create custom runner
    class CustomDutchSimulationRunner(SimulationRunner):
        def create_engine(self, analyzer=None, **kwargs):
            """Create engine with Dutch-aware impact model."""
            price_process = self.create_price_process()
            liquidity_model, _ = self.create_market_models()
            gas_model, filler_decision = self.create_cost_models()
            
            # Create Dutch-aware impact model with split=0
            impact_model = DutchImpact(
                spread=self.config.spread,
                gamma=0.0001,  # 0.01% impact per unit
                dutch_price_split=0.0  # Fill at limit price
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
        
        def run_simulation(self, strategy, analyzer=None, target_qty=1000.0, **engine_kwargs):
            """Run simulation with engine reference passed to strategy."""
            engine = self.create_engine(analyzer=analyzer, **engine_kwargs)
            
            # Pass engine reference to strategy if it supports it
            if hasattr(strategy, 'engine'):
                strategy.engine = engine
            
            # Run simulation using MatchingEngine API
            results = engine.run(
                algo=strategy,
                target_qty=target_qty,
                horizon=self.config.horizon
            )
            
            # Return analyzer results if available
            if analyzer and hasattr(analyzer, 'final'):
                analyzer_results = analyzer.final()
                analyzer_results.update(results)
                return analyzer_results
            else:
                return results
    
    # Create simulation runner
    runner = CustomDutchSimulationRunner(
        config=config,
        engine_class=CustomDutchEngine
    )
    
    # Create strategy (engine reference will be set by runner)
    strategy = FixedDutchStrategy(
        n_orders=n_orders,
        order_size=order_size,
        interval=interval,
        starting_offset_bps=starting_offset_bps,
        decay_bps_per_second=decay_bps_per_second,
        order_duration=order_duration,
        side=Side.SELL,
        analyzer=analyzer,
        engine=None  # Will be set by runner
    )
    
    print(f"\n🚀 Running Dutch order simulation...")
    
    # Run simulation
    results = runner.run_simulation(
        strategy=strategy,
        analyzer=analyzer,
        target_qty=n_orders * order_size
    )
    
    print(f"✅ Simulation completed!")
    
    # Extract results
    fills = analyzer.fills
    order_creations = analyzer.order_creations
    price_path = analyzer.price_path
    
    print(f"\n📊 Results Summary:")
    print(f"   • Total orders placed: {len(order_creations)}")
    print(f"   • Total fills: {len(fills)}")
    print(f"   • Fill rate: {len(fills)/len(order_creations)*100:.1f}%" if order_creations else "N/A")
    print(f"   • Price path points: {len(price_path)}")
    
    if fills:
        fill_prices = [f.price for f in fills]
        total_qty_filled = sum(f.qty for f in fills)
        avg_fill_price = np.mean(fill_prices)
        
        print(f"   • Total quantity filled: {total_qty_filled:,.0f} units")
        print(f"   • Average fill price: ${avg_fill_price:.2f}")
        print(f"   • Fill price range: ${min(fill_prices):.2f} - ${max(fill_prices):.2f}")
    
    # Create Dutch order analysis table
    print(f"\n📋 Dutch Order Analysis:")
    print("=" * 100)
    
    if fills and order_creations:
        # Create lookup for creation data
        creation_data = {order_id: (creation_time, market_price, start_limit) 
                        for creation_time, order_id, market_price, start_limit in order_creations}
        
        # Prepare table data
        table_data = []
        for fill in fills:
            if fill.order_id in creation_data:
                creation_time, creation_market_price, start_limit = creation_data[fill.order_id]
                
                # Calculate metrics
                time_to_fill = fill.timestamp - creation_time
                starting_offset_pct = (start_limit - creation_market_price) / creation_market_price * 100
                fill_vs_start_limit = fill.price - start_limit
                fill_vs_creation_mid = fill.price - creation_market_price
                
                table_data.append({
                    'order_id': fill.order_id,
                    'creation_time': creation_time,
                    'creation_mid': creation_market_price,
                    'start_limit': start_limit,
                    'fill_time': fill.timestamp,
                    'fill_price': fill.price,
                    'time_to_fill': time_to_fill,
                    'starting_offset_pct': starting_offset_pct,
                    'fill_vs_start_limit': fill_vs_start_limit,
                    'fill_vs_creation_mid': fill_vs_creation_mid
                })
        
        if table_data:
            # Print table header
            print(f"{'Order':<8} {'Create':<7} {'Mid@Create':<10} {'StartLimit':<10} {'FillTime':<8} {'FillPrice':<10} {'TimeToFill':<10} {'vs StartLimit':<12} {'vs CreateMid':<11}")
            print("-" * 100)
            
            # Print each row
            for row in table_data:
                print(f"{row['order_id']:<8} "
                      f"{row['creation_time']:<7.0f} "
                      f"${row['creation_mid']:<9.2f} "
                      f"${row['start_limit']:<9.2f} "
                      f"{row['fill_time']:<8.0f} "
                      f"${row['fill_price']:<9.2f} "
                      f"{row['time_to_fill']:<10.0f}s "
                      f"{row['fill_vs_start_limit']:<+11.2f} "
                      f"{row['fill_vs_creation_mid']:<+10.2f}")
            
            # Summary statistics
            avg_time_to_fill = np.mean([row['time_to_fill'] for row in table_data])
            avg_vs_start_limit = np.mean([row['fill_vs_start_limit'] for row in table_data])
            avg_vs_creation_mid = np.mean([row['fill_vs_creation_mid'] for row in table_data])
            
            print("-" * 100)
            print(f"Summary:")
            print(f"  • Average time to fill: {avg_time_to_fill:.0f} seconds")
            print(f"  • Average vs start limit: ${avg_vs_start_limit:+.2f} (should be ~$0 with split=0)")
            print(f"  • Average vs creation mid: ${avg_vs_creation_mid:+.2f}")
    
    # Create visualization
    print(f"\n📈 Creating Dutch order visualization...")
    create_dutch_visualization(fills, order_creations, price_path, config, strategy)
    
    return {
        'fills': fills,
        'order_creations': order_creations,
        'price_path': price_path,
        'results': results,
        'config': config
    }


def create_dutch_visualization(fills, order_creations, price_path, config, strategy):
    """Create comprehensive Dutch order visualization."""
    
    fig, axes = setup_figure(2, 2, (16, 12))
    fig.suptitle('Dutch Limit Orders on GBM: 50bps Start, 10bp/s Decay, Split=0', fontsize=14, fontweight='bold')
    
    # Convert time to minutes
    if price_path:
        times_min = [t/60 for t, _ in price_path]
        prices = [p for _, p in price_path]
    else:
        times_min, prices = [], []
    
    # Plot 1: Price path with order events and limit price evolution
    ax1 = axes[0, 0]
    
    # Plot price path
    if times_min and prices:
        ax1.plot(times_min, prices, 'b-', linewidth=1, alpha=0.7, label='GBM Price Path')
    
    # Plot order creation points
    if order_creations:
        creation_times = [t/60 for t, _, _, _ in order_creations]
        creation_mid_prices = [p for _, _, p, _ in order_creations]
        start_limits = [sl for _, _, _, sl in order_creations]
        
        ax1.scatter(creation_times, creation_mid_prices, s=40, color='orange', 
                   marker='^', edgecolors='black', linewidth=1, 
                   label='Order Created (Mid)', zorder=5)
        ax1.scatter(creation_times, start_limits, s=40, color='purple', 
                   marker='s', edgecolors='black', linewidth=1, 
                   label='Starting Limit Price', zorder=5)
    
    # Plot fill events
    if fills:
        fill_times = [f.timestamp/60 for f in fills]
        fill_prices = [f.price for f in fills]
        ax1.scatter(fill_times, fill_prices, s=50, color='red', 
                   marker='o', edgecolors='black', linewidth=1, 
                   label='Order Filled', zorder=10)
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Path with Dutch Order Events')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Limit price decay visualization
    ax2 = axes[0, 1]
    
    if order_creations:
        # Show theoretical limit price decay for first few orders
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, (creation_time, order_id, mid_price, start_limit) in enumerate(order_creations[:5]):
            # Calculate theoretical decay curve
            max_time = min(creation_time + strategy.order_duration, config.horizon)
            time_range = np.arange(creation_time, max_time, 1.0)
            
            limit_prices = []
            for t in time_range:
                time_elapsed = t - creation_time
                decay_amount = strategy.decay_bps_per_second * time_elapsed / 10000
                current_limit = start_limit * (1 - decay_amount)
                limit_prices.append(current_limit)
            
            color = colors[i % len(colors)]
            ax2.plot([t/60 for t in time_range], limit_prices, 
                    color=color, linewidth=2, alpha=0.7, 
                    label=f'{order_id}')
            
            # Mark creation point
            ax2.scatter([creation_time/60], [start_limit], 
                       s=50, color=color, marker='o', 
                       edgecolors='black', linewidth=1, zorder=10)
    
    # Add market mid price reference
    if times_min and prices:
        ax2.plot(times_min, prices, 'k--', linewidth=1, alpha=0.5, label='Market Mid')
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Limit Price ($)')
    ax2.set_title('Dutch Order Limit Price Decay (10bp/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Fill timing analysis
    ax3 = axes[1, 0]
    
    if fills and order_creations:
        creation_lookup = {oid: ct for ct, oid, _, _ in order_creations}
        fill_delays = []
        order_numbers = []
        
        for i, fill in enumerate(fills):
            if fill.order_id in creation_lookup:
                delay = fill.timestamp - creation_lookup[fill.order_id]
                fill_delays.append(delay)
                # Extract order number from dutch_X format
                order_numbers.append(int(fill.order_id.split('_')[-1]))
        
        if fill_delays:
            ax3.bar(order_numbers, fill_delays, color='skyblue', 
                   edgecolor='black', linewidth=1, alpha=0.7)
            
            # Add value labels
            for order_num, delay in zip(order_numbers, fill_delays):
                ax3.text(order_num, delay + max(fill_delays) * 0.01, 
                        f'{delay:.0f}s', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Order Number')
    ax3.set_ylabel('Time to Fill (seconds)')
    ax3.set_title('Dutch Order Fill Timing')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fill price vs limit price analysis
    ax4 = axes[1, 1]
    
    if fills and order_creations:
        creation_data = {oid: (ct, mp, sl) for ct, oid, mp, sl in order_creations}
        
        fill_vs_limit = []
        order_nums = []
        
        for fill in fills:
            if fill.order_id in creation_data:
                _, _, start_limit = creation_data[fill.order_id]
                diff = fill.price - start_limit
                fill_vs_limit.append(diff)
                # Extract order number from dutch_X format
                order_nums.append(int(fill.order_id.split('_')[-1]))
        
        if fill_vs_limit:
            colors = ['green' if x >= 0 else 'red' for x in fill_vs_limit]
            ax4.bar(order_nums, fill_vs_limit, color=colors, 
                   edgecolor='black', linewidth=1, alpha=0.7)
            
            # Add zero line
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            # Add value labels
            for order_num, diff in zip(order_nums, fill_vs_limit):
                ax4.text(order_num, diff + (max(fill_vs_limit) - min(fill_vs_limit)) * 0.01, 
                        f'${diff:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Order Number')
    ax4.set_ylabel('Fill Price - Start Limit ($)')
    ax4.set_title('Fill Price vs Starting Limit (Split=0)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = save_figure("dutch_orders_gbm_simulation.png", "results/order_debug")
    print(f"📁 Visualization saved to: {output_file}")


def main():
    """Main function."""
    return test_dutch_orders_with_gbm()


if __name__ == "__main__":
    main() 