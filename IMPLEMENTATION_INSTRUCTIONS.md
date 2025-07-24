# Execution‑Algorithm Simulator – **Implementation Blueprint**

> **Audience:** This document is written **for an autonomous coding AI agent**.  
> **Goal:** Generate a runnable Python codebase that can compare multiple execution algorithms under a stochastic price‑and‑liquidity environment (TWAP market orders, Dutch limits, Adaptive/Bayesian limits, etc.).  
> **Constraints:**  
> * Modular – every component must be swappable via dependency‑injection (no deep import chains).  
> * Reproducible – one YAML file defines an experiment; `python -m src.simulation <cfg>` runs it deterministically on a given seed.  
> * Test‑first – unit & property tests for every public interface.  

---

## 0. Quick Scaffold

├── pyproject.toml          # deps + tools (black, isort, mypy, pytest)
├── README.md               # quick‑start for humans
├── configs/                # YAML experiment recipes
├── src/
│   ├── core/               # pure dataclasses & enums
│   ├── market/             # price, spread, impact, liquidity
│   ├── cost/               # gas & fee models, filler decision
│   ├── strategy/           # execution algorithms (policies)
│   ├── engine/             # SimPy event loop + matching
│   ├── metrics/            # probes & report builders
│   └── simulation.py       # CLI entry point
└── tests/                  # pytest suites

> **Task 0.1** – generate the folders above (empty `__init__.py` in each).  
> **Task 0.2** – create `pyproject.toml` with: `python>=3.12`, `numpy`, `pandas`, `simpy`, `pydantic`, `pytest`, `hypothesis`, `pyyaml`, `typing‑extensions`, `ruff`, `mypy`.

---

## 1. Core Layer (`src/core/`)

### 1.1 Domain Objects

```python
# src/core/orders.py
from dataclasses import dataclass
from enum import Enum

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass(frozen=True, slots=True)
class Order:
    id: str
    side: Side
    qty: float
    limit_px: float | None        # None ⇒ market order
    placed_at: float              # simulation time (seconds)
    valid_to: float               # GTT time

# src/core/events.py
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    timestamp: float
    qty: float
    price: float
    gas_paid: float

Rule: No methods in core/*. Pure value objects only.
Unit test: Dataclasses must be hashable & order‑independent.

⸻

2. Market Layer (src/market/)

2.1 Interfaces (use Protocol from typing_extensions)

# src/market/protocols.py
from typing import Protocol

class PriceProcess(Protocol):
    def mid_price(self, t: float) -> float: ...

class LiquidityModel(Protocol):
    def crossed(self, side: Side, limit_px: float, mid_px: float) -> bool: ...

class ImpactModel(Protocol):
    def exec_price(self, order: Order, mid_px: float) -> float: ...

2.2 Reference Implementations
	•	gbm.py – Geometric Brownian Motion (μ, σ, dt)
	•	const_spread.py – fixed half‑spread; fill if abs(mid - limit) >= spread.
	•	linear_impact.py – market impact mid ± γ·qty.

Task 2.x – each implementation must accept a RandomState instance for reproducibility.

⸻

3. Cost Layer (src/cost/)

# gas_model.py
class GasModel(Protocol):
    def gas_fee(self, gas_used: int) -> float: ...

class Evm1559(GasModel):
    ...

# filler.py
class FillerDecision:
    """Return True if profit ≥ gas."""
    def should_fill(self, side: Side, limit_px: float, mid_px: float,
                    qty: float, gas_fee: float) -> bool:
        pnl = (mid_px - limit_px) * qty if side is Side.SELL else (limit_px - mid_px) * qty
        return pnl >= gas_fee


⸻

4. Strategy Layer (src/strategy/)

4.1 Common Protocol

class ExecutionAlgo(Protocol):
    def step(self,
             clock: float,
             broker_state: "BrokerState",
             market_state: "MarketSnapshot"
    ) -> list["OrderInstruction"]: ...

OrderInstruction = (place|cancel, order_id, kwargs…).

4.2 Algos to Implement

File	Description
twap_market.py	Post equal‑sized market orders at fixed intervals.
dutch_limit.py	Emit equal‑size limit; every second move price 1 tick toward contra‑side until filled.
adaptive_limit.py	Limit price drifts + size grows linearly; on (partial) fill, price jumps back k·spread.

Use config parameters (slice_qty, drift_rate, grow_rate, price_reversion, etc.).

⸻

5. Engine Layer (src/engine/)

5.1 SimPy Loop Sketch

import simpy

class MatchingEngine:
    def __init__(..., probes: list[Probe]):
        self.env = simpy.Environment()
        ...

    def run(self, algo: ExecutionAlgo, horizon: float):
        while self.env.now < horizon and not self.done:
            # 1. ask algo for instructions
            # 2. process instructions (place/cancel)
            # 3. check for fills via LiquidityModel / ImpactModel
            # 4. notify probes
            # 5. advance env by smallest next event (price tick, algo interval)

All mutable state lives in the engine.
Strategies cannot mutate engine internals—only send OrderInstructions.

Test: given a deterministic price path, order lifecycle is deterministic and total filled ≤ target qty.

⸻

6. Metrics Layer (src/metrics/)

class Probe(Protocol):
    def on_fill(self, fill: Fill): ...
    def on_step(self, t: float): ...
    def final(self) -> dict[str, Any]: ...

Reference probes:
	•	fill_timings.py – list of fill timestamp(s).
	•	fill_prices.py – list of fill price(s).
	•	slippage.py – volume‑weighted average vs. mid‑price at start.

All probes are attached in the YAML config.

⸻

7. Configuration (configs/)

Example twap_vs_dutch.yml:

run_name: twap_dutch_sample
horizon: 3600     # sec
random_seed: 42
price_process:
  name: GBM
  params: { mu: 0.00, sigma: 0.02, dt: 1 }
liquidity:
  name: ConstSpread
  params: { spread: 0.5 }
impact:
  name: LinearImpact
  params: { gamma: 1e-4 }
gas:
  name: Evm1559
  params: { base_fee: 5e-9, tip: 2e-9 }
strategies:
  - algo: TwapMarket
    params: { n_slices: 12 }
  - algo: DutchLimit
    params: { slice_qty: 10, drift: 0.5 }
probes:
  - FillTimings
  - FillPrices
paths: 500


⸻

8. CLI Entry‑Point (src/simulation.py)
	1.	Parse YAML ➜ instantiate objects via factory registry (REGISTRY: dict[str, type]).
	2.	Spawn multiprocessing.Pool if paths > 1.
	3.	Serialize probe results to ./results/<run_name>/run_<seed>.json.
	4.	Aggregate (mean, std) into summary.json.

⸻

9. Tests (tests/)

File	Purpose
test_core_hashable.py	Order & Fill are hashable, immutable.
test_dutch_limit_state.py	On static mid‑price, price drift hits opposite side in expected ticks.
test_engine_conservation.py	Filled qty ≤ original order qty for any Hypothesis price path.
test_regression_golden.py	Seed 42, cfg X must produce identical fills JSON (golden file).

Task 9.1 – integrate pytest --cov in CI stage (GitHub Actions template).

⸻

10. Development Checklist for the Agent
	1.	Generate project tree (Task 0.1).
	2.	Create pyproject.toml with pinned versions (Task 0.2).
	3.	Implement core dataclasses ➜ pass tests/test_core_hashable.py.
	4.	Create market protocols + GBM + ConstSpread.
	5.	Write engine skeleton that can simulate a single trivial strategy.
	6.	Add TWAP strategy ➜ regression test on 3‑step price path.
	7.	Add Dutch strategy ➜ unit test price drift logic.
	8.	Add Adaptive strategy.
	9.	Implement probes.
	10.	Finish CLI + YAML factory registry.
	11.	Write Hypothesis property tests.
	12.	CI pipeline (lint, mypy, tests, coverage ≥ 90 %).
	13.	Produce final README.md usage instructions.

⸻

11. ASCII Sequence Diagram (Engine–Strategy loop)

Strategy      Engine        Market Models
   |             |                |
   |---- step() ->|                |
   |<- fills -----|                |
   |              |-- mid(t) ----->|
   |              |<-- price ------|
   |              |-- crossed? --->|
   |              |<-- bool -------|
   |              |                |


⸻

12. Mermaid Component Diagram (for docs)

graph TD
  subgraph Core
    Orders
    Events
  end
  subgraph Market
    PriceProcess
    LiquidityModel
    ImpactModel
  end
  subgraph Cost
    GasModel
    FillerDecision
  end
  Strategy -->|OrderInstruction| Engine
  Engine --> Market
  Engine --> Cost
  Engine --> Core
  Engine --> Metrics


⸻

13. Coding Standards
	•	Type‑checked (mypy --strict).
	•	Formatted (black).
	•	Linted (ruff).
	•	No circular imports; enforce with pytest --import-mode=importlib.

⸻

14. Deliverables
	•	A git repo matching the scaffold.
	•	All tests green (pytest -q).
	•	Example run:

python -m src.simulation configs/twap_vs_dutch.yml

emits results/twap_dutch_sample/summary.json containing:

{
  "algo": "TwapMarket",
  "avg_fill_time": 178.3,
  "avg_fill_px": 100.25,
  ...
}



⸻

End of blueprint – begin implementation.

