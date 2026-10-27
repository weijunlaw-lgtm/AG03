# 🦅 MNQ EBP Strategy Engine (Micro E-mini Nasdaq-100 Futures)

A high-performance, institutional quantitative backtesting and grid-search engine for the **Micro E-mini Nasdaq-100 Futures (`MNQ`)** based on the **Engulfing / Sweep Bar Pattern (`EBP`)**.

Built with a **VectorBT + Numba (`@njit`) + pandas + numpy** hybrid architecture supporting **unlimited concurrent overlapping positions** subject to a strict **Dynamic Portfolio Contract Cap (50 contracts)**.

---

## 🏛️ Strategy Specifications & Rules

1. **Target Asset:** Micro E-mini Nasdaq-100 Futures (`MNQ`)
   - Tick Size: $0.25\text{ pts} = \$0.50\text{ USD}$
   - Point Value: $1.00\text{ pt} = \$2.00\text{ USD}$
   - Resolution: `5m` (5-minute intraday bars)
   - Reference Timezone: `America/New_York` (EST/EDT)
2. **Signal Pattern (Bar $t$ Close):**
   - **Bull EBP:** $\text{Low}_t < \text{Low}_{t-1} \land \text{Close}_t > \text{Open}_{t-1}$
   - **Bear EBP:** $\text{High}_t > \text{High}_{t-1} \land \text{Close}_t < \text{Open}_{t-1}$
   - **EMA Filter:** Long when $\text{Close}_t > \text{EMA}_n$; Short when $\text{Close}_t < \text{EMA}_n$.
   - **Session Window:** Only enter when `can_enter_mask` is True (prior to daily cutoff, e.g. 15:50 EST).
3. **Structural Stop & 50-Point Rule:**
   - Let $R_t = \text{High}_t - \text{Low}_t$.
   - If $R_t > 50.0\text{ pts}$: Stop anchor is $\text{Mid}_t = \frac{\text{High}_t + \text{Low}_t}{2.0}$.
   - Else: Long stop is $\text{Low}_t$; Short stop is $\text{High}_t$.
4. **Execution & Position Sizing (Open of Bar $t+1$):**
   - **Adverse Slippage:** $1.5\text{ ticks} = 0.375\text{ pts}$ on entries, stop-outs, and session flattens.
   - **Zero Lookahead:** Signals evaluated at Close of Bar $t$; fill price is $\text{Open}_{t+1} \pm \text{Slippage}$.
   - **Gap Guard:** If $\text{SL\_Dist} \le 0$, abort and skip trade.
   - **Arithmetic Half-Up Rounding:** $\text{Contracts} = \text{int}(\text{floor}(\frac{300.0}{\text{SL\_Dist} \times 2.0} + 0.5))$.
   - **Portfolio Contract Cap:** Max 50 aggregate open contracts across all active concurrent trade slots.
5. **Intrabar Pessimistic Priority Order:**
   - **Priority 1 (Stop-Loss):** Pessimistic tie-breaker if both SL and TP breached in same bar.
   - **Priority 2 (Take-Profit):** Limit fill at $\max(\text{Open}, \text{TP})$ / $\min(\text{Open}, \text{TP})$ without adverse slippage.
   - **Priority 3 (Flatten):** Hard flatten at daily prop firm cutoff (`16:05 EST` bar close before 16:10 EST).

---

## 🏗️ File Architecture

```
AG03/
├── config.py                 # StrategyConfig & GridSearchConfig dataclasses
├── data_loader.py            # Data ingestion, timezone normalization, session masks & synthetic generator
├── signals.py                # Vectorized Bull/Bear EBP, EMA trend filter & 50pt rule stops
├── execution_kernel.py       # Multi-slot Numba @njit execution kernel & MTM equity tracking
├── metrics_vbt.py            # VectorBT vbt.Trades wrapper & institutional/prop-firm metrics
├── run_grid_search.py        # 30-combination orthogonal grid search runner
├── nq_ebp_strategy.py        # Top-level unified strategy interface & CLI runner
├── tests/
│   └── test_nq_ebp_strategy.py # 14 comprehensive unit & integration tests
├── mnq_ebp_strategy_specification.md # Formal institutional specification
└── README.md                 # Documentation & usage instructions
```

---

## 🚀 Quickstart & Usage

### 1. Run Strategy Backtest (CLI)
```bash
python nq_ebp_strategy.py --days 15
```

### 2. Run Orthogonal Grid Search (30 Combinations)
```bash
python nq_ebp_strategy.py --grid
# or
python run_grid_search.py
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_nq_ebp_strategy.py
```

### 4. Programmatic Python Usage
```python
from nq_ebp_strategy import NQEBPStrategy, StrategyConfig

config = StrategyConfig(ema_period=20, rr_ratio=2.0)
strategy = NQEBPStrategy(config)

# Load data or generate synthetic MNQ data
df = strategy.load_data(days=20)

# Execute backtest
result = strategy.run(df)
print(result.summary())

# Access trade ledger and VectorBT trades object
print(result.trades_df.head())
vbt_trades = result.vbt_trades
```
