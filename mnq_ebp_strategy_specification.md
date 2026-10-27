# Role & Engineering Standards
You are an institutional quantitative developer. Write a clean, modular, and vectorized backtesting engine in Python using **VectorBT**, **Numba (`@njit`)**, **pandas**, and **numpy**.

## Core Engineering Mandates
1. **Zero Lookahead Bias (无未来函数):** All signals evaluated at the close of Bar $t$ must strictly use data up to Bar $t$. Trade entry fills must occur strictly at the **Open of Bar $t+1$**.
2. **Realistic Friction:** Subtract round-turn commissions and apply adverse bid-ask slippage on every entry, stop-loss, and session-flatten fill.
3. **Hybrid VectorBT + Numba Architecture:** Because this strategy allows **unlimited concurrent overlapping positions** (no `max_open_trades` limit) bounded by a **Dynamic Portfolio Contract Cap (50 MNQ contracts)**, standard `vbt.Portfolio.from_signals` cannot be used directly (as it nets positions and overwrites individual trade SL/TP levels). You must:
   - Pre-compute all indicators, EBP bar geometries, and session masks using vectorized `numpy`/`pandas`/`vbt.MA` arrays.
   - Pass the pre-computed arrays into a custom **Numba `@njit` Multi-Slot Execution Kernel** that tracks independent open trade slots bar-by-bar.
   - Feed the resulting trade records / PnL series back into **VectorBT** (`vbt.Portfolio` / `vbt.records`) and custom analytics wrappers to generate institutional and prop-firm metrics.
4. **Orthogonal Parameter Sweep:** Decouple strategy variables so the engine executes a fast multi-parameter Grid Search across all combinations.

---

# 1. Asset & Data Specifications
- **Target Asset:** Micro E-mini Nasdaq-100 Futures (`MNQ`)
  - Tick Size: $0.25\text{ points} = \$0.50\text{ USD}$
  - Point Value: $1.00\text{ point} = 4\text{ ticks} = \$2.00\text{ USD}$
- **Bar Resolution:** `5m` (5-minute intraday bars)
- **Historical Horizon:** 5 to 10 years of intraday OHLCV data
- **Timezone Normalization:** All timestamps must be converted to and evaluated in **`America/New_York` (EST/EDT)** to strictly align with CME / Prop Firm daily cutoffs across Daylight Saving Time transitions.

---

# 2. Vectorized Signal & Structural Logic (Evaluated at Close of Bar $t$)

## 2.1 EBP (Engulfing / Sweep Bar Pattern) Definition
- **BULL EBP on Bar $t$:**
  $$\text{Bull\_EBP}_t = (\text{Low}_t < \text{Low}_{t-1}) \land (\text{Close}_t > \text{Open}_{t-1})$$
- **BEAR EBP on Bar $t$:**
  $$\text{Bear\_EBP}_t = (\text{High}_t > \text{High}_{t-1}) \land (\text{Close}_t < \text{Open}_{t-1})$$

## 2.2 Exponential Moving Average (EMA) Filter
- Compute $\text{EMA}_{n, t}$ on $\text{Close}_t$ for sweepable periods $n \in [10, 15, 20]$ (Default $= 20$).
- **Long Entry Trigger (Bar $t$):**
  $$\text{Long\_Signal}_t = \text{Bull\_EBP}_t \land (\text{Close}_t > \text{EMA}_{n, t}) \land \text{can\_enter\_mask}_t$$
- **Short Entry Trigger (Bar $t$):**
  $$\text{Short\_Signal}_t = \text{Bear\_EBP}_t \land (\text{Close}_t < \text{EMA}_{n, t}) \land \text{can\_enter\_mask}_t$$

## 2.3 Stop-Loss Structural Price ($\text{SL\_Anchor}_t$) & 50-Point Rule
Let the total range of the signal candle Bar $t$ be:
$$R_t = \text{High}_t - \text{Low}_t$$
Let the 50% High-Low midpoint of Bar $t$ be:
$$\text{Mid}_t = \frac{\text{High}_t + \text{Low}_t}{2.0}$$
- **Long Structural Stop Price ($\text{SL\_Long}_t$):**
  $$\text{SL\_Long}_t = \begin{cases} \text{Mid}_t & \text{if } R_t > 50.0\text{ points} \\ \text{Low}_t & \text{if } R_t \le 50.0\text{ points} \end{cases}$$
- **Short Structural Stop Price ($\text{SL\_Short}_t$):**
  $$\text{SL\_Short}_t = \begin{cases} \text{Mid}_t & \text{if } R_t > 50.0\text{ points} \\ \text{High}_t & \text{if } R_t \le 50.0\text{ points} \end{cases}$$

---

# 3. Execution, Sizing & Portfolio Cap Engine (Numba `@njit` Kernel)

## 3.1 Session Windows & Daily Prop Cutoff (`America/New_York`)
- `can_enter_mask`: Boolean mask allowing new trade entries only prior to the daily entry cutoff (e.g., $\text{Time}_t < \text{15:50 EST}$, sweepable).
- `force_flatten_mask`: Boolean mask triggering a hard session exit before the prop firm overnight cutoff (default cutoff at **16:10 EST / 15:10 CT**; on `5m` bars, trigger flatten at the **16:05 EST** bar close so 100% of positions are flat prior to 16:10 EST).

## 3.2 Entry Fill & Position Sizing Math (At Open of Bar $t+1$)
When $\text{Long\_Signal}_t$ or $\text{Short\_Signal}_t$ is True, evaluate trade opening at the **Open of Bar $t+1$**:

1. **Adverse Entry Fill Price ($\text{Entry\_Price}$):**
   - $\text{Slippage\_Pts} = \text{Slippage\_Ticks} \times 0.25$ (Default: $1.5\text{ ticks} = 0.375\text{ points}$).
   - **Long Fill:** $\text{Entry\_Price} = \text{Open}_{t+1} + \text{Slippage\_Pts}$
   - **Short Fill:** $\text{Entry\_Price} = \text{Open}_{t+1} - \text{Slippage\_Pts}$

2. **Stop Distance ($\text{SL\_Dist\_Pts}$) & Invalid Gap Guard:**
   - **Long:** $\text{SL\_Dist\_Pts} = \text{Entry\_Price} - \text{SL\_Long}_t$
   - **Short:** $\text{SL\_Dist\_Pts} = \text{SL\_Short}_t - \text{Entry\_Price}$
   - *Guard:* If $\text{SL\_Dist\_Pts} \le 0.0$ (e.g., $\text{Open}_{t+1}$ gapped beyond the structural stop, or a >50pt candle closed beyond its 50% midpoint), **abort and skip the trade**.

3. **Contract Sizing & Strict Half-Up Rounding:**
   - Risk per trade target: $\text{Risk\_USD} = \$300.00$.
   - Raw contracts:
     $$\text{Raw\_Contracts} = \frac{300.0}{\text{SL\_Dist\_Pts} \times 2.0}$$
   - **Rounding Rule (Half-Up Arithmetic Rounding, NOT Python banker's rounding):**
     If the decimal portion is $\ge 0.5$, round up; otherwise round down (e.g., $4.24 \rightarrow 4$, $4.49 \rightarrow 4$, $4.50 \rightarrow 5$). Implement explicitly in Numba as:
     $$\text{Contracts} = \text{int}(\text{floor}(\text{Raw\_Contracts} + 0.5))$$

4. **Concurrent Position & Prop Firm Aggregate Contract Cap (`max_contracts = 50`):**
   - There is **no limit** on the number of concurrent open trades (`max_open_trades = Unlimited`).
   - However, total open contracts across all active running trades cannot exceed `max_prop_contracts = 50`.
   - Let $\text{Current\_Open\_Contracts} = \sum_{k \in \text{ActiveTrades}} \text{Contracts}_k$.
   - **Rejection Rule:** If $\text{Contracts} < 1$ **OR** $(\text{Current\_Open\_Contracts} + \text{Contracts}) > 50$:
     - **Do NOT enter the trade** (not even partially). Increment a diagnostic counter `skipped_by_contract_cap`.

5. **Take-Profit Price ($\text{TP\_Price}$):**
   - For sweepable $\text{RR\_Ratio} \in [1.0, 1.5, 2.0, 2.5, 3.0]$:
   - **Long TP:** $\text{TP\_Price} = \text{Entry\_Price} + (\text{SL\_Dist\_Pts} \times \text{RR\_Ratio})$
   - **Short TP:** $\text{TP\_Price} = \text{Entry\_Price} - (\text{SL\_Dist\_Pts} \times \text{RR\_Ratio})$

## 3.3 Intrabar Exit Priority (Evaluated on Every Bar $i \ge t+1$ for Every Active Slot)
For each active trade slot on Bar $i$ (including the entry bar $t+1$ after fill), check exits in this strict **pessimistic priority order**:

1. **Priority 1 — Stop-Loss Hit First (Pessimistic Intrabar Tie-Breaker):**
   - **Long:** If $\text{Low}_i \le \text{SL\_Price}$, exit at:
     $$\text{Exit\_Price} = \min(\text{Open}_i, \text{SL\_Price}) - \text{Slippage\_Pts}$$
   - **Short:** If $\text{High}_i \ge \text{SL\_Price}$, exit at:
     $$\text{Exit\_Price} = \max(\text{Open}_i, \text{SL\_Price}) + \text{Slippage\_Pts}$$
   - *(Note: If both SL and TP are breached within the same 5-minute bar, Priority 1 guarantees SL is triggered first).*

2. **Priority 2 — Take-Profit Hit:**
   - **Long:** Else if $\text{High}_i \ge \text{TP\_Price}$, exit at:
     $$\text{Exit\_Price} = \max(\text{Open}_i, \text{TP\_Price})$$
   - **Short:** Else if $\text{Low}_i \le \text{TP\_Price}$, exit at:
     $$\text{Exit\_Price} = \min(\text{Open}_i, \text{TP\_Price})$$

3. **Priority 3 — Daily Prop Cutoff Flatten:**
   - Else if `force_flatten_mask[i]` is True:
     - **Long:** $\text{Exit\_Price} = \text{Close}_i - \text{Slippage\_Pts}$
     - **Short:** $\text{Exit\_Price} = \text{Close}_i + \text{Slippage\_Pts}$

4. **Net Trade PnL Calculation:**
   - **Long Gross Points:** $\text{Gross\_Pts} = \text{Exit\_Price} - \text{Entry\_Price}$
   - **Short Gross Points:** $\text{Gross\_Pts} = \text{Entry\_Price} - \text{Exit\_Price}$
   - **Net USD PnL:**
     $$\text{Net\_PnL\_USD} = \left( \text{Gross\_Pts} \times 2.00 - \text{Commission\_RT} \right) \times \text{Contracts}$$
     *(Default `Commission_RT` = $\$1.24$ per contract round-turn, configurable).*

---

# 4. Orthogonal Parameter Matrix (Grid Search Variables)
Structure the runner to sweep all combinations of the following parameters:
- `ema_period`: `[10, 15, 20]` (Default: `20`)
- `rr_ratio`: `[1.0, 1.5, 2.0, 2.5, 3.0]`
- `entry_cutoff_est`: `['15:30', '15:55']` (Configurable time gate before the `16:05` flatten)
- **Fixed / Configurable Constants:**
  - `risk_per_trade_usd`: `300.0`
  - `max_prop_contracts`: `50`
  - `large_candle_threshold_pts`: `50.0`
  - `slippage_ticks_per_side`: `1.5` ($0.375\text{ points}$)
  - `commission_rt_per_contract`: `1.24` (USD)
  - `flatten_time_est`: `'16:05'` (Flattens before 16:10 EST / 15:10 CT prop deadline)

---

# 5. Code Deliverable & Required Output Metrics
Organize the codebase cleanly into modular files (`config.py`, `data_loader.py`, `signals.py`, `execution_kernel.py`, `metrics_vbt.py`, `run_grid_search.py`).

For every parameter combination in the Grid Search, reconstruct the equity curve and trade ledger in **VectorBT** and output a summary Leaderboard DataFrame containing:
1. **Standard Prop Firm & Institutional Metrics:**
   - Total Net Profit ($), Gross Profit ($), Gross Loss ($), Total Commissions Paid ($)
   - Profit Factor, Sharpe Ratio (annualized), Sortino Ratio, Calmar Ratio
   - Max End-of-Day (EOD) Drawdown ($) and Max Intraday Peak-to-Valley Trailing Drawdown ($)
   - Worst Single Day PnL ($) and Best Single Day PnL ($)
2. **Hold Time Analytics:**
   - Average Hold Time (in minutes) and Median Hold Time (in minutes)
3. **Position & Contract Sizing Analytics:**
   - Average Contract Size (Overall, Longs Only, Shorts Only)
   - Peak Concurrent Open Contracts Reached (`<= 50`)
   - Peak Concurrent Open Trade Slots
   - Total Signals Skipped Due to 50-Contract Cap (`skipped_by_contract_cap`)
4. **Directional Win Rate Breakdown:**
   - Overall Win Rate (%)
   - Long Win Rate (%) & Total Long Trades
   - Short Win Rate (%) & Total Short Trades
5. **Stop-Loss Geometry Metrics:**
   - Average Stop Distance (in MNQ points)
   - Percentage of Trades Using the `> 50pt` 50% Midpoint Rule (%)
6. **Trade Frequency:**
   - Average Executed Trades Per Active Trading Day