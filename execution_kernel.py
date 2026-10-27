"""
Numba-accelerated (@njit) multi-slot execution kernel for MNQ EBP strategy.

Enforces:
1. Zero lookahead bias: Signal evaluated at close of Bar t; entry fill at Open of Bar t+1.
2. Realistic adverse slippage on entries, stop-outs, and session flattens.
3. Unlimited concurrent overlapping positions bounded by 50-contract aggregate cap.
4. Strict half-up contract rounding: int(floor(raw_contracts + 0.5)).
5. Strict intrabar pessimistic priority order: Priority 1 (SL) -> Priority 2 (TP) -> Priority 3 (Flatten).
6. Full bar-by-bar Mark-to-Market (MTM) equity tracking for prop firm intraday trailing drawdown.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
from numba import njit

from config import StrategyConfig
from signals import StrategySignals


@dataclass
class ExecutionResult:
    """Complete ledger of executed trades and bar-by-bar portfolio state."""

    # Trade ledger arrays
    entry_bars: np.ndarray  # int64[:]
    exit_bars: np.ndarray  # int64[:]
    directions: np.ndarray  # int32[:] (1 = Long, -1 = Short)
    entry_prices: np.ndarray  # float64[:]
    exit_prices: np.ndarray  # float64[:]
    sl_prices: np.ndarray  # float64[:]
    tp_prices: np.ndarray  # float64[:]
    sl_dists: np.ndarray  # float64[:]
    contracts: np.ndarray  # int64[:]
    gross_pts: np.ndarray  # float64[:]
    net_pnl_usd: np.ndarray  # float64[:]
    commission_usd: np.ndarray  # float64[:]
    exit_reasons: np.ndarray  # int32[:] (1 = SL, 2 = TP, 3 = Flatten)
    used_midpoint_rule: np.ndarray  # int32[:] (1 if >50pt rule used, else 0)

    # Diagnostic & aggregate counters
    skipped_by_contract_cap: int
    peak_concurrent_contracts: int
    peak_concurrent_slots: int

    # Bar-by-bar portfolio series (length N)
    equity_realized: np.ndarray  # float64[:]
    equity_unrealized: np.ndarray  # float64[:]
    mtm_equity: np.ndarray  # float64[:]
    open_contracts_series: np.ndarray  # int64[:]


@njit
def _ebp_numba_kernel(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sl_long_arr: np.ndarray,
    sl_short_arr: np.ndarray,
    is_midpoint_arr: np.ndarray,
    force_flatten_arr: np.ndarray,
    rr_ratio: float,
    risk_per_trade_usd: float,
    max_prop_contracts: int,
    slippage_pts: float,
    commission_rt: float,
    point_value: float,
    initial_capital: float,
):
    n_bars = len(open_arr)
    max_trades = max(1000, n_bars)

    # Output trade ledger buffers
    out_entry_bar = np.empty(max_trades, dtype=np.int64)
    out_exit_bar = np.empty(max_trades, dtype=np.int64)
    out_direction = np.empty(max_trades, dtype=np.int32)
    out_entry_price = np.empty(max_trades, dtype=np.float64)
    out_exit_price = np.empty(max_trades, dtype=np.float64)
    out_sl_price = np.empty(max_trades, dtype=np.float64)
    out_tp_price = np.empty(max_trades, dtype=np.float64)
    out_sl_dist = np.empty(max_trades, dtype=np.float64)
    out_contracts = np.empty(max_trades, dtype=np.int64)
    out_gross_pts = np.empty(max_trades, dtype=np.float64)
    out_net_pnl_usd = np.empty(max_trades, dtype=np.float64)
    out_commission_usd = np.empty(max_trades, dtype=np.float64)
    out_exit_reason = np.empty(max_trades, dtype=np.int32)
    out_used_midpoint = np.empty(max_trades, dtype=np.int32)
    n_trades = 0

    # Active trade slots buffer
    max_slots = max(100, max_prop_contracts * 2)
    active_dir = np.empty(max_slots, dtype=np.int32)
    active_entry_bar = np.empty(max_slots, dtype=np.int64)
    active_entry_price = np.empty(max_slots, dtype=np.float64)
    active_sl_price = np.empty(max_slots, dtype=np.float64)
    active_tp_price = np.empty(max_slots, dtype=np.float64)
    active_sl_dist = np.empty(max_slots, dtype=np.float64)
    active_contracts = np.empty(max_slots, dtype=np.int64)
    active_midpoint_rule = np.empty(max_slots, dtype=np.int32)
    num_active = 0
    current_open_contracts = 0

    # Diagnostic trackers
    peak_concurrent_contracts = 0
    peak_concurrent_slots = 0
    skipped_by_contract_cap = 0

    # Bar-by-bar equity tracking series
    equity_realized = np.zeros(n_bars, dtype=np.float64)
    equity_unrealized = np.zeros(n_bars, dtype=np.float64)
    mtm_equity = np.zeros(n_bars, dtype=np.float64)
    open_contracts_series = np.zeros(n_bars, dtype=np.int64)
    cum_realized_pnl = 0.0

    # Bar-by-bar simulation loop
    for i in range(n_bars):
        # -------------------------------------------------------------
        # STEP 1: Evaluate trade entry at Open of Bar i (from Bar i-1 signal)
        # -------------------------------------------------------------
        if i > 0 and not force_flatten_arr[i] and i < n_bars - 1:
            sig_idx = i - 1
            has_long = long_signal[sig_idx]
            has_short = short_signal[sig_idx]

            if has_long and not has_short:
                # Long entry fill with adverse slippage
                entry_price = open_arr[i] + slippage_pts
                sl_struct = sl_long_arr[sig_idx]
                sl_dist = entry_price - sl_struct

                # Invalid gap guard: Open_{t+1} gapped beyond stop or sl_dist <= 0
                if open_arr[i] > sl_struct and sl_dist > 0.0:
                    raw_contracts = risk_per_trade_usd / (sl_dist * point_value)
                    # Strict half-up arithmetic rounding
                    contracts = int(math.floor(raw_contracts + 0.5))
                    # Rejection rule: Contracts < 1 OR (Current_Open_Contracts + Contracts) > 50
                    if contracts < 1 or (current_open_contracts + contracts) > max_prop_contracts:
                        skipped_by_contract_cap += 1
                    else:
                        # Enter long trade slot
                        tp_price = entry_price + (sl_dist * rr_ratio)
                        slot = num_active
                        active_dir[slot] = 1
                        active_entry_bar[slot] = i
                        active_entry_price[slot] = entry_price
                        active_sl_price[slot] = sl_struct
                        active_tp_price[slot] = tp_price
                        active_sl_dist[slot] = sl_dist
                        active_contracts[slot] = contracts
                        active_midpoint_rule[slot] = 1 if is_midpoint_arr[sig_idx] else 0
                        num_active += 1
                        current_open_contracts += contracts

            elif has_short and not has_long:
                # Short entry fill with adverse slippage
                entry_price = open_arr[i] - slippage_pts
                sl_struct = sl_short_arr[sig_idx]
                sl_dist = sl_struct - entry_price

                # Invalid gap guard: Open_{t+1} gapped beyond stop or sl_dist <= 0
                if open_arr[i] < sl_struct and sl_dist > 0.0:
                    raw_contracts = risk_per_trade_usd / (sl_dist * point_value)
                    # Strict half-up arithmetic rounding
                    contracts = int(math.floor(raw_contracts + 0.5))
                    # Rejection rule: Contracts < 1 OR (Current_Open_Contracts + Contracts) > 50
                    if contracts < 1 or (current_open_contracts + contracts) > max_prop_contracts:
                        skipped_by_contract_cap += 1
                    else:
                        # Enter short trade slot
                        tp_price = entry_price - (sl_dist * rr_ratio)
                        slot = num_active
                        active_dir[slot] = -1
                        active_entry_bar[slot] = i
                        active_entry_price[slot] = entry_price
                        active_sl_price[slot] = sl_struct
                        active_tp_price[slot] = tp_price
                        active_sl_dist[slot] = sl_dist
                        active_contracts[slot] = contracts
                        active_midpoint_rule[slot] = 1 if is_midpoint_arr[sig_idx] else 0
                        num_active += 1
                        current_open_contracts += contracts

        # Track concurrent peaks
        if current_open_contracts > peak_concurrent_contracts:
            peak_concurrent_contracts = current_open_contracts
        if num_active > peak_concurrent_slots:
            peak_concurrent_slots = num_active

        # -------------------------------------------------------------
        # STEP 2: Intrabar exit priority check for all active slots
        # Strict order: Priority 1 (SL) -> Priority 2 (TP) -> Priority 3 (Flatten)
        # -------------------------------------------------------------
        is_last_bar = (i == n_bars - 1)
        force_flat = force_flatten_arr[i] or is_last_bar

        slot_idx = 0
        while slot_idx < num_active:
            direction = active_dir[slot_idx]
            entry_b = active_entry_bar[slot_idx]
            entry_p = active_entry_price[slot_idx]
            sl_p = active_sl_price[slot_idx]
            tp_p = active_tp_price[slot_idx]
            sl_d = active_sl_dist[slot_idx]
            cnts = active_contracts[slot_idx]
            mid_r = active_midpoint_rule[slot_idx]

            exited = False
            exit_price = 0.0
            exit_reason = 0  # 1 = SL, 2 = TP, 3 = Flatten

            if direction == 1:  # LONG
                # Priority 1: Stop-Loss Hit
                if low_arr[i] <= sl_p:
                    exit_price = min(open_arr[i], sl_p) - slippage_pts
                    exit_reason = 1
                    exited = True
                # Priority 2: Take-Profit Hit (Limit fill)
                elif high_arr[i] >= tp_p:
                    exit_price = max(open_arr[i], tp_p)
                    exit_reason = 2
                    exited = True
                # Priority 3: Session Flatten Cutoff
                elif force_flat:
                    exit_price = close_arr[i] - slippage_pts
                    exit_reason = 3
                    exited = True

            else:  # SHORT
                # Priority 1: Stop-Loss Hit
                if high_arr[i] >= sl_p:
                    exit_price = max(open_arr[i], sl_p) + slippage_pts
                    exit_reason = 1
                    exited = True
                # Priority 2: Take-Profit Hit (Limit fill)
                elif low_arr[i] <= tp_p:
                    exit_price = min(open_arr[i], tp_p)
                    exit_reason = 2
                    exited = True
                # Priority 3: Session Flatten Cutoff
                elif force_flat:
                    exit_price = close_arr[i] + slippage_pts
                    exit_reason = 3
                    exited = True

            if exited:
                if direction == 1:
                    gross_pts = exit_price - entry_p
                else:
                    gross_pts = entry_p - exit_price

                comm_usd = commission_rt * cnts
                net_pnl = (gross_pts * point_value - commission_rt) * cnts
                cum_realized_pnl += net_pnl

                # Record trade
                if n_trades < max_trades:
                    out_entry_bar[n_trades] = entry_b
                    out_exit_bar[n_trades] = i
                    out_direction[n_trades] = direction
                    out_entry_price[n_trades] = entry_p
                    out_exit_price[n_trades] = exit_price
                    out_sl_price[n_trades] = sl_p
                    out_tp_price[n_trades] = tp_p
                    out_sl_dist[n_trades] = sl_d
                    out_contracts[n_trades] = cnts
                    out_gross_pts[n_trades] = gross_pts
                    out_net_pnl_usd[n_trades] = net_pnl
                    out_commission_usd[n_trades] = comm_usd
                    out_exit_reason[n_trades] = exit_reason
                    out_used_midpoint[n_trades] = mid_r
                    n_trades += 1

                # Release contracts and remove active slot (swap with last)
                current_open_contracts -= cnts
                last_idx = num_active - 1
                if slot_idx != last_idx:
                    active_dir[slot_idx] = active_dir[last_idx]
                    active_entry_bar[slot_idx] = active_entry_bar[last_idx]
                    active_entry_price[slot_idx] = active_entry_price[last_idx]
                    active_sl_price[slot_idx] = active_sl_price[last_idx]
                    active_tp_price[slot_idx] = active_tp_price[last_idx]
                    active_sl_dist[slot_idx] = active_sl_dist[last_idx]
                    active_contracts[slot_idx] = active_contracts[last_idx]
                    active_midpoint_rule[slot_idx] = active_midpoint_rule[last_idx]
                num_active -= 1
            else:
                slot_idx += 1

        # -------------------------------------------------------------
        # STEP 3: Mark-to-Market (MTM) calculation for remaining open positions
        # -------------------------------------------------------------
        unrealized_usd = 0.0
        for k in range(num_active):
            d_k = active_dir[k]
            ep_k = active_entry_price[k]
            c_k = active_contracts[k]
            if d_k == 1:
                u_pts = close_arr[i] - ep_k
            else:
                u_pts = ep_k - close_arr[i]
            unrealized_usd += (u_pts * point_value) * c_k

        equity_realized[i] = cum_realized_pnl
        equity_unrealized[i] = unrealized_usd
        mtm_equity[i] = initial_capital + cum_realized_pnl + unrealized_usd
        open_contracts_series[i] = current_open_contracts

    return (
        out_entry_bar[:n_trades],
        out_exit_bar[:n_trades],
        out_direction[:n_trades],
        out_entry_price[:n_trades],
        out_exit_price[:n_trades],
        out_sl_price[:n_trades],
        out_tp_price[:n_trades],
        out_sl_dist[:n_trades],
        out_contracts[:n_trades],
        out_gross_pts[:n_trades],
        out_net_pnl_usd[:n_trades],
        out_commission_usd[:n_trades],
        out_exit_reason[:n_trades],
        out_used_midpoint[:n_trades],
        skipped_by_contract_cap,
        peak_concurrent_contracts,
        peak_concurrent_slots,
        equity_realized,
        equity_unrealized,
        mtm_equity,
        open_contracts_series,
    )


def run_numba_execution(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    signals: StrategySignals,
    force_flatten_arr: np.ndarray,
    config: Optional[StrategyConfig] = None,
) -> ExecutionResult:
    """Wrapper that executes the Numba kernel and packages outputs into ExecutionResult."""
    cfg = config or StrategyConfig()

    (
        entry_bars,
        exit_bars,
        directions,
        entry_prices,
        exit_prices,
        sl_prices,
        tp_prices,
        sl_dists,
        contracts,
        gross_pts,
        net_pnl_usd,
        commission_usd,
        exit_reasons,
        used_midpoint,
        skipped_cap,
        peak_contracts,
        peak_slots,
        eq_real,
        eq_unreal,
        mtm_eq,
        open_cnts,
    ) = _ebp_numba_kernel(
        np.ascontiguousarray(open_arr, dtype=np.float64),
        np.ascontiguousarray(high_arr, dtype=np.float64),
        np.ascontiguousarray(low_arr, dtype=np.float64),
        np.ascontiguousarray(close_arr, dtype=np.float64),
        np.ascontiguousarray(signals.long_signal, dtype=np.bool_),
        np.ascontiguousarray(signals.short_signal, dtype=np.bool_),
        np.ascontiguousarray(signals.sl_long, dtype=np.float64),
        np.ascontiguousarray(signals.sl_short, dtype=np.float64),
        np.ascontiguousarray(signals.is_midpoint_rule, dtype=np.bool_),
        np.ascontiguousarray(force_flatten_arr, dtype=np.bool_),
        float(cfg.rr_ratio),
        float(cfg.risk_per_trade_usd),
        int(cfg.max_prop_contracts),
        float(cfg.slippage_pts),
        float(cfg.commission_rt_per_contract),
        float(cfg.point_value),
        float(cfg.initial_capital),
    )

    return ExecutionResult(
        entry_bars=entry_bars,
        exit_bars=exit_bars,
        directions=directions,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        sl_prices=sl_prices,
        tp_prices=tp_prices,
        sl_dists=sl_dists,
        contracts=contracts,
        gross_pts=gross_pts,
        net_pnl_usd=net_pnl_usd,
        commission_usd=commission_usd,
        exit_reasons=exit_reasons,
        used_midpoint_rule=used_midpoint,
        skipped_by_contract_cap=int(skipped_cap),
        peak_concurrent_contracts=int(peak_contracts),
        peak_concurrent_slots=int(peak_slots),
        equity_realized=eq_real,
        equity_unrealized=eq_unreal,
        mtm_equity=mtm_eq,
        open_contracts_series=open_cnts,
    )
