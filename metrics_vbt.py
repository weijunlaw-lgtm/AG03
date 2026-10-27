"""
VectorBT integration and institutional/prop-firm metric analytics for MNQ EBP.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import vectorbt as vbt

from config import StrategyConfig
from execution_kernel import ExecutionResult


def build_vbt_trades(
    df: pd.DataFrame, exec_res: ExecutionResult
) -> Optional[vbt.Trades]:
    """
    Construct a native VectorBT Trades object from execution records.
    Maps trade slots into vbt.portfolio.enums.trade_dt records.
    """
    n_trades = len(exec_res.entry_bars)
    if n_trades == 0:
        return None

    close_series = df["Close"]
    wrapper = vbt.ArrayWrapper.from_obj(close_series)

    rec_arr = np.empty(n_trades, dtype=vbt.portfolio.enums.trade_dt)
    point_val = 2.0

    for i in range(n_trades):
        # In VectorBT trade_dt: direction 0 = Long, 1 = Short
        vbt_direction = 0 if exec_res.directions[i] == 1 else 1
        entry_b = int(exec_res.entry_bars[i])
        exit_b = int(exec_res.exit_bars[i])
        entry_p = float(exec_res.entry_prices[i])
        exit_p = float(exec_res.exit_prices[i])
        cnts = float(exec_res.contracts[i])
        net_pnl = float(exec_res.net_pnl_usd[i])
        comm = float(exec_res.commission_usd[i])

        invested = entry_p * point_val * cnts
        ret = net_pnl / invested if invested > 0 else 0.0

        rec_arr[i] = (
            i,  # id
            0,  # col
            cnts,  # size
            entry_b,  # entry_idx
            entry_p,  # entry_price
            comm / 2.0,  # entry_fees
            exit_b,  # exit_idx
            exit_p,  # exit_price
            comm / 2.0,  # exit_fees
            net_pnl,  # pnl
            ret,  # return
            vbt_direction,  # direction
            1,  # status (1 = closed)
            -1,  # parent_id
        )

    return vbt.Trades(wrapper, rec_arr, close_series)


def trades_to_dataframe(
    df: pd.DataFrame, exec_res: ExecutionResult
) -> pd.DataFrame:
    """Format execution output arrays into a structured pandas DataFrame."""
    n_trades = len(exec_res.entry_bars)
    if n_trades == 0:
        return pd.DataFrame()

    timestamps = df.index
    reason_map = {1: "Stop-Loss", 2: "Take-Profit", 3: "Flatten"}

    records = []
    for i in range(n_trades):
        en_bar = exec_res.entry_bars[i]
        ex_bar = exec_res.exit_bars[i]
        records.append(
            {
                "trade_id": i + 1,
                "direction": "LONG" if exec_res.directions[i] == 1 else "SHORT",
                "entry_time": timestamps[en_bar],
                "exit_time": timestamps[ex_bar],
                "hold_time_minutes": (ex_bar - en_bar) * 5,
                "contracts": exec_res.contracts[i],
                "entry_price": round(exec_res.entry_prices[i], 4),
                "exit_price": round(exec_res.exit_prices[i], 4),
                "sl_price": round(exec_res.sl_prices[i], 4),
                "tp_price": round(exec_res.tp_prices[i], 4),
                "sl_distance_pts": round(exec_res.sl_dists[i], 4),
                "gross_pts": round(exec_res.gross_pts[i], 4),
                "net_pnl_usd": round(exec_res.net_pnl_usd[i], 2),
                "commission_usd": round(exec_res.commission_usd[i], 2),
                "exit_reason": reason_map.get(exec_res.exit_reasons[i], "Unknown"),
                "used_50pt_rule": bool(exec_res.used_midpoint_rule[i]),
            }
        )

    return pd.DataFrame(records)


def calculate_performance_metrics(
    df: pd.DataFrame, exec_res: ExecutionResult, config: Optional[StrategyConfig] = None
) -> Dict[str, Any]:
    """
    Compute all institutional and prop-firm metrics required by specification.
    """
    cfg = config or StrategyConfig()
    n_trades = len(exec_res.entry_bars)

    # Handle empty DataFrame safely
    if len(df) == 0:
        return {
            "total_trades": 0,
            "active_days": 0,
            "total_net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_commissions": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_eod_drawdown": 0.0,
            "max_intraday_drawdown": 0.0,
            "worst_single_day_pnl": 0.0,
            "best_single_day_pnl": 0.0,
            "avg_hold_time_minutes": 0.0,
            "median_hold_time_minutes": 0.0,
            "avg_contract_size": 0.0,
            "avg_long_contract_size": 0.0,
            "avg_short_contract_size": 0.0,
            "peak_concurrent_contracts": 0,
            "peak_concurrent_slots": 0,
            "skipped_by_contract_cap": exec_res.skipped_by_contract_cap if exec_res else 0,
            "overall_win_rate_pct": 0.0,
            "long_win_rate_pct": 0.0,
            "total_long_trades": 0,
            "short_win_rate_pct": 0.0,
            "total_short_trades": 0,
            "avg_sl_distance_pts": 0.0,
            "pct_midpoint_rule_trades": 0.0,
            "avg_trades_per_day": 0.0,
        }

    # 1. Financial totals & ratios
    if n_trades > 0:
        pnls = exec_res.net_pnl_usd
        total_net_profit = float(np.sum(pnls))
        gross_profit = float(np.sum(pnls[pnls > 0]))
        gross_loss = float(np.sum(pnls[pnls < 0]))
        total_commissions = float(np.sum(exec_res.commission_usd))
        abs_loss = abs(gross_loss)
        if abs_loss > 0:
            profit_factor = round(gross_profit / abs_loss, 4)
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        wins = np.sum(pnls > 0)
        overall_win_rate = round((wins / n_trades) * 100.0, 2)

        # Longs
        long_mask = exec_res.directions == 1
        total_longs = int(np.sum(long_mask))
        if total_longs > 0:
            long_wins = np.sum((pnls > 0) & long_mask)
            long_win_rate = round((long_wins / total_longs) * 100.0, 2)
            avg_long_contracts = round(float(np.mean(exec_res.contracts[long_mask])), 2)
        else:
            long_win_rate = 0.0
            avg_long_contracts = 0.0

        # Shorts
        short_mask = exec_res.directions == -1
        total_shorts = int(np.sum(short_mask))
        if total_shorts > 0:
            short_wins = np.sum((pnls > 0) & short_mask)
            short_win_rate = round((short_wins / total_shorts) * 100.0, 2)
            avg_short_contracts = round(float(np.mean(exec_res.contracts[short_mask])), 2)
        else:
            short_win_rate = 0.0
            avg_short_contracts = 0.0

        # Hold times
        hold_times = (exec_res.exit_bars - exec_res.entry_bars) * 5.0
        avg_hold_time = round(float(np.mean(hold_times)), 2)
        median_hold_time = round(float(np.median(hold_times)), 2)

        # Sizing
        avg_contract_size = round(float(np.mean(exec_res.contracts)), 2)

        # Stop-loss geometry
        avg_sl_dist = round(float(np.mean(exec_res.sl_dists)), 2)
        pct_midpoint = round(
            float(np.sum(exec_res.used_midpoint_rule) / n_trades) * 100.0, 2
        )
    else:
        total_net_profit = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        total_commissions = 0.0
        profit_factor = 0.0
        overall_win_rate = 0.0
        total_longs = 0
        long_win_rate = 0.0
        avg_long_contracts = 0.0
        total_shorts = 0
        short_win_rate = 0.0
        avg_short_contracts = 0.0
        avg_hold_time = 0.0
        median_hold_time = 0.0
        avg_contract_size = 0.0
        avg_sl_dist = 0.0
        pct_midpoint = 0.0

    # 2. Equity curve analytics (intraday & EOD drawdowns incorporated with initial capital peak)
    mtm_series = pd.Series(exec_res.mtm_equity, index=df.index)
    peak_mtm = np.maximum(cfg.initial_capital, mtm_series.cummax())
    intraday_dd = mtm_series - peak_mtm
    max_intraday_dd = round(float(abs(intraday_dd.min())), 2) if len(intraday_dd) > 0 else 0.0

    # EOD Drawdown & Daily PnL (incorporating Day 1 PnL vs initial capital)
    eod_equity = mtm_series.groupby(mtm_series.index.date).last()
    eod_peak = np.maximum(cfg.initial_capital, eod_equity.cummax())
    eod_dd = eod_equity - eod_peak
    max_eod_dd = round(float(abs(eod_dd.min())), 2) if len(eod_dd) > 0 else 0.0

    if len(eod_equity) > 0:
        first_day_pnl = eod_equity.iloc[0] - cfg.initial_capital
        subsequent_pnl = eod_equity.diff().iloc[1:]
        daily_pnl = pd.concat([pd.Series([first_day_pnl], index=[eod_equity.index[0]]), subsequent_pnl])
        worst_single_day = round(float(daily_pnl.min()), 2)
        best_single_day = round(float(daily_pnl.max()), 2)
    else:
        worst_single_day = 0.0
        best_single_day = 0.0

    # 3. Sharpe, Sortino, Calmar ratios (incorporating Day 1 returns vs initial capital)
    if len(eod_equity) > 0:
        first_day_ret = (eod_equity.iloc[0] - cfg.initial_capital) / cfg.initial_capital
        subsequent_rets = eod_equity.pct_change().iloc[1:]
        daily_rets = pd.concat([pd.Series([first_day_ret], index=[eod_equity.index[0]]), subsequent_rets]).dropna()
    else:
        daily_rets = pd.Series(dtype=np.float64)

    std_rets = float(daily_rets.std()) if len(daily_rets) > 1 else 0.0
    mean_rets = float(daily_rets.mean()) if len(daily_rets) > 0 else 0.0

    if std_rets > 1e-8:
        sharpe_ratio = round((mean_rets / std_rets) * np.sqrt(252), 4)
    else:
        sharpe_ratio = 0.0

    downside_rets = daily_rets[daily_rets < 0]
    if len(downside_rets) > 0:
        downside_std = float(np.sqrt(np.mean(downside_rets**2)))
        if downside_std > 1e-8:
            sortino_ratio = round((mean_rets / downside_std) * np.sqrt(252), 4)
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = float("inf") if mean_rets > 0 else 0.0

    # Calmar ratio
    total_ret = total_net_profit / cfg.initial_capital
    calendar_days = max(1, (df.index[-1].date() - df.index[0].date()).days + 1)
    annualized_return = total_ret * (365.25 / calendar_days)
    max_eod_dd_pct = max_eod_dd / cfg.initial_capital
    if max_eod_dd_pct > 1e-6:
        calmar_ratio = round(annualized_return / max_eod_dd_pct, 4)
    else:
        calmar_ratio = float("inf") if annualized_return > 0 else 0.0

    # 4. Trading frequency
    active_days = len(eod_equity)
    avg_trades_day = round(n_trades / active_days, 2) if active_days > 0 else 0.0

    return {
        "total_trades": n_trades,
        "active_days": active_days,
        "total_net_profit": round(total_net_profit, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "total_commissions": round(total_commissions, 2),
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_eod_drawdown": max_eod_dd,
        "max_intraday_drawdown": max_intraday_dd,
        "worst_single_day_pnl": worst_single_day,
        "best_single_day_pnl": best_single_day,
        "avg_hold_time_minutes": avg_hold_time,
        "median_hold_time_minutes": median_hold_time,
        "avg_contract_size": avg_contract_size,
        "avg_long_contract_size": avg_long_contracts,
        "avg_short_contract_size": avg_short_contracts,
        "peak_concurrent_contracts": exec_res.peak_concurrent_contracts,
        "peak_concurrent_slots": exec_res.peak_concurrent_slots,
        "skipped_by_contract_cap": exec_res.skipped_by_contract_cap,
        "overall_win_rate_pct": overall_win_rate,
        "long_win_rate_pct": long_win_rate,
        "total_long_trades": total_longs,
        "short_win_rate_pct": short_win_rate,
        "total_short_trades": total_shorts,
        "avg_sl_distance_pts": avg_sl_dist,
        "pct_midpoint_rule_trades": pct_midpoint,
        "avg_trades_per_day": avg_trades_day,
    }
